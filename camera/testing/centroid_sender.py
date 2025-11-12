#!/usr/bin/env python3
# ==============================================================
# File: centroid_track.py
# Purpose:
#   Moving-anchor tracker with explicit state machine:
#   - LOCKED: box at current anchor; when centroid exits, enter SEEKING
#   - SEEKING: wait until new position is "stable", then emit ONE delta
#              (old_anchor -> new_stable_centroid), re-anchor, back to LOCKED
#   Console prints show live prospective deltas in SEEKING and the actual
#   deltas when a command is emitted.
#
# SBGC transport:
#   - Protocol v1 framing: [0x3E][CMD][LEN][HDR_SUM][PAYLOAD...][PAY_SUM]
#   - CMD_CONTROL (ID 67) in ANGLE mode; ANGLE units are 14-bit (16384/360 deg)
#   - Payload axis order is ROLL, PITCH, YAW (int16 LE)
# ==============================================================

import os, sys, time, math, warnings, statistics
import cv2
import mediapipe as mp
from collections import deque

# --------- Paths: force to your repo folder ----------
BASE_DIR = os.path.expanduser('~/senior_design/A-dec-Senior-Design/camera/testing')
os.makedirs(BASE_DIR, exist_ok=True)

LOG_PATH  = os.path.join(BASE_DIR, 'face_track_log.txt')
TEST_LOG  = os.path.join(BASE_DIR, 'Board_Serial_Command_Test.txt')

print(f"# ADEC BASE_DIR = {BASE_DIR}")
print(f"# ADEC TEST_LOG = {TEST_LOG}")

# --------- Vision / tracker config ----------
CAM_INDEX = 0
FOV_H_DEG = 95.0
FOV_V_DEG = 60.0

# Stable box: fraction of HALF-frame sizes
STABLE_SCALAR   = 0.06    # tighten to 0.05 or 0.04 if too tolerant
TIME_TO_STABLE  = 0.45    # seconds new place must remain calm to re-anchor
WINDOW_SEC      = 0.6     # history window for stability metrics

# Stability gates (used in SEEKING)
VEL_THRESH_DEG_S  = 2.5   # median angular speed threshold
POS_STD_THRESH_PX = 2.5   # positional stddev threshold

# Send gating
SEND_TIME_LIMITER = 0.75  # min seconds between sends (rate limit)
MIN_STEP_DEG_YAW   = 0.5
MIN_STEP_DEG_PITCH = 0.5
MIN_STEP_DEG_ROLL  = 1.0

# Sign convention to match gimbal axes (tune as needed)
AXIS_SIGN = {"yaw": +1, "pitch": -1, "roll": +1}

# Serial / test control
TEST = 1                 # 1=write frames to TEST_LOG; 0=send over serial
SBGC_PORT = "/dev/ttyACM0"
SBGC_BAUD = 115200
SBGC_TIMEOUT_S = 0.05

DRAW = True
SMOOTH_ALPHA = 0.25
MAX_LOST_FRAMES = 10

# --------- Quiet noisy TF/MP logs ----------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")
sys.stderr = open(LOG_PATH, "w")

# ---------------- Utilities ----------------
def ema_point(curr, prev, alpha):
    if prev is None: return curr
    return (alpha*curr[0] + (1-alpha)*prev[0],
            alpha*curr[1] + (1-alpha)*prev[1])

def pixels_to_deg(dx_px, dy_px, w, h, fov_h, fov_v):
    half_w, half_h = w/2.0, h/2.0
    return (dx_px/half_w)*(fov_h/2.0), (dy_px/half_h)*(fov_v/2.0)

def angle_deg(p1, p2):
    return math.degrees(math.atan2(p2[1]-p1[1], p2[0]-p1[0]))

def build_stable_box(anchor_xy, w, h, scalar):
    cx, cy = anchor_xy
    half_w = scalar * (w/2.0)
    half_h = scalar * (h/2.0)
    return (cx - half_w, cy - half_h, cx + half_w, cy + half_h)

def inside_box(pt, box):
    x, y = pt; l, t, r, b = box
    return (l <= x <= r) and (t <= y <= b)

class TimedHist:
    def __init__(self, win_sec):
        self.win = win_sec
        self.buf = deque()
    def add(self, t, v): self.buf.append((t, v)); self._trim(t)
    def values(self): return [v for _, v in self.buf]
    def clear(self): self.buf.clear()
    def _trim(self, now):
        cut = now - self.win
        while self.buf and self.buf[0][0] < cut:
            self.buf.popleft()

# ---------------- SimpleBGC v1 framing ----------------
SBGC_START_V1    = 0x3E
SBGC_CMD_CONTROL = 67  # 0x43 (CMD_CONTROL)

def angle_deg_to_14bit(deg: float) -> int:
    """
    Convert degrees to SimpleBGC 'ANGLE' units (int16),
    where 1 LSB = 360/16384 = 0.02197265625 deg.
    """
    units_per_deg = 16384.0 / 360.0  # ≈ 45.511111...
    val = int(round(deg * units_per_deg))
    # clamp to int16
    return max(-32768, min(32767, val))

def sbgc_v1_header_checksum(cmd_id: int, payload_len: int) -> int:
    # sum of [CMD_ID + LEN] mod 256
    return (cmd_id + payload_len) & 0xFF

def sbgc_v1_payload_checksum(payload: bytes) -> int:
    # sum of all payload bytes mod 256
    return (sum(payload) & 0xFF)

def build_cmd_control_angles_v1(d_yaw_deg: float,
                                d_pitch_deg: float,
                                d_roll_deg: float) -> bytes:
    """
    Build CMD_CONTROL (ID 67) in ANGLE mode for all axes.
    ANGLE array order is ROLL, PITCH, YAW (int16, little-endian).
    We leave SPEED fields = 0 to use board profile speeds.
    """
    # Control mode: ANGLE for all axes. (Mode value is 2 in most firmwares.)
    CONTROL_MODE_ANGLE = 2

    # Convert degrees -> 14-bit angle units
    a_roll  = angle_deg_to_14bit(d_roll_deg)
    a_pitch = angle_deg_to_14bit(d_pitch_deg)
    a_yaw   = angle_deg_to_14bit(d_yaw_deg)

    # Legacy/common payload layout:
    # [MODE(1u), FLAGS(1u),
    #  SPEED_R(2s), SPEED_P(2s), SPEED_Y(2s),
    #  ANGLE_R(2s), ANGLE_P(2s), ANGLE_Y(2s)]
    payload = bytearray()
    payload += bytes([CONTROL_MODE_ANGLE])                  # MODE
    payload += bytes([0x00])                                # FLAGS
    payload += (0).to_bytes(2, 'little', signed=True)       # SPEED_R
    payload += (0).to_bytes(2, 'little', signed=True)       # SPEED_P
    payload += (0).to_bytes(2, 'little', signed=True)       # SPEED_Y
    payload += a_roll.to_bytes(2, 'little', signed=True)    # ANGLE_R
    payload += a_pitch.to_bytes(2, 'little', signed=True)   # ANGLE_P
    payload += a_yaw.to_bytes(2, 'little', signed=True)     # ANGLE_Y

    # Frame (Protocol v1)
    cmd = SBGC_CMD_CONTROL
    length = len(payload)
    header = bytearray([SBGC_START_V1, cmd, length, sbgc_v1_header_checksum(cmd, length)])
    frame  = header + payload + bytearray([sbgc_v1_payload_checksum(payload)])
    return bytes(frame)

# ------------- Output: send or write to file -------------
def write_test_line(frame_bytes, yaw_deg, pitch_deg, roll_deg):
    try:
        with open(TEST_LOG, 'a', buffering=1) as f:
            f.write(f"T={time.time():.3f} R={roll_deg:+.2f} P={pitch_deg:+.2f} Y={yaw_deg:+.2f} | ")
            f.write("FRAME=" + " ".join(f"{b:02X}" for b in frame_bytes) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return True
    except Exception as e:
        print(f"# TEST_FILE_ERROR: {e}")
        return False

def send_or_log_frame(yaw_deg, pitch_deg, roll_deg):
    frame_bytes = build_cmd_control_angles_v1(yaw_deg, pitch_deg, roll_deg)
    if TEST == 1:
        return write_test_line(frame_bytes, yaw_deg, pitch_deg, roll_deg)
    try:
        import serial
        with serial.Serial(SBGC_PORT, SBGC_BAUD, timeout=SBGC_TIMEOUT_S) as ser:
            ser.write(frame_bytes)
            # Optional: read a short response window for CMD_CONFIRM
            # resp = ser.read(64)
        return True
    except Exception as e:
        print(f"# SEND_ERROR: {e}")
        return False

# ---------------- Main loop (stateful) ----------------
def main():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("ERROR: Unable to open camera", CAM_INDEX); sys.exit(1)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # State
    t0 = time.time()
    prev_smoothed = None
    prev_roll_deg = None
    prev_time = None
    anchor = None
    last_send_time = 0.0
    lost = 0

    LOCKED, SEEKING = 0, 1
    state = LOCKED
    stable_timer = None

    pos_x = TimedHist(WINDOW_SEC)
    pos_y = TimedHist(WINDOW_SEC)
    vel_h = TimedHist(WINDOW_SEC)

    # Clear the test file at the start of every run
    try:
        with open(TEST_LOG, "w") as f:
            f.write("# SimpleBGC framed commands (TEST mode) — fresh run\n")
            f.write(f"# Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        print(f"# Cleared {TEST_LOG} for a new session.")
    except Exception as e:
        print(f"# ERROR clearing test log: {e}")

    print("# T roll pitch yaw can_send state (prospective during SEEKING; actual on SEND)")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("# WARN: frame grab failed"); break
        now = time.time()
        h, w = frame.shape[:2]

        # --- detect mouth centroid + roll
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        centroid = None
        roll_now_deg = None

        if res.multi_face_landmarks:
            fl = res.multi_face_landmarks[0]
            mouth_idxs = [13, 14, 61, 291]
            pts = []
            for idx in mouth_idxs:
                x = int(fl.landmark[idx].x * w)
                y = int(fl.landmark[idx].y * h)
                pts.append((x, y))
            if pts:
                cx = sum(p[0] for p in pts) / len(pts)
                cy = sum(p[1] for p in pts) / len(pts)
                centroid = (cx, cy)
                p_left  = (int(fl.landmark[61].x*w),  int(fl.landmark[61].y*h))
                p_right = (int(fl.landmark[291].x*w), int(fl.landmark[291].y*h))
                roll_now_deg = angle_deg(p_left, p_right)

        # Lost handling
        if centroid is None:
            lost += 1
            if lost > MAX_LOST_FRAMES:
                prev_smoothed = None
                prev_roll_deg = None
                prev_time = None
            T = now - t0
            print(f"{T:.3f} +0.000 +0.000 +0.000 0 {state}")
            if DRAW:
                cv2.imshow("Centroid Tracker (stateful)", frame)
                if cv2.waitKey(1) & 0xFF == 27: break
            continue
        lost = 0

        # Smooth centroid
        smoothed = ema_point(centroid, prev_smoothed, SMOOTH_ALPHA)

        # First-time anchor + timing init
        if anchor is None:
            anchor = smoothed
            if roll_now_deg is not None:
                prev_roll_deg = roll_now_deg
        if prev_time is None:
            prev_time = now

        # Build LOCKED box (at anchor)
        box = build_stable_box(anchor, w, h, STABLE_SCALAR)
        inside = inside_box(smoothed, box)

        # Compute dt-based velocity for stability
        dt = max(1e-6, now - prev_time)
        if prev_smoothed is None:
            speed = 0.0
        else:
            dx_px = smoothed[0] - prev_smoothed[0]
            dy_px = smoothed[1] - prev_smoothed[1]
            dvx, dvy = pixels_to_deg(dx_px, dy_px, w, h, FOV_H_DEG, FOV_V_DEG)
            speed = math.hypot(dvx, dvy) / dt  # deg/s
        prev_time = now
        prev_smoothed = smoothed

        # Default console outputs (prospective values)
        d_yaw = d_pitch = d_roll = 0.0
        can_send_flag = 0
        sent_this_frame = False

        # ---------- STATE MACHINE ----------
        if state == 0:  # LOCKED
            if not inside:
                state = 1  # SEEKING
                stable_timer = None
                pos_x.clear(); pos_y.clear(); vel_h.clear()
            # In LOCKED we do not move the box nor send anything
        else:  # SEEKING
            pos_x.add(now, smoothed[0]); pos_y.add(now, smoothed[1])
            vel_h.add(now, speed)

            xs, ys = pos_x.values(), pos_y.values()
            pos_std = 999.0
            if len(xs) >= 6 and len(ys) >= 6:
                pos_std = 0.5 * (statistics.pstdev(xs) + statistics.pstdev(ys))
            speeds = vel_h.values()
            vel_med = statistics.median(speeds) if len(speeds) >= 3 else 999.0
            is_stable_here = (vel_med < VEL_THRESH_DEG_S) and (pos_std < POS_STD_THRESH_PX)

            # Prospective deltas (old anchor -> current smoothed)
            dx_px = smoothed[0] - anchor[0]
            dy_px = smoothed[1] - anchor[1]
            d_yaw, d_pitch = pixels_to_deg(dx_px, dy_px, w, h, FOV_H_DEG, FOV_V_DEG)
            d_roll = 0.0
            if roll_now_deg is not None and prev_roll_deg is not None:
                d_roll = roll_now_deg - prev_roll_deg

            # Apply axis signs (for both print and send)
            d_yaw   = AXIS_SIGN["yaw"]   * d_yaw
            d_pitch = AXIS_SIGN["pitch"] * d_pitch
            d_roll  = AXIS_SIGN["roll"]  * d_roll

            # Stability dwell timer
            if is_stable_here:
                if stable_timer is None:
                    stable_timer = now
                stable_for = now - stable_timer
            else:
                stable_timer = None
                stable_for = 0.0

            can_time = (now - last_send_time) >= SEND_TIME_LIMITER
            can_send_flag = 1 if (stable_timer is not None and stable_for >= TIME_TO_STABLE and can_time) else 0

            # If we can send now, apply min-steps, emit, and re-anchor
            if can_send_flag:
                if abs(d_yaw)   < MIN_STEP_DEG_YAW:   d_yaw = 0.0
                if abs(d_pitch) < MIN_STEP_DEG_PITCH: d_pitch = 0.0
                if abs(d_roll)  < MIN_STEP_DEG_ROLL:  d_roll = 0.0

                # Only send if something nonzero remains
                if (d_yaw != 0.0) or (d_pitch != 0.0) or (d_roll != 0.0):
                    ok = send_or_log_frame(d_yaw, d_pitch, d_roll)
                    if ok:
                        last_send_time = now
                    sent_this_frame = True

                # Re-anchor regardless (we’ve “accepted” the new stable pose)
                anchor = smoothed
                if roll_now_deg is not None:
                    prev_roll_deg = roll_now_deg

                pos_x.clear(); pos_y.clear(); vel_h.clear()
                stable_timer = None
                state = 0  # back to LOCKED

        # -------- Telemetry --------
        T = now - t0
        if sent_this_frame or state == 1:
            print(f"{T:.3f} {d_roll:+.3f} {d_pitch:+.3f} {d_yaw:+.3f} {can_send_flag} {state}")
        else:
            print(f"{T:.3f} +0.000 +0.000 +0.000 0 {state}")

        # -------- UI --------
        if DRAW:
            l, t_, r, b = map(int, box)
            cv2.rectangle(frame, (l, t_), (r, b), (40, 220, 40), 1)
            cv2.drawMarker(frame, (int(anchor[0]), int(anchor[1])), (0, 200, 0),
                           cv2.MARKER_CROSS, 12, 2)
            cv2.circle(frame, (int(smoothed[0]), int(smoothed[1])), 4, (0, 0, 255), -1)

            state_txt = "LOCKED" if state == 0 else "SEEKING"
            cv2.putText(frame, f"state:{state_txt}", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40,220,40), 2, cv2.LINE_AA)
            if state == 1:
                cv2.putText(frame, f"dR:{d_roll:+.2f} dP:{d_pitch:+.2f} dY:{d_yaw:+.2f}",
                            (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40,220,40), 2, cv2.LINE_AA)
            cv2.imshow("Centroid Tracker (stateful)", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Stopped.")

if __name__ == "__main__":
    main()
