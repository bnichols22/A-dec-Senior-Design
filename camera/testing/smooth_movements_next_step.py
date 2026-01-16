#!/usr/bin/env python3
# ==============================================================
# File: full_tracking_smooth_takeover_A.py
# Purpose:
#   Center-locked tracking with continuous streaming setpoints,
#   but with a SAFE startup takeover to prevent a huge jump:
#     - On boot, read current gimbal angles.
#     - Initialize commanded angles so first command == current pose.
#     - Smooth streaming with slew-rate limiting.
# ==============================================================

import os, sys, time, math, warnings, statistics, ctypes
import cv2
import mediapipe as mp
from collections import deque

# --------- Paths ----------
BASE_DIR = os.path.expanduser('~/senior_design/A-dec-Senior-Design/camera/testing')
os.makedirs(BASE_DIR, exist_ok=True)

LOG_PATH  = os.path.join(BASE_DIR, 'face_track_log.txt')
TEST_LOG  = os.path.join(BASE_DIR, 'Board_Serial_Command_Test.txt')

print(f"# ADEC BASE_DIR = {BASE_DIR}")
print(f"# ADEC TEST_LOG = {TEST_LOG}")

# --------- Camera / vision ----------
CAM_INDEX = 0
FOV_H_DEG = 65.0
FOV_V_DEG = 48.75

# Stable center box (visual / state gating)
STABLE_SCALAR = 0.06
WINDOW_SEC    = 0.6

# Stop SEEKING when within this normalized radial distance from center
STABLE_STOP_SEEKING_THRESHOLD = 0.025

# Stability gates (avoid chasing jitter)
VEL_THRESH_DEG_S  = 2.5
POS_STD_THRESH_PX = 2.5

# Control gains
STEP_FRACTION = 0.15

# Stream setpoints while SEEKING
COMMAND_HZ = 60.0
COMMAND_PERIOD = 1.0 / COMMAND_HZ

# Deadband (deg) to avoid dithering
MIN_STEP_DEG_YAW   = 0.15
MIN_STEP_DEG_PITCH = 0.15
MIN_STEP_DEG_ROLL  = 0.15

# Slew-rate limits (deg/sec) => smoother = lower, faster = higher
MAX_RATE_YAW_DEG_S   = 25.0
MAX_RATE_PITCH_DEG_S = 25.0
MAX_RATE_ROLL_DEG_S  = 25.0

# Optional smoothing on the internal target angle
CMD_ANGLE_EMA_ALPHA = 0.35

# Soft-start takeover: ramp rate limits from low to full over this time
SOFT_START_SEC = 1.0
SOFT_START_RATE_SCALE = 0.25  # start at 25% of max rates

# Sign convention (keep your known-good mapping)
AXIS_SIGN = {"yaw": 1, "pitch": 1, "roll": 1}

# Gimbal / test control
TEST = 0  # 1=log only; 0=send via libsimplebgc.so

LIB_PATH = os.path.expanduser(
    "~/senior_design/A-dec-Senior-Design/camera/testing/SerialLibrary/serialAPI/libsimplebgc.so"
)

DRAW = True
SMOOTH_ALPHA = 0.25
MAX_LOST_FRAMES = 10

# --------- Quiet noisy logs ----------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")
sys.stderr = open(LOG_PATH, "w")

# ---------------- Utilities ----------------
def ema_point(curr, prev, alpha):
    if prev is None:
        return curr
    return (alpha*curr[0] + (1-alpha)*prev[0],
            alpha*curr[1] + (1-alpha)*prev[1])

def ema_scalar(curr, prev, alpha):
    if prev is None:
        return curr
    return alpha*curr + (1-alpha)*prev

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

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
    x, y = pt
    l, t, r, b = box
    return (l <= x <= r) and (t <= y <= b)

class TimedHist:
    def __init__(self, win_sec):
        self.win = win_sec
        self.buf = deque()
    def add(self, t, v):
        self.buf.append((t, v))
        self._trim(t)
    def values(self):
        return [v for _, v in self.buf]
    def clear(self):
        self.buf.clear()
    def _trim(self, now):
        cut = now - self.win
        while self.buf and self.buf[0][0] < cut:
            self.buf.popleft()

# ----------------------------------------------------------------------
# SBGC shim (ctypes)
# ----------------------------------------------------------------------
_bgc_lib = None
_bgc_initialized = False

_baseline_yaw_deg   = 0.0
_baseline_pitch_deg = 0.0
_baseline_roll_deg  = 0.0

def init_sbgc():
    """Load SBGC shim, init board, capture baseline."""
    global _bgc_lib, _bgc_initialized
    global _baseline_yaw_deg, _baseline_pitch_deg, _baseline_roll_deg

    if TEST == 1:
        print("# TEST mode: not loading SBGC library.")
        _bgc_lib = None
        _bgc_initialized = False
        _baseline_yaw_deg = 0.0
        _baseline_pitch_deg = 0.0
        _baseline_roll_deg = 0.0
        return

    try:
        _bgc_lib = ctypes.CDLL(LIB_PATH)
        print(f"# Loaded SBGC library from {LIB_PATH}")
    except OSError as e:
        print(f"# ERROR loading {LIB_PATH}: {e}")
        _bgc_lib = None
        _bgc_initialized = False
        return

    _bgc_lib.bgc_init.argtypes = []
    _bgc_lib.bgc_init.restype  = ctypes.c_int

    # order: (roll, pitch, yaw)
    _bgc_lib.bgc_control_angles.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float]
    _bgc_lib.bgc_control_angles.restype  = ctypes.c_int

    try:
        _bgc_lib.bgc_get_angles.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        _bgc_lib.bgc_get_angles.restype = ctypes.c_int
    except AttributeError:
        _bgc_lib.bgc_get_angles = None

    try:
        _bgc_lib.bgc_beep_once.argtypes = []
        _bgc_lib.bgc_beep_once.restype = ctypes.c_int
    except AttributeError:
        pass

    rc = _bgc_lib.bgc_init()
    if rc != 0:
        print(f"# ERROR: bgc_init() returned {rc}")
        _bgc_initialized = False
        return

    # Baseline = "zero-based origin"
    if _bgc_lib.bgc_get_angles is not None:
        yaw = ctypes.c_float()
        pitch = ctypes.c_float()
        roll = ctypes.c_float()
        rc2 = _bgc_lib.bgc_get_angles(ctypes.byref(yaw), ctypes.byref(pitch), ctypes.byref(roll))
        if rc2 == 0:
            _baseline_yaw_deg   = float(yaw.value)
            _baseline_pitch_deg = float(pitch.value)
            _baseline_roll_deg  = float(roll.value)
            print(f"# Baseline captured: yaw={_baseline_yaw_deg:.2f}, pitch={_baseline_pitch_deg:.2f}, roll={_baseline_roll_deg:.2f}")
        else:
            print(f"# WARN: bgc_get_angles() returned {rc2}, baseline=0")
            _baseline_yaw_deg = _baseline_pitch_deg = _baseline_roll_deg = 0.0
    else:
        print("# WARN: bgc_get_angles unavailable, baseline=0")
        _baseline_yaw_deg = _baseline_pitch_deg = _baseline_roll_deg = 0.0

    if hasattr(_bgc_lib, "bgc_beep_once"):
        try:
            _bgc_lib.bgc_beep_once()
        except Exception:
            pass

    _bgc_initialized = True
    print("# SBGC ready.")

def get_board_angles():
    """Read current board angles (yaw,pitch,roll). Returns None if unavailable."""
    if TEST == 1:
        return (0.0, 0.0, 0.0)
    if _bgc_lib is None or not _bgc_initialized:
        return None
    if not hasattr(_bgc_lib, "bgc_get_angles") or _bgc_lib.bgc_get_angles is None:
        return None

    yaw = ctypes.c_float()
    pitch = ctypes.c_float()
    roll = ctypes.c_float()
    rc = _bgc_lib.bgc_get_angles(ctypes.byref(yaw), ctypes.byref(pitch), ctypes.byref(roll))
    if rc != 0:
        return None
    return (float(yaw.value), float(pitch.value), float(roll.value))

def write_test_line(d_yaw, d_pitch, d_roll, abs_yaw, abs_pitch, abs_roll, board_yaw, board_pitch, board_roll):
    try:
        with open(TEST_LOG, 'a', buffering=1) as f:
            f.write(
                f"T={time.time():.3f} "
                f"dR={d_roll:+.3f} dP={d_pitch:+.3f} dY={d_yaw:+.3f} | "
                f"absR={abs_roll:+.3f} absP={abs_pitch:+.3f} absY={abs_yaw:+.3f} | "
                f"boardR={board_roll:+.3f} boardP={board_pitch:+.3f} boardY={board_yaw:+.3f}\n"
            )
            f.flush()
            os.fsync(f.fileno())
        return True
    except Exception as e:
        print(f"# TEST_FILE_ERROR: {e}")
        return False

def send_or_log_angles(d_yaw_deg, d_pitch_deg, d_roll_deg, abs_yaw_deg, abs_pitch_deg, abs_roll_deg):
    """Send zero-based absolute angles; board angles = baseline + abs."""
    board_yaw   = _baseline_yaw_deg   + abs_yaw_deg
    board_pitch = _baseline_pitch_deg + abs_pitch_deg
    board_roll  = _baseline_roll_deg  + abs_roll_deg

    if TEST == 1:
        return write_test_line(d_yaw_deg, d_pitch_deg, d_roll_deg,
                               abs_yaw_deg, abs_pitch_deg, abs_roll_deg,
                               board_yaw, board_pitch, board_roll)

    if _bgc_lib is None or not _bgc_initialized:
        print("# ERROR: SBGC not initialized.")
        return False

    rc = _bgc_lib.bgc_control_angles(
        ctypes.c_float(board_roll),
        ctypes.c_float(board_pitch),
        ctypes.c_float(board_yaw),
    )
    if rc != 0:
        print(f"# SEND_ERROR: bgc_control_angles() returned {rc}")
        return False
    return True

# ---------------- Main ----------------
def main():
    init_sbgc()

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("ERROR: Unable to open camera", CAM_INDEX)
        sys.exit(1)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Clear log
    try:
        with open(TEST_LOG, "w") as f:
            f.write("# Smooth takeover A — fresh run\n")
            f.write(f"# Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    except Exception as e:
        print(f"# ERROR clearing test log: {e}")

    # --- SAFE TAKEOVER INITIALIZATION ---
    # Ensure first commanded setpoint equals current gimbal pose to avoid snapping.
    cmd_yaw_deg = 0.0
    cmd_pitch_deg = 0.0
    cmd_roll_deg = 0.0

    last_sent_yaw = 0.0
    last_sent_pitch = 0.0
    last_sent_roll = 0.0

    # If we can read board angles, set zero-based command to match current pose:
    # abs = board - baseline
    board_angles = get_board_angles()
    if board_angles is not None:
        by, bp, br = board_angles
        cmd_yaw_deg   = by - _baseline_yaw_deg
        cmd_pitch_deg = bp - _baseline_pitch_deg
        cmd_roll_deg  = br - _baseline_roll_deg
        last_sent_yaw, last_sent_pitch, last_sent_roll = cmd_yaw_deg, cmd_pitch_deg, cmd_roll_deg
        # Send one "hold" command immediately (should be near no-op)
        send_or_log_angles(0.0, 0.0, 0.0, cmd_yaw_deg, cmd_pitch_deg, cmd_roll_deg)
        print("# Takeover: initialized commanded angles to current pose (no jump).")
    else:
        print("# Takeover: could not read current angles; using 0,0,0 (may jump).")

    sm_cmd_yaw = cmd_yaw_deg
    sm_cmd_pitch = cmd_pitch_deg
    sm_cmd_roll = cmd_roll_deg

    t0 = time.time()
    prev_smoothed = None
    prev_roll_deg = None
    prev_time = None
    anchor = None
    lost = 0

    LOCKED, SEEKING = 0, 1
    state = LOCKED

    pos_x = TimedHist(WINDOW_SEC)
    pos_y = TimedHist(WINDOW_SEC)
    vel_h = TimedHist(WINDOW_SEC)

    last_send_time = time.time()

    print("# T dR dP dY sent state r")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("# WARN: frame grab failed")
            break
        now = time.time()
        h, w = frame.shape[:2]

        if anchor is None:
            anchor = (w / 2.0, h / 2.0)

        # Detect mouth centroid + roll
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

        if centroid is None:
            lost += 1
            if lost > MAX_LOST_FRAMES:
                prev_smoothed = None
                prev_roll_deg = None
                prev_time = None
            T = now - t0
            print(f"{T:.3f} +0.000 +0.000 +0.000 0 {state} r=?")
            if DRAW:
                cv2.imshow("Smooth Takeover A", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            continue
        lost = 0

        smoothed = ema_point(centroid, prev_smoothed, SMOOTH_ALPHA)

        if prev_time is None:
            prev_time = now

        box = build_stable_box(anchor, w, h, STABLE_SCALAR)
        inside = inside_box(smoothed, box)

        dx_center = smoothed[0] - anchor[0]
        dy_center = smoothed[1] - anchor[1]
        norm_dx = dx_center / (w / 2.0)
        norm_dy = dy_center / (h / 2.0)
        radial_norm = math.hypot(norm_dx, norm_dy)
        within_stop = (radial_norm <= STABLE_STOP_SEEKING_THRESHOLD)

        dt = max(1e-6, now - prev_time)
        if prev_smoothed is None:
            speed = 0.0
        else:
            dx_px_dt = smoothed[0] - prev_smoothed[0]
            dy_px_dt = smoothed[1] - prev_smoothed[1]
            dvx, dvy = pixels_to_deg(dx_px_dt, dy_px_dt, w, h, FOV_H_DEG, FOV_V_DEG)
            speed = math.hypot(dvx, dvy) / dt

        prev_time = now
        prev_smoothed = smoothed

        d_yaw = d_pitch = d_roll = 0.0
        sent = 0

        # Soft-start rate scaling
        elapsed = now - t0
        if elapsed <= SOFT_START_SEC:
            frac = elapsed / max(1e-6, SOFT_START_SEC)
            rate_scale = SOFT_START_RATE_SCALE + (1.0 - SOFT_START_RATE_SCALE) * frac
        else:
            rate_scale = 1.0

        max_rate_yaw   = MAX_RATE_YAW_DEG_S   * rate_scale
        max_rate_pitch = MAX_RATE_PITCH_DEG_S * rate_scale
        max_rate_roll  = MAX_RATE_ROLL_DEG_S  * rate_scale

        # State machine
        if state == LOCKED:
            if not inside:
                state = SEEKING
                pos_x.clear(); pos_y.clear(); vel_h.clear()
                if roll_now_deg is not None and prev_roll_deg is None:
                    prev_roll_deg = roll_now_deg

        else:  # SEEKING
            pos_x.add(now, smoothed[0])
            pos_y.add(now, smoothed[1])
            vel_h.add(now, speed)

            xs, ys = pos_x.values(), pos_y.values()
            pos_std = 999.0
            if len(xs) >= 6 and len(ys) >= 6:
                pos_std = 0.5 * (statistics.pstdev(xs) + statistics.pstdev(ys))

            speeds = vel_h.values()
            vel_med = statistics.median(speeds) if len(speeds) >= 3 else 999.0
            stable_here = (vel_med < VEL_THRESH_DEG_S) and (pos_std < POS_STD_THRESH_PX)

            if within_stop:
                if roll_now_deg is not None:
                    prev_roll_deg = roll_now_deg
                pos_x.clear(); pos_y.clear(); vel_h.clear()
                state = LOCKED
            else:
                if (now - last_send_time) >= COMMAND_PERIOD and stable_here:
                    # Error -> degrees
                    full_d_yaw, full_d_pitch = pixels_to_deg(dx_center, dy_center, w, h, FOV_H_DEG, FOV_V_DEG)

                    d_roll_full = 0.0
                    if roll_now_deg is not None and prev_roll_deg is not None:
                        d_roll_full = roll_now_deg - prev_roll_deg

                    full_d_yaw   = AXIS_SIGN["yaw"]   * full_d_yaw
                    full_d_pitch = AXIS_SIGN["pitch"] * full_d_pitch
                    d_roll_full  = AXIS_SIGN["roll"]  * d_roll_full

                    # Micro-step
                    d_yaw   = full_d_yaw   * STEP_FRACTION
                    d_pitch = full_d_pitch * STEP_FRACTION
                    d_roll  = d_roll_full  * STEP_FRACTION

                    # Deadband
                    if abs(d_yaw)   < MIN_STEP_DEG_YAW:   d_yaw = 0.0
                    if abs(d_pitch) < MIN_STEP_DEG_PITCH: d_pitch = 0.0
                    if abs(d_roll)  < MIN_STEP_DEG_ROLL:  d_roll = 0.0

                    if (d_yaw != 0.0) or (d_pitch != 0.0) or (d_roll != 0.0):
                        # Update internal target
                        cmd_yaw_deg   += d_yaw
                        cmd_pitch_deg += d_pitch
                        cmd_roll_deg  += d_roll

                        # Smooth target a bit
                        sm_cmd_yaw   = ema_scalar(cmd_yaw_deg,   sm_cmd_yaw,   CMD_ANGLE_EMA_ALPHA)
                        sm_cmd_pitch = ema_scalar(cmd_pitch_deg, sm_cmd_pitch, CMD_ANGLE_EMA_ALPHA)
                        sm_cmd_roll  = ema_scalar(cmd_roll_deg,  sm_cmd_roll,  CMD_ANGLE_EMA_ALPHA)

                        # Slew limit the actual sent setpoint (deg per period)
                        max_step_yaw   = max_rate_yaw   * COMMAND_PERIOD
                        max_step_pitch = max_rate_pitch * COMMAND_PERIOD
                        max_step_roll  = max_rate_roll  * COMMAND_PERIOD

                        send_yaw   = clamp(sm_cmd_yaw,   last_sent_yaw   - max_step_yaw,   last_sent_yaw   + max_step_yaw)
                        send_pitch = clamp(sm_cmd_pitch, last_sent_pitch - max_step_pitch, last_sent_pitch + max_step_pitch)
                        send_roll  = clamp(sm_cmd_roll,  last_sent_roll  - max_step_roll,  last_sent_roll  + max_step_roll)

                        if send_or_log_angles(d_yaw, d_pitch, d_roll, send_yaw, send_pitch, send_roll):
                            last_send_time = now
                            last_sent_yaw, last_sent_pitch, last_sent_roll = send_yaw, send_pitch, send_roll
                            sent = 1

        T = now - t0
        print(f"{T:.3f} {d_roll:+.3f} {d_pitch:+.3f} {d_yaw:+.3f} {sent} {state} r={radial_norm:.3f}")

        if DRAW:
            l, t_, r, b = map(int, box)
            cv2.rectangle(frame, (l, t_), (r, b), (40, 220, 40), 1)
            cv2.drawMarker(frame, (int(anchor[0]), int(anchor[1])), (0, 200, 0),
                           cv2.MARKER_CROSS, 12, 2)
            cv2.circle(frame, (int(smoothed[0]), int(smoothed[1])), 4, (0, 0, 255), -1)

            state_txt = "LOCKED" if state == LOCKED else "SEEKING"
            cv2.putText(frame, f"state:{state_txt}", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40,220,40), 2, cv2.LINE_AA)
            cv2.putText(frame, f"r={radial_norm:.3f}", (10, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40,220,40), 2, cv2.LINE_AA)

            cv2.imshow("Smooth Takeover A", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Stopped.")

if __name__ == "__main__":
    main()
