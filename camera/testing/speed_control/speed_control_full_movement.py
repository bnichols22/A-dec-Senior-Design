#!/usr/bin/env python3
# ==============================================================
# File: full_tracking_speed.py
# Purpose:
#   Face tracker that drives gimbal using SPEED mode (deg/s),
#   but in SINGLE-MOVEMENT "pulses":
#     - When error is outside stable box, latch ONE move command:
#         (yaw_dps, pitch_dps, duration)
#     - Execute for that duration (with ramp-in/out for smoothness)
#     - Stop, re-check, and if still off, do another pulse.
#
# Why this helps:
#   - Still avoids the initial snap-to-zero bug (no absolute angle setpoints)
#   - Handles larger distances by letting one pulse run longer / faster
#   - If it overshoots, the next pulse corrects as a "new movement"
#
# Transport:
#   SimpleBGC SerialAPI shim must export:
#     - bgc_init()
#     - bgc_control_speeds(roll_dps, pitch_dps, yaw_dps)
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

# --------- Vision / tracker config ----------
CAM_INDEX = 0
FOV_H_DEG = 65.0
FOV_V_DEG = 48.75

STABLE_SCALAR = 0.06
WINDOW_SEC = 0.6
STABLE_STOP_SEEKING_THRESHOLD = 0.025

VEL_THRESH_DEG_S  = 2.5
POS_STD_THRESH_PX = 2.5

# ----------------------------
# SINGLE-MOVE PULSE TUNING
# ----------------------------

# Convert "deg error" -> desired speed (deg/s).
# Higher = faster, but more overshoot.
KP_YAW_DPS_PER_DEG   = 2.2
KP_PITCH_DPS_PER_DEG = 2.2

# Max speeds (deg/s)
MAX_DPS_YAW   = 70.0
MAX_DPS_PITCH = 70.0
MAX_DPS_ROLL  = 40.0

# Deadband (deg). If within this, we command 0 movement.
DEADBAND_DEG_YAW   = 0.35
DEADBAND_DEG_PITCH = 0.35

# Pulse duration bounds (seconds)
MIN_PULSE_S = 0.10   # don't do tiny flickers
MAX_PULSE_S = 0.90   # allow big moves, but cap so it can re-evaluate

# Smooth ramp time (seconds) at start/end of each pulse
RAMP_S = 0.12

# Command streaming rate (Hz) while pulse is active
COMMAND_HZ = 80.0
COMMAND_PERIOD = 1.0 / COMMAND_HZ

# Optional smoothing on *computed* speeds (small helps jitter)
CMD_SPEED_EMA_ALPHA = 0.20

AXIS_SIGN = {"yaw": 1, "pitch": 1, "roll": 1}

TEST = 0  # 1=log only; 0=send to gimbal
LIB_PATH = os.path.expanduser(
    "~/senior_design/A-dec-Senior-Design/camera/testing/SerialLibrary/serialAPI/libsimplebgc.so"
)

DRAW = True
SMOOTH_ALPHA = 0.25
MAX_LOST_FRAMES = 10

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")
sys.stderr = open(LOG_PATH, "w")


# ---------------- Utilities ----------------
def ema_point(curr, prev, alpha):
    if prev is None:
        return curr
    return (alpha * curr[0] + (1 - alpha) * prev[0],
            alpha * curr[1] + (1 - alpha) * prev[1])

def ema_scalar(curr, prev, alpha):
    if prev is None:
        return curr
    return alpha * curr + (1 - alpha) * prev

def clamp(val, lo, hi):
    return max(lo, min(hi, val))

def pixels_to_deg(dx_px, dy_px, w, h, fov_h, fov_v):
    half_w, half_h = w / 2.0, h / 2.0
    yaw_deg   = (dx_px / half_w) * (fov_h / 2.0)
    pitch_deg = (dy_px / half_h) * (fov_v / 2.0)
    return yaw_deg, pitch_deg

def build_stable_box(center_xy, w, h, scalar):
    cx, cy = center_xy
    half_w = scalar * (w / 2.0)
    half_h = scalar * (h / 2.0)
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

def ramp_factor(now, t_start, t_end, ramp_s):
    """
    0..1 ramp in/out for smooth pulses.
    """
    if t_end <= t_start:
        return 0.0
    # ramp in
    a = (now - t_start) / max(1e-6, ramp_s)
    # ramp out
    b = (t_end - now) / max(1e-6, ramp_s)
    return clamp(min(a, b, 1.0), 0.0, 1.0)


# ----------------------------------------------------------------------
# SBGC shim bindings (ctypes)
# ----------------------------------------------------------------------
_bgc_lib = None
_bgc_initialized = False

def init_sbgc():
    global _bgc_lib, _bgc_initialized

    if TEST == 1:
        print("# TEST mode: not loading SBGC library.")
        _bgc_lib = None
        _bgc_initialized = False
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
    _bgc_lib.bgc_init.restype = ctypes.c_int

    if not hasattr(_bgc_lib, "bgc_control_speeds"):
        print("# ERROR: libsimplebgc.so does not export bgc_control_speeds(). Rebuild the .so.")
        _bgc_initialized = False
        return

    _bgc_lib.bgc_control_speeds.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float]
    _bgc_lib.bgc_control_speeds.restype = ctypes.c_int

    rc = _bgc_lib.bgc_init()
    if rc != 0:
        print(f"# ERROR: bgc_init() returned {rc}")
        _bgc_initialized = False
        return

    _bgc_initialized = True
    print("# SBGC init OK (SPEED-mode, pulse movement).")

def send_or_log_speeds(roll_dps, pitch_dps, yaw_dps):
    if TEST == 1:
        try:
            with open(TEST_LOG, 'a', buffering=1) as f:
                f.write(f"T={time.time():.3f} roll_dps={roll_dps:+.2f} pitch_dps={pitch_dps:+.2f} yaw_dps={yaw_dps:+.2f}\n")
            return True
        except Exception as e:
            print(f"# TEST_FILE_ERROR: {e}")
            return False

    if _bgc_lib is None or not _bgc_initialized:
        print("# ERROR: SBGC shim not initialized.")
        return False

    rc = _bgc_lib.bgc_control_speeds(
        ctypes.c_float(roll_dps),
        ctypes.c_float(pitch_dps),
        ctypes.c_float(yaw_dps),
    )
    if rc != 0:
        print(f"# SEND_ERROR: bgc_control_speeds() returned {rc}")
        return False

    return True


# ---------------- Main loop ----------------
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

    # Clear test log
    try:
        with open(TEST_LOG, "w") as f:
            f.write("# SimpleBGC commands (SPEED mode pulses) — fresh run\n")
            f.write(f"# Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    except Exception:
        pass

    t0 = time.time()
    prev_smoothed = None
    prev_time = None
    lost = 0

    anchor = None

    LOCKED, SEEKING = 0, 1
    state = LOCKED

    pos_x = TimedHist(WINDOW_SEC)
    pos_y = TimedHist(WINDOW_SEC)
    vel_h = TimedHist(WINDOW_SEC)

    last_send_time = 0.0

    # --- Pulse state ---
    pulse_active = False
    pulse_end_time = 0.0
    pulse_start_time = 0.0
    pulse_yaw_dps_cmd = 0.0
    pulse_pitch_dps_cmd = 0.0
    pulse_roll_dps_cmd = 0.0

    # optional smoothing on commanded speeds
    smooth_yaw_dps = None
    smooth_pitch_dps = None
    smooth_roll_dps = None

    print("# T yaw_dps pitch_dps sent state pulse r")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("# WARN: frame grab failed")
            break

        now = time.time()
        h, w = frame.shape[:2]

        if anchor is None:
            anchor = (w / 2.0, h / 2.0)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        centroid = None
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

        if centroid is None:
            lost += 1

            # If we lose tracking, cancel any pulse and hold
            pulse_active = False
            pulse_yaw_dps_cmd = 0.0
            pulse_pitch_dps_cmd = 0.0
            pulse_roll_dps_cmd = 0.0

            if (now - last_send_time) >= COMMAND_PERIOD:
                send_or_log_speeds(0.0, 0.0, 0.0)
                last_send_time = now

            if lost > MAX_LOST_FRAMES:
                prev_smoothed = None
                prev_time = None

            if DRAW:
                cv2.imshow("Centroid Tracker (SPEED pulses)", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            continue

        lost = 0
        smoothed = ema_point(centroid, prev_smoothed, SMOOTH_ALPHA)

        if prev_time is None:
            prev_time = now
        dt = max(1e-6, now - prev_time)

        # stability velocity estimate (deg/s)
        if prev_smoothed is None:
            speed = 0.0
        else:
            dx_px_dt = smoothed[0] - prev_smoothed[0]
            dy_px_dt = smoothed[1] - prev_smoothed[1]
            dvx, dvy = pixels_to_deg(dx_px_dt, dy_px_dt, w, h, FOV_H_DEG, FOV_V_DEG)
            speed = math.hypot(dvx, dvy) / dt

        prev_time = now
        prev_smoothed = smoothed

        box = build_stable_box(anchor, w, h, STABLE_SCALAR)
        inside = inside_box(smoothed, box)

        dx_center = smoothed[0] - anchor[0]
        dy_center = smoothed[1] - anchor[1]

        norm_dx = dx_center / (w / 2.0)
        norm_dy = dy_center / (h / 2.0)
        radial_norm = math.hypot(norm_dx, norm_dy)
        within_stop_thresh = (radial_norm <= STABLE_STOP_SEEKING_THRESHOLD)

        pos_x.add(now, smoothed[0])
        pos_y.add(now, smoothed[1])
        vel_h.add(now, speed)

        xs, ys = pos_x.values(), pos_y.values()
        pos_std = 999.0
        if len(xs) >= 6 and len(ys) >= 6:
            pos_std = 0.5 * (statistics.pstdev(xs) + statistics.pstdev(ys))
        speeds = vel_h.values()
        vel_med = statistics.median(speeds) if len(speeds) >= 3 else 999.0

        too_wild = (vel_med > VEL_THRESH_DEG_S * 2.0) or (pos_std > POS_STD_THRESH_PX * 2.0)

        # state transitions
        if state == LOCKED:
            if not inside:
                state = SEEKING
        else:
            if within_stop_thresh:
                state = LOCKED

        # If locked, cancel pulse and hold
        if state == LOCKED or too_wild:
            pulse_active = False
            pulse_yaw_dps_cmd = 0.0
            pulse_pitch_dps_cmd = 0.0
            pulse_roll_dps_cmd = 0.0

        # If a pulse is active and has ended, stop and let next loop decide new pulse
        if pulse_active and now >= pulse_end_time:
            pulse_active = False
            pulse_yaw_dps_cmd = 0.0
            pulse_pitch_dps_cmd = 0.0
            pulse_roll_dps_cmd = 0.0

        # Start a NEW pulse only if:
        # - seeking, not wild, and no pulse active
        if (state == SEEKING) and (not too_wild) and (not pulse_active):
            err_yaw_deg, err_pitch_deg = pixels_to_deg(dx_center, dy_center, w, h, FOV_H_DEG, FOV_V_DEG)
            err_yaw_deg *= AXIS_SIGN["yaw"]
            err_pitch_deg *= AXIS_SIGN["pitch"]

            # Deadband
            if abs(err_yaw_deg) < DEADBAND_DEG_YAW:
                err_yaw_deg = 0.0
            if abs(err_pitch_deg) < DEADBAND_DEG_PITCH:
                err_pitch_deg = 0.0

            # If still meaningful error, plan one pulse
            if (err_yaw_deg != 0.0) or (err_pitch_deg != 0.0):
                # Compute desired speeds from error
                yaw_dps_cmd = clamp(KP_YAW_DPS_PER_DEG * err_yaw_deg, -MAX_DPS_YAW, +MAX_DPS_YAW)
                pitch_dps_cmd = clamp(KP_PITCH_DPS_PER_DEG * err_pitch_deg, -MAX_DPS_PITCH, +MAX_DPS_PITCH)
                roll_dps_cmd = 0.0

                # Compute duration from "how long at this speed to reduce error"
                # duration ~= max(|err|/|speed|) across axes (avoid div0)
                dur_y = abs(err_yaw_deg) / max(1e-6, abs(yaw_dps_cmd)) if yaw_dps_cmd != 0.0 else 0.0
                dur_p = abs(err_pitch_deg) / max(1e-6, abs(pitch_dps_cmd)) if pitch_dps_cmd != 0.0 else 0.0
                dur = max(dur_y, dur_p)

                # clamp duration
                dur = clamp(dur, MIN_PULSE_S, MAX_PULSE_S)

                # latch pulse
                pulse_active = True
                pulse_start_time = now
                pulse_end_time = now + dur
                pulse_yaw_dps_cmd = yaw_dps_cmd
                pulse_pitch_dps_cmd = pitch_dps_cmd
                pulse_roll_dps_cmd = roll_dps_cmd

        # Command to send this tick:
        yaw_out = 0.0
        pitch_out = 0.0
        roll_out = 0.0

        if pulse_active:
            # apply ramp in/out
            rf = ramp_factor(now, pulse_start_time, pulse_end_time, RAMP_S)
            yaw_out = pulse_yaw_dps_cmd * rf
            pitch_out = pulse_pitch_dps_cmd * rf
            roll_out = pulse_roll_dps_cmd * rf
        else:
            yaw_out = 0.0
            pitch_out = 0.0
            roll_out = 0.0

        # fixed-rate send + optional smoothing
        sent = 0
        if (now - last_send_time) >= COMMAND_PERIOD:
            smooth_yaw_dps = ema_scalar(yaw_out, smooth_yaw_dps, CMD_SPEED_EMA_ALPHA)
            smooth_pitch_dps = ema_scalar(pitch_out, smooth_pitch_dps, CMD_SPEED_EMA_ALPHA)
            smooth_roll_dps = ema_scalar(roll_out, smooth_roll_dps, CMD_SPEED_EMA_ALPHA)

            ok_send = send_or_log_speeds(smooth_roll_dps, smooth_pitch_dps, smooth_yaw_dps)
            if ok_send:
                last_send_time = now
                sent = 1

        T = now - t0
        print(f"{T:.3f} {yaw_out:+.2f} {pitch_out:+.2f} {sent} {state} {1 if pulse_active else 0} r={radial_norm:.3f}")

        if DRAW:
            l, t_, r, b = map(int, box)
            cv2.rectangle(frame, (l, t_), (r, b), (40, 220, 40), 1)
            cv2.drawMarker(frame, (int(anchor[0]), int(anchor[1])), (0, 200, 0),
                           cv2.MARKER_CROSS, 12, 2)
            cv2.circle(frame, (int(smoothed[0]), int(smoothed[1])), 4, (0, 0, 255), -1)

            state_txt = "LOCKED" if state == LOCKED else "SEEKING"
            pulse_txt = "PULSE" if pulse_active else "HOLD"
            cv2.putText(frame, f"state:{state_txt} {pulse_txt}", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40,220,40), 2, cv2.LINE_AA)
            cv2.putText(frame, f"r={radial_norm:.3f}", (10, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40,220,40), 2, cv2.LINE_AA)

            cv2.imshow("Centroid Tracker (SPEED pulses)", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    # stop motion on exit
    try:
        send_or_log_speeds(0.0, 0.0, 0.0)
    except Exception:
        pass

    cap.release()
    cv2.destroyAllWindows()
    print("Stopped.")


if __name__ == "__main__":
    main()
