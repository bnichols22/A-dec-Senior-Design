#!/usr/bin/env python3
# ==============================================================
# File: full_tracking.py
# Purpose:
#   Center-locked face tracker with smooth continuous control (Option E):
#   - Stable box is locked to the CENTER of the frame (optical axis).
#   - LOCKED: when the mouth centroid is close enough to center (stop threshold),
#             do nothing (hold position).
#   - SEEKING: when outside the stop threshold, continuously compute the error
#              (center -> mouth), filter it, and send smooth commands at a fixed
#              rate using a slew-rate limiter (no "wait for stable" gating).
#
# Key change vs your prior version:
#   - We DO NOT gate sends on "is_stable_here". Instead, we filter the
#     measured error and limit how fast commands can change (deg/s).
#   - This removes the stop/start cadence that looks choppy.
#
# Gimbal transport:
#   - Uses SimpleBGC SerialAPI C library exposed via simplebgc_shim.c
#   - Python works in ZERO-BASED ABSOLUTE angles:
#       * At startup, we read the current board angles and call that (0,0,0)
#       * We keep cmd_roll/pitch/yaw in this zero-based frame
#       * Before sending, we add the startup baseline back in so the board
#         gets its own absolute angles.
# ==============================================================

import os, sys, time, math, warnings, statistics, ctypes
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

# Best-guess FOV values (tune if needed)
FOV_H_DEG = 65.0
FOV_V_DEG = 48.75

# Stable (green) box size: fraction of HALF-frame sizes
STABLE_SCALAR = 0.06

# How close to center is "good enough" to stop SEEKING
# Normalized radial distance in [0,1]
STABLE_STOP_SEEKING_THRESHOLD = 0.025

# ---- Option E tuning knobs (smooth continuous control) ----
# Command rate (how often we send to the gimbal)
SEND_TIME_LIMITER = 0.04   # seconds between sends (try 0.03–0.06)

# Filter the measured angular error (deg) to remove jitter
ERROR_FILTER_ALPHA = 0.35  # 0..1, higher = more responsive, lower = smoother (try 0.25–0.5)

# Convert filtered error into a "desired step" each send (like a P gain)
# Smaller values = slower, smoother convergence. Larger = faster, can overshoot.
P_GAIN = 0.35              # dimensionless, try 0.2–0.6

# Limit how fast commanded angles are allowed to change (deg/s)
MAX_RATE_YAW_DEG_S   = 40.0
MAX_RATE_PITCH_DEG_S = 40.0
MAX_RATE_ROLL_DEG_S  = 60.0

# Small deadband to prevent buzzing near center (deg)
MIN_STEP_DEG_YAW   = 0.08
MIN_STEP_DEG_PITCH = 0.08
MIN_STEP_DEG_ROLL  = 0.10

# Sign convention to match gimbal axes (tune as needed)
AXIS_SIGN = {"yaw": 1, "pitch": 1, "roll": 1}

# Gimbal / test control
TEST = 0  # 1=log only; 0=send to gimbal via libsimplebgc.so

LIB_PATH = os.path.expanduser(
    "~/senior_design/A-dec-Senior-Design/camera/testing/SerialLibrary/serialAPI/libsimplebgc.so"
)

DRAW = True
SMOOTH_ALPHA = 0.25     # centroid EMA smoothing (image-space)
MAX_LOST_FRAMES = 10

# --------- Quiet noisy TF/MP logs ----------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")
sys.stderr = open(LOG_PATH, "w")

# ---------------- Utilities ----------------
def ema_point(curr, prev, alpha):
    """Exponential moving average for 2D points."""
    if prev is None:
        return curr
    return (alpha*curr[0] + (1-alpha)*prev[0],
            alpha*curr[1] + (1-alpha)*prev[1])

def ema_scalar(curr, prev, alpha):
    """Exponential moving average for scalars."""
    if prev is None:
        return curr
    return alpha*curr + (1-alpha)*prev

def pixels_to_deg(dx_px, dy_px, w, h, fov_h, fov_v):
    """Convert pixel offsets from center into approximate angular offsets (deg)."""
    half_w, half_h = w/2.0, h/2.0
    return (dx_px/half_w)*(fov_h/2.0), (dy_px/half_h)*(fov_v/2.0)

def angle_deg(p1, p2):
    """Angle (deg) between two points (for estimating roll)."""
    return math.degrees(math.atan2(p2[1]-p1[1], p2[0]-p1[0]))

def build_stable_box(anchor_xy, w, h, scalar):
    """Return a rectangle (l,t,r,b) centered at anchor_xy."""
    cx, cy = anchor_xy
    half_w = scalar * (w/2.0)
    half_h = scalar * (h/2.0)
    return (cx - half_w, cy - half_h, cx + half_w, cy + half_h)

def inside_box(pt, box):
    """True if point (x,y) is inside box (l,t,r,b)."""
    x, y = pt
    l, t, r, b = box
    return (l <= x <= r) and (t <= y <= b)

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def slew_limit(delta_deg, dt, max_rate_deg_s):
    """Clamp delta to a maximum allowed change based on deg/s rate limit."""
    max_delta = max_rate_deg_s * dt
    return clamp(delta_deg, -max_delta, +max_delta)

# ----------------------------------------------------------------------
# SBGC shim bindings (ctypes) + zero-based baseline
# ----------------------------------------------------------------------
_bgc_lib = None
_bgc_initialized = False

_baseline_yaw_deg   = 0.0
_baseline_pitch_deg = 0.0
_baseline_roll_deg  = 0.0

def init_sbgc():
    """Load libsimplebgc.so, init board, and capture baseline angles."""
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

    # NOTE: shim takes (roll_deg, pitch_deg, yaw_deg)
    _bgc_lib.bgc_control_angles.argtypes = [
        ctypes.c_float, ctypes.c_float, ctypes.c_float
    ]
    _bgc_lib.bgc_control_angles.restype = ctypes.c_int

    try:
        _bgc_lib.bgc_beep_once.argtypes = []
        _bgc_lib.bgc_beep_once.restype  = ctypes.c_int
    except AttributeError:
        pass

    try:
        _bgc_lib.bgc_get_angles.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        _bgc_lib.bgc_get_angles.restype = ctypes.c_int
    except AttributeError:
        _bgc_lib.bgc_get_angles = None

    rc = _bgc_lib.bgc_init()
    if rc != 0:
        print(f"# ERROR: bgc_init() returned {rc}")
        _bgc_initialized = False
        return

    if hasattr(_bgc_lib, "bgc_get_angles") and _bgc_lib.bgc_get_angles is not None:
        yaw = ctypes.c_float()
        pitch = ctypes.c_float()
        roll = ctypes.c_float()
        rc2 = _bgc_lib.bgc_get_angles(ctypes.byref(yaw), ctypes.byref(pitch), ctypes.byref(roll))
        if rc2 == 0:
            _baseline_yaw_deg   = float(yaw.value)
            _baseline_pitch_deg = float(pitch.value)
            _baseline_roll_deg  = float(roll.value)
            print("# Baseline angles captured from board:")
            print(f"#   yaw={_baseline_yaw_deg:.2f}, pitch={_baseline_pitch_deg:.2f}, roll={_baseline_roll_deg:.2f}")
        else:
            print(f"# WARN: bgc_get_angles() returned {rc2}, using baseline = 0,0,0")
            _baseline_yaw_deg = _baseline_pitch_deg = _baseline_roll_deg = 0.0
    else:
        print("# WARN: bgc_get_angles not available in shim; using baseline = 0,0,0")
        _baseline_yaw_deg = _baseline_pitch_deg = _baseline_roll_deg = 0.0

    if hasattr(_bgc_lib, "bgc_beep_once"):
        try:
            _bgc_lib.bgc_beep_once()
        except Exception:
            pass

    _bgc_initialized = True
    print("# SBGC initialization complete (zero-based frame set).")

# ----------------------------------------------------------------------
# Output: send or write to file (angles, not raw packets)
# ----------------------------------------------------------------------
def write_test_line(d_yaw, d_pitch, d_roll,
                    abs_yaw, abs_pitch, abs_roll,
                    board_yaw, board_pitch, board_roll):
    """Log command attempts to TEST_LOG instead of moving the gimbal."""
    try:
        with open(TEST_LOG, 'a', buffering=1) as f:
            f.write(
                f"T={time.time():.3f} "
                f"dR={d_roll:+.2f} dP={d_pitch:+.2f} dY={d_yaw:+.2f} | "
                f"absR={abs_roll:+.2f} absP={abs_pitch:+.2f} absY={abs_yaw:+.2f} | "
                f"boardR={board_roll:+.2f} boardP={board_pitch:+.2f} boardY={board_yaw:+.2f}\n"
            )
            f.flush()
            os.fsync(f.fileno())
        return True
    except Exception as e:
        print(f"# TEST_FILE_ERROR: {e}")
        return False

def send_or_log_angles(d_yaw_deg, d_pitch_deg, d_roll_deg,
                       abs_yaw_deg, abs_pitch_deg, abs_roll_deg):
    """
    Send angles to gimbal (or log in TEST mode).
    Uses zero-based software frame: board_angle = baseline + abs_angle.
    """
    board_yaw   = _baseline_yaw_deg   + abs_yaw_deg
    board_pitch = _baseline_pitch_deg + abs_pitch_deg
    board_roll  = _baseline_roll_deg  + abs_roll_deg

    if TEST == 1:
        return write_test_line(d_yaw_deg, d_pitch_deg, d_roll_deg,
                               abs_yaw_deg, abs_pitch_deg, abs_roll_deg,
                               board_yaw, board_pitch, board_roll)

    if _bgc_lib is None or not _bgc_initialized:
        print("# ERROR: SBGC shim not initialized.")
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

# ---------------- Main loop (stateful) ----------------
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

    # --- State ---
    t0 = time.time()
    prev_smoothed = None
    prev_time = None
    anchor = None  # locked to frame center

    lost = 0
    last_send_time = 0.0
    last_send_dt = SEND_TIME_LIMITER  # default fallback

    # zero-based absolute commanded angles (software frame)
    cmd_roll_deg  = 0.0
    cmd_pitch_deg = 0.0
    cmd_yaw_deg   = 0.0

    # Filtered error state (deg)
    filt_err_yaw = None
    filt_err_pitch = None
    filt_err_roll = None

    # Roll baseline when SEEKING starts
    roll_ref_deg = None

    LOCKED, SEEKING = 0, 1
    state = LOCKED

    try:
        with open(TEST_LOG, "w") as f:
            f.write("# SimpleBGC commands (Option E: filtered + slew-limited) — fresh run\n")
            f.write(f"# Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        print(f"# Cleared {TEST_LOG} for a new session.")
    except Exception as e:
        print(f"# ERROR clearing test log: {e}")

    print("# T dR dP dY sent state radial_norm")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("# WARN: frame grab failed")
            break

        now = time.time()
        h, w = frame.shape[:2]

        # Lock anchor to frame center once dimensions are known
        if anchor is None:
            anchor = (w / 2.0, h / 2.0)

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

        if centroid is None:
            lost += 1
            if lost > MAX_LOST_FRAMES:
                prev_smoothed = None
                prev_time = None
                filt_err_yaw = filt_err_pitch = filt_err_roll = None
                roll_ref_deg = None
                state = LOCKED
            if DRAW:
                cv2.imshow("Centroid Tracker (Option E)", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            continue
        lost = 0

        smoothed = ema_point(centroid, prev_smoothed, SMOOTH_ALPHA)

        if prev_time is None:
            prev_time = now
        dt_frame = max(1e-6, now - prev_time)
        prev_time = now
        prev_smoothed = smoothed

        # Stable (green) box, used mainly for visuals
        box = build_stable_box(anchor, w, h, STABLE_SCALAR)
        inside = inside_box(smoothed, box)

        # Stop threshold based on normalized radial distance
        dx_center = smoothed[0] - anchor[0]
        dy_center = smoothed[1] - anchor[1]
        norm_dx = dx_center / (w / 2.0)
        norm_dy = dy_center / (h / 2.0)
        radial_norm = math.hypot(norm_dx, norm_dy)
        within_stop_thresh = (radial_norm <= STABLE_STOP_SEEKING_THRESHOLD)

        # -------- State machine ----------
        if state == LOCKED:
            # If we're outside the stop threshold, begin SEEKING
            if not within_stop_thresh:
                state = SEEKING
                # Reset filtered errors when we start moving
                filt_err_yaw = filt_err_pitch = filt_err_roll = None
                # Set roll reference for this seeking episode
                roll_ref_deg = roll_now_deg if roll_now_deg is not None else None

        # Compute command deltas to send (deg)
        d_yaw = d_pitch = d_roll = 0.0
        sent = 0

        if state == SEEKING:
            if within_stop_thresh:
                state = LOCKED
                # Optional: clear filters so next move starts clean
                filt_err_yaw = filt_err_pitch = filt_err_roll = None
                roll_ref_deg = roll_now_deg if roll_now_deg is not None else roll_ref_deg
            else:
                # Compute instantaneous angular error from center
                err_yaw_deg, err_pitch_deg = pixels_to_deg(dx_center, dy_center, w, h, FOV_H_DEG, FOV_V_DEG)

                # Roll error relative to roll_ref_deg
                err_roll_deg = 0.0
                if roll_now_deg is not None and roll_ref_deg is not None:
                    err_roll_deg = roll_now_deg - roll_ref_deg

                # Apply axis signs
                err_yaw_deg   *= AXIS_SIGN["yaw"]
                err_pitch_deg *= AXIS_SIGN["pitch"]
                err_roll_deg  *= AXIS_SIGN["roll"]

                # Filter the error (removes jitter)
                filt_err_yaw   = ema_scalar(err_yaw_deg,   filt_err_yaw,   ERROR_FILTER_ALPHA)
                filt_err_pitch = ema_scalar(err_pitch_deg, filt_err_pitch, ERROR_FILTER_ALPHA)
                filt_err_roll  = ema_scalar(err_roll_deg,  filt_err_roll,  ERROR_FILTER_ALPHA)

                # Send at a fixed rate (but WITHOUT stability gating)
                can_time = (now - last_send_time) >= SEND_TIME_LIMITER
                if can_time:
                    send_dt = max(1e-6, now - last_send_time)
                    last_send_time = now
                    last_send_dt = send_dt

                    # Proportional step toward reducing error
                    step_yaw   = -P_GAIN * filt_err_yaw
                    step_pitch = -P_GAIN * filt_err_pitch
                    step_roll  = -P_GAIN * filt_err_roll

                    # Slew-rate limit (deg/s) so motion is smooth
                    step_yaw   = slew_limit(step_yaw,   send_dt, MAX_RATE_YAW_DEG_S)
                    step_pitch = slew_limit(step_pitch, send_dt, MAX_RATE_PITCH_DEG_S)
                    step_roll  = slew_limit(step_roll,  send_dt, MAX_RATE_ROLL_DEG_S)

                    # Deadband near zero to prevent buzzing
                    if abs(step_yaw)   < MIN_STEP_DEG_YAW:   step_yaw = 0.0
                    if abs(step_pitch) < MIN_STEP_DEG_PITCH: step_pitch = 0.0
                    if abs(step_roll)  < MIN_STEP_DEG_ROLL:  step_roll = 0.0

                    # Apply the step to commanded angles
                    if (step_yaw != 0.0) or (step_pitch != 0.0) or (step_roll != 0.0):
                        cmd_yaw_deg   += step_yaw
                        cmd_pitch_deg += step_pitch
                        cmd_roll_deg  += step_roll

                        ok_send = send_or_log_angles(
                            step_yaw, step_pitch, step_roll,
                            cmd_yaw_deg, cmd_pitch_deg, cmd_roll_deg
                        )
                        if ok_send:
                            d_yaw, d_pitch, d_roll = step_yaw, step_pitch, step_roll
                            sent = 1

        # -------- Telemetry --------
        T = now - t0
        print(f"{T:.3f} {d_roll:+.3f} {d_pitch:+.3f} {d_yaw:+.3f} {sent} {state} radial_norm={radial_norm:.3f}")

        # -------- UI --------
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
            if state == SEEKING:
                cv2.putText(frame, f"cmd dR:{d_roll:+.2f} dP:{d_pitch:+.2f} dY:{d_yaw:+.2f}",
                            (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40,220,40), 2, cv2.LINE_AA)

            cv2.imshow("Centroid Tracker (Option E)", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Stopped.")

if __name__ == "__main__":
    main()
