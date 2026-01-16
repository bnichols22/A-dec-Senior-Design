#!/usr/bin/env python3
# ==============================================================
# File: full_tracking_twoshot.py
# Purpose:
#   Center-locked tracker with explicit state machine:
#   - LOCKED: do nothing while mouth centroid is inside the stable box.
#   - SEEKING: when outside, correct the error using a two-shot move:
#       * If error is small: do a single move (micro-step).
#       * If error is large: do a big "first shot" (e.g., 75%),
#         then finish with smaller micro-steps until within stop threshold.
#
# Anchor is always the image center (optical axis).
#
# Gimbal transport:
#   - Uses SimpleBGC SerialAPI C library exposed via simplebgc_shim.c
#   - Python works in ZERO-BASED ABSOLUTE angles:
#       * At startup, we read the current board angles and call that (0,0,0)
#       * cmd_roll/pitch/yaw are kept in this zero-based frame
#       * Before sending, we add the startup baseline back in
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
WINDOW_SEC    = 0.6

# Stop condition: normalized radial distance (0..1)
STABLE_STOP_SEEKING_THRESHOLD = 0.025

# Stability gates (used in SEEKING — only send steps when motion isn't crazy)
VEL_THRESH_DEG_S  = 2.5
POS_STD_THRESH_PX = 2.5

# ---------------- Two-shot behavior ----------------
# If we're far from center (radial_norm >= this), do a 2-shot correction.
TWO_SHOT_RADIAL_TRIGGER = 0.080

# First shot fraction (big move). 0.75 = 75% of full offset.
FIRST_SHOT_FRACTION = 0.75

# After first shot, finish with smaller steps.
FINISH_STEP_FRACTION = 0.18

# If not doing 2-shot, use this as the regular micro-step fraction.
NORMAL_STEP_FRACTION = 0.15

# Send gating (rate limit between commands)
SEND_TIME_LIMITER = 0.085

# Ignore tiny commands
MIN_STEP_DEG_YAW   = 0.3
MIN_STEP_DEG_PITCH = 0.3
MIN_STEP_DEG_ROLL  = 0.3

# Sign convention to match gimbal axes (keep your known-good mapping)
AXIS_SIGN = {"yaw": 1, "pitch": 1, "roll": 1}

# Gimbal / test control
TEST = 0  # 1=log only; 0=send to gimbal via libsimplebgc.so

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
# SBGC shim bindings (ctypes) + zero-based baseline
# ----------------------------------------------------------------------
_bgc_lib = None
_bgc_initialized = False

_baseline_yaw_deg   = 0.0
_baseline_pitch_deg = 0.0
_baseline_roll_deg  = 0.0

def init_sbgc():
    """Load libsimplebgc.so, set prototypes, init board, and capture baseline."""
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
        rc2 = _bgc_lib.bgc_get_angles(
            ctypes.byref(yaw),
            ctypes.byref(pitch),
            ctypes.byref(roll),
        )
        if rc2 == 0:
            _baseline_yaw_deg   = float(yaw.value)
            _baseline_pitch_deg = float(pitch.value)
            _baseline_roll_deg  = float(roll.value)
            print(f"# Baseline angles captured from board:")
            print(f"#   yaw={_baseline_yaw_deg:.2f}, pitch={_baseline_pitch_deg:.2f}, roll={_baseline_roll_deg:.2f}")
        else:
            print(f"# WARN: bgc_get_angles() returned {rc2}, using baseline=0")
            _baseline_yaw_deg = _baseline_pitch_deg = _baseline_roll_deg = 0.0
    else:
        print("# WARN: bgc_get_angles not available in shim; using baseline=0")
        _baseline_yaw_deg = _baseline_pitch_deg = _baseline_roll_deg = 0.0

    if hasattr(_bgc_lib, "bgc_beep_once"):
        try:
            _bgc_lib.bgc_beep_once()
        except Exception:
            pass

    _bgc_initialized = True
    print("# SBGC initialization complete (zero-based frame set).")

def write_test_line(d_yaw, d_pitch, d_roll,
                    abs_yaw, abs_pitch, abs_roll,
                    board_yaw, board_pitch, board_roll):
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
    """Send absolute angles in zero-based software frame (board = baseline + abs)."""
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

    # Clear log
    try:
        with open(TEST_LOG, "w") as f:
            f.write("# Two-shot tracker — fresh run\n")
            f.write(f"# Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    except Exception as e:
        print(f"# ERROR clearing test log: {e}")

    # State
    t0 = time.time()
    prev_smoothed = None
    prev_roll_deg = None
    prev_time = None
    anchor = None
    last_send_time = 0.0
    lost = 0

    # Zero-based absolute commanded angles
    cmd_roll_deg  = 0.0
    cmd_pitch_deg = 0.0
    cmd_yaw_deg   = 0.0

    # Two-shot internal flag: did we already do the first big correction this SEEKING episode?
    did_first_shot = False

    LOCKED, SEEKING = 0, 1
    state = LOCKED

    pos_x = TimedHist(WINDOW_SEC)
    pos_y = TimedHist(WINDOW_SEC)
    vel_h = TimedHist(WINDOW_SEC)

    print("# T dR dP dY sent state radial twoShot")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("# WARN: frame grab failed")
            break
        now = time.time()
        h, w = frame.shape[:2]

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

        # Lost handling
        if centroid is None:
            lost += 1
            if lost > MAX_LOST_FRAMES:
                prev_smoothed = None
                prev_roll_deg = None
                prev_time = None
            if DRAW:
                cv2.imshow("Two-shot Tracker", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            continue
        lost = 0

        smoothed = ema_point(centroid, prev_smoothed, SMOOTH_ALPHA)

        if prev_time is None:
            prev_time = now

        box = build_stable_box(anchor, w, h, STABLE_SCALAR)
        inside = inside_box(smoothed, box)

        # radial_norm for stop decision + two-shot trigger
        dx_center = smoothed[0] - anchor[0]
        dy_center = smoothed[1] - anchor[1]
        norm_dx = dx_center / (w / 2.0)
        norm_dy = dy_center / (h / 2.0)
        radial_norm = math.hypot(norm_dx, norm_dy)
        within_stop_thresh = (radial_norm <= STABLE_STOP_SEEKING_THRESHOLD)

        # velocity for stability gating
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
        sent_flag = 0

        # ---------- STATE MACHINE ----------
        if state == LOCKED:
            did_first_shot = False
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
            is_stable_here = (vel_med < VEL_THRESH_DEG_S) and (pos_std < POS_STD_THRESH_PX)

            if within_stop_thresh:
                if roll_now_deg is not None:
                    prev_roll_deg = roll_now_deg
                pos_x.clear(); pos_y.clear(); vel_h.clear()
                state = LOCKED
                did_first_shot = False
            else:
                can_time = (now - last_send_time) >= SEND_TIME_LIMITER
                if is_stable_here and can_time:
                    # Full desired offset (center -> current)
                    full_d_yaw, full_d_pitch = pixels_to_deg(
                        dx_center, dy_center, w, h, FOV_H_DEG, FOV_V_DEG
                    )

                    d_roll_full = 0.0
                    if roll_now_deg is not None and prev_roll_deg is not None:
                        d_roll_full = roll_now_deg - prev_roll_deg

                    # Apply axis signs (keep your known-good behavior)
                    full_d_yaw   = AXIS_SIGN["yaw"]   * full_d_yaw
                    full_d_pitch = AXIS_SIGN["pitch"] * full_d_pitch
                    d_roll_full  = AXIS_SIGN["roll"]  * d_roll_full

                    # Decide which fraction to use:
                    # - If "far" AND we haven't done the first shot yet -> 75% first shot.
                    # - Else -> finishing micro-steps (or normal micro-step if never triggered).
                    two_shot_active = (radial_norm >= TWO_SHOT_RADIAL_TRIGGER)

                    if two_shot_active and (not did_first_shot):
                        frac = FIRST_SHOT_FRACTION
                        did_first_shot = True
                    else:
                        frac = FINISH_STEP_FRACTION if two_shot_active else NORMAL_STEP_FRACTION

                    d_yaw   = full_d_yaw   * frac
                    d_pitch = full_d_pitch * frac
                    d_roll  = d_roll_full  * frac

                    # Minimum thresholds to avoid dithering
                    if abs(d_yaw)   < MIN_STEP_DEG_YAW:   d_yaw = 0.0
                    if abs(d_pitch) < MIN_STEP_DEG_PITCH: d_pitch = 0.0
                    if abs(d_roll)  < MIN_STEP_DEG_ROLL:  d_roll = 0.0

                    if (d_yaw != 0.0) or (d_pitch != 0.0) or (d_roll != 0.0):
                        cmd_yaw_deg   += d_yaw
                        cmd_pitch_deg += d_pitch
                        cmd_roll_deg  += d_roll

                        ok = send_or_log_angles(
                            d_yaw, d_pitch, d_roll,
                            cmd_yaw_deg, cmd_pitch_deg, cmd_roll_deg
                        )
                        if ok:
                            last_send_time = now
                            sent_flag = 1

        # -------- Telemetry --------
        T = now - t0
        print(f"{T:.3f} {d_roll:+.3f} {d_pitch:+.3f} {d_yaw:+.3f} {sent_flag} {state} "
              f"r={radial_norm:.3f} twoShot={1 if radial_norm>=TWO_SHOT_RADIAL_TRIGGER else 0} firstDone={1 if did_first_shot else 0}")

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
            cv2.putText(frame, f"twoShot={(radial_norm>=TWO_SHOT_RADIAL_TRIGGER)} first={(did_first_shot)}",
                        (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40,220,40), 2, cv2.LINE_AA)

            cv2.imshow("Two-shot Tracker", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Stopped.")

if __name__ == "__main__":
    main()
