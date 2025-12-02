#!/usr/bin/env python3
# ==============================================================
# File: full_tracking.py
# Purpose:
#   Fixed-center tracker with micro-step control:
#   - The "anchor" is always the center of the frame.
#   - A green stable box is drawn around the center.#!/usr/bin/env python3
# ==============================================================
# File: full_tracking.py
# Purpose:
#   Fixed-center tracker with micro-step control:
#   - The "anchor" is always the center of the frame.
#   - A green stable box is drawn around the center.
#   - As long as the mouth centroid is inside the box, we do nothing.
#   - When the centroid exits the box, we:
#       * Compute the full offset (center -> current centroid)
#       * Send only a FRACTION of that as a micro-step
#       * Repeat while outside the box, until the centroid returns inside.
#
# Gimbal transport:
#   - Uses SimpleBGC SerialAPI C library exposed via simplebgc_shim.c
#   - Python works in ZERO-BASED ABSOLUTE angles:
#       * At startup, we read the current board angles and call that (0,0,0)
#       * We keep cmd_roll/pitch/yaw in this zero-based frame
#       * Before sending, we add the startup baseline back so the board
#         receives its own absolute angles.
# ==============================================================

import os, sys, time, math, warnings, ctypes
import cv2
import mediapipe as mp

# --------- Paths ----------
BASE_DIR = os.path.expanduser('~/senior_design/A-dec-Senior-Design/camera/testing')
os.makedirs(BASE_DIR, exist_ok=True)

LOG_PATH  = os.path.join(BASE_DIR, 'face_track_log.txt')
TEST_LOG  = os.path.join(BASE_DIR, 'Board_Serial_Command_Test.txt')

print(f"# ADEC BASE_DIR = {BASE_DIR}")
print(f"# ADEC TEST_LOG = {TEST_LOG}")

# --------- Vision / tracker config ----------
CAM_INDEX = 0

# FOV estimates (tuned from your FOV math)
FOV_H_DEG = 65.0
FOV_V_DEG = 48.75

# Stable box: fraction of HALF-frame sizes
STABLE_SCALAR = 0.06    # tighten to 0.05 or 0.04 if too tolerant

# Micro-step control (fraction of full offset per movement)
STEP_FRACTION = 0.2     # try 0.1–0.3

# Send gating
SEND_TIME_LIMITER = 0.25  # min seconds between sends
MIN_STEP_DEG_YAW   = 0.3
MIN_STEP_DEG_PITCH = 0.3
MIN_STEP_DEG_ROLL  = 1.0

# Clamp ranges for zero-based commanded angles (to avoid runaway spins)
MAX_YAW_ABS_DEG   = 80.0
MAX_PITCH_ABS_DEG = 45.0
MAX_ROLL_ABS_DEG  = 45.0

# Sign convention to match gimbal axes (tune as needed)
AXIS_SIGN = {"yaw": 1, "pitch": 1, "roll": 1}

# Gimbal / test control
TEST = 0  # 1=log only; 0=send to gimbal via libsimplebgc.so

# Path to the shared library we built from SerialAPI + shim
LIB_PATH = os.path.expanduser(
    "~/senior_design/A-dec-Senior-Design/camera/testing/SerialLibrary/serialAPI/libsimplebgc.so"
)

DRAW = True
SMOOTH_ALPHA = 0.25
MAX_LOST_FRAMES = 10

# --------- Quiet noisy TF/MP logs ----------
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

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

# ----------------------------------------------------------------------
# SBGC shim bindings (ctypes) + zero-based baseline
# ----------------------------------------------------------------------
_bgc_lib = None
_bgc_initialized = False

# baseline angles at startup (board frame)
_baseline_yaw_deg   = 0.0
_baseline_pitch_deg = 0.0
_baseline_roll_deg  = 0.0
_baseline_set       = False

def init_sbgc():
    """Load libsimplebgc.so, set prototypes, init board, and capture baseline."""
    global _bgc_lib, _bgc_initialized
    global _baseline_yaw_deg, _baseline_pitch_deg, _baseline_roll_deg, _baseline_set

    if TEST == 1:
        print("# TEST mode: not loading SBGC library.")
        _bgc_lib = None
        _bgc_initialized = False
        _baseline_yaw_deg = 0.0
        _baseline_pitch_deg = 0.0
        _baseline_roll_deg = 0.0
        _baseline_set = True
        return

    try:
        _bgc_lib = ctypes.CDLL(LIB_PATH)
        print(f"# Loaded SBGC library from {LIB_PATH}")
    except OSError as e:
        print(f"# ERROR loading {LIB_PATH}: {e}")
        _bgc_lib = None
        _bgc_initialized = False
        return

    # Prototypes from simplebgc_shim.c
    _bgc_lib.bgc_init.argtypes = []
    _bgc_lib.bgc_init.restype  = ctypes.c_int

    # IMPORTANT: shim takes (roll_deg, pitch_deg, yaw_deg)
    _bgc_lib.bgc_control_angles.argtypes = [
        ctypes.c_float, ctypes.c_float, ctypes.c_float
    ]
    _bgc_lib.bgc_control_angles.restype = ctypes.c_int

    # Optional helpers from shim
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

    # Initialize library (sets control config + motors ON inside shim)
    rc = _bgc_lib.bgc_init()
    if rc != 0:
        print(f"# ERROR: bgc_init() returned {rc}")
        _bgc_initialized = False
        return

    # Capture baseline board angles at startup
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
            _baseline_set       = True
            print("# Baseline angles captured from board:")
            print(f"#   yaw={_baseline_yaw_deg:.2f}, "
                  f"pitch={_baseline_pitch_deg:.2f}, "
                  f"roll={_baseline_roll_deg:.2f}")
        else:
            print(f"# WARN: bgc_get_angles() returned {rc2}, using baseline = 0,0,0")
            _baseline_yaw_deg = 0.0
            _baseline_pitch_deg = 0.0
            _baseline_roll_deg = 0.0
            _baseline_set = True
    else:
        print("# WARN: bgc_get_angles not available in shim; using baseline = 0,0,0")
        _baseline_yaw_deg = 0.0
        _baseline_pitch_deg = 0.0
        _baseline_roll_deg = 0.0
        _baseline_set = True

    # Optional beep
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
    """
    Log attempted commands to TEST_LOG instead of moving gimbal.
    """
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
    If TEST=1: only log to file.
    If TEST=0: send angles to gimbal via bgc_control_angles(),
               using zero-based software frame:
                   board_angle = baseline_angle + abs_angle
    """
    global _baseline_yaw_deg, _baseline_pitch_deg, _baseline_roll_deg

    # Compute board-frame absolute angles by adding baseline
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

    # Shim order: (roll, pitch, yaw)
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

    # State
    t0 = time.time()
    prev_smoothed = None
    prev_time = None
    last_send_time = 0.0
    lost = 0

    # Roll reference (we store the roll when face is centered)
    roll_anchor_deg = None

    # Zero-based absolute commanded angles (software frame)
    cmd_roll_deg  = 0.0
    cmd_pitch_deg = 0.0
    cmd_yaw_deg   = 0.0

    LOCKED, SEEKING = 0, 1
    state = LOCKED

    # Clear the test file at the start of every run
    try:
        with open(TEST_LOG, "w") as f:
            f.write("# SimpleBGC commands (zero-based + micro-steps, center-anchored) — fresh run\n")
            f.write(f"# Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        print(f"# Cleared {TEST_LOG} for a new session.")
    except Exception as e:
        print(f"# ERROR clearing test log: {e}")

    print("# T roll pitch yaw state")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("# WARN: frame grab failed")
            break
        now = time.time()
        h, w = frame.shape[:2]

        # Fixed anchor at center of frame
        center_anchor = (w / 2.0, h / 2.0)
        box = build_stable_box(center_anchor, w, h, STABLE_SCALAR)

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
                prev_time = None
                roll_anchor_deg = None
            T = now - t0
            print(f"{T:.3f} +0.000 +0.000 +0.000 {state}")
            if DRAW:
                cv2.imshow("Centroid Tracker (center-anchored)", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            continue
        lost = 0

        # Smooth centroid
        smoothed = ema_point(centroid, prev_smoothed, SMOOTH_ALPHA)

        # Init timing
        if prev_time is None:
            prev_time = now
        dt = max(1e-6, now - prev_time)
        prev_time = now
        prev_smoothed = smoothed

        # Is centroid inside the center box?
        inside = inside_box(smoothed, box)

        # Default console outputs
        d_yaw = d_pitch = d_roll = 0.0
        sent_this_frame = False

        if inside:
            state = LOCKED
            # Refresh roll reference when centered
            if roll_now_deg is not None:
                roll_anchor_deg = roll_now_deg
        else:
            state = SEEKING

            # Full desired offset from center -> smoothed
            dx_px = smoothed[0] - center_anchor[0]
            dy_px = smoothed[1] - center_anchor[1]
            full_d_yaw, full_d_pitch = pixels_to_deg(dx_px, dy_px, w, h,
                                                     FOV_H_DEG, FOV_V_DEG)

            # Roll offset relative to roll when centered
            d_roll_full = 0.0
            if roll_now_deg is not None and roll_anchor_deg is not None:
                d_roll_full = roll_now_deg - roll_anchor_deg

            # Apply axis signs
            full_d_yaw   = AXIS_SIGN["yaw"]   * full_d_yaw
            full_d_pitch = AXIS_SIGN["pitch"] * full_d_pitch
            d_roll_full  = AXIS_SIGN["roll"]  * d_roll_full

            # Micro-step towards the center
            d_yaw   = full_d_yaw   * STEP_FRACTION
            d_pitch = full_d_pitch * STEP_FRACTION
            d_roll  = d_roll_full  * STEP_FRACTION

            # Enforce minimum movement thresholds
            if abs(d_yaw)   < MIN_STEP_DEG_YAW:   d_yaw = 0.0
            if abs(d_pitch) < MIN_STEP_DEG_PITCH: d_pitch = 0.0
            if abs(d_roll)  < MIN_STEP_DEG_ROLL:  d_roll = 0.0

            can_time = (now - last_send_time) >= SEND_TIME_LIMITER

            if can_time and ((d_yaw != 0.0) or (d_pitch != 0.0) or (d_roll != 0.0)):
                # Update zero-based commanded angles and clamp
                cmd_yaw_deg   = clamp(cmd_yaw_deg   + d_yaw,
                                      -MAX_YAW_ABS_DEG,   MAX_YAW_ABS_DEG)
                cmd_pitch_deg = clamp(cmd_pitch_deg + d_pitch,
                                      -MAX_PITCH_ABS_DEG, MAX_PITCH_ABS_DEG)
                cmd_roll_deg  = clamp(cmd_roll_deg  + d_roll,
                                      -MAX_ROLL_ABS_DEG,  MAX_ROLL_ABS_DEG)

                ok = send_or_log_angles(
                    d_yaw, d_pitch, d_roll,
                    cmd_yaw_deg, cmd_pitch_deg, cmd_roll_deg
                )
                if ok:
                    last_send_time = now
                    sent_this_frame = True

        # -------- Telemetry --------
        T = now - t0
        if sent_this_frame or state == SEEKING:
            print(f"{T:.3f} {d_roll:+.3f} {d_pitch:+.3f} {d_yaw:+.3f} {state}")
        else:
            print(f"{T:.3f} +0.000 +0.000 +0.000 {state}")

        # -------- UI --------
        if DRAW:
            l, t_, r, b = map(int, box)
            # Green box at CENTER of frame
            cv2.rectangle(frame, (l, t_), (r, b), (40, 220, 40), 1)

            # Cross at anchor = center
            cv2.drawMarker(
                frame,
                (int(center_anchor[0]), int(center_anchor[1])),
                (0, 200, 0),
                cv2.MARKER_CROSS, 12, 2
            )

            # Red dot at smoothed mouth location
            cv2.circle(frame, (int(smoothed[0]), int(smoothed[1])), 4, (0, 0, 255), -1)

            state_txt = "LOCKED" if state == LOCKED else "SEEKING"
            cv2.putText(frame, f"state:{state_txt}", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40,220,40), 2, cv2.LINE_AA)
            if state == SEEKING:
                cv2.putText(frame, f"dR:{d_roll:+.2f} dP:{d_pitch:+.2f} dY:{d_yaw:+.2f}",
                            (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (40,220,40), 2, cv2.LINE_AA)

            cv2.imshow("Centroid Tracker (center-anchored)", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Stopped.")

if __name__ == "__main__":
    main()

#   - As long as the mouth centroid is inside the box, we do nothing.
#   - When the centroid exits the box, we:
#       * Compute the full offset (center -> current centroid)
#       * Send only a FRACTION of that as a micro-step
#       * Repeat while outside the box, until the centroid returns inside.
#
# Gimbal transport:
#   - Uses SimpleBGC SerialAPI C library exposed via simplebgc_shim.c
#   - Python works in ZERO-BASED ABSOLUTE angles:
#       * At startup, we read the current board angles and call that (0,0,0)
#       * We keep cmd_roll/pitch/yaw in this zero-based frame
#       * Before sending, we add the startup baseline back so the board
#         receives its own absolute angles.
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

# FOV estimates (can still be tuned; micro-steps make you less sensitive)
FOV_H_DEG = 65.0
FOV_V_DEG = 48.75

# Stable box: fraction of HALF-frame sizes
STABLE_SCALAR   = 0.06    # tighten to 0.05 or 0.04 if too tolerant
WINDOW_SEC      = 0.6     # history window for stability metrics

# Stability gates (used while SEEKING)
VEL_THRESH_DEG_S  = 2.5   # median angular speed threshold
POS_STD_THRESH_PX = 2.5   # positional stddev threshold

# Micro-step control
STEP_FRACTION = 0.2       # 20% of the full offset per step (try 0.1–0.3)

# Send gating
SEND_TIME_LIMITER = 0.25  # min seconds between sends
MIN_STEP_DEG_YAW   = 0.3
MIN_STEP_DEG_PITCH = 0.3
MIN_STEP_DEG_ROLL  = 1.0

# Sign convention to match gimbal axes (tune as needed)
AXIS_SIGN = {"yaw": 1, "pitch": 1, "roll": 1}

# Gimbal / test control
TEST = 0  # 1 = log only; 0 = send to gimbal via libsimplebgc.so

# Path to the shared library we built from SerialAPI + shim
LIB_PATH = os.path.expanduser(
    "~/senior_design/A-dec-Senior-Design/camera/testing/SerialLibrary/serialAPI/libsimplebgc.so"
)

DRAW = True
SMOOTH_ALPHA = 0.25
MAX_LOST_FRAMES = 10

# --------- Quiet noisy TF/MP logs ----------
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

# baseline angles at startup (board frame)
_baseline_yaw_deg   = 0.0
_baseline_pitch_deg = 0.0
_baseline_roll_deg  = 0.0
_baseline_set       = False

def init_sbgc():
    """Load libsimplebgc.so, set prototypes, init board, and capture baseline."""
    global _bgc_lib, _bgc_initialized
    global _baseline_yaw_deg, _baseline_pitch_deg, _baseline_roll_deg, _baseline_set

    if TEST == 1:
        print("# TEST mode: not loading SBGC library.")
        _bgc_lib = None
        _bgc_initialized = False
        _baseline_yaw_deg = 0.0
        _baseline_pitch_deg = 0.0
        _baseline_roll_deg = 0.0
        _baseline_set = True
        return

    try:
        _bgc_lib = ctypes.CDLL(LIB_PATH)
        print(f"# Loaded SBGC library from {LIB_PATH}")
    except OSError as e:
        print(f"# ERROR loading {LIB_PATH}: {e}")
        _bgc_lib = None
        _bgc_initialized = False
        return

    # Prototypes from simplebgc_shim.c
    _bgc_lib.bgc_init.argtypes = []
    _bgc_lib.bgc_init.restype  = ctypes.c_int

    # NOTE: shim takes (roll_deg, pitch_deg, yaw_deg)
    _bgc_lib.bgc_control_angles.argtypes = [
        ctypes.c_float, ctypes.c_float, ctypes.c_float
    ]
    _bgc_lib.bgc_control_angles.restype = ctypes.c_int

    # Optional helpers from shim
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

    # Initialize library (sets control config + motors ON inside shim)
    rc = _bgc_lib.bgc_init()
    if rc != 0:
        print(f"# ERROR: bgc_init() returned {rc}")
        _bgc_initialized = False
        return

    # Capture baseline board angles at startup
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
            _baseline_set       = True
            print("# Baseline angles captured from board:")
            print(f"#   yaw={_baseline_yaw_deg:.2f}, "
                  f"pitch={_baseline_pitch_deg:.2f}, "
                  f"roll={_baseline_roll_deg:.2f}")
        else:
            print(f"# WARN: bgc_get_angles() returned {rc2}, using baseline = 0,0,0")
            _baseline_yaw_deg = 0.0
            _baseline_pitch_deg = 0.0
            _baseline_roll_deg = 0.0
            _baseline_set = True
    else:
        print("# WARN: bgc_get_angles not available in shim; using baseline = 0,0,0")
        _baseline_yaw_deg = 0.0
        _baseline_pitch_deg = 0.0
        _baseline_roll_deg = 0.0
        _baseline_set = True

    # Optional beep
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
    """
    Log attempted commands to TEST_LOG instead of moving gimbal.
    """
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
    If TEST=1: only log to file.
    If TEST=0: send angles to gimbal via bgc_control_angles(),
               using zero-based software frame:
                   board_angle = baseline_angle + abs_angle
    """
    global _baseline_yaw_deg, _baseline_pitch_deg, _baseline_roll_deg

    # Compute board-frame absolute angles by adding baseline
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

    # Shim order: (roll, pitch, yaw)
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

    # State
    t0 = time.time()
    prev_smoothed = None
    prev_time = None
    last_send_time = 0.0
    lost = 0

    # Roll reference (we store the roll when face is centered)
    roll_anchor_deg = None

    # Zero-based absolute commanded angles (software frame)
    cmd_roll_deg  = 0.0
    cmd_pitch_deg = 0.0
    cmd_yaw_deg   = 0.0

    LOCKED, SEEKING = 0, 1
    state = LOCKED

    pos_x = TimedHist(WINDOW_SEC)
    pos_y = TimedHist(WINDOW_SEC)
    vel_h = TimedHist(WINDOW_SEC)

    # Clear the test file at the start of every run
    try:
        with open(TEST_LOG, "w") as f:
            f.write("# SimpleBGC commands (TEST/zero-based, micro-steps, center-anchored) — fresh run\n")
            f.write(f"# Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        print(f"# Cleared {TEST_LOG} for a new session.")
    except Exception as e:
        print(f"# ERROR clearing test log: {e}")

    print("# T roll pitch yaw state")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("# WARN: frame grab failed")
            break
        now = time.time()
        h, w = frame.shape[:2]

        # Fixed anchor at center of frame
        center_anchor = (w / 2.0, h / 2.0)
        box = build_stable_box(center_anchor, w, h, STABLE_SCALAR)

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
                prev_time = None
                roll_anchor_deg = None
            T = now - t0
            print(f"{T:.3f} +0.000 +0.000 +0.000 {state}")
            if DRAW:
                cv2.imshow("Centroid Tracker (center-anchored)", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            continue
        lost = 0

        # Smooth centroid
        smoothed = ema_point(centroid, prev_smoothed, SMOOTH_ALPHA)

        # Init timing
        if prev_time is None:
            prev_time = now

        # Is centroid inside the center box?
        inside = inside_box(smoothed, box)

        # Compute dt-based velocity for stability metrics
        dt = max(1e-6, now - prev_time)
        if prev_smoothed is None:
            speed = 0.0
        else:
            dx_px_dt = smoothed[0] - prev_smoothed[0]
            dy_px_dt = smoothed[1] - prev_smoothed[1]
            dvx, dvy = pixels_to_deg(dx_px_dt, dy_px_dt, w, h, FOV_H_DEG, FOV_V_DEG)
            speed = math.hypot(dvx, dvy) / dt  # deg/s
        prev_time = now
        prev_smoothed = smoothed

        # Default console outputs
        d_yaw = d_pitch = d_roll = 0.0
        sent_this_frame = False

        # Update histories
        if inside:
            state = LOCKED
            pos_x.clear()
            pos_y.clear()
            vel_h.clear()
            # When centered, refresh our roll reference
            if roll_now_deg is not None:
                roll_anchor_deg = roll_now_deg
        else:
            state = SEEKING
            pos_x.add(now, smoothed[0])
            pos_y.add(now, smoothed[1])
            vel_h.add(now, speed)

        # Only compute moves when SEEKING
        if state == SEEKING:
            xs, ys = pos_x.values(), pos_y.values()
            pos_std = 999.0
            if len(xs) >= 6 and len(ys) >= 6:
                pos_std = 0.5 * (statistics.pstdev(xs) + statistics.pstdev(ys))
            speeds = vel_h.values()
            vel_med = statistics.median(speeds) if len(speeds) >= 3 else 999.0
            is_stable_here = (vel_med < VEL_THRESH_DEG_S) and (pos_std < POS_STD_THRESH_PX)

            # Full desired offset from center -> smoothed
            dx_px = smoothed[0] - center_anchor[0]
            dy_px = smoothed[1] - center_anchor[1]
            full_d_yaw, full_d_pitch = pixels_to_deg(dx_px, dy_px, w, h, FOV_H_DEG, FOV_V_DEG)

            # Roll offset relative to roll when centered
            d_roll_full = 0.0
            if roll_now_deg is not None and roll_anchor_deg is not None:
                d_roll_full = roll_now_deg - roll_anchor_deg

            # Apply axis signs
            full_d_yaw   = AXIS_SIGN["yaw"]   * full_d_yaw
            full_d_pitch = AXIS_SIGN["pitch"] * full_d_pitch
            d_roll_full  = AXIS_SIGN["roll"]  * d_roll_full

            can_time = (now - last_send_time) >= SEND_TIME_LIMITER

            if is_stable_here and can_time:
                # Micro-step towards the center
                d_yaw   = full_d_yaw   * STEP_FRACTION
                d_pitch = full_d_pitch * STEP_FRACTION
                d_roll  = d_roll_full  * STEP_FRACTION

                # Enforce minimum movement thresholds
                if abs(d_yaw)   < MIN_STEP_DEG_YAW:   d_yaw = 0.0
                if abs(d_pitch) < MIN_STEP_DEG_PITCH: d_pitch = 0.0
                if abs(d_roll)  < MIN_STEP_DEG_ROLL:  d_roll = 0.0

                if (d_yaw != 0.0) or (d_pitch != 0.0) or (d_roll != 0.0):
                    # Update zero-based commanded angles
                    cmd_yaw_deg   += d_yaw
                    cmd_pitch_deg += d_pitch
                    cmd_roll_deg  += d_roll

                    ok = send_or_log_angles(
                        d_yaw, d_pitch, d_roll,
                        cmd_yaw_deg, cmd_pitch_deg, cmd_roll_deg
                    )
                    if ok:
                        last_send_time = now
                        sent_this_frame = True

        # -------- Telemetry --------
        T = now - t0
        if sent_this_frame or state == SEEKING:
            print(f"{T:.3f} {d_roll:+.3f} {d_pitch:+.3f} {d_yaw:+.3f} {state}")
        else:
            print(f"{T:.3f} +0.000 +0.000 +0.000 {state}")

        # -------- UI --------
        if DRAW:
            l, t_, r, b = map(int, box)
            # Green box at CENTER of frame
            cv2.rectangle(frame, (l, t_), (r, b), (40, 220, 40), 1)

            # Cross at anchor = center
            cv2.drawMarker(
                frame,
                (int(center_anchor[0]), int(center_anchor[1])),
                (0, 200, 0),
                cv2.MARKER_CROSS, 12, 2
            )

            # Red dot at smoothed mouth location
            cv2.circle(frame, (int(smoothed[0]), int(smoothed[1])), 4, (0, 0, 255), -1)

            state_txt = "LOCKED" if state == LOCKED else "SEEKING"
            cv2.putText(frame, f"state:{state_txt}", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40,220,40), 2, cv2.LINE_AA)
            if state == SEEKING:
                cv2.putText(frame, f"dR:{d_roll:+.2f} dP:{d_pitch:+.2f} dY:{d_yaw:+.2f}",
                            (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (40,220,40), 2, cv2.LINE_AA)

            cv2.imshow("Centroid Tracker (center-anchored)", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Stopped.")

if __name__ == "__main__":
    main()
