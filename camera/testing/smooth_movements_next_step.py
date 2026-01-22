#!/usr/bin/env python3
# ==============================================================
# File: full_tracking.py
# Purpose:
#   Center-locked stable box face tracker with a simple state machine:
#   - LOCKED: do nothing while face is inside the center stable box.
#   - SEEKING: while face is outside, compute an angular error from center
#              and drive the gimbal back using smooth continuous setpoint streaming.
#
# Key fix in this version:
#   - Poll board angles ONCE at startup (retries + debug)
#   - Initialize cmd_* to those angles so first send is a HOLD,
#     avoiding the "first movement jumps wrong" bug.
#
# Transport:
#   - SimpleBGC SerialAPI shim:
#       bgc_init()
#       bgc_control_angles(roll, pitch, yaw)
#       bgc_get_angles(&roll,&pitch,&yaw)   <-- required for this file
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

# Stable box dimensions (fraction of HALF-frame)
STABLE_SCALAR = 0.06
WINDOW_SEC = 0.6

# Stop SEEKING when close enough to center (normalized radial distance)
STABLE_STOP_SEEKING_THRESHOLD = 0.025

# Stability gates (used only as a "don't move if it's insane" block)
VEL_THRESH_DEG_S  = 2.5
POS_STD_THRESH_PX = 2.5

# Continuous control strength (bigger = faster convergence)
STEP_FRACTION = 0.22  # try 0.18–0.30

# Command streaming rate (Hz)
COMMAND_HZ = 45.0     # try 35–60
COMMAND_PERIOD = 1.0 / COMMAND_HZ

# Smooth the commanded angles (0=no smoothing, 1=very slow)
CMD_ANGLE_EMA_ALPHA = 0.35  # try 0.25–0.55

# Minimum degrees to send (reduce to avoid "stall then jump")
MIN_STEP_DEG_YAW   = 0.10
MIN_STEP_DEG_PITCH = 0.10
MIN_STEP_DEG_ROLL  = 0.15

# Optional per-axis max step per command (prevents sudden jumps)
MAX_STEP_DEG_YAW   = 2.0
MAX_STEP_DEG_PITCH = 2.0
MAX_STEP_DEG_ROLL  = 2.0

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
    """Exponential moving average for 2D points."""
    if prev is None:
        return curr
    return (alpha * curr[0] + (1 - alpha) * prev[0],
            alpha * curr[1] + (1 - alpha) * prev[1])


def ema_scalar(curr, prev, alpha):
    """Exponential moving average for scalars."""
    if prev is None:
        return curr
    return alpha * curr + (1 - alpha) * prev


def clamp(val, lo, hi):
    """Clamp val into [lo, hi]."""
    return max(lo, min(hi, val))


def pixels_to_deg(dx_px, dy_px, w, h, fov_h, fov_v):
    """Convert pixel offsets to approximate angular offsets using camera FOV."""
    half_w, half_h = w / 2.0, h / 2.0
    yaw_deg   = (dx_px / half_w) * (fov_h / 2.0)
    pitch_deg = (dy_px / half_h) * (fov_v / 2.0)
    return yaw_deg, pitch_deg


def angle_deg(p1, p2):
    """Angle (deg) of vector p1->p2."""
    return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))


def build_stable_box(center_xy, w, h, scalar):
    """Build a rectangle centered at center_xy sized by scalar of half-frame."""
    cx, cy = center_xy
    half_w = scalar * (w / 2.0)
    half_h = scalar * (h / 2.0)
    return (cx - half_w, cy - half_h, cx + half_w, cy + half_h)


def inside_box(pt, box):
    """Return True if point pt lies inside box (l,t,r,b)."""
    x, y = pt
    l, t, r, b = box
    return (l <= x <= r) and (t <= y <= b)


class TimedHist:
    """Time-windowed history buffer for stability metrics."""
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
# SBGC shim bindings (ctypes)
# ----------------------------------------------------------------------
_bgc_lib = None
_bgc_initialized = False


def get_board_angles_debug(retries=30, delay_s=0.05):
    """
    Poll angles from the board via shim function:

        int bgc_get_angles(float *roll, float *pitch, float *yaw)

    Returns (roll,pitch,yaw) floats on success, else None.

    This is *debuggy* on purpose: it prints symbol presence and rc codes.
    """
    global _bgc_lib, _bgc_initialized

    if TEST == 1:
        print("# get_board_angles: TEST=1 -> returning 0,0,0")
        return (0.0, 0.0, 0.0)

    if _bgc_lib is None or not _bgc_initialized:
        print("# get_board_angles: lib not initialized")
        return None

    if not hasattr(_bgc_lib, "bgc_get_angles"):
        print("# get_board_angles: bgc_get_angles symbol NOT FOUND in libsimplebgc.so")
        return None

    roll = ctypes.c_float()
    pitch = ctypes.c_float()
    yaw = ctypes.c_float()

    last_rc = None
    for attempt in range(retries):
        rc = _bgc_lib.bgc_get_angles(ctypes.byref(roll), ctypes.byref(pitch), ctypes.byref(yaw))
        last_rc = rc

        if rc == 0:
            r = float(roll.value)
            p = float(pitch.value)
            y = float(yaw.value)
            print(f"# get_board_angles: OK on attempt {attempt+1} -> roll={r:.3f}, pitch={p:.3f}, yaw={y:.3f}")
            return (r, p, y)

        if attempt < 5 or attempt == retries - 1:
            print(f"# get_board_angles: rc={rc} (attempt {attempt+1}/{retries})")
        time.sleep(delay_s)

    print(f"# get_board_angles: FAILED after {retries} attempts, last rc={last_rc}")
    return None


def init_sbgc():
    """Load SBGC library and initialize board."""
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

    # Required symbols
    _bgc_lib.bgc_init.argtypes = []
    _bgc_lib.bgc_init.restype = ctypes.c_int

    # NOTE: shim expects (roll, pitch, yaw)
    _bgc_lib.bgc_control_angles.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float]
    _bgc_lib.bgc_control_angles.restype = ctypes.c_int

    # Optional (but we want it)
    if hasattr(_bgc_lib, "bgc_get_angles"):
        _bgc_lib.bgc_get_angles.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        _bgc_lib.bgc_get_angles.restype = ctypes.c_int

    rc = _bgc_lib.bgc_init()
    if rc != 0:
        print(f"# ERROR: bgc_init() returned {rc}")
        _bgc_initialized = False
        return

    _bgc_initialized = True
    print("# SBGC initialization complete.")


def write_test_line(abs_roll, abs_pitch, abs_yaw):
    """Log command attempts to TEST_LOG instead of moving gimbal."""
    try:
        with open(TEST_LOG, 'a', buffering=1) as f:
            f.write(
                f"T={time.time():.3f} "
                f"cmdR={abs_roll:+.3f} cmdP={abs_pitch:+.3f} cmdY={abs_yaw:+.3f}\n"
            )
            f.flush()
            os.fsync(f.fileno())
        return True
    except Exception as e:
        print(f"# TEST_FILE_ERROR: {e}")
        return False


def send_or_log_angles(abs_roll_deg, abs_pitch_deg, abs_yaw_deg):
    """
    Send absolute angles (roll,pitch,yaw) to gimbal, or log if TEST=1.
    """
    if TEST == 1:
        return write_test_line(abs_roll_deg, abs_pitch_deg, abs_yaw_deg)

    if _bgc_lib is None or not _bgc_initialized:
        print("# ERROR: SBGC shim not initialized.")
        return False

    rc = _bgc_lib.bgc_control_angles(
        ctypes.c_float(abs_roll_deg),
        ctypes.c_float(abs_pitch_deg),
        ctypes.c_float(abs_yaw_deg),
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

    # Clear test log
    try:
        with open(TEST_LOG, "w") as f:
            f.write("# SimpleBGC commands (continuous streaming) — fresh run\n")
            f.write(f"# Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    except Exception:
        pass

    # ------------------------------------------------------------------
    # NEW: one-shot poll of board angles to initialize command state
    # ------------------------------------------------------------------
    initial_angles = get_board_angles_debug(retries=40, delay_s=0.05)
    if initial_angles is None:
        print("# WARNING: Could not poll angles from board.")
        print("#          Tracker will still run, but first movement may jump if")
        print("#          your gimbal is NOT already at (0,0,0) in the board's angle frame.")
        init_roll = 0.0
        init_pitch = 0.0
        init_yaw = 0.0
    else:
        init_roll, init_pitch, init_yaw = initial_angles

        # If you're *always* getting zeros, that may be normal for frame-referenced angles.
        # Still useful: it makes the first command a HOLD(0,0,0) instead of a random jump.
        if abs(init_roll) < 1e-3 and abs(init_pitch) < 1e-3 and abs(init_yaw) < 1e-3:
            print("# NOTE: Polled angles are ~0,0,0.")
            print("#       This can be normal if the board reports stabilized camera angles.")
            print("#       If you need motor/encoder positions, you must read different RT data fields.")

    # Soft-start: send one hold command immediately (optional but helps)
    if TEST != 1:
        print(f"# Sending initial HOLD: roll={init_roll:.3f} pitch={init_pitch:.3f} yaw={init_yaw:.3f}")
        send_or_log_angles(init_roll, init_pitch, init_yaw)
        time.sleep(0.05)

    t0 = time.time()
    prev_smoothed = None
    prev_time = None
    lost = 0

    anchor = None  # fixed at frame center

    # Software-frame absolute commands (initialize to board angles to avoid first jump)
    cmd_roll = init_roll
    cmd_pitch = init_pitch
    cmd_yaw = init_yaw

    # Smoothed command outputs
    smooth_cmd_roll = init_roll
    smooth_cmd_pitch = init_pitch
    smooth_cmd_yaw = init_yaw

    LOCKED, SEEKING = 0, 1
    state = LOCKED

    pos_x = TimedHist(WINDOW_SEC)
    pos_y = TimedHist(WINDOW_SEC)
    vel_h = TimedHist(WINDOW_SEC)

    last_send_time = 0.0

    print("# T dR dP dY sent state radial_norm")

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
            if DRAW:
                cv2.imshow("Centroid Tracker (Option 1)", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            continue
        lost = 0

        smoothed = ema_point(centroid, prev_smoothed, SMOOTH_ALPHA)

        if prev_time is None:
            prev_time = now
        dt = max(1e-6, now - prev_time)

        # Speed estimate for stability (deg/s)
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

        # Update stability history (used only to BLOCK when crazy)
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

        # --- State transitions ---
        if state == LOCKED:
            if not inside:
                state = SEEKING
        else:
            if within_stop_thresh:
                state = LOCKED

        # --- Control: continuously stream while SEEKING (unless too wild) ---
        d_yaw = d_pitch = d_roll = 0.0
        sent = 0

        if state == SEEKING and not too_wild:
            full_d_yaw, full_d_pitch = pixels_to_deg(dx_center, dy_center, w, h, FOV_H_DEG, FOV_V_DEG)

            full_d_yaw   *= AXIS_SIGN["yaw"]
            full_d_pitch *= AXIS_SIGN["pitch"]

            d_yaw = clamp(full_d_yaw * STEP_FRACTION,       -MAX_STEP_DEG_YAW,   +MAX_STEP_DEG_YAW)
            d_pitch = clamp(full_d_pitch * STEP_FRACTION,   -MAX_STEP_DEG_PITCH, +MAX_STEP_DEG_PITCH)

            # Roll disabled in this tracker
            d_roll = 0.0

            if abs(d_yaw)   < MIN_STEP_DEG_YAW:   d_yaw = 0.0
            if abs(d_pitch) < MIN_STEP_DEG_PITCH: d_pitch = 0.0
            if abs(d_roll)  < MIN_STEP_DEG_ROLL:  d_roll = 0.0

            cmd_yaw   += d_yaw
            cmd_pitch += d_pitch
            cmd_roll  += d_roll

        # --- Fixed-rate streaming ---
        if (now - last_send_time) >= COMMAND_PERIOD:
            smooth_cmd_yaw   = ema_scalar(cmd_yaw,   smooth_cmd_yaw,   CMD_ANGLE_EMA_ALPHA)
            smooth_cmd_pitch = ema_scalar(cmd_pitch, smooth_cmd_pitch, CMD_ANGLE_EMA_ALPHA)
            smooth_cmd_roll  = ema_scalar(cmd_roll,  smooth_cmd_roll,  CMD_ANGLE_EMA_ALPHA)

            ok_send = send_or_log_angles(smooth_cmd_roll, smooth_cmd_pitch, smooth_cmd_yaw)
            if ok_send:
                last_send_time = now
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

            cv2.imshow("Centroid Tracker (Option 1)", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Stopped.")


if __name__ == "__main__":
    main()
