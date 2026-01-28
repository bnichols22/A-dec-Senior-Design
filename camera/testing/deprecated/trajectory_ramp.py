#!/usr/bin/env python3
# ==============================================================
# Option 2: Queued mini-trajectory ramp
#   - Keeps your SEEKING "micro-step" decision behavior
#   - But executes each decided micro-step as N small setpoint updates
#     spaced over time to look continuous.
# ==============================================================

import os, sys, time, math, warnings, statistics, ctypes
import cv2
import mediapipe as mp
from collections import deque

BASE_DIR = os.path.expanduser('~/senior_design/A-dec-Senior-Design/camera/testing')
os.makedirs(BASE_DIR, exist_ok=True)

LOG_PATH  = os.path.join(BASE_DIR, 'face_track_log.txt')
TEST_LOG  = os.path.join(BASE_DIR, 'Board_Serial_Command_Test.txt')

print(f"# ADEC BASE_DIR = {BASE_DIR}")
print(f"# ADEC TEST_LOG = {TEST_LOG}")

CAM_INDEX = 0
FOV_H_DEG = 65.0
FOV_V_DEG = 48.75

STABLE_SCALAR = 0.06
WINDOW_SEC = 0.6

STABLE_STOP_SEEKING_THRESHOLD = 0.025

VEL_THRESH_DEG_S  = 2.5
POS_STD_THRESH_PX = 2.5

STEP_FRACTION = 0.22

# Decision limiter (how often we decide a new move)
DECISION_TIME_LIMITER = 0.07  # try 0.05–0.10

# Ramp execution: how many sub-steps and how fast
RAMP_SUBSTEPS = 6             # try 4–10
RAMP_DT = 0.02                # 20ms between substeps => ~50Hz during ramp

MIN_STEP_DEG_YAW   = 0.10
MIN_STEP_DEG_PITCH = 0.10
MIN_STEP_DEG_ROLL  = 0.15

MAX_STEP_DEG_YAW   = 2.0
MAX_STEP_DEG_PITCH = 2.0
MAX_STEP_DEG_ROLL  = 2.0

AXIS_SIGN = {"yaw": 1, "pitch": 1, "roll": 1}

TEST = 0
LIB_PATH = os.path.expanduser(
    "~/senior_design/A-dec-Senior-Design/camera/testing/SerialLibrary/serialAPI/libsimplebgc.so"
)

DRAW = True
SMOOTH_ALPHA = 0.25
MAX_LOST_FRAMES = 10

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")
sys.stderr = open(LOG_PATH, "w")


def ema_point(curr, prev, alpha):
    if prev is None:
        return curr
    return (alpha * curr[0] + (1 - alpha) * prev[0],
            alpha * curr[1] + (1 - alpha) * prev[1])


def clamp(val, lo, hi):
    return max(lo, min(hi, val))


def pixels_to_deg(dx_px, dy_px, w, h, fov_h, fov_v):
    half_w, half_h = w / 2.0, h / 2.0
    yaw_deg   = (dx_px / half_w) * (fov_h / 2.0)
    pitch_deg = (dy_px / half_h) * (fov_v / 2.0)
    return yaw_deg, pitch_deg


def angle_deg(p1, p2):
    return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))


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


_bgc_lib = None
_bgc_initialized = False

_baseline_yaw_deg   = 0.0
_baseline_pitch_deg = 0.0
_baseline_roll_deg  = 0.0


def init_sbgc():
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
    _bgc_lib.bgc_init.restype = ctypes.c_int

    _bgc_lib.bgc_control_angles.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float]
    _bgc_lib.bgc_control_angles.restype = ctypes.c_int

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

    if _bgc_lib.bgc_get_angles is not None:
        yaw = ctypes.c_float()
        pitch = ctypes.c_float()
        roll = ctypes.c_float()
        rc2 = _bgc_lib.bgc_get_angles(ctypes.byref(yaw), ctypes.byref(pitch), ctypes.byref(roll))
        if rc2 == 0:
            _baseline_yaw_deg = float(yaw.value)
            _baseline_pitch_deg = float(pitch.value)
            _baseline_roll_deg = float(roll.value)
            print("# Baseline angles captured from board:")
            print(f"#   yaw={_baseline_yaw_deg:.2f}, pitch={_baseline_pitch_deg:.2f}, roll={_baseline_roll_deg:.2f}")
        else:
            print(f"# WARN: bgc_get_angles() returned {rc2}, using baseline = 0,0,0")

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

    try:
        with open(TEST_LOG, "w") as f:
            f.write("# SimpleBGC commands (ramp queue) — fresh run\n")
            f.write(f"# Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    except Exception:
        pass

    t0 = time.time()
    prev_smoothed = None
    prev_time = None
    lost = 0

    anchor = None

    cmd_roll = 0.0
    cmd_pitch = 0.0
    cmd_yaw = 0.0

    LOCKED, SEEKING = 0, 1
    state = LOCKED

    pos_x = TimedHist(WINDOW_SEC)
    pos_y = TimedHist(WINDOW_SEC)
    vel_h = TimedHist(WINDOW_SEC)

    last_decision_time = 0.0

    # Ramp queue holds tuples: (send_time, abs_yaw, abs_pitch, abs_roll)
    ramp_queue = deque()

    print("# T dR dP dY sent state radial_norm queue_len")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("# WARN: frame grab failed")
            break

        now = time.time()
        h, w = frame.shape[:2]

        if anchor is None:
            anchor = (w / 2.0, h / 2.0)

        # --- Face detection (MediaPipe) ---
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
            if lost > MAX_LOST_FRAMES:
                prev_smoothed = None
                prev_time = None
                ramp_queue.clear()
            if DRAW:
                cv2.imshow("Centroid Tracker (Option 2)", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            continue
        lost = 0

        smoothed = ema_point(centroid, prev_smoothed, SMOOTH_ALPHA)

        if prev_time is None:
            prev_time = now
        dt = max(1e-6, now - prev_time)

        # Speed estimate (deg/s)
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

        # Stability metrics (same as your logic)
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

        # State transitions
        if state == LOCKED:
            if not inside:
                state = SEEKING
        else:
            if within_stop_thresh:
                state = LOCKED
                ramp_queue.clear()

        d_yaw = d_pitch = d_roll = 0.0
        sent = 0

        # 1) Execute queued ramp substeps if ready
        while ramp_queue and ramp_queue[0][0] <= now:
            _, abs_yaw, abs_pitch, abs_roll = ramp_queue.popleft()
            ok_send = send_or_log_angles(0.0, 0.0, 0.0, abs_yaw, abs_pitch, abs_roll)
            if ok_send:
                sent = 1

        # 2) If SEEKING and queue is empty, decide a new micro-step and enqueue a ramp
        can_decide = (now - last_decision_time) >= DECISION_TIME_LIMITER
        if state == SEEKING and not ramp_queue and can_decide and is_stable_here:
            full_d_yaw, full_d_pitch = pixels_to_deg(dx_center, dy_center, w, h, FOV_H_DEG, FOV_V_DEG)

            full_d_yaw *= AXIS_SIGN["yaw"]
            full_d_pitch *= AXIS_SIGN["pitch"]

            d_yaw = clamp(full_d_yaw * STEP_FRACTION,   -MAX_STEP_DEG_YAW,   +MAX_STEP_DEG_YAW)
            d_pitch = clamp(full_d_pitch * STEP_FRACTION, -MAX_STEP_DEG_PITCH, +MAX_STEP_DEG_PITCH)
            d_roll = 0.0

            if abs(d_yaw)   < MIN_STEP_DEG_YAW:   d_yaw = 0.0
            if abs(d_pitch) < MIN_STEP_DEG_PITCH: d_pitch = 0.0
            if abs(d_roll)  < MIN_STEP_DEG_ROLL:  d_roll = 0.0

            # Only enqueue if there is real movement
            if (d_yaw != 0.0) or (d_pitch != 0.0) or (d_roll != 0.0):
                # Target end command after applying this micro-step
                target_yaw = cmd_yaw + d_yaw
                target_pitch = cmd_pitch + d_pitch
                target_roll = cmd_roll + d_roll

                # Build a ramp from current cmd -> target cmd
                # We keep cmd_* as the "end-of-ramp" value for bookkeeping
                cmd_yaw = target_yaw
                cmd_pitch = target_pitch
                cmd_roll = target_roll

                start_time = now + RAMP_DT
                for i in range(1, RAMP_SUBSTEPS + 1):
                    u = i / float(RAMP_SUBSTEPS)
                    abs_yaw = (1 - u) * (cmd_yaw - d_yaw) + u * cmd_yaw
                    abs_pitch = (1 - u) * (cmd_pitch - d_pitch) + u * cmd_pitch
                    abs_roll = (1 - u) * (cmd_roll - d_roll) + u * cmd_roll
                    ramp_queue.append((start_time + (i - 1) * RAMP_DT, abs_yaw, abs_pitch, abs_roll))

                last_decision_time = now

        # Telemetry
        T = now - t0
        print(f"{T:.3f} {d_roll:+.3f} {d_pitch:+.3f} {d_yaw:+.3f} {sent} {state} r={radial_norm:.3f} q={len(ramp_queue)}")

        # UI
        if DRAW:
            l, t_, r, b = map(int, box)
            cv2.rectangle(frame, (l, t_), (r, b), (40, 220, 40), 1)
            cv2.drawMarker(frame, (int(anchor[0]), int(anchor[1])), (0, 200, 0),
                           cv2.MARKER_CROSS, 12, 2)
            cv2.circle(frame, (int(smoothed[0]), int(smoothed[1])), 4, (0, 0, 255), -1)

            state_txt = "LOCKED" if state == LOCKED else "SEEKING"
            cv2.putText(frame, f"state:{state_txt}", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40,220,40), 2, cv2.LINE_AA)
            cv2.putText(frame, f"r={radial_norm:.3f} q={len(ramp_queue)}", (10, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40,220,40), 2, cv2.LINE_AA)

            cv2.imshow("Centroid Tracker (Option 2)", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Stopped.")


if __name__ == "__main__":
    main()
