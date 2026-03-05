# ==============================================================
# File: demo_face_track.py
# Purpose:
#   Track the patient's mouth in the camera image and drive a
#   SimpleBGC gimbal so that the mouth is centered on the
#   optical axis. The algorithm uses a small "stable" box
#   at the center of the frame and a simple two-state
#   (LOCKED/SEEKING) state machine:
#
#   - LOCKED:
#       * The mouth centroid is inside the stable box.
#       * No gimbal commands are sent.
#
#   - SEEKING:
#       * The mouth centroid is outside the stable box.
#       * Compute the offset from the center, convert to
#         gimbal angles, and send only a fraction of that
#         offset as a "micro-step" command.
#       * Repeat micro-steps until the mouth is within a
#         tighter threshold around the center, then return
#         to LOCKED.
#
#   The stable box is locked to the CENTER of the image and
#   never moves with the mouth. When the mouth is inside this
#   center box, the flashlight (co-aligned with the camera)
#   is approximately pointed at the mouth.
#
#   The gimbal is controlled in a ZERO-BASED ABSOLUTE frame:
#     - At startup, we read the board's current yaw/pitch/roll
#       and treat that as (0,0,0) in software.
#     - We track commanded angles in that software frame.
#     - Before sending, we add the baseline back in so the board
#       receives angles in its own absolute frame.
# ==============================================================

import os
import sys
import time
import math
import warnings
import statistics
import ctypes
from collections import deque

import cv2
import mediapipe as mp

# --------- Paths: force to your repo folder ----------
# Base directory for logs and test files.
BASE_OUTPUT_DIR = os.path.expanduser('~/senior_design/A-dec-Senior-Design/camera/testing')
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# Path for detailed face-tracking / diagnostic logs.
FACE_TRACK_LOG_PATH = os.path.join(BASE_OUTPUT_DIR, 'face_track_log.txt')

# Path for logging gimbal commands when running in test mode.
GIMBAL_TEST_LOG_PATH = os.path.join(BASE_OUTPUT_DIR, 'Board_Serial_Command_Test.txt')

print(f"# ADEC BASE_OUTPUT_DIR = {BASE_OUTPUT_DIR}")
print(f"# ADEC GIMBAL_TEST_LOG_PATH = {GIMBAL_TEST_LOG_PATH}")

# --------- Vision / tracker config ----------
# Index of the camera device for OpenCV.
CAMERA_INDEX = 0

# Horizontal / vertical camera field-of-view in degrees (tuned by calibration).
CAMERA_FOV_HORIZONTAL_DEG = 65.0
CAMERA_FOV_VERTICAL_DEG = 48.75

# Stable box size as a fraction of HALF-frame width/height.
# Example: 0.06 -> box is ~12% of frame width/height.
STABLE_BOX_SCALE = 0.06

# Time window (seconds) over which we compute stability statistics
# for position and velocity in SEEKING state.
STABILITY_WINDOW_SEC = 0.6

# How close to the frame center (normalized) is "good enough" to stop SEEKING.
# This is a radial distance in normalized units, where 1.0 corresponds
# roughly to "half frame" in each axis.
CENTER_STOP_THRESHOLD_NORM = 0.025

# Thresholds for deciding when motion is "stable enough" to send a micro-step.
# We require low angular velocity + low positional jitter.
STABILITY_VEL_THRESHOLD_DEG_S = 2.5   # median angular speed threshold (deg/s)
STABILITY_POS_STD_THRESHOLD_PX = 2.5  # average std dev of x/y position (pixels)

# Micro-step control: fraction of the full offset applied per command.
# Example: 0.15 -> move 15% of the required yaw/pitch/roll difference per step.
MICRO_STEP_FRACTION = 0.15

# Rate limit for sending gimbal commands (seconds between commands).
SEND_RATE_LIMIT_SEC = 0.085

# Minimum step magnitudes (deg) below which commands are discarded
# to avoid tiny, jittery movements near the target.
MIN_YAW_STEP_DEG = 0.3
MIN_PITCH_STEP_DEG = 0.3
MIN_ROLL_STEP_DEG = 0.3

# Axis sign convention to match the physical gimbal orientation.
AXIS_SIGN_CONVENTION = {
    "yaw": 1,
    "pitch": 1,
    "roll": 1,
}

# If 1: log commands to file only; if 0: send commands to the gimbal.
TEST_MODE = 0

# Path to the SimpleBGC shared library (built from SerialAPI + shim).
SBGC_LIBRARY_PATH = os.path.expanduser(
    "~/senior_design/A-dec-Senior-Design/camera/testing/SerialLibrary/serialAPI/libsimplebgc.so"
)

# If True, show OpenCV debug window(s) with overlay graphics.
DISPLAY_DEBUG_WINDOWS = True

# Exponential moving average smoothing factor for the mouth centroid.
# Smaller values = smoother but more laggy.
CENTROID_EMA_ALPHA = 0.25

# Number of consecutive frames with no detected face before we reset tracking state.
MAX_CONSECUTIVE_LOST_FRAMES = 10

# --------- Quiet noisy TF/MP logs ----------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")
sys.stderr = open(FACE_TRACK_LOG_PATH, "w")


# ---------------- Utilities ----------------
def smooth_point_ema(current_point, previous_point, alpha):
    """
    Compute an exponential moving average (EMA) of a 2D point.

    Args:
        current_point (tuple[float, float]): Current (x, y) point.
        previous_point (tuple[float, float] or None): Previous EMA output.
        alpha (float): EMA weight on the current point in [0, 1].

    Returns:
        tuple[float, float]: Smoothed (x, y) point.
    """
    if previous_point is None:
        return current_point

    return (
        alpha * current_point[0] + (1 - alpha) * previous_point[0],
        alpha * current_point[1] + (1 - alpha) * previous_point[1],
    )


def pixels_to_degrees(dx_pixels, dy_pixels, frame_width, frame_height,
                      fov_horizontal_deg, fov_vertical_deg):
    """
    Convert pixel offsets from the frame center to angular offsets in degrees.

    The mapping assumes a pinhole camera model, treating the camera FOV as
    linearly proportional to the displacement from the image center.

    Args:
        dx_pixels (float): Horizontal pixel offset from frame center.
        dy_pixels (float): Vertical pixel offset from frame center.
        frame_width (int): Width of the frame in pixels.
        frame_height (int): Height of the frame in pixels.
        fov_horizontal_deg (float): Horizontal field-of-view (degrees).
        fov_vertical_deg (float): Vertical field-of-view (degrees).

    Returns:
        tuple[float, float]: (delta_yaw_deg, delta_pitch_deg).
    """
    half_width = frame_width / 2.0
    half_height = frame_height / 2.0

    yaw_deg = (dx_pixels / half_width) * (fov_horizontal_deg / 2.0)
    pitch_deg = (dy_pixels / half_height) * (fov_vertical_deg / 2.0)
    return yaw_deg, pitch_deg


def angle_between_points_deg(point1, point2):
    """
    Compute the angle (in degrees) between two points, using atan2.

    Primarily used to estimate roll angle based on left/right mouth landmarks.

    Args:
        point1 (tuple[float, float]): First point (x, y).
        point2 (tuple[float, float]): Second point (x, y).

    Returns:
        float: Angle in degrees from point1 to point2.
    """
    return math.degrees(math.atan2(point2[1] - point1[1],
                                   point2[0] - point1[0]))


def compute_stable_box(anchor_xy, frame_width, frame_height, scale):
    """
    Build a rectangular "stable region" box centered at anchor_xy.

    The box size is defined as a fraction of the half-frame dimensions.

    Args:
        anchor_xy (tuple[float, float]): (x, y) center for the box.
        frame_width (int): Width of the frame in pixels.
        frame_height (int): Height of the frame in pixels.
        scale (float): Fraction of half-frame size to use as half-box size.

    Returns:
        tuple[float, float, float, float]: (left, top, right, bottom) coordinates.
    """
    center_x, center_y = anchor_xy
    half_box_width = scale * (frame_width / 2.0)
    half_box_height = scale * (frame_height / 2.0)

    return (
        center_x - half_box_width,
        center_y - half_box_height,
        center_x + half_box_width,
        center_y + half_box_height,
    )


def is_point_inside_box(point_xy, box):
    """
    Check if a point lies inside a given axis-aligned bounding box.

    Args:
        point_xy (tuple[float, float]): (x, y) point to test.
        box (tuple[float, float, float, float]): (left, top, right, bottom) box.

    Returns:
        bool: True if the point is inside or on the boundary of the box.
    """
    x, y = point_xy
    left, top, right, bottom = box
    return (left <= x <= right) and (top <= y <= bottom)


class TimedHistory:
    """
    Fixed-duration time history buffer for scalar values.

    This class keeps (timestamp, value) pairs in a deque and automatically
    discards entries older than a specified time window.
    """

    def __init__(self, window_seconds):
        """
        Initialize the history with a given time window.

        Args:
            window_seconds (float): Length of the window in seconds.
        """
        self.window_seconds = window_seconds
        self.buffer = deque()

    def add(self, timestamp, value):
        """
        Add a new (timestamp, value) pair to the history.

        Args:
            timestamp (float): Time in seconds (e.g., time.time()).
            value (float): Scalar value to store.
        """
        self.buffer.append((timestamp, value))
        self._trim(timestamp)

    def values(self):
        """
        Get the current list of values still inside the time window.

        Returns:
            list[float]: List of stored scalar values.
        """
        return [value for _, value in self.buffer]

    def clear(self):
        """
        Clear all entries from the history.
        """
        self.buffer.clear()

    def _trim(self, current_time):
        """
        Remove entries older than the configured window.

        Args:
            current_time (float): Current time used as cutoff reference.
        """
        cutoff_time = current_time - self.window_seconds
        while self.buffer and self.buffer[0][0] < cutoff_time:
            self.buffer.popleft()


# ----------------------------------------------------------------------
# SimpleBGC shim bindings (ctypes) + zero-based angle baseline
# ----------------------------------------------------------------------
_sbgc_lib = None
_sbgc_initialized = False

# Baseline angles at startup (board frame). These represent the
# board's yaw/pitch/roll in degrees at the moment we start, which
# we treat as (0,0,0) in software.
_sbgc_baseline_yaw_deg = 0.0
_sbgc_baseline_pitch_deg = 0.0
_sbgc_baseline_roll_deg = 0.0
_sbgc_baseline_set = False


def init_simplebgc():
    """
    Initialize the SimpleBGC library, motors, and zero-based baseline.

    - Loads the libsimplebgc shared library.
    - Calls bgc_init() to configure and enable the gimbal.
    - Optionally beeps via bgc_beep_once().
    - Reads the current yaw/pitch/roll using bgc_get_angles() and
      stores them as baseline, establishing the software (0,0,0).
    """
    global _sbgc_lib, _sbgc_initialized
    global _sbgc_baseline_yaw_deg, _sbgc_baseline_pitch_deg, _sbgc_baseline_roll_deg
    global _sbgc_baseline_set

    if TEST_MODE == 1:
        print("# TEST mode: not loading SBGC library.")
        _sbgc_lib = None
        _sbgc_initialized = False
        _sbgc_baseline_yaw_deg = 0.0
        _sbgc_baseline_pitch_deg = 0.0
        _sbgc_baseline_roll_deg = 0.0
        _sbgc_baseline_set = True
        return

    try:
        _sbgc_lib = ctypes.CDLL(SBGC_LIBRARY_PATH)
        print(f"# Loaded SBGC library from {SBGC_LIBRARY_PATH}")
    except OSError as exc:
        print(f"# ERROR loading {SBGC_LIBRARY_PATH}: {exc}")
        _sbgc_lib = None
        _sbgc_initialized = False
        return

    # Prototypes from simplebgc_shim.c
    _sbgc_lib.bgc_init.argtypes = []
    _sbgc_lib.bgc_init.restype = ctypes.c_int

    # NOTE: shim takes arguments in (roll_deg, pitch_deg, yaw_deg) order.
    _sbgc_lib.bgc_control_angles.argtypes = [
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
    ]
    _sbgc_lib.bgc_control_angles.restype = ctypes.c_int

    # Optional helpers from shim (beep)
    try:
        _sbgc_lib.bgc_beep_once.argtypes = []
        _sbgc_lib.bgc_beep_once.restype = ctypes.c_int
    except AttributeError:
        pass  # not fatal if not present

    # Optional helper to read current gimbal angles
    try:
        _sbgc_lib.bgc_get_angles.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        _sbgc_lib.bgc_get_angles.restype = ctypes.c_int
    except AttributeError:
        _sbgc_lib.bgc_get_angles = None

    # Initialize library (sets control config + motors ON inside shim)
    init_result = _sbgc_lib.bgc_init()
    if init_result != 0:
        print(f"# ERROR: bgc_init() returned {init_result}")
        _sbgc_initialized = False
        return

    # Capture baseline board angles at startup
    if hasattr(_sbgc_lib, "bgc_get_angles") and _sbgc_lib.bgc_get_angles is not None:
        yaw = ctypes.c_float()
        pitch = ctypes.c_float()
        roll = ctypes.c_float()
        get_result = _sbgc_lib.bgc_get_angles(
            ctypes.byref(yaw),
            ctypes.byref(pitch),
            ctypes.byref(roll),
        )
        if get_result == 0:
            _sbgc_baseline_yaw_deg = float(yaw.value)
            _sbgc_baseline_pitch_deg = float(pitch.value)
            _sbgc_baseline_roll_deg = float(roll.value)
            _sbgc_baseline_set = True
            print("# Baseline angles captured from board:")
            print(f"#   yaw={_sbgc_baseline_yaw_deg:.2f}, "
                  f"pitch={_sbgc_baseline_pitch_deg:.2f}, "
                  f"roll={_sbgc_baseline_roll_deg:.2f}")
        else:
            print(f"# WARN: bgc_get_angles() returned {get_result}, using baseline = 0,0,0")
            _sbgc_baseline_yaw_deg = 0.0
            _sbgc_baseline_pitch_deg = 0.0
            _sbgc_baseline_roll_deg = 0.0
            _sbgc_baseline_set = True
    else:
        print("# WARN: bgc_get_angles not available in shim; using baseline = 0,0,0")
        _sbgc_baseline_yaw_deg = 0.0
        _sbgc_baseline_pitch_deg = 0.0
        _sbgc_baseline_roll_deg = 0.0
        _sbgc_baseline_set = True

    # Optional: audible beep to signal ready
    if hasattr(_sbgc_lib, "bgc_beep_once"):
        try:
            _sbgc_lib.bgc_beep_once()
        except Exception:  # noqa: BLE001
            pass

    _sbgc_initialized = True
    print("# SBGC initialization complete (zero-based frame set).")


# ----------------------------------------------------------------------
# Output: send or write to file (angles, not raw packets)
# ----------------------------------------------------------------------
def write_test_command_log_entry(delta_yaw_deg, delta_pitch_deg, delta_roll_deg,
                                 abs_yaw_deg, abs_pitch_deg, abs_roll_deg,
                                 board_yaw_deg, board_pitch_deg, board_roll_deg):
    """
    Append a human-readable gimbal command entry to the test log file.

    This is used when TEST_MODE == 1, so that commands can be inspected
    without actually moving the gimbal motors.

    Args:
        delta_yaw_deg (float): Incremental yaw change commanded (software frame).
        delta_pitch_deg (float): Incremental pitch change commanded.
        delta_roll_deg (float): Incremental roll change commanded.
        abs_yaw_deg (float): Absolute yaw in software zero-based frame.
        abs_pitch_deg (float): Absolute pitch in software zero-based frame.
        abs_roll_deg (float): Absolute roll in software zero-based frame.
        board_yaw_deg (float): Board-frame absolute yaw angle.
        board_pitch_deg (float): Board-frame absolute pitch angle.
        board_roll_deg (float): Board-frame absolute roll angle.

    Returns:
        bool: True if write succeeded, False on error.
    """
    try:
        with open(GIMBAL_TEST_LOG_PATH, 'a', buffering=1) as log_file:
            log_file.write(
                f"T={time.time():.3f} "
                f"dR={delta_roll_deg:+.2f} dP={delta_pitch_deg:+.2f} dY={delta_yaw_deg:+.2f} | "
                f"absR={abs_roll_deg:+.2f} absP={abs_pitch_deg:+.2f} absY={abs_yaw_deg:+.2f} | "
                f"boardR={board_roll_deg:+.2f} boardP={board_pitch_deg:+.2f} "
                f"boardY={board_yaw_deg:+.2f}\n"
            )
            log_file.flush()
            os.fsync(log_file.fileno())
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"# TEST_FILE_ERROR: {exc}")
        return False


def send_or_log_gimbal_angles(delta_yaw_deg, delta_pitch_deg, delta_roll_deg,
                              abs_yaw_deg, abs_pitch_deg, abs_roll_deg):
    """
    Send yaw/pitch/roll angles to the gimbal or log them in test mode.

    In test mode (TEST_MODE == 1), the command is appended to the test log.
    Otherwise, the function computes the board-frame angles by adding the
    startup baseline and calls bgc_control_angles().

    Args:
        delta_yaw_deg (float): Incremental yaw change (software frame).
        delta_pitch_deg (float): Incremental pitch change.
        delta_roll_deg (float): Incremental roll change.
        abs_yaw_deg (float): Absolute yaw in software zero-based frame.
        abs_pitch_deg (float): Absolute pitch in software zero-based frame.
        abs_roll_deg (float): Absolute roll in software zero-based frame.

    Returns:
        bool: True if the command (or log) succeeded, False on error.
    """
    global _sbgc_baseline_yaw_deg, _sbgc_baseline_pitch_deg, _sbgc_baseline_roll_deg

    # Compute board-frame absolute angles by adding baseline
    board_yaw_deg = _sbgc_baseline_yaw_deg + abs_yaw_deg
    board_pitch_deg = _sbgc_baseline_pitch_deg + abs_pitch_deg
    board_roll_deg = _sbgc_baseline_roll_deg + abs_roll_deg

    if TEST_MODE == 1:
        return write_test_command_log_entry(
            delta_yaw_deg, delta_pitch_deg, delta_roll_deg,
            abs_yaw_deg, abs_pitch_deg, abs_roll_deg,
            board_yaw_deg, board_pitch_deg, board_roll_deg
        )

    if _sbgc_lib is None or not _sbgc_initialized:
        print("# ERROR: SBGC shim not initialized.")
        return False

    # Shim order: (roll, pitch, yaw)
    send_result = _sbgc_lib.bgc_control_angles(
        ctypes.c_float(board_roll_deg),
        ctypes.c_float(board_pitch_deg),
        ctypes.c_float(board_yaw_deg),
    )
    if send_result != 0:
        print(f"# SEND_ERROR: bgc_control_angles() returned {send_result}")
        return False

    return True


# ---------------- Main loop (stateful) ----------------
def main():
    """
    Run the mouth-tracking and gimbal-control loop.

    High-level steps:
    1. Initialize SimpleBGC and set the zero-based baseline.
    2. Initialize MediaPipe FaceMesh for mouth landmark tracking.
    3. Open the camera, lock a small "stable box" at the center.
    4. In each frame:
         - Detect the mouth and compute its centroid and roll.
         - Smooth the centroid with an EMA.
         - Compute distance from the center; update a simple state machine:
             * LOCKED: mouth is near center; send no commands.
             * SEEKING: mouth is off-center; send micro-steps until
               close enough, then return to LOCKED.
         - Draw overlay graphics and print telemetry for debugging.
    """
    init_simplebgc()

    mediapipe_face_mesh_module = mp.solutions.face_mesh
    face_mesh_detector = mediapipe_face_mesh_module.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    video_capture = cv2.VideoCapture(CAMERA_INDEX)
    if not video_capture.isOpened():
        print("ERROR: Unable to open camera", CAMERA_INDEX)
        sys.exit(1)
    video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # State variables
    start_time = time.time()
    prev_smoothed_centroid = None
    prev_face_roll_deg = None
    prev_frame_time = None
    frame_center_anchor = None      # Locked to center of frame
    last_command_time = 0.0
    consecutive_lost_frames = 0

    # Zero-based absolute commanded angles (software frame)
    commanded_roll_deg = 0.0
    commanded_pitch_deg = 0.0
    commanded_yaw_deg = 0.0

    LOCKED, SEEKING = 0, 1
    tracker_state = LOCKED

    # Time-based histories for position and velocity
    history_x_pixels = TimedHistory(STABILITY_WINDOW_SEC)
    history_y_pixels = TimedHistory(STABILITY_WINDOW_SEC)
    history_velocity_deg_s = TimedHistory(STABILITY_WINDOW_SEC)

    # Clear the test file at the start of every run
    try:
        with open(GIMBAL_TEST_LOG_PATH, "w") as log_file:
            log_file.write("# SimpleBGC commands (TEST/zero-based, micro-steps) — fresh run\n")
            log_file.write(f"# Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        print(f"# Cleared {GIMBAL_TEST_LOG_PATH} for a new session.")
    except Exception as exc:  # noqa: BLE001
        print(f"# ERROR clearing test log: {exc}")

    print("# T roll pitch yaw can_send state radial_norm "
          "(prospective during SEEKING; actual on SEND)")

    while True:
        ok, frame = video_capture.read()
        if not ok:
            print("# WARN: frame grab failed")
            break

        now = time.time()
        frame_height, frame_width = frame.shape[:2]

        # --------- LOCK ANCHOR TO CENTER OF FRAME ---------
        if frame_center_anchor is None:
            # First time we know width/height, set anchor to optical axis center.
            frame_center_anchor = (frame_width / 2.0, frame_height / 2.0)

        # --- Detect mouth centroid + roll using MediaPipe FaceMesh ---
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_mesh_result = face_mesh_detector.process(rgb_frame)
        mouth_centroid = None
        current_face_roll_deg = None

        if face_mesh_result.multi_face_landmarks:
            face_landmarks = face_mesh_result.multi_face_landmarks[0]

            # Use a small subset of mouth landmarks to estimate mouth center.
            mouth_landmark_ids = [13, 14, 61, 291]
            mouth_points = []
            for landmark_idx in mouth_landmark_ids:
                x_pixel = int(face_landmarks.landmark[landmark_idx].x * frame_width)
                y_pixel = int(face_landmarks.landmark[landmark_idx].y * frame_height)
                mouth_points.append((x_pixel, y_pixel))

            if mouth_points:
                mouth_cx = sum(p[0] for p in mouth_points) / len(mouth_points)
                mouth_cy = sum(p[1] for p in mouth_points) / len(mouth_points)
                mouth_centroid = (mouth_cx, mouth_cy)

                # Approximate roll using left/right mouth corners
                left_mouth_corner = (
                    int(face_landmarks.landmark[61].x * frame_width),
                    int(face_landmarks.landmark[61].y * frame_height),
                )
                right_mouth_corner = (
                    int(face_landmarks.landmark[291].x * frame_width),
                    int(face_landmarks.landmark[291].y * frame_height),
                )
                current_face_roll_deg = angle_between_points_deg(
                    left_mouth_corner, right_mouth_corner
                )

        # Lost-face handling
        if mouth_centroid is None:
            consecutive_lost_frames += 1
            if consecutive_lost_frames > MAX_CONSECUTIVE_LOST_FRAMES:
                prev_smoothed_centroid = None
                prev_face_roll_deg = None
                prev_frame_time = None
            elapsed = now - start_time
            print(f"{elapsed:.3f} +0.000 +0.000 +0.000 0 {tracker_state} radial_norm=0.000")
            if DISPLAY_DEBUG_WINDOWS:
                cv2.imshow("Centroid Tracker (stateful)", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            continue

        consecutive_lost_frames = 0

        # Smooth centroid with EMA
        smoothed_centroid = smooth_point_ema(
            mouth_centroid,
            prev_smoothed_centroid,
            CENTROID_EMA_ALPHA,
        )

        # Timing initialization
        if prev_frame_time is None:
            prev_frame_time = now

        # Build stable box at fixed center anchor
        stable_box = compute_stable_box(
            frame_center_anchor,
            frame_width,
            frame_height,
            STABLE_BOX_SCALE,
        )
        is_inside_stable_box = is_point_inside_box(smoothed_centroid, stable_box)

        # Compute normalized radial distance from frame center (0 = center).
        dx_center = smoothed_centroid[0] - frame_center_anchor[0]
        dy_center = smoothed_centroid[1] - frame_center_anchor[1]
        norm_dx = dx_center / (frame_width / 2.0)
        norm_dy = dy_center / (frame_height / 2.0)
        radial_norm = math.hypot(norm_dx, norm_dy)
        within_stop_threshold = (radial_norm <= CENTER_STOP_THRESHOLD_NORM)

        # Compute dt-based velocity for stability metrics
        delta_t = max(1e-6, now - prev_frame_time)
        if prev_smoothed_centroid is None:
            angular_speed = 0.0
        else:
            dx_px_dt = smoothed_centroid[0] - prev_smoothed_centroid[0]
            dy_px_dt = smoothed_centroid[1] - prev_smoothed_centroid[1]
            dv_yaw_deg, dv_pitch_deg = pixels_to_degrees(
                dx_px_dt, dy_px_dt,
                frame_width, frame_height,
                CAMERA_FOV_HORIZONTAL_DEG, CAMERA_FOV_VERTICAL_DEG,
            )
            angular_speed = math.hypot(dv_yaw_deg, dv_pitch_deg) / delta_t

        prev_frame_time = now
        prev_smoothed_centroid = smoothed_centroid

        # Default console outputs (prospective values; overwritten if we send)
        delta_yaw_step_deg = 0.0
        delta_pitch_step_deg = 0.0
        delta_roll_step_deg = 0.0
        can_send_flag = 0
        sent_this_frame = False

        # ---------- STATE MACHINE ----------
        if tracker_state == LOCKED:
            # If mouth leaves the stable box, begin SEEKING.
            if not is_inside_stable_box:
                tracker_state = SEEKING
                history_x_pixels.clear()
                history_y_pixels.clear()
                history_velocity_deg_s.clear()
                # On first departure from center, capture roll reference.
                if current_face_roll_deg is not None and prev_face_roll_deg is None:
                    prev_face_roll_deg = current_face_roll_deg
        else:  # SEEKING
            # Update histories for stability metrics
            history_x_pixels.add(now, smoothed_centroid[0])
            history_y_pixels.add(now, smoothed_centroid[1])
            history_velocity_deg_s.add(now, angular_speed)

            xs = history_x_pixels.values()
            ys = history_y_pixels.values()
            position_std = 999.0
            if len(xs) >= 6 and len(ys) >= 6:
                position_std = 0.5 * (statistics.pstdev(xs) + statistics.pstdev(ys))

            speeds = history_velocity_deg_s.values()
            if len(speeds) >= 3:
                median_speed = statistics.median(speeds)
            else:
                median_speed = 999.0

            is_stable_here = (
                median_speed < STABILITY_VEL_THRESHOLD_DEG_S
                and position_std < STABILITY_POS_STD_THRESHOLD_PX
            )

            # Stop SEEKING when we are within the tighter radial threshold,
            # not just when we are barely inside the box.
            if within_stop_threshold:
                if current_face_roll_deg is not None:
                    prev_face_roll_deg = current_face_roll_deg
                history_x_pixels.clear()
                history_y_pixels.clear()
                history_velocity_deg_s.clear()
                tracker_state = LOCKED
            else:
                # Compute full desired offsets (center anchor -> current centroid)
                dx_pixels = smoothed_centroid[0] - frame_center_anchor[0]
                dy_pixels = smoothed_centroid[1] - frame_center_anchor[1]

                full_delta_yaw_deg, full_delta_pitch_deg = pixels_to_degrees(
                    dx_pixels, dy_pixels,
                    frame_width, frame_height,
                    CAMERA_FOV_HORIZONTAL_DEG, CAMERA_FOV_VERTICAL_DEG,
                )

                # Roll offset from initial roll in this SEEKING episode
                full_delta_roll_deg = 0.0
                if current_face_roll_deg is not None and prev_face_roll_deg is not None:
                    full_delta_roll_deg = current_face_roll_deg - prev_face_roll_deg

                # Apply axis sign convention
                full_delta_yaw_deg *= AXIS_SIGN_CONVENTION["yaw"]
                full_delta_pitch_deg *= AXIS_SIGN_CONVENTION["pitch"]
                full_delta_roll_deg *= AXIS_SIGN_CONVENTION["roll"]

                # Send micro-steps only when motion is stable and rate-limit allows
                can_time = (now - last_command_time) >= SEND_RATE_LIMIT_SEC
                if is_stable_here and can_time:
                    # Take a fraction of the full offset as a micro-step
                    delta_yaw_step_deg = full_delta_yaw_deg * MICRO_STEP_FRACTION
                    delta_pitch_step_deg = full_delta_pitch_deg * MICRO_STEP_FRACTION
                    delta_roll_step_deg = full_delta_roll_deg * MICRO_STEP_FRACTION

                    # Enforce minimum movement thresholds
                    if abs(delta_yaw_step_deg) < MIN_YAW_STEP_DEG:
                        delta_yaw_step_deg = 0.0
                    if abs(delta_pitch_step_deg) < MIN_PITCH_STEP_DEG:
                        delta_pitch_step_deg = 0.0
                    if abs(delta_roll_step_deg) < MIN_ROLL_STEP_DEG:
                        delta_roll_step_deg = 0.0

                    if (delta_yaw_step_deg != 0.0 or
                        delta_pitch_step_deg != 0.0 or
                            delta_roll_step_deg != 0.0):
                        # Update zero-based commanded angles (software frame)
                        commanded_yaw_deg += delta_yaw_step_deg
                        commanded_pitch_deg += delta_pitch_step_deg
                        commanded_roll_deg += delta_roll_step_deg

                        ok = send_or_log_gimbal_angles(
                            delta_yaw_step_deg,
                            delta_pitch_step_deg,
                            delta_roll_step_deg,
                            commanded_yaw_deg,
                            commanded_pitch_deg,
                            commanded_roll_deg,
                        )
                        if ok:
                            last_command_time = now
                            sent_this_frame = True
                            can_send_flag = 1

        # -------- Telemetry / debug print --------
        elapsed = now - start_time
        if sent_this_frame or tracker_state == SEEKING:
            print(
                f"{elapsed:.3f} "
                f"{delta_roll_step_deg:+.3f} "
                f"{delta_pitch_step_deg:+.3f} "
                f"{delta_yaw_step_deg:+.3f} "
                f"{can_send_flag} {tracker_state} "
                f"radial_norm={radial_norm:.3f}"
            )
        else:
            print(
                f"{elapsed:.3f} "
                f"+0.000 +0.000 +0.000 0 {tracker_state} "
                f"radial_norm={radial_norm:.3f}"
            )

        # -------- UI / visualization --------
        if DISPLAY_DEBUG_WINDOWS:
            left, top, right, bottom = map(int, stable_box)
            # Green stable box locked at center (anchor)
            cv2.rectangle(frame, (left, top), (right, bottom), (40, 220, 40), 1)
            # Crosshair at the anchor = frame center
            cv2.drawMarker(
                frame,
                (int(frame_center_anchor[0]), int(frame_center_anchor[1])),
                (0, 200, 0),
                cv2.MARKER_CROSS,
                12,
                2,
            )
            # Red dot for mouth centroid
            cv2.circle(
                frame,
                (int(smoothed_centroid[0]), int(smoothed_centroid[1])),
                4,
                (0, 0, 255),
                -1,
            )

            state_text = "LOCKED" if tracker_state == LOCKED else "SEEKING"
            cv2.putText(
                frame,
                f"state:{state_text}",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (40, 220, 40),
                2,
                cv2.LINE_AA,
            )
            if tracker_state == SEEKING:
                cv2.putText(
                    frame,
                    f"dR:{delta_roll_step_deg:+.2f} "
                    f"dP:{delta_pitch_step_deg:+.2f} "
                    f"dY:{delta_yaw_step_deg:+.2f}",
                    (10, 48),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (40, 220, 40),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"r={radial_norm:.3f}",
                    (10, 72),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (40, 220, 40),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Centroid Tracker (center-locked box)", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    video_capture.release()
    cv2.destroyAllWindows()
    print("Stopped.")


if __name__ == "__main__":
    main()
