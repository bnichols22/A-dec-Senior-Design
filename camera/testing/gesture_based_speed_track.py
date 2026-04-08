#!/usr/bin/env python3
# ==============================================================
# File: speed_control_compliance_dynbox.py
# Purpose:
#   Smooth face tracker that drives gimbal using SPEED mode (deg/s).
#
# ADDITIONS:
#   (1) Dynamic stable box sizing (discrete ranges) based on face distance
#       proxy using FaceMesh eye distance (in pixels). Closer face => bigger box.
#   (2) Gesture-driven tracking mode control:
#       - Pinch + index finger -> fingertip tracking
#       - Fist -> lock in place
#       - Four fingers -> resume mouth tracking
#
# Motor lib usage:
#   SimpleBGC SerialAPI shim:
#     - bgc_init()
#     - bgc_control_speeds(roll_dps, pitch_dps, yaw_dps)
#     - bgc_set_motors(on_off)   (0=off, 1=on)
# ==============================================================

import os, sys, time, math, warnings, statistics, ctypes
import cv2
import mediapipe as mp
from collections import deque
import json
import board
import busio
from adafruit_ads1x15 import ADS1115, AnalogIn, ads1x15

# --------- Paths and Filename ----------
file_name = "gesture_based_speed_track.py"
BASE_DIR = os.path.expanduser('~/senior_design/A-dec-Senior-Design/camera/testing')
os.makedirs(BASE_DIR, exist_ok=True)

LOG_PATH  = os.path.join(BASE_DIR, 'face_track_log.txt')
TEST_LOG  = os.path.join(BASE_DIR, 'Board_Serial_Command_Test.txt')

LIB_PATH = os.path.expanduser(
    "~/senior_design/A-dec-Senior-Design/camera/testing/SerialLibrary/serialAPI/libsimplebgc.so"
)

CAMERA_PROFILE_DIR = os.path.join(BASE_DIR, "camera_profiles")
# Photos from the center-facing camera are stored here during the gesture photo sequence.
PATIENT_PHOTO_DIR = os.path.join(BASE_DIR, "patient_photo")
os.makedirs(PATIENT_PHOTO_DIR, exist_ok=True)

# --------- Vision / tracker config ----------
WIDE_ANGLE_CAM_INDEX = 0
CENTER_CAM_INDEX = 1
FOV_H_DEG = 65.0
FOV_V_DEG = 48.75

# --- Stable box base ---
# NOTE: this value is now a "middle" default; actual box scalar is chosen by ranges below.
STABLE_SCALAR_DEFAULT = 0.06

WINDOW_SEC = 0.6
STABLE_STOP_SEEKING_THRESHOLD = 0.025

VEL_THRESH_DEG_S  = 2.5
POS_STD_THRESH_PX = 2.5

# Speed controller tuning
# Convert deg error to deg/s command with gain KP.
KP_YAW_DPS_PER_DEG   = 1.25
KP_PITCH_DPS_PER_DEG = 1.25

# Limits on commanded speeds (deg/s)
MAX_DPS_YAW   = 80.0
MAX_DPS_PITCH = 80.0
MAX_DPS_ROLL  = 60.0

# Deadband (deg error) to avoid micro-jitter
DEADBAND_DEG_YAW   = 0.25
DEADBAND_DEG_PITCH = 0.25

# Command streaming rate (Hz)
COMMAND_HZ = 500.0
COMMAND_PERIOD = 1.0 / COMMAND_HZ

# Smooth commanded speeds (0=no smoothing, 1=very slow)
CMD_SPEED_EMA_ALPHA = 0.35

# Axis sign values (trying to keep them all + for ease of conceptualization)
AXIS_SIGN = {"yaw": 1, "pitch": 1, "roll": 1}

# Capture vars
MAX_STORED_FRAMES = 1

# If this is True, it makes a realtime window on monitor, otherwise it does not
DRAW_FRAME_RT = True
# If this is True it will print telemtry
PRINT_TELEMETRY = False

# Smoothing alpha val and max # of lost frames
SMOOTH_ALPHA = 0.25
MAX_LOST_FRAMES = 10

# Environment logging stuff
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")


# ==============================================================
# NEW: Discrete stable-box scaling by "distance" (eye pixel distance)
# ==============================================================
# Landmarks: 33 (left eye outer corner), 263 (right eye outer corner)
EYE_L_IDX = 33
EYE_R_IDX = 263

# Tune these thresholds for your camera/resolution:
# - If these feel off: print eye_dist_px and adjust.
EYE_DIST_FAR_MAX   = 45.0   # <= this => FAR (face small/far)
EYE_DIST_NEAR_MIN  = 85.0   # >= this => NEAR (face big/close)
# Middle is between those.

# Stable box scalars for each range
STABLE_SCALAR_FAR  = 0.035  # small box when face is far
STABLE_SCALAR_MID  = 0.060  # default
STABLE_SCALAR_NEAR = 0.095  # larger box when face is close

# Prevent flicker: require N consecutive frames to accept a new range
RANGE_SWITCH_FRAMES = 8

# ==============================================================
# NEW: Light mode ADC config
# ==============================================================
LIGHT_MODE_THRESHOLD_VOLTS = 0.8

LIGHT_OFF_MODE     = "LIGHT_OFF"
YELLOW_LIGHT_MODE  = "YELLOW_LIGHT"
LOWEST_LIGHT_MODE  = "LOWEST_LIGHT"
MEDIUM_LIGHT_MODE  = "MEDIUM_LIGHT"
HIGHEST_LIGHT_MODE = "HIGHEST_LIGHT"

LIGHT_MODE_TO_PROFILE = {
    HIGHEST_LIGHT_MODE: "wide_angle_full_light_on.json",
    MEDIUM_LIGHT_MODE:  "wide_angle_medium_light_on.json",
    LOWEST_LIGHT_MODE:  "wide_angle_low_light_on.json",
    YELLOW_LIGHT_MODE:  "wide_angle_yellow_light_on.json",
    LIGHT_OFF_MODE:     "wide_angle_light_off.json",
}

GESTURE_TRACK_MOUTH = 0
GESTURE_TRACK_PINCH = 1
GESTURE_LOCKED = 2

HAND_DETECTION_CONFIDENCE = 0.75
HAND_TRACKING_CONFIDENCE = 0.75
PINCH_ENTER_FRAMES = 5
FOUR_ENTER_FRAMES = 5
TWO_ENTER_FRAMES = 5
PINCH_DISTANCE_RATIO = 0.28
FIST_ENTER_FRAMES = 5
PHOTO_COUNTDOWN_SEC = 2.0

FINGER_SMOOTH_ALPHA = 0.45
FINGER_CMD_SPEED_EMA_ALPHA = 0.55
FINGER_STABLE_SCALAR_MULT = 0.60
FINGER_STOP_SEEKING_THRESHOLD = 0.010
FINGER_DEADBAND_DEG_YAW = 0.08
FINGER_DEADBAND_DEG_PITCH = 0.08
FINGER_KP_YAW_DPS_PER_DEG = 1.65
FINGER_KP_PITCH_DPS_PER_DEG = 1.65


# ---------------- Helper Functions ----------------

def ema_point(current_smooth, previous_smooth, alpha):
    if previous_smooth is None:
        return current_smooth
    return (alpha * current_smooth[0] + (1 - alpha) * previous_smooth[0],
            alpha * current_smooth[1] + (1 - alpha) * previous_smooth[1])

def ema_scalar(current, previous, alpha):
    if previous is None:
        return current
    return alpha * current + (1 - alpha) * previous

def clamp(value, min_val, max_val):
    return max(min_val, min(max_val, value))

def pixels_to_deg(pixal_change_x, pixal_change_y, frame_width, frame_height, fov_horizontal, fov_verticle):
    half_width, half_height = frame_width / 2.0, frame_height / 2.0
    yaw_deg   = (pixal_change_x / half_width) * (fov_horizontal / 2.0)
    pitch_deg = (pixal_change_y / half_height) * (fov_verticle / 2.0)
    return yaw_deg, pitch_deg

def build_stable_box(center_point, frame_width, frame_height, scalar):
    center_x, center_y = center_point
    half_width = scalar * (frame_width / 2.0)
    half_height = scalar * (frame_height / 2.0)
    return (center_x - half_width, center_y - half_height, center_x + half_width, center_y + half_height)

def inside_box(pt, stable_box):
    x, y = pt
    l, t, r, b = stable_box
    return (l <= x <= r) and (t <= y <= b)

def estimate_eye_dist_px(face_landmarks, frame_width, frame_height):
    """Return eye distance (px) using FaceMesh landmarks. Returns None if unavailable."""
    try:
        lx = face_landmarks.landmark[EYE_L_IDX].x * frame_width
        ly = face_landmarks.landmark[EYE_L_IDX].y * frame_height
        rx = face_landmarks.landmark[EYE_R_IDX].x * frame_width
        ry = face_landmarks.landmark[EYE_R_IDX].y * frame_height
        return math.hypot(rx - lx, ry - ly)
    except Exception:
        return None

def range_from_eye_dist(eye_dist_px):
    """Map eye distance to discrete distance ranges."""
    if eye_dist_px is None:
        return "MID"  # safe default
    if eye_dist_px <= EYE_DIST_FAR_MAX:
        return "FAR"
    if eye_dist_px >= EYE_DIST_NEAR_MIN:
        return "NEAR"
    return "MID"

def scalar_for_range(rng):
    if rng == "FAR":
        return STABLE_SCALAR_FAR
    if rng == "NEAR":
        return STABLE_SCALAR_NEAR
    return STABLE_SCALAR_MID

# If the histogram was removed, this coudl be deleted
class TimedHistogram:

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

    def _trim(self, current_time):
        cut = current_time - self.win
        while self.buf and self.buf[0][0] < cut:
            self.buf.popleft()

def update_camera_settings(camera, filename, profile_dir):
    try:
        if os.path.isabs(filename):
            profile_path = filename
        else:
            profile_path = os.path.join(profile_dir, filename)

        with open(profile_path, 'r') as f:
            camera_settings = json.load(f)

        # If a camera doesn't support a property, cv2 usually just ignores it or returns false.
        if "auto_exposure" in camera_settings:
            camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, camera_settings["auto_exposure"])
        if "exposure" in camera_settings:
            camera.set(cv2.CAP_PROP_EXPOSURE, camera_settings["exposure"])
        if "brightness" in camera_settings:
            camera.set(cv2.CAP_PROP_BRIGHTNESS, camera_settings["brightness"])
        if "contrast" in camera_settings:
            camera.set(cv2.CAP_PROP_CONTRAST, camera_settings["contrast"])
        if "gain" in camera_settings:
            camera.set(cv2.CAP_PROP_GAIN, camera_settings["gain"])
        if "saturation" in camera_settings:
            camera.set(cv2.CAP_PROP_SATURATION, camera_settings["saturation"])

        print(f"Loaded camera profile: {profile_path}")
        return True
    except Exception as camera_update_error:
        print(f"Error loading profile: {camera_update_error}")
        return False

def read_light_mode(adc_channels, off_threshold_volts):
    v0 = adc_channels["A0"].voltage
    v1 = adc_channels["A1"].voltage
    v2 = adc_channels["A2"].voltage
    v3 = adc_channels["A3"].voltage

    voltage_map = {
        YELLOW_LIGHT_MODE:  v0,
        LOWEST_LIGHT_MODE:  v1,
        MEDIUM_LIGHT_MODE:  v2,
        HIGHEST_LIGHT_MODE: v3,
    }

    active_modes = {mode_name: voltage for mode_name, voltage in voltage_map.items() if voltage >= off_threshold_volts}

    if not active_modes:
        return LIGHT_OFF_MODE, (v0, v1, v2, v3)

    selected_mode = max(active_modes, key=active_modes.get)
    return selected_mode, (v0, v1, v2, v3)

def update_camera_profile_from_light_mode(camera, current_light_mode, previous_light_mode, profile_dir):
    if current_light_mode == previous_light_mode:
        return previous_light_mode

    profile_filename = LIGHT_MODE_TO_PROFILE.get(current_light_mode)
    if profile_filename is None:
        print(f"Warning: No profile mapped for light mode {current_light_mode}")
        return previous_light_mode

    profile_loaded = update_camera_settings(camera, profile_filename, profile_dir)
    if profile_loaded:
        print(f"Light mode changed: {previous_light_mode} -> {current_light_mode}")
        return current_light_mode

    return previous_light_mode

def landmark_to_pixel(landmark, frame_width, frame_height):
    return (landmark.x * frame_width, landmark.y * frame_height)

def finger_extended(tip_point, pip_point):
    return tip_point[1] < pip_point[1]

def detect_hand_gestures(hand_results, frame_width, frame_height):
    pinch_detected = False
    pinch_start_point = None
    index_tip_point = None
    two_detected = False
    four_detected = False
    best_pinch_ratio = 999.0
    fist_detected = False

    if not hand_results.multi_hand_landmarks:
        return pinch_detected, pinch_start_point, index_tip_point, two_detected, four_detected, fist_detected

    for hand_landmarks in hand_results.multi_hand_landmarks:
        lm = hand_landmarks.landmark

        thumb_tip = landmark_to_pixel(lm[4], frame_width, frame_height)
        thumb_ip = landmark_to_pixel(lm[3], frame_width, frame_height)
        thumb_mcp = landmark_to_pixel(lm[2], frame_width, frame_height)

        index_tip = landmark_to_pixel(lm[8], frame_width, frame_height)
        index_pip = landmark_to_pixel(lm[6], frame_width, frame_height)
        index_mcp = landmark_to_pixel(lm[5], frame_width, frame_height)

        middle_tip = landmark_to_pixel(lm[12], frame_width, frame_height)
        middle_pip = landmark_to_pixel(lm[10], frame_width, frame_height)

        ring_tip = landmark_to_pixel(lm[16], frame_width, frame_height)
        ring_pip = landmark_to_pixel(lm[14], frame_width, frame_height)

        pinky_tip = landmark_to_pixel(lm[20], frame_width, frame_height)
        pinky_pip = landmark_to_pixel(lm[18], frame_width, frame_height)
        pinky_mcp = landmark_to_pixel(lm[17], frame_width, frame_height)

        index_tip_point = index_tip

        palm_width = max(1.0, math.hypot(index_mcp[0] - pinky_mcp[0], index_mcp[1] - pinky_mcp[1]))
        pinch_distance = math.hypot(thumb_tip[0] - index_tip[0], thumb_tip[1] - index_tip[1])
        pinch_ratio = pinch_distance / palm_width
        thumb_to_index_mcp = math.hypot(thumb_tip[0] - index_mcp[0], thumb_tip[1] - index_mcp[1])

        index_extended = finger_extended(index_tip, index_pip)
        middle_extended = finger_extended(middle_tip, middle_pip)
        ring_extended = finger_extended(ring_tip, ring_pip)
        pinky_extended = finger_extended(pinky_tip, pinky_pip)

        non_index_closed = (not middle_extended and not ring_extended and not pinky_extended)

        if pinch_ratio < PINCH_DISTANCE_RATIO and non_index_closed and pinch_ratio < best_pinch_ratio:
            best_pinch_ratio = pinch_ratio
            pinch_detected = True
            pinch_start_point = ((thumb_tip[0] + index_tip[0]) / 2.0, (thumb_tip[1] + index_tip[1]) / 2.0)

        thumb_tucked = (
            thumb_to_index_mcp < (0.95 * palm_width) and
            thumb_tip[1] > thumb_ip[1]
        )
        # Four fingers extended with thumb tucked resumes normal mouth tracking.
        hand_four = (
            index_extended and
            middle_extended and
            ring_extended and
            pinky_extended and
            thumb_tucked
        )
        if hand_four:
            four_detected = True

        # Two fingers extended with thumb tucked starts the patient photo countdown.
        hand_two = (
            index_extended and
            middle_extended and
            not ring_extended and
            not pinky_extended and
            thumb_tucked
        )
        if hand_two:
            two_detected = True

        hand_fist = (
            not index_extended and
            not middle_extended and
            not ring_extended and
            not pinky_extended
        )
        if hand_fist:
            fist_detected = True

    return pinch_detected, pinch_start_point, index_tip_point, two_detected, four_detected, fist_detected


# ----------------------------------------------------------------------
# SBGC shim bindings (ctypes)
# ----------------------------------------------------------------------
motor_library = None
motor_library_initialized = False

def init_sbgc():
    global motor_library, motor_library_initialized

    try:
        motor_library = ctypes.CDLL(LIB_PATH)
        print(f"Loaded SBGC library from {LIB_PATH}")
    except OSError as e:
        # Return w/ Error code if we cannot open the .so
        print(f"init_sbgc: error loading {LIB_PATH}: {e}")
        motor_library = None
        motor_library_initialized = False
        return

    motor_library.bgc_init.argtypes = []
    motor_library.bgc_init.restype = ctypes.c_int

    motor_library.bgc_control_speeds.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float]
    motor_library.bgc_control_speeds.restype = ctypes.c_int

    # NEW: needed for compliance mode
    if hasattr(motor_library, "bgc_set_motors"):
        motor_library.bgc_set_motors.argtypes = [ctypes.c_int]
        motor_library.bgc_set_motors.restype = ctypes.c_int
    else:
        print("init_sbgc: WARNING libsimplebgc.so missing bgc_set_motors(). Compliance mode will not work.")

    status_code = motor_library.bgc_init()
    if status_code != 0:
        print(f"init_sbgc: error bgc_init() returned {status_code}")
        motor_library_initialized = False
        return

    motor_library_initialized = True
    print("Initialized the library")

def send_speeds(roll_dps, pitch_dps, yaw_dps):
    if motor_library is None or not motor_library_initialized:
        print("send_speeds: cannot send because lib is not initialied or setup")
        return False

    # send the speeds to the motors
    status_code = motor_library.bgc_control_speeds(
        ctypes.c_float(roll_dps),
        ctypes.c_float(pitch_dps),
        ctypes.c_float(yaw_dps),
    )
    if status_code != 0:
        print(f"send_speeds: bgc_control_speeds: bgc_control_speeds() returned {status_code}, non-zero status fail")
        return False

    return True

def set_motors(on_off: int):
    """
    Motors control:
      - on_off=0 -> motors OFF
      - on_off=1 -> motors ON
    """
    if motor_library is None or not motor_library_initialized:
        print("set_motors: cannot send because lib is not initialied or setup")
        return False

    if not hasattr(motor_library, "bgc_set_motors"):
        print("set_motors: bgc_set_motors() missing from libsimplebgc.so")
        return False

    rc = motor_library.bgc_set_motors(ctypes.c_int(int(on_off)))
    if rc != 0:
        print(f"set_motors: bgc_set_motors({on_off}) returned {rc}")
        return False
    return True

def reset_tracking_state(next_state):
    return None, None, 0, next_state

def update_gesture_mode(
    gesture_mode,
    pinch_detected,
    index_tip_point,
    four_detected,
    fist_detected,
    pinch_counter,
    four_counter,
    fist_counter,
    pinch_point,
    seeking_state,
    locked_state
):
    reset_state = None

    if fist_detected:
        fist_counter += 1
        if fist_counter >= FIST_ENTER_FRAMES:
            pinch_counter = 0
            four_counter = 0
            fist_counter = 0
            if gesture_mode != GESTURE_LOCKED:
                gesture_mode = GESTURE_LOCKED
                pinch_point = None
                reset_state = locked_state
            return (
                gesture_mode,
                pinch_counter,
                four_counter,
                fist_counter,
                pinch_point,
                reset_state,
            )
    else:
        fist_counter = 0

    if four_detected:
        four_counter += 1
        if four_counter >= FOUR_ENTER_FRAMES:
            pinch_counter = 0
            four_counter = 0
            if gesture_mode != GESTURE_TRACK_MOUTH:
                gesture_mode = GESTURE_TRACK_MOUTH
                pinch_point = None
                reset_state = locked_state
            return (
                gesture_mode,
                pinch_counter,
                four_counter,
                fist_counter,
                pinch_point,
                reset_state,
            )
    else:
        four_counter = 0

    if gesture_mode == GESTURE_TRACK_PINCH and index_tip_point is not None:
        pinch_point = index_tip_point

    if pinch_detected and index_tip_point is not None:
        pinch_counter += 1
        if pinch_counter >= PINCH_ENTER_FRAMES:
            pinch_point = index_tip_point
            pinch_counter = 0
            four_counter = 0
            fist_counter = 0
            if gesture_mode != GESTURE_TRACK_PINCH:
                gesture_mode = GESTURE_TRACK_PINCH
                reset_state = seeking_state
    else:
        pinch_counter = 0

    return (
        gesture_mode,
        pinch_counter,
        four_counter,
        fist_counter,
        pinch_point,
        reset_state,
    )

def get_mouth_centroid_and_eye_dist(processed_image, frame_width, frame_height):
    centroid = None
    eye_dist_px = None

    if not processed_image.multi_face_landmarks:
        return centroid, eye_dist_px

    patient_face = processed_image.multi_face_landmarks[0]
    mouth_idxs = [13, 14, 61, 291]
    mouth_points = []

    for idx in mouth_idxs:
        x = int(patient_face.landmark[idx].x * frame_width)
        y = int(patient_face.landmark[idx].y * frame_height)
        mouth_points.append((x, y))

    if mouth_points:
        centroid_x = sum(p[0] for p in mouth_points) / len(mouth_points)
        centroid_y = sum(p[1] for p in mouth_points) / len(mouth_points)
        centroid = (centroid_x, centroid_y)

    eye_dist_px = estimate_eye_dist_px(patient_face, frame_width, frame_height)
    return centroid, eye_dist_px

def update_stable_range_state(desired_range, stable_range, pending_range, pending_count):
    if desired_range != stable_range:
        if pending_range != desired_range:
            pending_range = desired_range
            pending_count = 1
        else:
            pending_count += 1

        if pending_count >= RANGE_SWITCH_FRAMES:
            stable_range = desired_range
            pending_range = None
            pending_count = 0
    else:
        pending_range = None
        pending_count = 0

    stable_scalar = scalar_for_range(stable_range)
    return stable_range, pending_range, pending_count, stable_scalar

def send_zero_if_due(current_time, last_send_time, should_send=True):
    if (current_time - last_send_time) >= COMMAND_PERIOD:
        if should_send:
            send_speeds(0.0, 0.0, 0.0)
        return current_time
    return last_send_time

def draw_hand_landmarks(frame, hand_results, mp_drawing, mp_drawing_styles, hand_connections):
    if not hand_results.multi_hand_landmarks:
        return

    for hand_landmarks in hand_results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            hand_connections,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

def show_runtime_frame(window_name, frame, overlay_lines):
    if not DRAW_FRAME_RT:
        return False

    for text, position, color, scale in overlay_lines:
        cv2.putText(
            frame,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            2,
            cv2.LINE_AA
        )

    cv2.imshow(window_name, frame)
    return (cv2.waitKey(1) & 0xFF) == 27

def capture_patient_photo(center_camera):
    if center_camera is None:
        print("capture_patient_photo: center_camera unavailable")
        return False, None

    # Grab a single still frame from the center camera and save it with a unique timestamp.
    frame_read, center_frame = center_camera.read()
    if not frame_read:
        print("capture_patient_photo: failed to read frame from center_camera")
        return False, None

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    milliseconds = int((time.time() % 1.0) * 1000)
    photo_path = os.path.join(PATIENT_PHOTO_DIR, f"patient_photo_{timestamp}_{milliseconds:03d}.jpg")
    photo_written = cv2.imwrite(photo_path, center_frame)
    if not photo_written:
        print(f"capture_patient_photo: failed to write photo to {photo_path}")
        return False, None

    print(f"Saved patient photo: {photo_path}")
    return True, photo_path


# ---------------- Main loop ----------------
def main():

    # ======= Setup =======
    init_sbgc()

    mp_face_mesh = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hand_connections = mp_hands.HAND_CONNECTIONS
    window_name = f"Image playback using: {file_name}"

    # Can adjust confidence as and detection as needed
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )

    hands = mp_hands.Hands(
        model_complexity=1,
        max_num_hands=2,
        min_detection_confidence=HAND_DETECTION_CONFIDENCE,
        min_tracking_confidence=HAND_TRACKING_CONFIDENCE
    )

    # Create capture device object and verify it constructed
    face_track_cam = cv2.VideoCapture(WIDE_ANGLE_CAM_INDEX)
    if not face_track_cam.isOpened():
        print(f'main: Error Unable to open camera from {WIDE_ANGLE_CAM_INDEX}')
        # Exit here because we cannot run without camera
        sys.exit(1)

    center_camera = cv2.VideoCapture(CENTER_CAM_INDEX)
    if not center_camera.isOpened():
        print(f"main: Warning unable to open center_camera from {CENTER_CAM_INDEX}")
        center_camera.release()
        center_camera = None

    # Create ADC object and channels
    i2c = busio.I2C(board.SCL, board.SDA)
    ads = ADS1115(i2c)
    ads.gain = 1

    adc_channels = {
        "A0": AnalogIn(ads, ads1x15.Pin.A0),
        "A1": AnalogIn(ads, ads1x15.Pin.A1),
        "A2": AnalogIn(ads, ads1x15.Pin.A2),
        "A3": AnalogIn(ads, ads1x15.Pin.A3),
    }

    current_light_mode = LIGHT_OFF_MODE
    previous_light_mode = None
    previous_light_mode = update_camera_profile_from_light_mode(
        face_track_cam,
        current_light_mode,
        previous_light_mode,
        CAMERA_PROFILE_DIR
    )

    # Set the max number of stored frames allowed
    face_track_cam.set(cv2.CAP_PROP_BUFFERSIZE, MAX_STORED_FRAMES)
    if center_camera is not None:
        center_camera.set(cv2.CAP_PROP_BUFFERSIZE, MAX_STORED_FRAMES)

    # Clear test log
    try:
        with open(LOG_PATH, "w") as log_file:
            log_file.write(f"Filename: {file_name}\n")
            log_file.write(f"# Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"--------------------------------------------------------\n")
        sys.stderr = log_file
    except Exception:
        print("Unable to open log file")
        pass

    initial_time = time.time()
    prev_smoothed = None
    prev_time = None
    consecutive_lost_frames = 0

    anchor = None

    # State names
    LOCKED, SEEKING = 0, 1
    state = LOCKED

    pos_x = TimedHistogram(WINDOW_SEC)
    pos_y = TimedHistogram(WINDOW_SEC)
    vel_h = TimedHistogram(WINDOW_SEC)

    last_send_time = 0.0

    # smoothed speed outputs
    smooth_yaw_dps = None
    smooth_pitch_dps = None
    smooth_roll_dps = None

    # ==============================================================
    # NEW: Dynamic stable-box state
    # ==============================================================
    stable_range = "MID"
    pending_range = None
    pending_count = 0
    stable_scalar = STABLE_SCALAR_DEFAULT

    # Start gesture control in the locked state, equivalent to beginning with a fist.
    gesture_mode = GESTURE_LOCKED
    pinch_counter = 0
    two_counter = 0
    four_counter = 0
    fist_counter = 0
    pinch_point = None
    # Photo capture is handled as a temporary overlay state so the main loop keeps running.
    photo_countdown_active = False
    photo_countdown_start = None
    photo_return_mode = GESTURE_LOCKED
    photo_trigger_armed = True

    # ======= Main Loop =======
    while True:
        yaw_dps = 0.0
        pitch_dps = 0.0
        roll_dps = 0.0

        frame_read, frame = face_track_cam.read()
        if not frame_read:
            # Note: log_file is only set if file open succeeded above
            try:
                log_file.write(f"main: frame grab failed\n")
            except Exception:
                pass
            break

        current_time = time.time()
        frame_height, frame_width = frame.shape[:2]

        # Update camera profile from ADC light mode every loop
        current_light_mode, light_mode_voltages = read_light_mode(adc_channels, LIGHT_MODE_THRESHOLD_VOLTS)
        previous_light_mode = update_camera_profile_from_light_mode(
            face_track_cam,
            current_light_mode,
            previous_light_mode,
            CAMERA_PROFILE_DIR
        )

        if anchor is None:
            anchor = (frame_width / 2.0, frame_height / 2.0)

        # Normal tracking mode
        # get the capture from camera
        rgb_frame_cap = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # process the image and set landmarks
        processed_image = face_mesh.process(rgb_frame_cap)
        hand_results = hands.process(rgb_frame_cap)

        pinch_detected, _pinch_start_point, index_tip_point, two_detected, four_detected, fist_detected = detect_hand_gestures(
            hand_results,
            frame_width,
            frame_height
        )

        if not two_detected:
            two_counter = 0
            photo_trigger_armed = True

        if photo_countdown_active:
            # Hold the head still while counting down; this avoids blocking the rest of the loop.
            last_send_time = send_zero_if_due(current_time, last_send_time)

            elapsed_photo_time = current_time - photo_countdown_start
            remaining_photo_time = max(0.0, PHOTO_COUNTDOWN_SEC - elapsed_photo_time)

            if elapsed_photo_time >= PHOTO_COUNTDOWN_SEC:
                photo_ok, photo_path = capture_patient_photo(center_camera)
                photo_countdown_active = False
                gesture_mode = photo_return_mode
                prev_smoothed, prev_time, consecutive_lost_frames, state = reset_tracking_state(LOCKED)
            draw_hand_landmarks(frame, hand_results, mp_drawing, mp_drawing_styles, hand_connections)
            if show_runtime_frame(
                window_name,
                frame,
                [
                    ("gesture:PHOTO_COUNTDOWN", (10, 24), (0, 255, 255), 0.55),
                    (f"Taking photo in {remaining_photo_time:.1f}s", (10, 48), (0, 255, 255), 0.55),
                    (f"light_mode:{current_light_mode}", (10, 72), (255, 255, 255), 0.55),
                ],
            ):
                break
            if photo_countdown_active:
                continue

        if photo_trigger_armed and two_detected:
            two_counter += 1
            if two_counter >= TWO_ENTER_FRAMES:
                # Remember the current tracking mode so we can resume it after the photo is taken.
                photo_countdown_active = True
                photo_countdown_start = current_time
                photo_return_mode = gesture_mode
                photo_trigger_armed = False
                two_counter = 0
                prev_smoothed, prev_time, consecutive_lost_frames, state = reset_tracking_state(LOCKED)
                continue

        gesture_mode, pinch_counter, four_counter, fist_counter, pinch_point, reset_state = update_gesture_mode(
            gesture_mode,
            pinch_detected,
            index_tip_point,
            four_detected,
            fist_detected,
            pinch_counter,
            four_counter,
            fist_counter,
            pinch_point,
            SEEKING,
            LOCKED
        )
        if reset_state is not None:
            prev_smoothed, prev_time, consecutive_lost_frames, state = reset_tracking_state(reset_state)

        centroid, eye_dist_px = get_mouth_centroid_and_eye_dist(processed_image, frame_width, frame_height)

        if gesture_mode == GESTURE_TRACK_PINCH and pinch_point is not None:
            centroid = pinch_point

        if gesture_mode == GESTURE_LOCKED:
            last_send_time = send_zero_if_due(current_time, last_send_time)

            draw_hand_landmarks(frame, hand_results, mp_drawing, mp_drawing_styles, hand_connections)
            if show_runtime_frame(
                window_name,
                frame,
                [
                    ("gesture:LOCKED", (10, 24), (0, 255, 255), 0.55),
                    ("Pinch to track finger | Show 4 to mouth track", (10, 48), (0, 255, 255), 0.55),
                    ("Show 2 to take patient photo", (10, 72), (0, 255, 255), 0.55),
                    (f"light_mode:{current_light_mode}", (10, 96), (255, 255, 255), 0.55),
                ],
            ):
                break
            continue

        # Handle if no centroid was found
        if centroid is None:
            consecutive_lost_frames += 1
            # if we lose tracking hold
            last_send_time = send_zero_if_due(current_time, last_send_time)

            # Check how many consecutive_lost_frames frames we have
            if consecutive_lost_frames > MAX_LOST_FRAMES:
                prev_smoothed = None
                prev_time = None

            draw_hand_landmarks(frame, hand_results, mp_drawing, mp_drawing_styles, hand_connections)
            if show_runtime_frame(
                window_name,
                frame,
                [
                    (f"light_mode:{current_light_mode}", (10, 24), (255, 255, 255), 0.55),
                ],
            ):
                break
            # Go back to top of while loop
            continue

        # Reset consecutive_lost_frames and smoothed if we got face points
        consecutive_lost_frames = 0
        point_alpha = FINGER_SMOOTH_ALPHA if gesture_mode == GESTURE_TRACK_PINCH else SMOOTH_ALPHA
        smoothed = ema_point(centroid, prev_smoothed, point_alpha)

        if prev_time is None:
            prev_time = current_time
        # ensure the change in time is non-zero
        delta_time = max(1e-6, current_time - prev_time)

        # Find the angular change between frames and convert to speed in deg/s
        if prev_smoothed is None:
            speed = 0.0
        else:
            pixal_displacement_x = smoothed[0] - prev_smoothed[0]
            pixal_displacement_y = smoothed[1] - prev_smoothed[1]
            # Get displacement in x and y in degrees
            dvx, dvy = pixels_to_deg(pixal_displacement_x, pixal_displacement_y, frame_width, frame_height, FOV_H_DEG, FOV_V_DEG)
            # Convert the displacement into speed
            speed = math.hypot(dvx, dvy) / delta_time

        # Set timing and smoothed the previous values for next loop
        prev_time = current_time
        prev_smoothed = smoothed

        # Update stable box scalar using discrete ranges + debounce
        desired_range = range_from_eye_dist(eye_dist_px)
        stable_range, pending_range, pending_count, stable_scalar = update_stable_range_state(
            desired_range,
            stable_range,
            pending_range,
            pending_count
        )

        current_stable_scalar = stable_scalar
        current_stop_threshold = STABLE_STOP_SEEKING_THRESHOLD

        if gesture_mode == GESTURE_TRACK_PINCH:
            current_stable_scalar *= FINGER_STABLE_SCALAR_MULT
            current_stop_threshold = FINGER_STOP_SEEKING_THRESHOLD

        # Build our stable box (dynamic scalar)
        stable_box = build_stable_box(anchor, frame_width, frame_height, current_stable_scalar)
        # Determine if we are in the stable region
        in_stable_region = inside_box(smoothed, stable_box)

        # Offset of centroid from center of frame
        dx_center = smoothed[0] - anchor[0]
        dy_center = smoothed[1] - anchor[1]

        # Normalizes the offset error
        norm_dx = dx_center / (frame_width / 2.0)
        norm_dy = dy_center / (frame_height / 2.0)
        # Find radial distance from center
        radial_norm = math.hypot(norm_dx, norm_dy)
        # Check if we are close enough to stop
        within_stop_threshold = (radial_norm <= current_stop_threshold)

        ### Compute Jitter using timed histogram to set too_wild var (may be able to be removed) ###

        # Add values to the histogram
        pos_x.add(current_time, smoothed[0])
        pos_y.add(current_time, smoothed[1])
        vel_h.add(current_time, speed)

        xs, ys = pos_x.values(), pos_y.values()
        pos_std = 999.0
        if len(xs) >= 6 and len(ys) >= 6:
            pos_std = 0.5 * (statistics.pstdev(xs) + statistics.pstdev(ys))
        speeds = vel_h.values()
        vel_med = statistics.median(speeds) if len(speeds) >= 3 else 999.0

        # If this is 1, the gimbal will not move and is in place as a precaution to stop the gimbal from chasing error
        too_wild = (vel_med > VEL_THRESH_DEG_S * 1000.0) or (pos_std > POS_STD_THRESH_PX * 1000.0)

        ###                                                                                      ###

        # Set States
        if state == LOCKED:
            if not in_stable_region:
                state = SEEKING
        else:
            if within_stop_threshold:
                state = LOCKED

        # Comput the speed commands to send
        if state == SEEKING and not too_wild:
            err_yaw_deg, err_pitch_deg = pixels_to_deg(dx_center, dy_center, frame_width, frame_height, FOV_H_DEG, FOV_V_DEG)
            err_yaw_deg *= AXIS_SIGN["yaw"]
            err_pitch_deg *= AXIS_SIGN["pitch"]

            deadband_yaw = FINGER_DEADBAND_DEG_YAW if gesture_mode == GESTURE_TRACK_PINCH else DEADBAND_DEG_YAW
            deadband_pitch = FINGER_DEADBAND_DEG_PITCH if gesture_mode == GESTURE_TRACK_PINCH else DEADBAND_DEG_PITCH

            # Deadband
            if abs(err_yaw_deg) < deadband_yaw:
                err_yaw_deg = 0.0
            if abs(err_pitch_deg) < deadband_pitch:
                err_pitch_deg = 0.0

            kp_yaw = FINGER_KP_YAW_DPS_PER_DEG if gesture_mode == GESTURE_TRACK_PINCH else KP_YAW_DPS_PER_DEG
            kp_pitch = FINGER_KP_PITCH_DPS_PER_DEG if gesture_mode == GESTURE_TRACK_PINCH else KP_PITCH_DPS_PER_DEG

            yaw_dps = clamp(kp_yaw * err_yaw_deg, -MAX_DPS_YAW, +MAX_DPS_YAW)
            pitch_dps = clamp(kp_pitch * err_pitch_deg, -MAX_DPS_PITCH, +MAX_DPS_PITCH)
            roll_dps = 0.0  # keep roll off unless you want it

        else:
            # LOCKED or too_wild so hold and do nothing
            yaw_dps = 0.0
            pitch_dps = 0.0
            roll_dps = 0.0

        # smooth and send the speeds to the controller
        sent = 0
        # Check if enough time has passed since last send
        if (current_time - last_send_time) >= COMMAND_PERIOD:
            cmd_alpha = FINGER_CMD_SPEED_EMA_ALPHA if gesture_mode == GESTURE_TRACK_PINCH else CMD_SPEED_EMA_ALPHA

            smooth_yaw_dps = ema_scalar(yaw_dps, smooth_yaw_dps, cmd_alpha)
            smooth_pitch_dps = ema_scalar(pitch_dps, smooth_pitch_dps, cmd_alpha)
            smooth_roll_dps = ema_scalar(roll_dps, smooth_roll_dps, cmd_alpha)

            ok_send = send_speeds(smooth_roll_dps, smooth_pitch_dps, smooth_yaw_dps)
            if ok_send:
                last_send_time = current_time
                sent = 1

        # Only print telemetry if desired
        if PRINT_TELEMETRY:
            eye_str = f"{eye_dist_px:.1f}" if eye_dist_px is not None else "None"
            print(f"{current_time - initial_time:.3f} {yaw_dps:+.2f} {pitch_dps:+.2f} {sent} {state} r={radial_norm:.3f} eye={eye_str} box={current_stable_scalar:.3f} {stable_range} light_mode={current_light_mode} A0={light_mode_voltages[0]:.3f} A1={light_mode_voltages[1]:.3f} A2={light_mode_voltages[2]:.3f} A3={light_mode_voltages[3]:.3f}")

        # This draws out the frame for seeing the tracking in real time and has no effect on the algorithm
        if DRAW_FRAME_RT:
            l, t_, r, b = map(int, stable_box)
            cv2.rectangle(frame, (l, t_), (r, b), (40, 220, 40), 1)
            cv2.drawMarker(frame, (int(anchor[0]), int(anchor[1])), (0, 200, 0),
                           cv2.MARKER_CROSS, 12, 2)
            cv2.circle(frame, (int(smoothed[0]), int(smoothed[1])), 4, (0, 0, 255), -1)

            draw_hand_landmarks(frame, hand_results, mp_drawing, mp_drawing_styles, hand_connections)

            if pinch_point is not None and gesture_mode == GESTURE_TRACK_PINCH:
                cv2.circle(frame, (int(pinch_point[0]), int(pinch_point[1])), 8, (255, 0, 255), -1)

            eye_str = f"{eye_dist_px:.1f}" if eye_dist_px is not None else "None"
            state_txt = "LOCKED" if state == LOCKED else "SEEKING"
            gesture_txt = "MOUTH"
            if gesture_mode == GESTURE_TRACK_PINCH:
                gesture_txt = "INDEX_TIP"
            elif gesture_mode == GESTURE_LOCKED:
                gesture_txt = "LOCKED"

            if show_runtime_frame(
                window_name,
                frame,
                [
                    (f"state:{state_txt}", (10, 24), (40, 220, 40), 0.55),
                    (f"Radial distance = {radial_norm:.3f}", (10, 48), (40, 220, 40), 0.55),
                    (f"eye_px={eye_str} box={current_stable_scalar:.3f} {stable_range}", (10, 72), (40, 220, 40), 0.55),
                    (f"light_mode:{current_light_mode}", (10, 96), (255, 255, 255), 0.55),
                    (f"gesture:{gesture_txt}", (10, 120), (255, 255, 255), 0.55),
                    ("Pinch -> fingertip | Fist -> lock | Four -> mouth", (10, 144), (255, 255, 255), 0.55),
                    ("Two -> 2s countdown then center photo", (10, 168), (255, 255, 255), 0.55),
                ],
            ):
                break

    # stop motion on exit
    try:
        # Send a hold command to the motors
        send_speeds(0.0, 0.0, 0.0)
        # Stop motors on end of program
        set_motors(0)
    except Exception:
        pass

    hands.close()
    face_mesh.close()
    face_track_cam.release()
    if center_camera is not None:
        center_camera.release()
    cv2.destroyAllWindows()

    print("Stopped.")


if __name__ == "__main__":
    main()
