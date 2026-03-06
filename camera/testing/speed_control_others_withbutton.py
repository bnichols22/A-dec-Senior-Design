#!/usr/bin/env python3
# ==============================================================
# File: speed_control_compliance_dynbox.py
# Purpose:
#   Smooth face tracker that drives gimbal using SPEED mode (deg/s).
#
# ADDITIONS:
#   (1) Dynamic stable box sizing (discrete ranges) based on face distance
#       proxy using FaceMesh eye distance (in pixels). Closer face => bigger box.
#   (2) Compliance + lock modes on 'c' key, using a 3-press cycle:
#       - Press 'c' once  -> MOTORS OFF (compliance; user can reposition by hand)
#       - Press 'c' twice -> MOTORS ON, HOLD (0 speed) but TRACKING DISABLED
#       - Press 'c' third -> TRACKING ENABLED again (normal operation)
#       Then repeat.
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
import lgpio
import json

# --------- Paths and Filename ----------
file_name = "speed_control_compliance_dynbox.py"
BASE_DIR = os.path.expanduser('~/senior_design/A-dec-Senior-Design/camera/testing')
os.makedirs(BASE_DIR, exist_ok=True)

LOG_PATH  = os.path.join(BASE_DIR, 'face_track_log.txt')
TEST_LOG  = os.path.join(BASE_DIR, 'Board_Serial_Command_Test.txt')

LIB_PATH = os.path.expanduser(
    "~/senior_design/A-dec-Senior-Design/camera/testing/SerialLibrary/serialAPI/libsimplebgc.so"
)

CAMERA_PROFILE_DIR = os.path.join(BASE_DIR, "camera_profiles")

# --------- Vision / tracker config ----------
CAM_INDEX = 0
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
COMMAND_HZ = 150.0
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

def button_pressed_edge(now, gpio_handle, button_pin, button_debounce_sec, last_button_event_time, prev_button_level):
    level = lgpio.gpio_read(gpio_handle, button_pin)

    pressed_event = False
    if prev_button_level == 1 and level == 0:
        if (now - last_button_event_time) >= button_debounce_sec:
            last_button_event_time = now
            pressed_event = True

    prev_button_level = level
    return pressed_event, last_button_event_time, prev_button_level

def handle_mode_cycle(mode, prev_smoothed, prev_time, consecutive_lost_frames, state, TRACKING_ENABLED, COMPLIANCE_MOTORS_OFF, HOLD_MOTORS_ON_NO_TRACK, LOCKED):
    if mode == TRACKING_ENABLED:
        # Press once -> motors OFF (compliance)
        send_speeds(0.0, 0.0, 0.0)
        set_motors(0)
        mode = COMPLIANCE_MOTORS_OFF

        prev_smoothed = None
        prev_time = None
        consecutive_lost_frames = 0
        state = LOCKED
        return mode, prev_smoothed, prev_time, consecutive_lost_frames, state

    if mode == COMPLIANCE_MOTORS_OFF:
        # Press twice -> motors ON, HOLD (tracking still disabled)
        set_motors(1)
        send_speeds(0.0, 0.0, 0.0)
        send_speeds(0.0, 0.0, 0.0)
        mode = HOLD_MOTORS_ON_NO_TRACK

        prev_smoothed = None
        prev_time = None
        consecutive_lost_frames = 0
        state = LOCKED
        return mode, prev_smoothed, prev_time, consecutive_lost_frames, state

    # Press third -> tracking enabled again
    mode = TRACKING_ENABLED
    prev_smoothed = None
    prev_time = None
    consecutive_lost_frames = 0
    state = LOCKED
    return mode, prev_smoothed, prev_time, consecutive_lost_frames, state

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
        camera.set(cv2.CAP_PROP_EXPOSURE, camera_settings["exposure"])
        camera.set(cv2.CAP_PROP_BRIGHTNESS, camera_settings["brightness"])
        camera.set(cv2.CAP_PROP_CONTRAST, camera_settings["contrast"])
        print(f"Loaded camera profile: {profile_path}")
    except Exception as camera_update_error:
        print(f"Error loading profile: {camera_update_error}")


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


# ---------------- Main loop ----------------
def main():

    # ======= Setup =======
    init_sbgc()

    mp_face_mesh = mp.solutions.face_mesh
    # Can adjust confidence as and detection as needed
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )

    # Create capture device object and verify it constructed
    face_track_cam = cv2.VideoCapture(CAM_INDEX)
    if not face_track_cam.isOpened():
        print(f'main: Error Unable to open camera from {CAM_INDEX}')
        # Exit here because we cannot run without camera
        sys.exit(1)

    CAMERA_PROFILE_NAME = "wide_angle_full_light_off.json"
    update_camera_settings(face_track_cam, CAMERA_PROFILE_NAME, CAMERA_PROFILE_DIR)

    # Set the max number of stored frames allowed
    face_track_cam.set(cv2.CAP_PROP_BUFFERSIZE, MAX_STORED_FRAMES)

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
    # NEW: Compliance / lock / tracking state (3-press 'c' cycle)
    # ==============================================================
    TRACKING_ENABLED = 0
    COMPLIANCE_MOTORS_OFF = 1
    HOLD_MOTORS_ON_NO_TRACK = 2
    mode = TRACKING_ENABLED

    # ==============================================================
    # NEW: Dynamic stable-box state
    # ==============================================================
    stable_range = "MID"
    pending_range = None
    pending_count = 0
    stable_scalar = STABLE_SCALAR_DEFAULT

    # GPIO 6 pushbutton input (internal pull-up)
    BUTTON_PIN = 6
    BUTTON_DEBOUNCE_SEC = 0.20
    last_button_event_time = 0.0
    prev_button_level = 1

    gpio_handle = None
    try:
        gpio_handle = lgpio.gpiochip_open(0)
        if hasattr(lgpio, "SET_PULL_UP"):
            lgpio.gpio_claim_input(gpio_handle, BUTTON_PIN, lgpio.SET_PULL_UP)
        else:
            lgpio.gpio_claim_input(gpio_handle, BUTTON_PIN)
    except Exception as e:
        print(f"main: lgpio init failed: {e}")
        gpio_handle = None

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

        if anchor is None:
            anchor = (frame_width / 2.0, frame_height / 2.0)

        # Do this only if the function exists for hadning button changes
        if gpio_handle is not None:
            button_event, last_button_event_time, prev_button_level = button_pressed_edge(
                current_time,
                gpio_handle,
                BUTTON_PIN,
                BUTTON_DEBOUNCE_SEC,
                last_button_event_time,
                prev_button_level
            )
            if button_event:
                mode, prev_smoothed, prev_time, consecutive_lost_frames, state = handle_mode_cycle(
                    mode,
                    prev_smoothed,
                    prev_time,
                    consecutive_lost_frames,
                    state,
                    TRACKING_ENABLED,
                    COMPLIANCE_MOTORS_OFF,
                    HOLD_MOTORS_ON_NO_TRACK,
                    LOCKED
                )
                
        # Handle compliance/hold modes before doing face tracking
        if mode != TRACKING_ENABLED:
            # In these modes, we do NOT compute face tracking commands.
            # We either have motors OFF (compliance) or motors ON holding (0 speed).
            if (current_time - last_send_time) >= COMMAND_PERIOD:
                # HOLD mode wants a steady 0-speed stream.
                # Compliance mode does not need commands, but this keeps timing consistent.
                if mode == HOLD_MOTORS_ON_NO_TRACK:
                    send_speeds(0.0, 0.0, 0.0)
                last_send_time = current_time

            if DRAW_FRAME_RT:
                if mode == COMPLIANCE_MOTORS_OFF:
                    msg = "COMPLIANCE (motors OFF) - press 'c' to lock motors ON"
                    color = (0, 0, 255)
                else:
                    msg = "LOCKED HOLD (motors ON) - press 'c' to reenable tracking"
                    color = (0, 200, 255)

                cv2.putText(frame, msg, (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
                cv2.imshow(f"Image playback using: {file_name}", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
            else:
                # No UI window: cannot read 'c' reliably, so just keep holding.
                pass

            # Skip to next frame
            continue

        # Normal tracking mode
        # get the capture from camera
        rgb_frame_cap = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # process the image and set landmarks
        processed_image = face_mesh.process(rgb_frame_cap)

        centroid = None
        eye_dist_px = None

        # Get the points from the face mesh and average to get centroid tuple
        if processed_image.multi_face_landmarks:
            # Only 1 face tracked, assumed to be patient face at index 0
            patient_face = processed_image.multi_face_landmarks[0]

            # known point ids: center upper lip, lower center lip, left mouth corner, right mouth corner
            mouth_idxs = [13, 14, 61, 291]

            mouth_points = []
            for idx in mouth_idxs:
                x = int(patient_face.landmark[idx].x * frame_width)
                y = int(patient_face.landmark[idx].y * frame_height)
                mouth_points.append((x, y))

            if mouth_points:
                centroid_x = sum(p[0] for p in mouth_points) / len(mouth_points)
                centroid_y = sum(p[1] for p in mouth_points) / len(mouth_points)
                # Make tuple of averaged x and y vals to get mouth center centroid
                centroid = (centroid_x, centroid_y)

            # Get distance proximity for dynamic stable-box sizing
            eye_dist_px = estimate_eye_dist_px(patient_face, frame_width, frame_height)

        # Handle if no centroid was found
        if centroid is None:
            consecutive_lost_frames += 1
            # if we lose tracking hold
            if (current_time - last_send_time) >= COMMAND_PERIOD:
                send_speeds(0.0, 0.0, 0.0)
                last_send_time = current_time

            # Check how many consecutive_lost_frames frames we have
            if consecutive_lost_frames > MAX_LOST_FRAMES:
                prev_smoothed = None
                prev_time = None

            if DRAW_FRAME_RT:
                cv2.imshow(f"Image playback using: {file_name}", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
            # Go back to top of while loop
            continue

        # Reset consecutive_lost_frames and smoothed if we got face points
        consecutive_lost_frames = 0
        smoothed = ema_point(centroid, prev_smoothed, SMOOTH_ALPHA)

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
                stable_scalar = scalar_for_range(stable_range)
        else:
            pending_range = None
            pending_count = 0
            stable_scalar = scalar_for_range(stable_range)

        # Build our stable box (dynamic scalar)
        stable_box = build_stable_box(anchor, frame_width, frame_height, stable_scalar)
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
        within_stop_threshold = (radial_norm <= STABLE_STOP_SEEKING_THRESHOLD)

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

        # If this is 0, the gimbal will not move and is in place as a precaution to stop the gimbal from chasing error
        too_wild = (vel_med > VEL_THRESH_DEG_S * 2.0) or (pos_std > POS_STD_THRESH_PX * 2.0)

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

            # Deadband
            if abs(err_yaw_deg) < DEADBAND_DEG_YAW:
                err_yaw_deg = 0.0
            if abs(err_pitch_deg) < DEADBAND_DEG_PITCH:
                err_pitch_deg = 0.0


            yaw_dps = clamp(KP_YAW_DPS_PER_DEG * err_yaw_deg, -MAX_DPS_YAW, +MAX_DPS_YAW)
            pitch_dps = clamp(KP_PITCH_DPS_PER_DEG * err_pitch_deg, -MAX_DPS_PITCH, +MAX_DPS_PITCH)
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
            smooth_yaw_dps = ema_scalar(yaw_dps, smooth_yaw_dps, CMD_SPEED_EMA_ALPHA)
            smooth_pitch_dps = ema_scalar(pitch_dps, smooth_pitch_dps, CMD_SPEED_EMA_ALPHA)
            smooth_roll_dps = ema_scalar(roll_dps, smooth_roll_dps, CMD_SPEED_EMA_ALPHA)

            ok_send = send_speeds(smooth_roll_dps, smooth_pitch_dps, smooth_yaw_dps)
            if ok_send:
                last_send_time = current_time
                sent = 1

        # Only print telemetry if desired
        if PRINT_TELEMETRY:
            eye_str = f"{eye_dist_px:.1f}" if eye_dist_px is not None else "None"
            print(f"{current_time - initial_time:.3f} {yaw_dps:+.2f} {pitch_dps:+.2f} {sent} {state} r={radial_norm:.3f} eye={eye_str} box={stable_scalar:.3f} {stable_range}")

        # This draws out the frame for seeing the tracking in real time and has no effect on the algorithm
        if DRAW_FRAME_RT:
            l, t_, r, b = map(int, stable_box)
            cv2.rectangle(frame, (l, t_), (r, b), (40, 220, 40), 1)
            cv2.drawMarker(frame, (int(anchor[0]), int(anchor[1])), (0, 200, 0),
                           cv2.MARKER_CROSS, 12, 2)
            cv2.circle(frame, (int(smoothed[0]), int(smoothed[1])), 4, (0, 0, 255), -1)

            if state == LOCKED:
                state_txt = "LOCKED"
            else:
                state_txt = "SEEKING"

            cv2.putText(frame, f"state:{state_txt}", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40,220,40), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Radial distance = {radial_norm:.3f}", (10, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40,220,40), 2, cv2.LINE_AA)

            # NEW: show dynamic stable-box info
            eye_str = f"{eye_dist_px:.1f}" if eye_dist_px is not None else "None"
            cv2.putText(frame, f"eye_px={eye_str} box={stable_scalar:.3f} {stable_range}", (10, 72),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40,220,40), 2, cv2.LINE_AA)

            cv2.putText(frame, "Press 'c' -> compliance/lock cycle", (10, 96),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2, cv2.LINE_AA)

            cv2.imshow(f"Image playback using: {file_name}", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == ord('c'):
                CAMERA_PROFILE_NAME = "wide_angle_full_light_on.json"
                update_camera_settings(face_track_cam, CAMERA_PROFILE_NAME, CAMERA_PROFILE_DIR)
            if key == ord('d'):
                CAMERA_PROFILE_NAME = "wide_angle_full_light_off.json"
                update_camera_settings(face_track_cam, CAMERA_PROFILE_NAME, CAMERA_PROFILE_DIR)
                
    # stop motion on exit
    try:
        # Send a hold command to the motors
        send_speeds(0.0, 0.0, 0.0)
        # Stop motors on end of program
        set_motors(0)
    except Exception:
        pass

    face_track_cam.release()
    cv2.destroyAllWindows()

    if gpio_handle is not None:
        try:
            lgpio.gpio_free(gpio_handle, BUTTON_PIN)
        except Exception:
            pass
        try:
            lgpio.gpiochip_close(gpio_handle)
        except Exception:
            pass

    print("Stopped.")


if __name__ == "__main__":
    main()