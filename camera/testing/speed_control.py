#!/usr/bin/env python3
# ==============================================================
# File: speed_control.py
# Purpose:
#   Smooth face tracker that drives gimbal using SPEED mode (deg/s).
#
# Motor lib usage:
#   SimpleBGC SerialAPI shim:
#     - bgc_init()
#     - bgc_control_speeds(roll_dps, pitch_dps, yaw_dps)
# ==============================================================

import os, sys, time, math, warnings, statistics, ctypes
import cv2
import mediapipe as mp
from collections import deque

# --------- Paths + Filename ----------
file_name = "speed_control.py"
BASE_DIR = os.path.expanduser('~/senior_design/A-dec-Senior-Design/camera/testing')
os.makedirs(BASE_DIR, exist_ok=True)

LOG_PATH  = os.path.join(BASE_DIR, 'face_track_log.txt')
TEST_LOG  = os.path.join(BASE_DIR, 'Board_Serial_Command_Test.txt')

LIB_PATH = os.path.expanduser(
    "~/senior_design/A-dec-Senior-Design/camera/testing/SerialLibrary/serialAPI/libsimplebgc.so"
)

# --------- Vision / tracker config ----------
CAM_INDEX = 0
FOV_H_DEG = 65.0
FOV_V_DEG = 48.75

STABLE_SCALAR = 0.06
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
CMD_SPEED_EMA_ALPHA = 0.35  # 

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


# ---------------- Helper Functions ----------------

def ema_point(curr, prev, alpha):
    if prev is None:
        return curr
    return (alpha * curr[0] + (1 - alpha) * prev[0],
            alpha * curr[1] + (1 - alpha) * prev[1])

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

def angle_deg(point1, point2):
    return math.degrees(math.atan2(point2[1] - point1[1], point2[0] - point1[0]))

def build_stable_box(center_point, frame_width, frame_height, scalar):
    center_x, center_y = center_point
    half_width = scalar * (frame_width / 2.0)
    half_height = scalar * (frame_height / 2.0)
    return (center_x - half_width, center_y - half_height, center_x + half_width, center_y + half_height)

def inside_box(pt, stable_box):
    x, y = pt
    l, t, r, b = stable_box
    return (l <= x <= r) and (t <= y <= b)

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

def update_camera_settings(camera, filename):

    if not os.path.exists(filename):
        print(f"Error: File {filename} not found.")
        return None
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        print(f"--> Profile LOADED from {filename}")
        return data
    except Exception as e:
        print(f"Error loading profile: {e}")
        return None

    # If a camera doesn't support a property, cv2 usually just ignores it or returns false.
    camera.set(cv2.CAP_PROP_EXPOSURE, current_settings["exposure"])
    camera.set(cv2.CAP_PROP_BRIGHTNESS, current_settings["brightness"])
    camera.set(cv2.CAP_PROP_CONTRAST, current_settings["contrast"])

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
    capture_dev = cv2.VideoCapture(CAM_INDEX)
    if not capture_dev.isOpened():
        print(f'main: Error Unable to open camera from {CAM_INDEX}')
        # Exit here because we cannot run without camera
        sys.exit(1)
    
    # Set the max number of stored frames allowed
    capture_dev.set(cv2.CAP_PROP_BUFFERSIZE, MAX_STORED_FRAMES)

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

    # ======= Main Loop =======
    while True:
        yaw_dps = 0.0
        pitch_dps = 0.0
        roll_dps = 0.0

        frame_read, frame = capture_dev.read()
        if not frame_read:
            log_file.write(f"main: frame grab failed\n")
            break

        current_time = time.time()
        frame_height, frame_width = frame.shape[:2]

        if anchor is None:
            anchor = (frame_width / 2.0, frame_height / 2.0)

        # get the capture from camera
        rgb_frame_cap = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # process the image detecting faces and setting landmarks
        processed_image = face_mesh.process(rgb_frame_cap)

        centroid = None

        # Get the points from the face mesh and average to get centroid tuple
        if processed_image.multi_face_landmarks:
            # Only 1 face tracked, assumed to be patient face at index 0
            patient_face = processed_image.multi_face_landmarks[0]
            # known point ids: center upper lip, lower center lip, left mouth corner, right mouth corner
            mouth_idxs = [13, 14, 61, 291]
            # 
            mouth_points = []
            for idx in mouth_idxs:
                x = int(patient_face.landmark[idx].x * frame_width)
                y = int(patient_face.landmark[idx].y * frame_height)
                mouth_points.append((x, y))
            if mouth_points:
                centroid_x = sum(x[0] for x in mouth_points) / len(mouth_points)
                centroid_y = sum(y[1] for y in mouth_points) / len(mouth_points)
                # Make tuple of averaged x and y vals to get mouth center centroid
                centroid = (centroid_x, centroid_y)
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
                if cv2.waitKey(1) & 0xFF == 27:
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

        # Build our stable box
        stable_box = build_stable_box(anchor, frame_width, frame_height, STABLE_SCALAR)
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
            print(f"{current_time - initial_time:.3f} {yaw_dps:+.2f} {pitch_dps:+.2f} {sent} {state} r={radial_norm:.3f}")

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

            cv2.imshow(f"Image playback using: {file_name}", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    # stop motion on exit
    try:
        # Send a hold command to the motors
        send_speeds(0.0, 0.0, 0.0)
        # Stop motors on end of program
        motor_library.bgc_set_motors(0)
    except Exception:
        pass

    capture_dev.release()
    cv2.destroyAllWindows()
    print("Stopped.")


if __name__ == "__main__":
    main()
