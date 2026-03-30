#!/usr/bin/env python3
# ==============================================================
# File: speed_control_compliance_dynbox_qt.py
# Purpose:
#   Smooth face tracker that drives gimbal using SPEED mode (deg/s).
#   Wrapped in a PyQt5 Application.
# ==============================================================

import os, sys, time, math, warnings, statistics, ctypes
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import lgpio
import json
import board
import busio
from adafruit_ads1x15 import ADS1115, AnalogIn, ads1x15

# --- PyQt5 Imports ---
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QHBoxLayout
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap

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

STABLE_SCALAR_DEFAULT = 0.06
WINDOW_SEC = 0.6
STABLE_STOP_SEEKING_THRESHOLD = 0.025
VEL_THRESH_DEG_S  = 2.5
POS_STD_THRESH_PX = 2.5

KP_YAW_DPS_PER_DEG   = 1.25
KP_PITCH_DPS_PER_DEG = 1.25

MAX_DPS_YAW   = 80.0
MAX_DPS_PITCH = 80.0
MAX_DPS_ROLL  = 60.0

DEADBAND_DEG_YAW   = 0.25
DEADBAND_DEG_PITCH = 0.25

COMMAND_HZ = 500.0
COMMAND_PERIOD = 1.0 / COMMAND_HZ
CMD_SPEED_EMA_ALPHA = 0.35

AXIS_SIGN = {"yaw": 1, "pitch": 1, "roll": 1}
MAX_STORED_FRAMES = 1
PRINT_TELEMETRY = False

SMOOTH_ALPHA = 0.25
MAX_LOST_FRAMES = 10

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

EYE_L_IDX = 33
EYE_R_IDX = 263
EYE_DIST_FAR_MAX   = 45.0
EYE_DIST_NEAR_MIN  = 85.0

STABLE_SCALAR_FAR  = 0.035
STABLE_SCALAR_MID  = 0.060
STABLE_SCALAR_NEAR = 0.095
RANGE_SWITCH_FRAMES = 8

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

# ---------------- Helper Functions ----------------
# (Kept identical to original script)
def ema_point(current_smooth, previous_smooth, alpha):
    if previous_smooth is None: return current_smooth
    return (alpha * current_smooth[0] + (1 - alpha) * previous_smooth[0],
            alpha * current_smooth[1] + (1 - alpha) * previous_smooth[1])

def ema_scalar(current, previous, alpha):
    if previous is None: return current
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
    try:
        lx = face_landmarks.landmark[EYE_L_IDX].x * frame_width
        ly = face_landmarks.landmark[EYE_L_IDX].y * frame_height
        rx = face_landmarks.landmark[EYE_R_IDX].x * frame_width
        ry = face_landmarks.landmark[EYE_R_IDX].y * frame_height
        return math.hypot(rx - lx, ry - ly)
    except Exception:
        return None

def range_from_eye_dist(eye_dist_px):
    if eye_dist_px is None: return "MID"
    if eye_dist_px <= EYE_DIST_FAR_MAX: return "FAR"
    if eye_dist_px >= EYE_DIST_NEAR_MIN: return "NEAR"
    return "MID"

def scalar_for_range(rng):
    if rng == "FAR": return STABLE_SCALAR_FAR
    if rng == "NEAR": return STABLE_SCALAR_NEAR
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
        send_speeds(0.0, 0.0, 0.0)
        set_motors(0)
        mode = COMPLIANCE_MOTORS_OFF
        prev_smoothed = None
        prev_time = None
        consecutive_lost_frames = 0
        state = LOCKED
        return mode, prev_smoothed, prev_time, consecutive_lost_frames, state

    if mode == COMPLIANCE_MOTORS_OFF:
        set_motors(1)
        send_speeds(0.0, 0.0, 0.0)
        send_speeds(0.0, 0.0, 0.0)
        mode = HOLD_MOTORS_ON_NO_TRACK
        prev_smoothed = None
        prev_time = None
        consecutive_lost_frames = 0
        state = LOCKED
        return mode, prev_smoothed, prev_time, consecutive_lost_frames, state

    mode = TRACKING_ENABLED
    prev_smoothed = None
    prev_time = None
    consecutive_lost_frames = 0
    state = LOCKED
    return mode, prev_smoothed, prev_time, consecutive_lost_frames, state

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
        profile_path = filename if os.path.isabs(filename) else os.path.join(profile_dir, filename)
        with open(profile_path, 'r') as f:
            camera_settings = json.load(f)
        if "auto_exposure" in camera_settings: camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, camera_settings["auto_exposure"])
        if "exposure" in camera_settings: camera.set(cv2.CAP_PROP_EXPOSURE, camera_settings["exposure"])
        if "brightness" in camera_settings: camera.set(cv2.CAP_PROP_BRIGHTNESS, camera_settings["brightness"])
        if "contrast" in camera_settings: camera.set(cv2.CAP_PROP_CONTRAST, camera_settings["contrast"])
        if "gain" in camera_settings: camera.set(cv2.CAP_PROP_GAIN, camera_settings["gain"])
        if "saturation" in camera_settings: camera.set(cv2.CAP_PROP_SATURATION, camera_settings["saturation"])
        print(f"Loaded camera profile: {profile_path}")
        return True
    except Exception as camera_update_error:
        print(f"Error loading profile: {camera_update_error}")
        return False

def read_light_mode(adc_channels, off_threshold_volts):
    v0, v1, v2, v3 = adc_channels["A0"].voltage, adc_channels["A1"].voltage, adc_channels["A2"].voltage, adc_channels["A3"].voltage
    voltage_map = {
        YELLOW_LIGHT_MODE:  v0,
        LOWEST_LIGHT_MODE:  v1,
        MEDIUM_LIGHT_MODE:  v2,
        HIGHEST_LIGHT_MODE: v3,
    }
    active_modes = {mode_name: voltage for mode_name, voltage in voltage_map.items() if voltage >= off_threshold_volts}
    if not active_modes: return LIGHT_OFF_MODE, (v0, v1, v2, v3)
    return max(active_modes, key=active_modes.get), (v0, v1, v2, v3)

def update_camera_profile_from_light_mode(camera, current_light_mode, previous_light_mode, profile_dir):
    if current_light_mode == previous_light_mode: return previous_light_mode
    profile_filename = LIGHT_MODE_TO_PROFILE.get(current_light_mode)
    if profile_filename is None: return previous_light_mode
    if update_camera_settings(camera, profile_filename, profile_dir):
        print(f"Light mode changed: {previous_light_mode} -> {current_light_mode}")
        return current_light_mode
    return previous_light_mode

# ---------------- SBGC shim bindings ----------------
motor_library = None
motor_library_initialized = False

def init_sbgc():
    global motor_library, motor_library_initialized
    try:
        motor_library = ctypes.CDLL(LIB_PATH)
        print(f"Loaded SBGC library from {LIB_PATH}")
    except OSError as e:
        print(f"init_sbgc: error loading {LIB_PATH}: {e}")
        motor_library, motor_library_initialized = None, False
        return

    motor_library.bgc_init.argtypes = []
    motor_library.bgc_init.restype = ctypes.c_int
    motor_library.bgc_control_speeds.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float]
    motor_library.bgc_control_speeds.restype = ctypes.c_int

    if hasattr(motor_library, "bgc_set_motors"):
        motor_library.bgc_set_motors.argtypes = [ctypes.c_int]
        motor_library.bgc_set_motors.restype = ctypes.c_int
    else:
        print("init_sbgc: WARNING libsimplebgc.so missing bgc_set_motors(). Compliance mode will not work.")

    if motor_library.bgc_init() != 0:
        print(f"init_sbgc: error bgc_init() failed")
        motor_library_initialized = False
        return
    motor_library_initialized = True

def send_speeds(roll_dps, pitch_dps, yaw_dps):
    if motor_library is None or not motor_library_initialized: return False
    return motor_library.bgc_control_speeds(ctypes.c_float(roll_dps), ctypes.c_float(pitch_dps), ctypes.c_float(yaw_dps)) == 0

def set_motors(on_off: int):
    if motor_library is None or not motor_library_initialized: return False
    if not hasattr(motor_library, "bgc_set_motors"): return False
    return motor_library.bgc_set_motors(ctypes.c_int(int(on_off))) == 0


# ==============================================================
# WORKER THREAD (Replaces the while True loop)
# ==============================================================
class VideoWorker(QThread):
    # Signals to send data back to the GUI thread safely
    change_pixmap_signal = pyqtSignal(np.ndarray)
    
    def __init__(self):
        super().__init__()
        self._run_flag = True
        
        # External Trigger flag for GUI button or keypress
        self.trigger_mode_cycle = False

    def run(self):
        init_sbgc()
        
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )

        face_track_cam = cv2.VideoCapture(CAM_INDEX)
        if not face_track_cam.isOpened():
            print(f'Error Unable to open camera from {CAM_INDEX}')
            return

        frame_width = int(face_track_cam.get(3))
        frame_height = int(face_track_cam.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('face_track_demo.avi', fourcc, 20.0, (frame_width, frame_height))

        # Hardware init
        try:
            i2c = busio.I2C(board.SCL, board.SDA)
            ads = ADS1115(i2c)
            ads.gain = 1
            adc_channels = {
                "A0": AnalogIn(ads, ads1x15.Pin.A0), "A1": AnalogIn(ads, ads1x15.Pin.A1),
                "A2": AnalogIn(ads, ads1x15.Pin.A2), "A3": AnalogIn(ads, ads1x15.Pin.A3),
            }
        except Exception as e:
            print(f"ADC init failed: {e}")
            adc_channels = None

        current_light_mode = LIGHT_OFF_MODE
        previous_light_mode = update_camera_profile_from_light_mode(face_track_cam, current_light_mode, None, CAMERA_PROFILE_DIR)

        face_track_cam.set(cv2.CAP_PROP_BUFFERSIZE, MAX_STORED_FRAMES)

        initial_time = time.time()
        prev_smoothed, prev_time = None, None
        consecutive_lost_frames = 0
        anchor = None

        LOCKED, SEEKING = 0, 1
        state = LOCKED

        pos_x = TimedHistogram(WINDOW_SEC)
        pos_y = TimedHistogram(WINDOW_SEC)
        vel_h = TimedHistogram(WINDOW_SEC)

        last_send_time = 0.0
        smooth_yaw_dps = smooth_pitch_dps = smooth_roll_dps = None

        TRACKING_ENABLED = 0
        COMPLIANCE_MOTORS_OFF = 1
        HOLD_MOTORS_ON_NO_TRACK = 2
        mode = TRACKING_ENABLED

        stable_range = "MID"
        pending_range = None
        pending_count = 0
        stable_scalar = STABLE_SCALAR_DEFAULT

        BUTTON_PIN = 6
        BUTTON_DEBOUNCE_SEC = 0.20
        last_button_event_time = 0.0
        prev_button_level = 1

        try:
            gpio_handle = lgpio.gpiochip_open(0)
            if hasattr(lgpio, "SET_PULL_UP"):
                lgpio.gpio_claim_input(gpio_handle, BUTTON_PIN, lgpio.SET_PULL_UP)
            else:
                lgpio.gpio_claim_input(gpio_handle, BUTTON_PIN)
        except Exception as e:
            print(f"lgpio init failed: {e}")
            gpio_handle = None

        # Main Processing Loop
        while self._run_flag:
            yaw_dps = pitch_dps = roll_dps = 0.0

            frame_read, frame = face_track_cam.read()
            if not frame_read: break

            current_time = time.time()
            
            if adc_channels:
                current_light_mode, light_mode_voltages = read_light_mode(adc_channels, LIGHT_MODE_THRESHOLD_VOLTS)
                previous_light_mode = update_camera_profile_from_light_mode(face_track_cam, current_light_mode, previous_light_mode, CAMERA_PROFILE_DIR)

            if anchor is None:
                anchor = (frame_width / 2.0, frame_height / 2.0)

            # Check Hardware GPIO button or GUI Software Trigger
            button_event = False
            if gpio_handle is not None:
                button_event, last_button_event_time, prev_button_level = button_pressed_edge(
                    current_time, gpio_handle, BUTTON_PIN, BUTTON_DEBOUNCE_SEC, last_button_event_time, prev_button_level
                )
            
            # Combine Hardware button and Software GUI button
            if button_event or self.trigger_mode_cycle:
                self.trigger_mode_cycle = False # Reset software trigger
                mode, prev_smoothed, prev_time, consecutive_lost_frames, state = handle_mode_cycle(
                    mode, prev_smoothed, prev_time, consecutive_lost_frames, state,
                    TRACKING_ENABLED, COMPLIANCE_MOTORS_OFF, HOLD_MOTORS_ON_NO_TRACK, LOCKED
                )

            # Draw overlays depending on modes
            if mode != TRACKING_ENABLED:
                if (current_time - last_send_time) >= COMMAND_PERIOD:
                    if mode == HOLD_MOTORS_ON_NO_TRACK:
                        send_speeds(0.0, 0.0, 0.0)
                    last_send_time = current_time

                msg = "COMPLIANCE (motors OFF)" if mode == COMPLIANCE_MOTORS_OFF else "LOCKED HOLD (motors ON)"
                color = (0, 0, 255) if mode == COMPLIANCE_MOTORS_OFF else (0, 200, 255)
                
                cv2.putText(frame, msg, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
                cv2.putText(frame, f"light_mode:{current_light_mode}", (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2, cv2.LINE_AA)
                
                # Emit frame to GUI and continue
                self.change_pixmap_signal.emit(frame)
                continue

            # Tracking logic
            rgb_frame_cap = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_image = face_mesh.process(rgb_frame_cap)

            centroid = eye_dist_px = None

            if processed_image.multi_face_landmarks:
                patient_face = processed_image.multi_face_landmarks[0]
                mouth_idxs = [13, 14, 61, 291]
                mouth_points = [(int(patient_face.landmark[i].x * frame_width), int(patient_face.landmark[i].y * frame_height)) for i in mouth_idxs]

                if mouth_points:
                    centroid = (sum(p[0] for p in mouth_points) / len(mouth_points), sum(p[1] for p in mouth_points) / len(mouth_points))

                eye_dist_px = estimate_eye_dist_px(patient_face, frame_width, frame_height)

            if centroid is None:
                consecutive_lost_frames += 1
                if (current_time - last_send_time) >= COMMAND_PERIOD:
                    send_speeds(0.0, 0.0, 0.0)
                    last_send_time = current_time

                if consecutive_lost_frames > MAX_LOST_FRAMES:
                    prev_smoothed = prev_time = None

                cv2.putText(frame, f"light_mode:{current_light_mode}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2, cv2.LINE_AA)
                self.change_pixmap_signal.emit(frame)
                continue

            consecutive_lost_frames = 0
            smoothed = ema_point(centroid, prev_smoothed, SMOOTH_ALPHA)
            
            if prev_time is None: prev_time = current_time
            delta_time = max(1e-6, current_time - prev_time)

            if prev_smoothed is None:
                speed = 0.0
            else:
                dvx, dvy = pixels_to_deg(smoothed[0] - prev_smoothed[0], smoothed[1] - prev_smoothed[1], frame_width, frame_height, FOV_H_DEG, FOV_V_DEG)
                speed = math.hypot(dvx, dvy) / delta_time

            prev_time, prev_smoothed = current_time, smoothed

            # Box logic
            desired_range = range_from_eye_dist(eye_dist_px)
            if desired_range != stable_range:
                if pending_range != desired_range:
                    pending_range, pending_count = desired_range, 1
                else:
                    pending_count += 1

                if pending_count >= RANGE_SWITCH_FRAMES:
                    stable_range, pending_range, pending_count = desired_range, None, 0
                    stable_scalar = scalar_for_range(stable_range)
            else:
                pending_range, pending_count = None, 0
                stable_scalar = scalar_for_range(stable_range)

            stable_box = build_stable_box(anchor, frame_width, frame_height, stable_scalar)
            in_stable_region = inside_box(smoothed, stable_box)

            dx_center = smoothed[0] - anchor[0]
            dy_center = smoothed[1] - anchor[1]
            radial_norm = math.hypot(dx_center / (frame_width / 2.0), dy_center / (frame_height / 2.0))
            within_stop_threshold = (radial_norm <= STABLE_STOP_SEEKING_THRESHOLD)

            pos_x.add(current_time, smoothed[0])
            pos_y.add(current_time, smoothed[1])
            vel_h.add(current_time, speed)

            xs, ys = pos_x.values(), pos_y.values()
            pos_std = 0.5 * (statistics.pstdev(xs) + statistics.pstdev(ys)) if len(xs) >= 6 and len(ys) >= 6 else 999.0
            speeds = vel_h.values()
            vel_med = statistics.median(speeds) if len(speeds) >= 3 else 999.0

            too_wild = (vel_med > VEL_THRESH_DEG_S * 100.0) or (pos_std > POS_STD_THRESH_PX * 100.0)

            if state == LOCKED:
                if not in_stable_region: state = SEEKING
            else:
                if within_stop_threshold: state = LOCKED

            if state == SEEKING and not too_wild:
                err_yaw_deg, err_pitch_deg = pixels_to_deg(dx_center, dy_center, frame_width, frame_height, FOV_H_DEG, FOV_V_DEG)
                err_yaw_deg *= AXIS_SIGN["yaw"]
                err_pitch_deg *= AXIS_SIGN["pitch"]

                if abs(err_yaw_deg) < DEADBAND_DEG_YAW: err_yaw_deg = 0.0
                if abs(err_pitch_deg) < DEADBAND_DEG_PITCH: err_pitch_deg = 0.0

                yaw_dps = clamp(KP_YAW_DPS_PER_DEG * err_yaw_deg, -MAX_DPS_YAW, +MAX_DPS_YAW)
                pitch_dps = clamp(KP_PITCH_DPS_PER_DEG * err_pitch_deg, -MAX_DPS_PITCH, +MAX_DPS_PITCH)
            
            if (current_time - last_send_time) >= COMMAND_PERIOD:
                smooth_yaw_dps = ema_scalar(yaw_dps, smooth_yaw_dps, CMD_SPEED_EMA_ALPHA)
                smooth_pitch_dps = ema_scalar(pitch_dps, smooth_pitch_dps, CMD_SPEED_EMA_ALPHA)
                smooth_roll_dps = ema_scalar(roll_dps, smooth_roll_dps, CMD_SPEED_EMA_ALPHA)

                if send_speeds(smooth_roll_dps, smooth_pitch_dps, smooth_yaw_dps):
                    last_send_time = current_time

            # Drawing
            out.write(frame)
            l, t_, r, b = map(int, stable_box)
            cv2.rectangle(frame, (l, t_), (r, b), (40, 220, 40), 1)
            cv2.drawMarker(frame, (int(anchor[0]), int(anchor[1])), (0, 200, 0), cv2.MARKER_CROSS, 12, 2)
            cv2.circle(frame, (int(smoothed[0]), int(smoothed[1])), 4, (0, 0, 255), -1)

            state_txt = "LOCKED" if state == LOCKED else "SEEKING"
            cv2.putText(frame, f"state:{state_txt}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40,220,40), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Radial distance = {radial_norm:.3f}", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40,220,40), 2, cv2.LINE_AA)
            eye_str = f"{eye_dist_px:.1f}" if eye_dist_px is not None else "None"
            cv2.putText(frame, f"eye_px={eye_str} box={stable_scalar:.3f} {stable_range}", (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40,220,40), 2, cv2.LINE_AA)
            cv2.putText(frame, f"light_mode:{current_light_mode}", (10, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(frame, "Press 'c' or UI button -> compliance/lock cycle", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2, cv2.LINE_AA)

            # Emit the drawn frame
            self.change_pixmap_signal.emit(frame)

        # Cleanup
        send_speeds(0.0, 0.0, 0.0)
        set_motors(0)
        face_track_cam.release()
        out.release()
        if gpio_handle is not None:
            lgpio.gpiochip_close(gpio_handle)

    def stop(self):
        self._run_flag = False
        self.wait()

    def cycle_mode(self):
        # Trigger the mode change via software
        self.trigger_mode_cycle = True

# ==============================================================
# MAIN GUI WINDOW
# ==============================================================
class GimbalApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gimbal Tracking Application")
        self.resize(800, 600)
        
        # Main Layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Video Label
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: black;")
        self.layout.addWidget(self.image_label)

        # Controls Layout
        self.controls_layout = QHBoxLayout()
        
        self.btn_cycle_mode = QPushButton("Cycle Compliance Mode ('C')", self)
        self.btn_cycle_mode.clicked.connect(self.trigger_cycle)
        
        self.btn_quit = QPushButton("Quit", self)
        self.btn_quit.clicked.connect(self.close_app)

        self.controls_layout.addWidget(self.btn_cycle_mode)
        self.controls_layout.addWidget(self.btn_quit)
        self.layout.addLayout(self.controls_layout)

        # Start the Background Worker Thread
        self.thread = VideoWorker()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)
        
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def trigger_cycle(self):
        """Send signal to worker thread to cycle hardware states"""
        self.thread.cycle_mode()

    # Native Keyboard Events
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_C:
            self.trigger_cycle()
        elif event.key() == Qt.Key_Escape:
            self.close_app()
            
    def close_app(self):
        self.thread.stop()
        self.close()

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GimbalApp()
    window.show()
    sys.exit(app.exec_())