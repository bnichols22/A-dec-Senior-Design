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
#       - Fist -> lock in place / hold position
#       - Four fingers -> resume mouth tracking
#       - Two fingers -> start photo countdown and temporarily capture with face-track camera
#       - Thumbs up -> re-enable motors
#       - Two closed fists on screen -> cut motors
#
# Motor lib usage:
#   SimpleBGC SerialAPI shim:
#     - bgc_init()
#     - bgc_control_speeds(roll_dps, pitch_dps, yaw_dps)
#     - bgc_set_motors(on_off)   (0=off, 1=on)
# ==============================================================

import os, sys, time, math, warnings, statistics, ctypes, threading, wave
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import json
# ADC hardware is removed for now, so keep the light-mode ADC imports disabled.
# import board
# import busio
# from adafruit_ads1x15 import ADS1115, AnalogIn, ads1x15

try:
    import pyaudio
except ImportError:
    pyaudio = None

try:
    from vosk import Model, KaldiRecognizer
except ImportError:
    Model = None
    KaldiRecognizer = None

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
PATIENT_PHOTO_DIR = os.path.join(BASE_DIR, "patient_photo")
os.makedirs(PATIENT_PHOTO_DIR, exist_ok=True)
POSTER_CAPTURE_DIR = os.path.join(BASE_DIR, "poster_captures")
os.makedirs(POSTER_CAPTURE_DIR, exist_ok=True)
AUDIO_RECORD_DIR = os.path.join(BASE_DIR, "audio_recordings")
os.makedirs(AUDIO_RECORD_DIR, exist_ok=True)
TRANSCRIPT_DIR = os.path.join(BASE_DIR, "transcript")
VOSK_MODEL_DIR = os.path.join(TRANSCRIPT_DIR, "vosk_model_heavy")

# --------- Vision / tracker config ----------
FACE_TRACK_CAM_INDEX = 2
CENTER_CAM_INDEX = 0
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
CENTER_PHOTO_FLUSH_FRAMES = 8
CENTER_PHOTO_FLUSH_DELAY_SEC = 0.02

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

# Tracking-anchor vertical compensation (pixels).
# The face-tracking camera sits above the true light center, so the desired
# mouth position in the camera image should be lower than image center.
# We keep this discrete by distance bucket to avoid making the controller
# continuously re-target itself.
#
# Horizontal compensation is a small fixed offset because the camera is only
# slightly left/right of the true light center compared with the stronger
# vertical offset from being mounted above it.
ANCHOR_X_OFFSET_PX = -10.0
ANCHOR_Y_OFFSET_FAR_PX = 34.0
ANCHOR_Y_OFFSET_MID_PX = 54.0
ANCHOR_Y_OFFSET_NEAR_PX = 74.0

# Prevent flicker: require N consecutive frames to accept a new range
RANGE_SWITCH_FRAMES = 8

# ==============================================================
# NEW: Light mode ADC config
# ==============================================================
ADC_LIGHT_MODE_ENABLED = False
PROFILE_SWITCHING_ENABLED = False
LIGHT_MODE_THRESHOLD_VOLTS = 2.0

LIGHT_OFF_MODE     = "LIGHT_OFF"
YELLOW_LIGHT_MODE  = "YELLOW_LIGHT"
LOWEST_LIGHT_MODE  = "LOWEST_LIGHT"
MEDIUM_LIGHT_MODE  = "MEDIUM_LIGHT"
HIGHEST_LIGHT_MODE = "HIGHEST_LIGHT"
DEFAULT_CENTER_LIGHT_MODE = HIGHEST_LIGHT_MODE

CENTER_CAM_LIGHT_MODE_TO_PROFILE = {
    LIGHT_OFF_MODE: "zoom_loff.json",
    YELLOW_LIGHT_MODE: "zoom_lon.json",
    LOWEST_LIGHT_MODE: "zoom_lon.json",
    MEDIUM_LIGHT_MODE: "zoom_lon.json",
    HIGHEST_LIGHT_MODE: "zoom_lon.json",
}

GESTURE_TRACK_MOUTH = 0
GESTURE_TRACK_PINCH = 1
GESTURE_LOCKED = 2

HAND_DETECTION_CONFIDENCE = 0.75
HAND_TRACKING_CONFIDENCE = 0.75
PINCH_ENTER_FRAMES = 5
FOUR_ENTER_FRAMES = 3
TWO_ENTER_FRAMES = 3
PINCH_DISTANCE_RATIO = 0.28
FIST_ENTER_FRAMES = 5
PHOTO_COUNTDOWN_SEC = 2.0
THUMB_TUCK_RATIO = 1.15
THUMB_OPEN_RATIO = 1.35
FOUR_THUMB_MAX_RATIO = 1.10
FOUR_THUMB_CLOSED_RATIO = 0.60
FIST_CURLED_RATIO = 0.85
THUMBS_UP_ENTER_FRAMES = 4
TWO_FISTS_ENTER_FRAMES = 4
THUMBS_UP_HEIGHT_RATIO = 0.25
THUMBS_UP_CLEARANCE_RATIO = 0.10
MOTOR_GESTURE_COOLDOWN_SEC = 1.0
THREE_ENTER_FRAMES = 4
MIN_AUDIO_RECORDING_DURATION_SEC = 5.0
MIN_AUDIO_RESTART_DELAY_SEC = 3.0
QR_WB_STEP = 100.0
QR_WB_TOLERANCE = 2.0
QR_WB_DEFAULT_TEMP = 4500.0
QR_WB_MIN_TEMP = 2000.0
QR_WB_MAX_TEMP = 10000.0
QR_WB_APPLY_INTERVAL_SEC = 0.75
RECORDING_AUDIO_RATE = 44100
RECORDING_AUDIO_CHANNELS = 1

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

def get_color_imbalance(roi):
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    white_pixel_count = np.count_nonzero(mask == 255)
    if white_pixel_count == 0:
        raise ValueError("QR ROI did not contain detectable white pixels")

    mean_b = np.mean(roi[:, :, 0][mask == 255])
    mean_r = np.mean(roi[:, :, 2][mask == 255])
    return mean_b, mean_r

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

def anchor_y_offset_for_range(rng):
    if rng == "FAR":
        return ANCHOR_Y_OFFSET_FAR_PX
    if rng == "NEAR":
        return ANCHOR_Y_OFFSET_NEAR_PX
    return ANCHOR_Y_OFFSET_MID_PX

def build_tracking_anchor(frame_width, frame_height, rng):
    return (
        (frame_width / 2.0) + ANCHOR_X_OFFSET_PX,
        (frame_height / 2.0) + anchor_y_offset_for_range(rng),
    )

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
            if "auto_exposure" not in camera_settings:
                camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            camera.set(cv2.CAP_PROP_EXPOSURE, camera_settings["exposure"])
        if "brightness" in camera_settings:
            camera.set(cv2.CAP_PROP_BRIGHTNESS, camera_settings["brightness"])
        if "contrast" in camera_settings:
            camera.set(cv2.CAP_PROP_CONTRAST, camera_settings["contrast"])
        if "gain" in camera_settings:
            camera.set(cv2.CAP_PROP_GAIN, camera_settings["gain"])
        if "saturation" in camera_settings:
            camera.set(cv2.CAP_PROP_SATURATION, camera_settings["saturation"])
        if "focus" in camera_settings:
            camera.set(cv2.CAP_PROP_FOCUS, camera_settings["focus"])
        if "auto_white_balance" in camera_settings:
            camera.set(cv2.CAP_PROP_AUTO_WB, camera_settings["auto_white_balance"])
        if "white_balance" in camera_settings:
            camera.set(cv2.CAP_PROP_WB_TEMPERATURE, camera_settings["white_balance"])
        if "white_balance_temperature" in camera_settings:
            camera.set(cv2.CAP_PROP_WB_TEMPERATURE, camera_settings["white_balance_temperature"])

        print(f"Loaded camera profile: {profile_path}")
        return True
    except Exception as camera_update_error:
        print(f"Error loading profile: {camera_update_error}")
        return False

def read_light_mode(adc_channels, off_threshold_volts):
    try:
        v0 = adc_channels["A0"].voltage
        v1 = adc_channels["A1"].voltage
        v2 = adc_channels["A2"].voltage
        v3 = adc_channels["A3"].voltage
    except Exception as adc_read_error:
        print(f"ADC read unavailable, disabling light-mode updates: {adc_read_error}")
        return None, None

    voltage_map = {
        YELLOW_LIGHT_MODE:  v0,
        LOWEST_LIGHT_MODE:  v1,
        MEDIUM_LIGHT_MODE:  v2,
        HIGHEST_LIGHT_MODE: v3,
    }

    active_modes = {mode_name: voltage for mode_name, voltage in voltage_map.items() if voltage < off_threshold_volts}

    if not active_modes:
        return LIGHT_OFF_MODE, (v0, v1, v2, v3)

    selected_mode = min(active_modes, key=active_modes.get)
    return selected_mode, (v0, v1, v2, v3)

def update_camera_profile_from_light_mode(camera, current_light_mode, previous_light_mode, profile_dir, profile_map):
    if current_light_mode == previous_light_mode:
        return previous_light_mode

    profile_filename = profile_map.get(current_light_mode)
    if profile_filename is None:
        print(f"Warning: No profile mapped for light mode {current_light_mode}")
        return previous_light_mode

    profile_loaded = update_camera_settings(camera, profile_filename, profile_dir)
    if profile_loaded:
        print(f"Light mode changed: {previous_light_mode} -> {current_light_mode}")
        return current_light_mode

    return previous_light_mode

def init_camera_white_balance(camera):
    try:
        camera.set(cv2.CAP_PROP_AUTO_WB, 0.0)
    except Exception:
        pass

    current_temp = camera.get(cv2.CAP_PROP_WB_TEMPERATURE)
    if current_temp is None or current_temp <= 0 or current_temp == -1:
        current_temp = QR_WB_DEFAULT_TEMP

    try:
        camera.set(cv2.CAP_PROP_WB_TEMPERATURE, current_temp)
    except Exception:
        pass

    return float(current_temp)

def apply_qr_white_balance(camera, frame, qr_detector, current_temp):
    qr_found, _decoded_info, points, _ = qr_detector.detectAndDecodeMulti(frame)
    if not qr_found or points is None or len(points) == 0:
        return current_temp, False, "QR WB idle"

    pts = points[0].astype(int)
    x, y, w, h = cv2.boundingRect(pts)
    frame_height, frame_width = frame.shape[:2]
    x = max(0, x)
    y = max(0, y)
    w = min(w, frame_width - x)
    h = min(h, frame_height - y)
    if w <= 0 or h <= 0:
        return current_temp, True, "QR WB skipped: invalid ROI"

    roi = frame[y:y + h, x:x + w]

    try:
        mean_b, mean_r = get_color_imbalance(roi)
    except Exception as qr_error:
        return current_temp, True, f"QR WB failed: {qr_error}"

    diff = mean_r - mean_b
    if abs(diff) <= QR_WB_TOLERANCE:
        return current_temp, True, f"QR WB locked at {current_temp:.0f}K"

    if mean_r > mean_b:
        current_temp -= QR_WB_STEP
    else:
        current_temp += QR_WB_STEP

    current_temp = clamp(current_temp, QR_WB_MIN_TEMP, QR_WB_MAX_TEMP)
    camera.set(cv2.CAP_PROP_WB_TEMPERATURE, current_temp)
    return current_temp, True, f"QR WB tuning {current_temp:.0f}K"

def init_adc_channels():
    if not ADC_LIGHT_MODE_ENABLED:
        print("ADC light-mode input disabled; center camera will stay on the default light-on profile.")
        return None

    try:
        i2c = busio.I2C(board.SCL, board.SDA)
        ads = ADS1115(i2c)
        ads.gain = 1
        return {
            "A0": AnalogIn(ads, ads1x15.Pin.A0),
            "A1": AnalogIn(ads, ads1x15.Pin.A1),
            "A2": AnalogIn(ads, ads1x15.Pin.A2),
            "A3": AnalogIn(ads, ads1x15.Pin.A3),
        }
    except Exception as adc_error:
        print(f"ADC init unavailable, continuing without light-mode updates: {adc_error}")
        return None

def landmark_to_pixel(landmark, frame_width, frame_height):
    return (landmark.x * frame_width, landmark.y * frame_height)

def finger_extended(tip_point, pip_point):
    return tip_point[1] < pip_point[1]

class AudioRecorder:

    def __init__(self, audio_filename):
        self.audio_filename = audio_filename
        self.open = False
        self.rate = RECORDING_AUDIO_RATE
        self.frames_per_buffer = 1024
        self.channels = RECORDING_AUDIO_CHANNELS
        self.format = pyaudio.paInt16 if pyaudio is not None else None
        self.audio = None
        self.stream = None
        self.audio_frames = []
        self.audio_thread = None

    def start(self):
        if pyaudio is None:
            raise RuntimeError("PyAudio is not installed; audio recording is unavailable.")

        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.frames_per_buffer,
        )
        self.open = True
        self.audio_thread = threading.Thread(target=self.record, daemon=True)
        self.audio_thread.start()

    def record(self):
        self.stream.start_stream()
        while self.open:
            data = self.stream.read(self.frames_per_buffer, exception_on_overflow=False)
            self.audio_frames.append(data)

    def stop(self):
        if not self.open:
            return

        self.open = False
        if self.audio_thread is not None:
            self.audio_thread.join(timeout=2.0)

        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
        if self.audio is not None:
            sample_width = self.audio.get_sample_size(self.format)
            self.audio.terminate()
        else:
            sample_width = 2

        wave_file = wave.open(self.audio_filename, "wb")
        wave_file.setnchannels(self.channels)
        wave_file.setsampwidth(sample_width)
        wave_file.setframerate(self.rate)
        wave_file.writeframes(b"".join(self.audio_frames))
        wave_file.close()

class AudioSessionRecorder:

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.active = False
        self.audio_recorder = None
        self.base_name = None
        self.audio_path = None
        self.transcript_path = None
        self.status_message = "idle"

    def start(self):
        if self.active:
            return False, "recording already active"

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.base_name = f"audio_recording_{timestamp}"
        self.audio_path = os.path.join(self.output_dir, f"{self.base_name}.wav")
        self.transcript_path = os.path.join(self.output_dir, f"{self.base_name}_transcript.txt")

        try:
            self.audio_recorder = AudioRecorder(self.audio_path)
            self.audio_recorder.start()
        except Exception as audio_error:
            self.audio_recorder = None
            self.status_message = f"audio start failed: {audio_error}"
            return False, self.status_message

        self.active = True
        self.status_message = f"recording {self.base_name}"
        print(f"Started audio recording: {self.base_name}")
        return True, self.status_message

    def stop(self):
        if not self.active:
            return False, "recording not active"

        self.active = False

        if self.audio_recorder is not None:
            self.audio_recorder.stop()
            self.audio_recorder = None

        transcript_ok, transcript_message = self._generate_transcript()

        message_parts = [f"saved audio {self.audio_path}"]
        if transcript_message:
            message_parts.append(transcript_message)
        self.status_message = " | ".join(part for part in message_parts if part)
        print(f"Stopped audio recording: {self.status_message}")
        return transcript_ok, self.status_message

    def _generate_transcript(self):
        if Model is None or KaldiRecognizer is None:
            return False, "transcript skipped: vosk is not installed in the active Python environment"

        if not os.path.isdir(VOSK_MODEL_DIR):
            return False, f"transcript skipped: Vosk model folder not found at {VOSK_MODEL_DIR}"

        try:
            model = Model(VOSK_MODEL_DIR)
            with wave.open(self.audio_path, "rb") as wav_file:
                if (
                    wav_file.getnchannels() != 1 or
                    wav_file.getsampwidth() != 2 or
                    wav_file.getcomptype() != "NONE"
                ):
                    return False, "transcript skipped: audio file must be mono PCM wav for Vosk"

                recognizer = KaldiRecognizer(model, wav_file.getframerate())
                recognizer.SetWords(True)

                transcript_parts = []
                while True:
                    data = wav_file.readframes(4000)
                    if len(data) == 0:
                        break
                    if recognizer.AcceptWaveform(data):
                        partial_result = json.loads(recognizer.Result())
                        transcript_parts.append(partial_result.get("text", ""))

                final_result = json.loads(recognizer.FinalResult())
                transcript_parts.append(final_result.get("text", ""))

            transcript_text = " ".join(filter(None, transcript_parts)).strip()
            with open(self.transcript_path, "w", encoding="utf-8") as transcript_file:
                transcript_file.write(transcript_text)
        except Exception as transcript_error:
            return False, f"transcript failed: {transcript_error}"

        return True, f"saved transcript {self.transcript_path}"

def detect_hand_gestures(hand_results, frame_width, frame_height):
    pinch_detected = False
    pinch_start_point = None
    index_tip_point = None
    two_detected = False
    four_detected = False
    three_detected = False
    three_and_fist_detected = False
    best_pinch_ratio = 999.0
    fist_count = 0
    three_count = 0
    thumbs_up_detected = False
    single_fist_detected = False
    two_fists_detected = False

    if not hand_results.multi_hand_landmarks:
        return pinch_detected, pinch_start_point, index_tip_point, two_detected, four_detected, three_detected, three_and_fist_detected, single_fist_detected, thumbs_up_detected, two_fists_detected

    for hand_landmarks in hand_results.multi_hand_landmarks:
        lm = hand_landmarks.landmark

        wrist = landmark_to_pixel(lm[0], frame_width, frame_height)

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
        thumb_to_pinky_mcp = math.hypot(thumb_tip[0] - pinky_mcp[0], thumb_tip[1] - pinky_mcp[1])

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
            thumb_to_index_mcp < (THUMB_TUCK_RATIO * palm_width) or
            thumb_tip[1] > thumb_ip[1]
        )
        thumb_not_wide_open = thumb_to_index_mcp < (THUMB_OPEN_RATIO * palm_width)
        thumb_neutral_for_four = thumb_to_index_mcp < (FOUR_THUMB_MAX_RATIO * palm_width)
        thumb_closed_for_four = (
            thumb_neutral_for_four and
            thumb_to_pinky_mcp < (FOUR_THUMB_CLOSED_RATIO * palm_width)
        )
        index_curled = math.hypot(index_tip[0] - index_mcp[0], index_tip[1] - index_mcp[1]) < (FIST_CURLED_RATIO * palm_width)
        middle_curled = math.hypot(middle_tip[0] - index_mcp[0], middle_tip[1] - index_mcp[1]) < (FIST_CURLED_RATIO * palm_width)
        ring_curled = math.hypot(ring_tip[0] - pinky_mcp[0], ring_tip[1] - pinky_mcp[1]) < (FIST_CURLED_RATIO * palm_width)
        pinky_curled = math.hypot(pinky_tip[0] - pinky_mcp[0], pinky_tip[1] - pinky_mcp[1]) < (FIST_CURLED_RATIO * palm_width)

        # Exactly four fingers resumes mouth tracking; an open five-finger hand should not trigger it.
        hand_four = (
            index_extended and
            middle_extended and
            ring_extended and
            pinky_extended and
            thumb_closed_for_four
        )
        if hand_four:
            four_detected = True

        # Two fingers extended starts the patient photo countdown.
        # Allow a relaxed thumb so the gesture is easier to hold.
        hand_two = (
            index_extended and
            middle_extended and
            not ring_extended and
            not pinky_extended and
            thumb_not_wide_open and
            not pinch_detected
        )
        if hand_two:
            two_detected = True

        hand_three = (
            index_extended and
            middle_extended and
            ring_extended and
            not pinky_extended and
            thumb_not_wide_open and
            not pinch_detected
        )
        if hand_three:
            three_count += 1

        hand_fist = (
            not index_extended and
            not middle_extended and
            not ring_extended and
            not pinky_extended and
            thumb_tucked and
            index_curled and
            middle_curled and
            ring_curled and
            pinky_curled
        )
        if hand_fist:
            fist_count += 1

        thumb_extended_up = (
            thumb_tip[1] < thumb_mcp[1] and
            thumb_tip[1] < wrist[1]
        )
        thumb_above_index_mcp = thumb_tip[1] < (index_mcp[1] - (THUMBS_UP_CLEARANCE_RATIO * palm_width))
        thumb_height = abs(thumb_tip[1] - thumb_mcp[1])
        hand_thumbs_up = (
            thumb_extended_up and
            thumb_above_index_mcp and
            thumb_height > (THUMBS_UP_HEIGHT_RATIO * palm_width) and
            not index_extended and
            not middle_extended and
            not ring_extended and
            not pinky_extended and
            not hand_fist and
            not pinch_detected
        )
        if hand_thumbs_up:
            thumbs_up_detected = True

    three_detected = (three_count >= 1)
    three_and_fist_detected = (three_count >= 1 and fist_count >= 1)
    single_fist_detected = (fist_count == 1)
    two_fists_detected = (fist_count >= 2)

    return pinch_detected, pinch_start_point, index_tip_point, two_detected, four_detected, three_detected, three_and_fist_detected, single_fist_detected, thumbs_up_detected, two_fists_detected


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
    single_fist_detected,
    pinch_counter,
    four_counter,
    fist_counter,
    pinch_point,
    seeking_state,
    locked_state
):
    reset_state = None

    if single_fist_detected:
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

def save_poster_capture(frame):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    milliseconds = int((time.time() % 1.0) * 1000)
    capture_path = os.path.join(POSTER_CAPTURE_DIR, f"poster_capture_{timestamp}_{milliseconds:03d}.jpg")
    if cv2.imwrite(capture_path, frame):
        print(f"Saved poster capture: {capture_path}")
        return True

    print(f"Failed to save poster capture: {capture_path}")
    return False

def show_runtime_frame(window_name, frame, overlay_lines, poster_frame=None):
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
    key = cv2.waitKey(1) & 0xFF
    if key == ord("c"):
        save_poster_capture(poster_frame if poster_frame is not None else frame)
    return key == 27

def status_overlay(recording_active):
    return [
        (f"recording:{'ON' if recording_active else 'OFF'}", (10, 24), (255, 255, 255), 0.55),
    ]

def capture_patient_photo(center_cam):
    if center_cam is None:
        print("capture_patient_photo: center_cam unavailable")
        return False, None

    # Drain buffered center-camera frames so two-finger capture saves the current view.
    frame_read = False
    photo_frame = None
    for _ in range(CENTER_PHOTO_FLUSH_FRAMES):
        center_cam.grab()
        time.sleep(CENTER_PHOTO_FLUSH_DELAY_SEC)

    frame_read, photo_frame = center_cam.read()
    if not frame_read:
        print("capture_patient_photo: failed to read frame from center_cam")
        return False, None

    photo_frame = cv2.flip(photo_frame, -1)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    milliseconds = int((time.time() % 1.0) * 1000)
    photo_path = os.path.join(PATIENT_PHOTO_DIR, f"patient_photo_{timestamp}_{milliseconds:03d}.jpg")
    photo_written = cv2.imwrite(photo_path, photo_frame)
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
    qr_detector = cv2.QRCodeDetector()

    face_track_cam = cv2.VideoCapture(FACE_TRACK_CAM_INDEX)
    if not face_track_cam.isOpened():
        print(f'main: Error Unable to open camera from {FACE_TRACK_CAM_INDEX}')
        sys.exit(1)

    center_cam = cv2.VideoCapture(CENTER_CAM_INDEX)
    if not center_cam.isOpened():
        print(f'main: Error Unable to open center camera from {CENTER_CAM_INDEX}')
        sys.exit(1)

    adc_channels = init_adc_channels()

    current_light_mode = DEFAULT_CENTER_LIGHT_MODE
    previous_center_light_mode = None
    previous_center_light_mode = update_camera_profile_from_light_mode(
        center_cam,
        current_light_mode,
        previous_center_light_mode,
        CAMERA_PROFILE_DIR,
        CENTER_CAM_LIGHT_MODE_TO_PROFILE
    )

    # Set the max number of stored frames allowed
    face_track_cam.set(cv2.CAP_PROP_BUFFERSIZE, MAX_STORED_FRAMES)
    center_cam.set(cv2.CAP_PROP_BUFFERSIZE, MAX_STORED_FRAMES)
    current_wb_temp = init_camera_white_balance(face_track_cam)
    last_qr_wb_apply_time = -999.0
    qr_wb_status = "QR WB idle"

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
    motors_enabled = bool(set_motors(1))
    last_motors_off_time = -999.0

    # Start gesture control in the locked state, equivalent to beginning with a fist.
    gesture_mode = GESTURE_LOCKED
    pinch_counter = 0
    two_counter = 0
    four_counter = 0
    fist_counter = 0
    thumbs_up_counter = 0
    two_fists_counter = 0
    three_counter = 0
    three_trigger_armed = True
    pinch_point = None
    av_recorder = AudioSessionRecorder(AUDIO_RECORD_DIR)
    recording_started_at = None
    last_audio_state_change_time = -999.0
    # Photo capture is handled as a temporary overlay state so the main loop keeps running.
    photo_countdown_active = False
    photo_countdown_start = None
    photo_return_mode = GESTURE_LOCKED
    photo_trigger_armed = True

    # ======= Main Loop =======
    try:
        while True:
            yaw_dps = 0.0
            pitch_dps = 0.0
            roll_dps = 0.0

            frame_read, frame = face_track_cam.read()
            if not frame_read:
                # Note: log_file is only set if file open succeeded above
                try:
                    log_file.write("main: frame grab failed\n")
                except Exception:
                    pass
                break

            current_time = time.time()
            frame_height, frame_width = frame.shape[:2]
            # ADC/profile switching is disabled for now; keep center camera on light-on profile.
            if PROFILE_SWITCHING_ENABLED and adc_channels is not None:
                read_mode, read_voltages = read_light_mode(adc_channels, LIGHT_MODE_THRESHOLD_VOLTS)
                if read_mode is None:
                    adc_channels = None
                    current_light_mode = DEFAULT_CENTER_LIGHT_MODE
                    light_mode_voltages = (0.0, 0.0, 0.0, 0.0)
                else:
                    current_light_mode = read_mode
                    light_mode_voltages = read_voltages
                    previous_center_light_mode = update_camera_profile_from_light_mode(
                        center_cam,
                        current_light_mode,
                        previous_center_light_mode,
                        CAMERA_PROFILE_DIR,
                        CENTER_CAM_LIGHT_MODE_TO_PROFILE
                    )
            else:
                current_light_mode = DEFAULT_CENTER_LIGHT_MODE
                light_mode_voltages = (0.0, 0.0, 0.0, 0.0)

            # Normal tracking mode
            # get the capture from camera
            rgb_frame_cap = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # process the image and set landmarks
            processed_image = face_mesh.process(rgb_frame_cap)
            hand_results = hands.process(rgb_frame_cap)

            qr_wb_status = "QR WB idle"
            if gesture_mode == GESTURE_LOCKED and (current_time - last_qr_wb_apply_time) >= QR_WB_APPLY_INTERVAL_SEC:
                current_wb_temp, qr_detected_now, qr_wb_status = apply_qr_white_balance(
                    face_track_cam,
                    frame,
                    qr_detector,
                    current_wb_temp
                )
                if qr_detected_now:
                    last_qr_wb_apply_time = current_time

            pinch_detected, _pinch_start_point, index_tip_point, two_detected, four_detected, three_detected, three_and_fist_detected, single_fist_detected, thumbs_up_detected, two_fists_detected = detect_hand_gestures(
                hand_results,
                frame_width,
                frame_height
            )

            if three_detected:
                three_counter += 1
                if three_counter >= THREE_ENTER_FRAMES and three_trigger_armed:
                    if not av_recorder.active:
                        can_restart = (current_time - last_audio_state_change_time) >= MIN_AUDIO_RESTART_DELAY_SEC
                        if not can_restart:
                            three_counter = 0
                            three_trigger_armed = False
                            continue

                        started_ok, _started_message = av_recorder.start()
                        if started_ok:
                            recording_started_at = current_time
                            last_audio_state_change_time = current_time
                            three_trigger_armed = False
                    elif (
                        recording_started_at is not None and
                        (current_time - recording_started_at) >= MIN_AUDIO_RECORDING_DURATION_SEC and
                        (current_time - last_audio_state_change_time) >= MIN_AUDIO_RECORDING_DURATION_SEC
                    ):
                        av_recorder.stop()
                        recording_started_at = None
                        last_audio_state_change_time = current_time
                        three_trigger_armed = False
                    three_counter = 0
            else:
                three_counter = 0
                three_trigger_armed = True

            if two_fists_detected:
                two_fists_counter += 1
                if two_fists_counter >= TWO_FISTS_ENTER_FRAMES and motors_enabled:
                    send_speeds(0.0, 0.0, 0.0)
                    motors_enabled = not bool(set_motors(0))
                    last_motors_off_time = current_time
                    two_fists_counter = 0
                    gesture_mode = GESTURE_LOCKED
                    prev_smoothed, prev_time, consecutive_lost_frames, state = reset_tracking_state(LOCKED)
            else:
                two_fists_counter = 0

            if thumbs_up_detected:
                thumbs_up_counter += 1
                if (
                    thumbs_up_counter >= THUMBS_UP_ENTER_FRAMES and
                    not motors_enabled and
                    (current_time - last_motors_off_time) >= MOTOR_GESTURE_COOLDOWN_SEC
                ):
                    if set_motors(1):
                        motors_enabled = True
                        thumbs_up_counter = 0
                        prev_smoothed, prev_time, consecutive_lost_frames, state = reset_tracking_state(LOCKED)
            else:
                thumbs_up_counter = 0

            if not two_detected:
                two_counter = 0
                photo_trigger_armed = True

            if photo_countdown_active:
                # Hold the head still while counting down; this avoids blocking the rest of the loop.
                last_send_time = send_zero_if_due(current_time, last_send_time, motors_enabled)

                elapsed_photo_time = current_time - photo_countdown_start
                remaining_photo_time = max(0.0, PHOTO_COUNTDOWN_SEC - elapsed_photo_time)

                if elapsed_photo_time >= PHOTO_COUNTDOWN_SEC:
                    photo_ok, photo_path = capture_patient_photo(center_cam)
                    photo_countdown_active = False
                    gesture_mode = photo_return_mode
                    prev_smoothed, prev_time, consecutive_lost_frames, state = reset_tracking_state(LOCKED)
                draw_hand_landmarks(frame, hand_results, mp_drawing, mp_drawing_styles, hand_connections)
                poster_frame = frame.copy()
                if show_runtime_frame(
                    window_name,
                    frame,
                    status_overlay(av_recorder.active),
                    poster_frame,
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
                single_fist_detected,
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
                last_send_time = send_zero_if_due(current_time, last_send_time, motors_enabled)

                draw_hand_landmarks(frame, hand_results, mp_drawing, mp_drawing_styles, hand_connections)
                poster_frame = frame.copy()
                if show_runtime_frame(
                    window_name,
                    frame,
                    status_overlay(av_recorder.active),
                    poster_frame,
                ):
                    break
                continue

            # Handle if no centroid was found
            if centroid is None:
                consecutive_lost_frames += 1
                # if we lose tracking hold
                last_send_time = send_zero_if_due(current_time, last_send_time, motors_enabled)

                # Check how many consecutive_lost_frames frames we have
                if consecutive_lost_frames > MAX_LOST_FRAMES:
                    prev_smoothed = None
                    prev_time = None

                draw_hand_landmarks(frame, hand_results, mp_drawing, mp_drawing_styles, hand_connections)
                poster_frame = frame.copy()
                if show_runtime_frame(
                    window_name,
                    frame,
                    status_overlay(av_recorder.active),
                    poster_frame,
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

            anchor = build_tracking_anchor(frame_width, frame_height, stable_range)
            anchor_x_offset_px = ANCHOR_X_OFFSET_PX
            anchor_y_offset_px = anchor_y_offset_for_range(stable_range)

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

                if motors_enabled:
                    ok_send = send_speeds(smooth_roll_dps, smooth_pitch_dps, smooth_yaw_dps)
                    if ok_send:
                        last_send_time = current_time
                        sent = 1
                else:
                    last_send_time = current_time

            # Only print telemetry if desired
            if PRINT_TELEMETRY:
                eye_str = f"{eye_dist_px:.1f}" if eye_dist_px is not None else "None"
                print(f"{current_time - initial_time:.3f} {yaw_dps:+.2f} {pitch_dps:+.2f} {sent} {state} r={radial_norm:.3f} eye={eye_str} box={current_stable_scalar:.3f} {stable_range} anchor_xoff={anchor_x_offset_px:.1f} anchor_yoff={anchor_y_offset_px:.1f} motors={'ON' if motors_enabled else 'OFF'} light_mode={current_light_mode} A0={light_mode_voltages[0]:.3f} A1={light_mode_voltages[1]:.3f} A2={light_mode_voltages[2]:.3f} A3={light_mode_voltages[3]:.3f}")

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

                poster_frame = frame.copy()

                if show_runtime_frame(
                    window_name,
                    frame,
                    status_overlay(av_recorder.active),
                    poster_frame,
                ):
                    break

    finally:
        # stop motion on exit
        try:
            # Send a hold command to the motors
            send_speeds(0.0, 0.0, 0.0)
            # Stop motors on end of program
            set_motors(0)
        except Exception:
            pass

        try:
            if av_recorder.active:
                av_recorder.stop()
        except Exception as recorder_error:
            print(f"Recorder cleanup error: {recorder_error}")

        hands.close()
        face_mesh.close()
        face_track_cam.release()
        center_cam.release()
        cv2.destroyAllWindows()

        print("Stopped.")


if __name__ == "__main__":
    try:
        main()
    except Exception as main_error:
        print(f"Fatal error: {main_error}")
        raise
