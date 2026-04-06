#!/usr/bin/env python3
"""
==============================================================
UNIFIED GIMBAL CONTROLLER
==============================================================
Combines multiple tools into a single, integrated application:6
  1. Core gimbal movement (face tracking)
  2. White balance calibration (QR code)
  3. Hand gesture recognition
  4. Audio recording (transcription support)

Author: Senior Design Team
==============================================================
"""

import os
import sys
import json
import time
import math
import warnings
import statistics
import ctypes
from collections import deque
from enum import Enum
from typing import Optional, Tuple, List, Dict

import cv2
import mediapipe as mp
import numpy as np

try:
    import lgpio
except ImportError:
    lgpio = None

try:
    import pyaudio
    import wave
except ImportError:
    pyaudio = None
    wave = None


# ==============================================================
# CONFIGURATION & ENUMS
# ==============================================================

class GimbalMode(Enum):
    """Motor control modes"""
    TRACKING_ENABLED = 0
    COMPLIANCE_MOTORS_OFF = 1
    HOLD_MOTORS_ON_NO_TRACK = 2


class TrackingState(Enum):
    """Face tracking states"""
    LOCKED = 0
    SEEKING = 1


class RangeType(Enum):
    """Distance range categories"""
    FAR = "FAR"
    MID = "MID"
    NEAR = "NEAR"


class Config:
    """Application configuration"""
    
    # Paths
    BASE_DIR = os.path.expanduser('~/senior_design/A-dec-Senior-Design/camera/testing')
    LOG_PATH = os.path.join(BASE_DIR, 'face_track_log.txt')
    TEST_LOG = os.path.join(BASE_DIR, 'Board_Serial_Command_Test.txt')
    AUDIO_OUTPUT = os.path.join(BASE_DIR, 'transcript', 'my_recording.wav')
    
    LIB_PATH = os.path.expanduser(
        "~/senior_design/A-dec-Senior-Design/camera/testing/SerialLibrary/serialAPI/libsimplebgc.so"
    )
    
    # Camera settings
    CAM_INDEX = 0
    FOV_H_DEG = 65.0
    FOV_V_DEG = 48.75
    MAX_STORED_FRAMES = 1
    
    # Stable box thresholds
    STABLE_SCALAR_DEFAULT = 0.06
    EYE_L_IDX = 33
    EYE_R_IDX = 263
    EYE_DIST_FAR_MAX = 45.0
    EYE_DIST_NEAR_MIN = 85.0
    STABLE_SCALAR_FAR = 0.035
    STABLE_SCALAR_MID = 0.060
    STABLE_SCALAR_NEAR = 0.095
    RANGE_SWITCH_FRAMES = 8
    
    # Tracking parameters
    WINDOW_SEC = 0.6
    SMOOTH_ALPHA = 0.25
    MAX_LOST_FRAMES = 10
    STABLE_STOP_SEEKING_THRESHOLD = 0.025
    VEL_THRESH_DEG_S = 2.5
    POS_STD_THRESH_PX = 2.5
    
    # Speed control
    KP_YAW_DPS_PER_DEG = 1.25
    KP_PITCH_DPS_PER_DEG = 1.25
    MAX_DPS_YAW = 80.0
    MAX_DPS_PITCH = 80.0
    MAX_DPS_ROLL = 60.0
    DEADBAND_DEG_YAW = 0.25
    DEADBAND_DEG_PITCH = 0.25
    COMMAND_HZ = 150.0
    CMD_SPEED_EMA_ALPHA = 0.35
    
    # Audio recording
    AUDIO_FORMAT = 'paInt16'
    AUDIO_CHANNELS = 1
    AUDIO_RATE = 16000
    AUDIO_CHUNK = 1024
    
    # Button control
    BUTTON_PIN = 6
    BUTTON_DEBOUNCE_SEC = 0.20
    
    # UI
    DRAW_FRAME_RT = True
    PRINT_TELEMETRY = False
    
    @staticmethod
    def setup_directories():
        """Create required directories"""
        os.makedirs(Config.BASE_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(Config.AUDIO_OUTPUT), exist_ok=True)


# ==============================================================
# UTILITY FUNCTIONS
# ==============================================================

def ema_point(curr: Tuple[float, float], prev: Optional[Tuple[float, float]], 
              alpha: float) -> Tuple[float, float]:
    """Exponential Moving Average for points"""
    if prev is None:
        return curr
    return (alpha * curr[0] + (1 - alpha) * prev[0],
            alpha * curr[1] + (1 - alpha) * prev[1])


def ema_scalar(current: float, previous: Optional[float], alpha: float) -> float:
    """Exponential Moving Average for scalars"""
    if previous is None:
        return current
    return alpha * current + (1 - alpha) * previous


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between bounds"""
    return max(min_val, min(max_val, value))


def pixels_to_deg(px_x: float, px_y: float, frame_w: int, frame_h: int, 
                  fov_h: float, fov_v: float) -> Tuple[float, float]:
    """Convert pixel displacement to degrees"""
    half_w, half_h = frame_w / 2.0, frame_h / 2.0
    yaw_deg = (px_x / half_w) * (fov_h / 2.0)
    pitch_deg = (px_y / half_h) * (fov_v / 2.0)
    return yaw_deg, pitch_deg


def build_stable_box(center: Tuple[float, float], frame_w: int, frame_h: int, 
                     scalar: float) -> Tuple[float, float, float, float]:
    """Create stable box around center"""
    cx, cy = center
    half_w = scalar * (frame_w / 2.0)
    half_h = scalar * (frame_h / 2.0)
    return (cx - half_w, cy - half_h, cx + half_w, cy + half_h)


def inside_box(pt: Tuple[float, float], box: Tuple[float, float, float, float]) -> bool:
    """Check if point is inside box"""
    x, y = pt
    l, t, r, b = box
    return (l <= x <= r) and (t <= y <= b)


# ==============================================================
# TIMED HISTOGRAM (for jitter detection)
# ==============================================================

class TimedHistogram:
    """Time-windowed histogram for statistics"""
    
    def __init__(self, window_sec: float):
        self.window_sec = window_sec
        self.buffer = deque()
    
    def add(self, timestamp: float, value: float):
        """Add value with timestamp"""
        self.buffer.append((timestamp, value))
        self._trim(timestamp)
    
    def values(self) -> List[float]:
        """Get all values in window"""
        return [v for _, v in self.buffer]
    
    def clear(self):
        """Clear buffer"""
        self.buffer.clear()
    
    def _trim(self, current_time: float):
        """Remove expired entries"""
        cutoff = current_time - self.window_sec
        while self.buffer and self.buffer[0][0] < cutoff:
            self.buffer.popleft()


# ==============================================================
# GIMBAL MOTOR CONTROL (SimpleBGC)
# ==============================================================

class GimbalMotor:
    """SimpleBGC motor library interface"""
    
    def __init__(self, lib_path: str):
        self.lib_path = lib_path
        self.motor_lib = None
        self.initialized = False
        self._init_library()
    
    def _init_library(self) -> bool:
        """Initialize SimpleBGC library"""
        try:
            self.motor_lib = ctypes.CDLL(self.lib_path)
            print(f"[Motor] Loaded library from {self.lib_path}")
        except OSError as e:
            print(f"[Motor] Error loading library: {e}")
            return False
        
        # Setup function signatures
        self.motor_lib.bgc_init.argtypes = []
        self.motor_lib.bgc_init.restype = ctypes.c_int
        
        self.motor_lib.bgc_control_speeds.argtypes = [
            ctypes.c_float, ctypes.c_float, ctypes.c_float
        ]
        self.motor_lib.bgc_control_speeds.restype = ctypes.c_int
        
        if hasattr(self.motor_lib, "bgc_set_motors"):
            self.motor_lib.bgc_set_motors.argtypes = [ctypes.c_int]
            self.motor_lib.bgc_set_motors.restype = ctypes.c_int
        else:
            print("[Motor] WARNING: bgc_set_motors not found")
        
        # Initialize
        status = self.motor_lib.bgc_init()
        if status != 0:
            print(f"[Motor] bgc_init() returned {status}")
            return False
        
        self.initialized = True
        print("[Motor] Initialized successfully")
        return True
    
    def send_speeds(self, roll_dps: float, pitch_dps: float, yaw_dps: float) -> bool:
        """Send speed commands to gimbal"""
        if not self.initialized:
            return False
        
        status = self.motor_lib.bgc_control_speeds(
            ctypes.c_float(roll_dps),
            ctypes.c_float(pitch_dps),
            ctypes.c_float(yaw_dps)
        )
        return status == 0
    
    def set_motors(self, enabled: bool) -> bool:
        """Enable/disable motors"""
        if not self.initialized or not hasattr(self.motor_lib, "bgc_set_motors"):
            return False
        
        status = self.motor_lib.bgc_set_motors(ctypes.c_int(1 if enabled else 0))
        return status == 0


# ==============================================================
# FACE TRACKER (Gimbal Control)
# ==============================================================

class FaceTracker:
    """Face tracking and gimbal control"""
    
    def __init__(self, config: Config):
        self.config = config
        self.motor = GimbalMotor(config.LIB_PATH)
        
        # MediaPipe
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Camera
        self.camera = cv2.VideoCapture(config.CAM_INDEX)
        if not self.camera.isOpened():
            raise RuntimeError(f"Cannot open camera {config.CAM_INDEX}")
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, config.MAX_STORED_FRAMES)
        
        # Tracking state
        self.mode = GimbalMode.TRACKING_ENABLED
        self.state = TrackingState.LOCKED
        self.prev_smoothed = None
        self.prev_time = None
        self.consecutive_lost = 0
        self.anchor = None
        
        # Stable box
        self.stable_range = RangeType.MID
        self.pending_range = None
        self.pending_count = 0
        self.stable_scalar = config.STABLE_SCALAR_MID
        
        # Speed smoothing
        self.smooth_yaw_dps = None
        self.smooth_pitch_dps = None
        self.smooth_roll_dps = None
        self.last_send_time = 0.0
        
        # Statistics
        self.pos_x_hist = TimedHistogram(config.WINDOW_SEC)
        self.pos_y_hist = TimedHistogram(config.WINDOW_SEC)
        self.vel_hist = TimedHistogram(config.WINDOW_SEC)
        
        # GPIO button
        self.gpio_handle = None
        self.prev_button_level = 1
        self.last_button_time = 0.0
        if lgpio:
            self._init_gpio()
    
    def _init_gpio(self):
        """Initialize GPIO for button control"""
        try:
            self.gpio_handle = lgpio.gpiochip_open(0)
            if hasattr(lgpio, "SET_PULL_UP"):
                lgpio.gpio_claim_input(self.gpio_handle, self.config.BUTTON_PIN, 
                                      lgpio.SET_PULL_UP)
            else:
                lgpio.gpio_claim_input(self.gpio_handle, self.config.BUTTON_PIN)
            print("[FaceTracker] GPIO initialized")
        except Exception as e:
            print(f"[FaceTracker] GPIO init failed: {e}")
            self.gpio_handle = None
    
    def _estimate_eye_distance(self, landmarks, frame_w: int, frame_h: int) -> Optional[float]:
        """Estimate face distance using eye landmark distance"""
        try:
            lx = landmarks.landmark[self.config.EYE_L_IDX].x * frame_w
            ly = landmarks.landmark[self.config.EYE_L_IDX].y * frame_h
            rx = landmarks.landmark[self.config.EYE_R_IDX].x * frame_w
            ry = landmarks.landmark[self.config.EYE_R_IDX].y * frame_h
            return math.hypot(rx - lx, ry - ly)
        except Exception:
            return None
    
    def _range_from_eye_distance(self, eye_dist: Optional[float]) -> RangeType:
        """Map eye distance to range category"""
        if eye_dist is None:
            return RangeType.MID
        if eye_dist <= self.config.EYE_DIST_FAR_MAX:
            return RangeType.FAR
        if eye_dist >= self.config.EYE_DIST_NEAR_MIN:
            return RangeType.NEAR
        return RangeType.MID
    
    def _scalar_for_range(self, rng: RangeType) -> float:
        """Get stable box scalar for range"""
        scalars = {
            RangeType.FAR: self.config.STABLE_SCALAR_FAR,
            RangeType.MID: self.config.STABLE_SCALAR_MID,
            RangeType.NEAR: self.config.STABLE_SCALAR_NEAR,
        }
        return scalars.get(rng, self.config.STABLE_SCALAR_MID)
    
    def _check_button_press(self, current_time: float) -> bool:
        """Check if button was pressed"""
        if self.gpio_handle is None:
            return False
        
        try:
            level = lgpio.gpio_read(self.gpio_handle, self.config.BUTTON_PIN)
            pressed = (self.prev_button_level == 1 and level == 0 and
                      (current_time - self.last_button_time) >= self.config.BUTTON_DEBOUNCE_SEC)
            
            if pressed:
                self.last_button_time = current_time
            
            self.prev_button_level = level
            return pressed
        except Exception:
            return False
    
    def _cycle_mode(self):
        """Cycle through gimbal modes"""
        if self.mode == GimbalMode.TRACKING_ENABLED:
            self.motor.set_motors(False)
            self.motor.send_speeds(0.0, 0.0, 0.0)
            self.mode = GimbalMode.COMPLIANCE_MOTORS_OFF
        elif self.mode == GimbalMode.COMPLIANCE_MOTORS_OFF:
            self.motor.set_motors(True)
            self.motor.send_speeds(0.0, 0.0, 0.0)
            self.mode = GimbalMode.HOLD_MOTORS_ON_NO_TRACK
        else:
            self.mode = GimbalMode.TRACKING_ENABLED
        
        self._reset_tracking()
    
    def _reset_tracking(self):
        """Reset tracking variables"""
        self.prev_smoothed = None
        self.prev_time = None
        self.consecutive_lost = 0
        self.state = TrackingState.LOCKED
    
    def _get_mouth_centroid(self, landmarks, frame_w: int, frame_h: int) -> Optional[Tuple[float, float]]:
        """Extract mouth centroid from face landmarks"""
        mouth_idxs = [13, 14, 61, 291]
        mouth_points = []
        
        for idx in mouth_idxs:
            x = int(landmarks.landmark[idx].x * frame_w)
            y = int(landmarks.landmark[idx].y * frame_h)
            mouth_points.append((x, y))
        
        if mouth_points:
            cx = sum(p[0] for p in mouth_points) / len(mouth_points)
            cy = sum(p[1] for p in mouth_points) / len(mouth_points)
            return (cx, cy)
        return None
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """Process single frame and return tracking data"""
        current_time = time.time()
        frame_h, frame_w = frame.shape[:2]
        
        # Initialize anchor
        if self.anchor is None:
            self.anchor = (frame_w / 2.0, frame_h / 2.0)
        
        # Check for mode change button
        if self._check_button_press(current_time):
            self._cycle_mode()
        
        # Handle non-tracking modes
        if self.mode != GimbalMode.TRACKING_ENABLED:
            if (current_time - self.last_send_time) >= (1.0 / self.config.COMMAND_HZ):
                if self.mode == GimbalMode.HOLD_MOTORS_ON_NO_TRACK:
                    self.motor.send_speeds(0.0, 0.0, 0.0)
                self.last_send_time = current_time
            
            status = "COMPLIANCE (motors OFF)" if self.mode == GimbalMode.COMPLIANCE_MOTORS_OFF \
                    else "LOCKED HOLD (motors ON)"
            return {"frame": frame, "status": status, "mode": self.mode.name}
        
        # TRACKING MODE
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        centroid = None
        eye_dist = None
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            centroid = self._get_mouth_centroid(landmarks, frame_w, frame_h)
            eye_dist = self._estimate_eye_distance(landmarks, frame_w, frame_h)
        
        # Handle lost face
        if centroid is None:
            self.consecutive_lost += 1
            if (current_time - self.last_send_time) >= (1.0 / self.config.COMMAND_HZ):
                self.motor.send_speeds(0.0, 0.0, 0.0)
                self.last_send_time = current_time
            
            if self.consecutive_lost > self.config.MAX_LOST_FRAMES:
                self._reset_tracking()
            
            return {"frame": frame, "centroid": None, "status": "No face detected"}
        
        # Reset lost counter
        self.consecutive_lost = 0
        smoothed = ema_point(centroid, self.prev_smoothed, self.config.SMOOTH_ALPHA)
        
        if self.prev_time is None:
            self.prev_time = current_time
        
        delta_time = max(1e-6, current_time - self.prev_time)
        
        # Estimate speed
        if self.prev_smoothed is None:
            speed = 0.0
        else:
            dpx = smoothed[0] - self.prev_smoothed[0]
            dpy = smoothed[1] - self.prev_smoothed[1]
            dvx, dvy = pixels_to_deg(dpx, dpy, frame_w, frame_h, 
                                     self.config.FOV_H_DEG, self.config.FOV_V_DEG)
            speed = math.hypot(dvx, dvy) / delta_time
        
        self.prev_time = current_time
        self.prev_smoothed = smoothed
        
        # Update stable box
        desired_range = self._range_from_eye_distance(eye_dist)
        if desired_range != self.stable_range:
            if self.pending_range != desired_range:
                self.pending_range = desired_range
                self.pending_count = 1
            else:
                self.pending_count += 1
            
            if self.pending_count >= self.config.RANGE_SWITCH_FRAMES:
                self.stable_range = desired_range
                self.pending_range = None
                self.pending_count = 0
                self.stable_scalar = self._scalar_for_range(self.stable_range)
        else:
            self.pending_range = None
            self.pending_count = 0
            self.stable_scalar = self._scalar_for_range(self.stable_range)
        
        # Check tracking boundaries
        stable_box = build_stable_box(self.anchor, frame_w, frame_h, self.stable_scalar)
        in_stable = inside_box(smoothed, stable_box)
        
        dx = smoothed[0] - self.anchor[0]
        dy = smoothed[1] - self.anchor[1]
        
        norm_dx = dx / (frame_w / 2.0)
        norm_dy = dy / (frame_h / 2.0)
        radial = math.hypot(norm_dx, norm_dy)
        within_stop = radial <= self.config.STABLE_STOP_SEEKING_THRESHOLD
        
        # Compute statistics
        self.pos_x_hist.add(current_time, smoothed[0])
        self.pos_y_hist.add(current_time, smoothed[1])
        self.vel_hist.add(current_time, speed)
        
        xs = self.pos_x_hist.values()
        ys = self.pos_y_hist.values()
        pos_std = 999.0
        if len(xs) >= 6 and len(ys) >= 6:
            pos_std = 0.5 * (statistics.pstdev(xs) + statistics.pstdev(ys))
        
        speeds = self.vel_hist.values()
        vel_med = statistics.median(speeds) if len(speeds) >= 3 else 999.0
        
        too_wild = (vel_med > self.config.VEL_THRESH_DEG_S * 2.0) or \
                   (pos_std > self.config.POS_STD_THRESH_PX * 2.0)
        
        # Update state
        if self.state == TrackingState.LOCKED:
            if not in_stable:
                self.state = TrackingState.SEEKING
        else:
            if within_stop:
                self.state = TrackingState.LOCKED
        
        # Compute commands
        yaw_dps = pitch_dps = roll_dps = 0.0
        
        if self.state == TrackingState.SEEKING and not too_wild:
            err_yaw, err_pitch = pixels_to_deg(dx, dy, frame_w, frame_h,
                                               self.config.FOV_H_DEG, self.config.FOV_V_DEG)
            
            if abs(err_yaw) < self.config.DEADBAND_DEG_YAW:
                err_yaw = 0.0
            if abs(err_pitch) < self.config.DEADBAND_DEG_PITCH:
                err_pitch = 0.0
            
            yaw_dps = clamp(self.config.KP_YAW_DPS_PER_DEG * err_yaw,
                           -self.config.MAX_DPS_YAW, self.config.MAX_DPS_YAW)
            pitch_dps = clamp(self.config.KP_PITCH_DPS_PER_DEG * err_pitch,
                             -self.config.MAX_DPS_PITCH, self.config.MAX_DPS_PITCH)
            roll_dps = 0.0
        
        # Send speeds periodically
        if (current_time - self.last_send_time) >= (1.0 / self.config.COMMAND_HZ):
            self.smooth_yaw_dps = ema_scalar(yaw_dps, self.smooth_yaw_dps, 
                                            self.config.CMD_SPEED_EMA_ALPHA)
            self.smooth_pitch_dps = ema_scalar(pitch_dps, self.smooth_pitch_dps,
                                              self.config.CMD_SPEED_EMA_ALPHA)
            self.smooth_roll_dps = ema_scalar(roll_dps, self.smooth_roll_dps,
                                             self.config.CMD_SPEED_EMA_ALPHA)
            
            self.motor.send_speeds(self.smooth_roll_dps, self.smooth_pitch_dps, self.smooth_yaw_dps)
            self.last_send_time = current_time
        
        return {
            "frame": frame,
            "centroid": smoothed,
            "eye_distance": eye_dist,
            "state": self.state.name,
            "stable_range": self.stable_range.name,
            "radial_distance": radial,
            "stable_box": stable_box,
            "anchor": self.anchor,
            "status": "Tracking",
        }
    
    def draw_debug(self, result: Dict) -> np.ndarray:
        """Draw debug information on frame"""
        frame = result["frame"].copy()
        
        if "centroid" in result and result["centroid"]:
            cx, cy = result["centroid"]
            if "stable_box" in result:
                l, t, r, b = map(int, result["stable_box"])
                cv2.rectangle(frame, (l, t), (r, b), (40, 220, 40), 1)
            
            if "anchor" in result:
                ax, ay = result["anchor"]
                cv2.drawMarker(frame, (int(ax), int(ay)), (0, 200, 0),
                              cv2.MARKER_CROSS, 12, 2)
            
            cv2.circle(frame, (int(cx), int(cy)), 4, (0, 0, 255), -1)
            
            # Draw info
            state_txt = result.get("state", "UNKNOWN")
            cv2.putText(frame, f"State: {state_txt}", (10, 24),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 220, 40), 2)
            
            dist = result.get("radial_distance", 0)
            cv2.putText(frame, f"Radial: {dist:.3f}", (10, 48),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 220, 40), 2)
            
            eye_d = result.get("eye_distance")
            eye_str = f"{eye_d:.1f}" if eye_d else "None"
            range_n = result.get("stable_range", "MID")
            scalar = self.stable_scalar
            cv2.putText(frame, f"Eye: {eye_str} Box: {scalar:.3f} {range_n}", (10, 72),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 220, 40), 2)
        
        mode_str = result.get("status", "Unknown")
        color = (0, 0, 255) if "OFF" in mode_str else (0, 255, 0)
        cv2.putText(frame, f"Mode: {mode_str}", (10, 96),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        
        return frame
    
    def shutdown(self):
        """Cleanup resources"""
        try:
            self.motor.send_speeds(0.0, 0.0, 0.0)
            self.motor.set_motors(False)
        except Exception:
            pass
        
        if self.gpio_handle is not None:
            try:
                lgpio.gpio_free(self.gpio_handle, self.config.BUTTON_PIN)
                lgpio.gpiochip_close(self.gpio_handle)
            except Exception:
                pass
        
        self.camera.release()
        cv2.destroyAllWindows()


# ==============================================================
# WHITE BALANCE CALIBRATOR
# ==============================================================

class WhiteBalanceCalibrator:
    """QR code-based white balance calibration"""
    
    def __init__(self, camera_index: int = 0):
        self.camera = cv2.VideoCapture(camera_index)
        self.camera.set(cv2.CAP_PROP_AUTO_WB, 0.0)
        
        self.current_temp = self.camera.get(cv2.CAP_PROP_WB_TEMPERATURE)
        if self.current_temp <= 0 or self.current_temp == -1:
            self.current_temp = 4500.0
        
        self.camera.set(cv2.CAP_PROP_WB_TEMPERATURE, self.current_temp)
        self.qr_detector = cv2.QRCodeDetector()
        self.calibrated = False
        
        print("[WhiteBalance] Initialized")
        print("  Press 'c' to tune white balance")
        print("  Press 'r' to reset to 4500K")
        print("  Press 'q' to quit")
    
    def _get_color_imbalance(self, roi: np.ndarray) -> Tuple[float, float]:
        """Analyze color imbalance in QR code"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        mean_b = np.mean(roi[:, :, 0][mask == 255])
        mean_r = np.mean(roi[:, :, 2][mask == 255])
        
        return mean_b, mean_r
    
    def _tune_temperature(self, roi: np.ndarray) -> bool:
        """Run tuning feedback loop"""
        try:
            mean_b, mean_r = self._get_color_imbalance(roi)
            
            step = 100
            tolerance = 2.0
            diff = mean_r - mean_b
            
            if abs(diff) > tolerance:
                if mean_r > mean_b:
                    self.current_temp -= step
                else:
                    self.current_temp += step
                
                self.current_temp = max(2000.0, min(10000.0, self.current_temp))
                self.camera.set(cv2.CAP_PROP_WB_TEMPERATURE, self.current_temp)
                print(f"  Tuning... {self.current_temp:.0f}K | R: {mean_r:.1f}, B: {mean_b:.1f}")
                self.calibrated = False
                return False
            else:
                self.calibrated = True
                print(f"  Locked at {self.current_temp:.0f}K")
                return True
        except Exception as e:
            print(f"  Tuning error: {e}")
            return False
    
    def process_frame(self) -> Tuple[bool, np.ndarray]:
        """Process one frame and return (running, frame)"""
        ret, frame = self.camera.read()
        if not ret:
            return False, None
        
        display = frame.copy()
        ret_qr, _, points, _ = self.qr_detector.detectAndDecodeMulti(frame)
        
        if ret_qr:
            points = points.astype(int)
            for pts in points:
                cv2.polylines(display, [pts], True, (0, 255, 0), 2)
                if not self.calibrated:
                    cv2.putText(display, "QR Detected! Hold 'c' to Tune", (10, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Status
        status_color = (0, 255, 0) if self.calibrated else (0, 165, 255)
        cv2.putText(display, f"Temp: {self.current_temp:.0f}K", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        return True, display
    
    def handle_input(self, key: int, frame: np.ndarray, ret_qr: bool, points=None) -> bool:
        """Handle keyboard input. Returns True to continue."""
        if key == ord('c') and ret_qr and points is not None:
            pts = points[0]
            x, y, w, h = cv2.boundingRect(pts)
            if x >= 0 and y >= 0 and w > 0 and h > 0:
                roi = frame[y:y+h, x:x+w]
                self._tune_temperature(roi)
        elif key == ord('r'):
            self.current_temp = 4500.0
            self.camera.set(cv2.CAP_PROP_WB_TEMPERATURE, self.current_temp)
            self.calibrated = False
            print("[WhiteBalance] Reset to 4500K")
        elif key == ord('q'):
            return False
        
        return True
    
    def shutdown(self):
        """Cleanup"""
        self.camera.release()
        cv2.destroyAllWindows()


# ==============================================================
# HAND GESTURE TRACKER
# ==============================================================

class HandGestureTracker:
    """Hand recognition and gesture detection"""
    
    # Finger landmark indices
    FINGER_TIPS = [8, 12, 16, 20]
    THUMB_TIP = 4
    FINGER_NAMES = ["Index", "Middle", "Ring", "Pinky"]
    
    def __init__(self, camera_index: int = 0):
        self.camera = cv2.VideoCapture(camera_index)
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        print("[HandTracker] Initialized")
    
    def _detect_fingers(self, hand_landmarks, hand_label: str) -> List[str]:
        """Detect which fingers are open"""
        fingers = []
        
        # Thumb
        thumb_tip_x = hand_landmarks.landmark[self.THUMB_TIP].x
        thumb_ip_x = hand_landmarks.landmark[3].x
        
        if hand_label == "Left":
            if thumb_tip_x > thumb_ip_x:
                fingers.append("Thumb")
        else:
            if thumb_tip_x < thumb_ip_x:
                fingers.append("Thumb")
        
        # Fingers
        for i, tip_idx in enumerate(self.FINGER_TIPS):
            tip_y = hand_landmarks.landmark[tip_idx].y
            pip_y = hand_landmarks.landmark[tip_idx - 2].y
            
            if tip_y < pip_y:
                fingers.append(self.FINGER_NAMES[i])
        
        return fingers
    
    def process_frame(self) -> Tuple[bool, np.ndarray, List[Dict]]:
        """Process one frame. Returns (success, frame, hands_data)"""
        ret, image = self.camera.read()
        if not ret:
            return False, None, []
        
        image = cv2.flip(image, 1)
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        image.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        hands_data = []
        
        if results.multi_hand_landmarks:
            for hand_idx, landmarks in enumerate(results.multi_hand_landmarks):
                hand_label = results.multi_handedness[hand_idx].classification[0].label
                fingers = self._detect_fingers(landmarks, hand_label)
                
                hands_data.append({
                    "label": hand_label,
                    "fingers": fingers,
                    "landmarks": landmarks
                })
                
                # Draw
                self.mp_drawing.draw_landmarks(
                    image, landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Display
                text_y = 50 + (hand_idx * 50)
                fingers_str = ", ".join(fingers)
                cv2.putText(image, f"{hand_label}: {fingers_str}", (10, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return True, image, hands_data
    
    def shutdown(self):
        """Cleanup"""
        self.hands.close()
        self.camera.release()
        cv2.destroyAllWindows()


# ==============================================================
# AUDIO RECORDER
# ==============================================================

class AudioRecorder:
    """Audio recording for transcription"""
    
    def __init__(self, config: Config):
        if pyaudio is None:
            raise RuntimeError("PyAudio not installed")
        
        self.config = config
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.frames = []
        self.is_recording = False
        
        print("[AudioRecorder] Initialized")
        print(f"  Output: {config.AUDIO_OUTPUT}")
    
    def start_recording(self):
        """Start recording"""
        try:
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=self.config.AUDIO_CHANNELS,
                rate=self.config.AUDIO_RATE,
                input=True,
                frames_per_buffer=self.config.AUDIO_CHUNK
            )
            self.frames = []
            self.is_recording = True
            print("[AudioRecorder] Recording started")
            return True
        except Exception as e:
            print(f"[AudioRecorder] Start error: {e}")
            return False
    
    def stop_recording(self):
        """Stop and save recording"""
        if not self.is_recording:
            return False
        
        try:
            self.stream.stop_stream()
            self.stream.close()
            self.is_recording = False
            
            # Save to file
            with wave.open(self.config.AUDIO_OUTPUT, 'wb') as wf:
                wf.setnchannels(self.config.AUDIO_CHANNELS)
                wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
                wf.setframerate(self.config.AUDIO_RATE)
                wf.writeframes(b''.join(self.frames))
            
            print(f"[AudioRecorder] Saved to {self.config.AUDIO_OUTPUT}")
            return True
        except Exception as e:
            print(f"[AudioRecorder] Stop error: {e}")
            return False
    
    def record_chunk(self) -> bool:
        """Record one audio chunk"""
        if not self.is_recording:
            return False
        
        try:
            data = self.stream.read(self.config.AUDIO_CHUNK, 
                                   exception_on_overflow=False)
            self.frames.append(data)
            return True
        except Exception as e:
            print(f"[AudioRecorder] Chunk error: {e}")
            return False
    
    def shutdown(self):
        """Cleanup"""
        if self.is_recording:
            self.stop_recording()
        
        if self.stream is not None:
            try:
                self.stream.close()
            except Exception:
                pass
        
        self.audio.terminate()


# ==============================================================
# MAIN APPLICATION
# ==============================================================

class GimbalControlApplication:
    """Main unified application"""
    
    def __init__(self):
        Config.setup_directories()
        self.config = Config()
        
        self.face_tracker = None
        self.white_balance = None
        self.hand_tracker = None
        self.audio_recorder = None
        
        print("\n" + "="*60)
        print("UNIFIED GIMBAL CONTROLLER")
        print("="*60)
    
    def show_menu(self) -> str:
        """Display main menu"""
        print("\n" + "="*60)
        print("MAIN MENU")
        print("="*60)
        print("1. Run Face Tracking (Gimbal Control)")
        print("2. White Balance Calibration (QR Code)")
        print("3. Hand Gesture Recognition")
        print("4. Audio Recording")
        print("5. Full Integration (Face + Hand + Audio)")
        print("6. Exit")
        print("="*60)
        return input("Select option (1-6): ").strip()
    
    def run_face_tracking(self):
        """Run face tracking gimbal control"""
        print("\n[Starting Face Tracking]")
        print("Controls:")
        print("  Press 'c' to cycle gimbal modes")
        print("  Press ESC to exit")
        print("-" * 60)
        
        try:
            self.face_tracker = FaceTracker(self.config)
            
            while True:
                try:
                    ret, frame = self.face_tracker.camera.read()
                    if not ret:
                        break
                    
                    result = self.face_tracker.process_frame()
                    
                    if self.config.DRAW_FRAME_RT:
                        display = self.face_tracker.draw_debug(result)
                        cv2.imshow("Face Tracking - Gimbal Control", display)
                        
                        key = cv2.waitKey(1) & 0xFF
                        if key == 27:  # ESC
                            break
                
                except Exception as e:
                    print(f"Error in frame processing: {e}")
                    break
        
        except Exception as e:
            print(f"Error initializing face tracker: {e}")
        
        finally:
            if self.face_tracker:
                self.face_tracker.shutdown()
                self.face_tracker = None
            cv2.destroyAllWindows()
            print("[Face Tracking Ended]")
    
    def run_white_balance(self):
        """Run white balance calibrator"""
        print("\n[Starting White Balance Calibration]")
        print("-" * 60)
        
        try:
            self.white_balance = WhiteBalanceCalibrator()
            
            while True:
                ret, frame = self.white_balance.camera.read()
                if not ret:
                    break
                
                display = frame.copy()
                ret_qr, _, points, _ = self.white_balance.qr_detector.detectAndDecodeMulti(frame)
                
                if ret_qr:
                    points = points.astype(int)
                    for pts in points:
                        cv2.polylines(display, [pts], True, (0, 255, 0), 2)
                
                status_color = (0, 255, 0) if self.white_balance.calibrated else (0, 165, 255)
                cv2.putText(display, f"Temp: {self.white_balance.current_temp:.0f}K", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                
                cv2.imshow("White Balance Calibration", display)
                
                key = cv2.waitKey(1) & 0xFF
                if not self.white_balance.handle_input(key, frame, ret_qr, points):
                    break
        
        except Exception as e:
            print(f"Error in white balance: {e}")
        
        finally:
            if self.white_balance:
                self.white_balance.shutdown()
                self.white_balance = None
            print("[White Balance Ended]")
    
    def run_hand_tracking(self):
        """Run hand gesture recognition"""
        print("\n[Starting Hand Gesture Recognition]")
        print("Controls:")
        print("  Press ESC to exit")
        print("-" * 60)
        
        try:
            self.hand_tracker = HandGestureTracker()
            
            while True:
                ret, frame, hands = self.hand_tracker.process_frame()
                if not ret:
                    break
                
                cv2.imshow("Hand Gesture Recognition", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
        
        except Exception as e:
            print(f"Error in hand tracking: {e}")
        
        finally:
            if self.hand_tracker:
                self.hand_tracker.shutdown()
                self.hand_tracker = None
            print("[Hand Tracking Ended]")
    
    def run_audio_recording(self):
        """Run audio recorder"""
        print("\n[Starting Audio Recording]")
        print("-" * 60)
        
        try:
            self.audio_recorder = AudioRecorder(self.config)
            
            input("Press ENTER to start recording...")
            
            if not self.audio_recorder.start_recording():
                return
            
            try:
                while True:
                    self.audio_recorder.record_chunk()
                    time.sleep(0.001)
            except KeyboardInterrupt:
                print("\nStopping...")
            
            self.audio_recorder.stop_recording()
        
        except Exception as e:
            print(f"Error in audio recording: {e}")
        
        finally:
            if self.audio_recorder:
                self.audio_recorder.shutdown()
                self.audio_recorder = None
            print("[Audio Recording Ended]")
    
    def run_full_integration(self):
        """Run full integrated mode with all features"""
        print("\n[Starting Full Integration Mode]")
        print("Features active: Face Tracking, Hand Gestures, Audio")
        print("Controls:")
        print("  Press 'c' to cycle gimbal modes")
        print("  Press 'r' to start/stop audio recording")
        print("  Press ESC to exit")
        print("-" * 60)
        
        try:
            self.face_tracker = FaceTracker(self.config)
            self.hand_tracker = HandGestureTracker()
            
            try:
                self.audio_recorder = AudioRecorder(self.config)
            except Exception as e:
                print(f"Audio not available: {e}")
                self.audio_recorder = None
            
            recording = False
            
            while True:
                # Face tracking
                ret_face, face_frame = self.face_tracker.camera.read()
                if not ret_face:
                    break
                
                face_result = self.face_tracker.process_frame()
                display = self.face_tracker.draw_debug(face_result)
                
                # Hand tracking (on same frame)
                ret_hand, _, hands = self.hand_tracker.process_frame()
                
                if ret_hand:
                    # Resize hand frame to match face frame for overlay
                    hand_frame = cv2.resize(_, display.shape[1::-1])
                    # Draw hand info on top
                    cv2.putText(display, f"Hands detected: {len(hands)}", (10, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                # Audio status
                if self.audio_recorder:
                    status = "Recording" if recording else "Ready"
                    color = (0, 0, 255) if recording else (0, 255, 0)
                    cv2.putText(display, f"Audio: {status}", (10, 144),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                cv2.imshow("Full Integration", display)
                
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == ord('r') and self.audio_recorder:
                    if not recording:
                        self.audio_recorder.start_recording()
                        recording = True
                        print("Recording started...")
                    else:
                        self.audio_recorder.stop_recording()
                        recording = False
                        print("Recording stopped...")
        
        except Exception as e:
            print(f"Error in full integration: {e}")
        
        finally:
            if self.face_tracker:
                self.face_tracker.shutdown()
                self.face_tracker = None
            if self.hand_tracker:
                self.hand_tracker.shutdown()
                self.hand_tracker = None
            if self.audio_recorder:
                self.audio_recorder.shutdown()
                self.audio_recorder = None
            cv2.destroyAllWindows()
            print("[Full Integration Ended]")
    
    def run(self):
        """Main application loop"""
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        warnings.filterwarnings("ignore")
        
        while True:
            choice = self.show_menu()
            
            if choice == "1":
                self.run_face_tracking()
            elif choice == "2":
                self.run_white_balance()
            elif choice == "3":
                self.run_hand_tracking()
            elif choice == "4":
                self.run_audio_recording()
            elif choice == "5":
                self.run_full_integration()
            elif choice == "6":
                print("\nExiting... Goodbye!")
                break
            else:
                print("Invalid selection. Please try again.")


# ==============================================================
# ENTRY POINT
# ==============================================================

if __name__ == "__main__":
    app = GimbalControlApplication()
    try:
        app.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
