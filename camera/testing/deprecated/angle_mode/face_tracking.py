# ==============================================================
# File: face_tracking.py
# Path: ~/senior_design/camera/testing/face_tracking.py
# Description:
#   Tracks a user’s face (specifically the mouth region) using
#   MediaPipe FaceMesh and OpenCV. Draws tracking markers on
#   the live camera feed and logs runtime info to file.
# ==============================================================

import cv2
import mediapipe as mp
import os
import sys
import warnings
import time
from collections import deque

# ==============================================================
# Logging Setup
# ==============================================================
# Define log file path and redirect stderr to log output
LOG_PATH = os.path.expanduser("~/senior_design/camera/testing/face_track_log.txt")
sys.stderr = open(LOG_PATH, "w")

# Suppress verbose TensorFlow / MediaPipe startup logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

# ==============================================================
# Initialize MediaPipe FaceMesh
# ==============================================================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,          # Continuous video stream
    max_num_faces=1,                  # Detect one face only
    refine_landmarks=True,            # More precise landmarking
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Drawing utilities for visual debugging (optional)
mp_drawing = mp.solutions.drawing_utils
landmark_style = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# ==============================================================
# Helper Function: Smooth coordinates
# ==============================================================
def smooth_point(current_point, previous_point, alpha=0.2):
    """
    Smooth a moving coordinate (Exponential Moving Average).
    Args:
        current_point: tuple (x, y) - Current landmark location.
        previous_point: tuple (x, y) - Previous smoothed location.
        alpha: float - Smoothing factor (0-1).
    Returns:
        tuple (smoothed_x, smoothed_y)
    """
    if previous_point is None:
        return current_point
    smoothed_x = alpha * current_point[0] + (1 - alpha) * previous_point[0]
    smoothed_y = alpha * current_point[1] + (1 - alpha) * previous_point[1]
    return int(smoothed_x), int(smoothed_y)

# ==============================================================
# Camera Initialization
# ==============================================================
# Capture from default camera (index 0). Change index if needed.
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Error: Unable to access camera.")
    sys.exit(1)

# Track last processed frame time
last_frame_time = time.time()
smoothed_mouth_center = None

# ==============================================================
# Main Tracking Loop
# ==============================================================
while camera.isOpened():
    success, frame = camera.read()
    if not success:
        print("Warning: Camera frame not read successfully.")
        break

    # Convert to RGB for MediaPipe (expects RGB images)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    # If a face is detected, process landmarks
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Key indices for mouth landmarks
            mouth_indices = [13, 14, 61, 291]
            mouth_points = []

            # Convert landmark coordinates to pixel positions
            for idx in mouth_indices:
                x = int(face_landmarks.landmark[idx].x * frame.shape[1])
                y = int(face_landmarks.landmark[idx].y * frame.shape[0])
                mouth_points.append((x, y))
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)  # Green dots

            # Compute smoothed mouth center
            if mouth_points:
                avg_x = sum(p[0] for p in mouth_points) // len(mouth_points)
                avg_y = sum(p[1] for p in mouth_points) // len(mouth_points)
                smoothed_mouth_center = smooth_point((avg_x, avg_y), smoothed_mouth_center)

                # Draw a red crosshair on the mouth center
                cv2.drawMarker(frame, smoothed_mouth_center, (0, 0, 255),
                               markerType=cv2.MARKER_CROSS, markerSize=15,
                               thickness=2, line_type=cv2.LINE_AA)

    # Show live video feed with tracking markers
    cv2.imshow("Mouth Tracking", frame)

    # Press ESC to exit
    if cv2.waitKey(5) & 0xFF == 27:
        break

# ==============================================================
# Cleanup
# ==============================================================
camera.release()
cv2.destroyAllWindows()
print("Tracking stopped. Camera released successfully.")
