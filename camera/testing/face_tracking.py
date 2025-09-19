# face_track.py
# File path: ~/senior_design/camera/testing/face_track.py

import cv2
import mediapipe as mp
import time
import os
import sys
import warnings
from collections import deque

# ------------------------------
# Setup logging to file
# ------------------------------
LOG_PATH = os.path.expanduser("~/senior_design/camera/testing/face_track_log.txt")
sys.stderr = open(LOG_PATH, "w")

# Suppress TensorFlow/MediaPipe warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

# ------------------------------
# Initialize MediaPipe FaceMesh
# ------------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Drawing utilities
mp_drawing = mp.solutions.drawing_utils
draw_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# ------------------------------
# Smoothing with EMA
# ------------------------------
def smooth_point(point, prev_point, alpha=0.2):
    if prev_point is None:
        return point
    x = alpha * point[0] + (1 - alpha) * prev_point[0]
    y = alpha * point[1] + (1 - alpha) * prev_point[1]
    return int(x), int(y)

# ------------------------------
# Camera Setup
# ------------------------------
cap = cv2.VideoCapture(0)  # Change to correct camera index if needed

prev_time = time.time()
fps_deque = deque(maxlen=10)
smoothed_center = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Mouth landmark indices
            mouth_indices = [13, 14, 61, 291]
            mouth_points = []
            for idx in mouth_indices:
                x = int(face_landmarks.landmark[idx].x * frame.shape[1])
                y = int(face_landmarks.landmark[idx].y * frame.shape[0])
                mouth_points.append((x, y))
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # Center of mouth
            if mouth_points:
                center_x = sum(p[0] for p in mouth_points) // len(mouth_points)
                center_y = sum(p[1] for p in mouth_points) // len(mouth_points)
                smoothed_center = smooth_point((center_x, center_y), smoothed_center)

                # Draw smoothed crosshair
                cv2.drawMarker(frame, smoothed_center, (0, 0, 255),
                               markerType=cv2.MARKER_CROSS, markerSize=15,
                               thickness=2, line_type=cv2.LINE_AA)

    # FPS calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if curr_time > prev_time else 0
    prev_time = curr_time
    fps_deque.append(fps)
    avg_fps = sum(fps_deque) / len(fps_deque)

    cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Face/Mouth Tracking", frame)
    if cv2.waitKey(5) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
