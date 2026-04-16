#!/usr/bin/env python3
"""
Simple viewer for the center camera feed.

Camera mapping:
  - index 0: center camera
  - index 2: wide-angle tracking camera
"""

import sys
import cv2
import json
import os
import time


CENTER_CAM_INDEX = 0
WINDOW_NAME = "Center Camera Feed"
BASE_DIR = os.path.expanduser("~/senior_design/A-dec-Senior-Design/camera/testing")
CAMERA_PROFILE_DIR = os.path.join(BASE_DIR, "camera_profiles")
STARTUP_PROFILE = "zoom_lon.json"
POSTER_CAPTURE_DIR = os.path.join(BASE_DIR, "poster_captures")
os.makedirs(POSTER_CAPTURE_DIR, exist_ok=True)


def update_camera_settings(camera, filename, profile_dir):
    try:
        profile_path = os.path.join(profile_dir, filename)
        with open(profile_path, "r", encoding="utf-8") as profile_file:
            camera_settings = json.load(profile_file)

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
        if "auto_white_balance" in camera_settings:
            camera.set(cv2.CAP_PROP_AUTO_WB, camera_settings["auto_white_balance"])
        if "white_balance_temperature" in camera_settings:
            camera.set(cv2.CAP_PROP_WB_TEMPERATURE, camera_settings["white_balance_temperature"])

        print(f"Loaded center camera profile: {profile_path}")
        return True
    except Exception as profile_error:
        print(f"Warning: failed to load center camera profile {filename}: {profile_error}")
        return False


def save_poster_capture(frame):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    milliseconds = int((time.time() % 1.0) * 1000)
    capture_path = os.path.join(POSTER_CAPTURE_DIR, f"center_camera_capture_{timestamp}_{milliseconds:03d}.jpg")
    if cv2.imwrite(capture_path, frame):
        print(f"Saved center camera capture: {capture_path}")
        return True

    print(f"Failed to save center camera capture: {capture_path}")
    return False


def main():
    camera = cv2.VideoCapture(CENTER_CAM_INDEX)
    if not camera.isOpened():
        print(f"Error: unable to open center camera at index {CENTER_CAM_INDEX}")
        sys.exit(1)

    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    update_camera_settings(camera, STARTUP_PROFILE, CAMERA_PROFILE_DIR)

    try:
        while True:
            frame_read, frame = camera.read()
            frame = cv2.flip(frame, 0)
            if not frame_read:
                print("Error: failed to read a frame from the center camera")
                break

            poster_frame = frame.copy()
            cv2.putText(
                frame,
                f"Center camera (index {CENTER_CAM_INDEX})",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("c"):
                save_poster_capture(poster_frame)
            if key == 27 or key == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
