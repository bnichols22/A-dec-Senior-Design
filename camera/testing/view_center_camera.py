#!/usr/bin/env python3
"""
Simple viewer for the center camera feed.

Camera mapping:
  - index 0: center camera
  - index 2: wide-angle tracking camera
"""

import sys
import cv2


CENTER_CAM_INDEX = 0
WINDOW_NAME = "Center Camera Feed"


def main():
    camera = cv2.VideoCapture(CENTER_CAM_INDEX)
    if not camera.isOpened():
        print(f"Error: unable to open center camera at index {CENTER_CAM_INDEX}")
        sys.exit(1)

    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    try:
        while True:
            frame_read, frame = camera.read()
            frame = cv2.flip(frame, 0)
            if not frame_read:
                print("Error: failed to read a frame from the center camera")
                break
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
            if key == 27 or key == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
