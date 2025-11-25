#!/usr/bin/env python3
"""
fov_tester.py

- Initializes SimpleBGC via libsimplebgc.so
- Turns motors ON
- Opens USB camera, grabs one frame, displays it
- Saves the frame to: ./FOV_TEST_SNAPSHOT/snapshot_<timestamp>.png
- Waits for key press, then shuts everything down
"""

import os
import time
import ctypes
import cv2
from datetime import datetime

# --------- Paths / config ---------
LIB_PATH = os.path.expanduser(
    "~/senior_design/A-dec-Senior-Design/camera/testing/SerialLibrary/serialAPI/libsimplebgc.so"
)

CAM_INDEX = 0  # USB camera index (usually 0)

# Directory to save the snapshot
SNAPSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FOV_TEST_SNAPSHOT")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# --------- Load library & bind functions ---------
if not os.path.exists(LIB_PATH):
    print(f"# ERROR: libsimplebgc.so not found at {LIB_PATH}")
    raise SystemExit(1)

print(f"# Using SBGC library at: {LIB_PATH}")
lib = ctypes.CDLL(LIB_PATH)

# Bind shim function signatures
lib.bgc_init.argtypes = []
lib.bgc_init.restype  = ctypes.c_int

lib.bgc_set_motors.argtypes = [ctypes.c_int]
lib.bgc_set_motors.restype  = ctypes.c_int

lib.bgc_deinit.argtypes = []
lib.bgc_deinit.restype  = None


def bgc_init():
    rc = lib.bgc_init()
    print(f"# bgc_init() -> {rc}")
    return rc

def bgc_set_motors(on: bool):
    rc = lib.bgc_set_motors(1 if on else 0)
    print(f"# bgc_set_motors({on}) -> {rc}")
    return rc

def bgc_deinit():
    print("# bgc_deinit()")
    lib.bgc_deinit()


def main():
    # 1) Init gimbal
    if bgc_init() != 0:
        print("# ERROR: bgc_init failed, aborting.")
        return

    # 2) Turn motors ON
    bgc_set_motors(True)
    time.sleep(1.0)

    # 3) Open camera
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"# ERROR: Unable to open camera index {CAM_INDEX}")
        bgc_set_motors(False)
        bgc_deinit()
        return

    # 4) Grab one frame
    ok, frame = cap.read()
    if not ok or frame is None:
        print("# ERROR: Failed to grab frame from camera.")
        cap.release()
        bgc_set_motors(False)
        bgc_deinit()
        return

    # ---- SAVE SNAPSHOT ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"snapshot_{timestamp}.png"
    save_path = os.path.join(SNAPSHOT_DIR, filename)

    cv2.imwrite(save_path, frame)
    print(f"# Snapshot saved to: {save_path}")

    # ---- DISPLAY SNAPSHOT ----
    cv2.imshow("Gimbal Camera Snapshot", frame)
    print("# Press any key in the image window to exit...")
    cv2.waitKey(0)

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    bgc_set_motors(False)
    bgc_deinit()

    print("# Done.")

if __name__ == "__main__":
    main()
