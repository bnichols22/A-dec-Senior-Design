#!/usr/bin/env python3
"""
libraryBGCtester.py

Tester for libsimplebgc.so + simplebgc_shim.c, with automatic
yaw-zeroing: the very first yaw command is treated as yaw = 0°.
All future commands are relative to that first yaw.

Directory structure:

    SerialLibrary/
        libraryBGCtester.py
        serialAPI/
            libsimplebgc.so
"""

import os
import time
import ctypes

# ----------------------------------------------------------
# Locate shared library
# ----------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
LIB_PATH = os.path.join(HERE, "serialAPI", "libsimplebgc.so")

print(f"# Using SBGC library: {LIB_PATH}")

if not os.path.exists(LIB_PATH):
    print("# ERROR: libsimplebgc.so not found.")
    raise SystemExit(1)

# ----------------------------------------------------------
# Load library and prototypes
# ----------------------------------------------------------
lib = ctypes.CDLL(LIB_PATH)

lib.bgc_init.argtypes = []
lib.bgc_init.restype  = ctypes.c_int

lib.bgc_set_motors.argtypes = [ctypes.c_int]
lib.bgc_set_motors.restype  = ctypes.c_int

lib.bgc_control_angles.argtypes = [
    ctypes.c_float,  # roll
    ctypes.c_float,  # pitch
    ctypes.c_float   # yaw (deg)
]
lib.bgc_control_angles.restype = ctypes.c_int

lib.bgc_deinit.argtypes = []
lib.bgc_deinit.restype  = None

# ----------------------------------------------------------
# Tracking the first-yaw-zero baseline
# ----------------------------------------------------------
yaw_zero_baseline = 0.0
first_yaw_initialized = False

def apply_first_yaw_zero(yaw_deg: float) -> float:
    """
    On first call: record incoming yaw as baseline.
    On subsequent calls: subtract baseline so that first yaw is interpreted as 0.
    """
    global first_yaw_initialized, yaw_zero_baseline

    if not first_yaw_initialized:
        yaw_zero_baseline = yaw_deg
        first_yaw_initialized = True
        print(f"# First yaw detected = {yaw_deg:+.2f}° — setting as yaw=0 reference.")
        return 0.0

    # Convert all future yaws into relative coordinates
    effective_yaw = yaw_deg - yaw_zero_baseline
    return effective_yaw


# ----------------------------------------------------------
# Wrapper functions
# ----------------------------------------------------------
def bgc_init():
    rc = lib.bgc_init()
    print(f"# bgc_init() -> {rc}")
    return rc

def bgc_set_motors(on: bool):
    rc = lib.bgc_set_motors(1 if on else 0)
    print(f"# bgc_set_motors({on}) -> {rc}")
    return rc

def bgc_control_angles(roll_deg: float, pitch_deg: float, yaw_deg: float):
    """
    Yaw passed in is absolute user-requested yaw.
    We convert yaw to a coordinate where the first yaw is zero.
    """
    effective_yaw = apply_first_yaw_zero(yaw_deg)

    print(f"# Request: roll={roll_deg:+.1f}, pitch={pitch_deg:+.1f}, yaw_in={yaw_deg:+.1f}")
    print(f"# Effective yaw (relative to first): {effective_yaw:+.1f}")

    rc = lib.bgc_control_angles(
        ctypes.c_float(roll_deg),
        ctypes.c_float(pitch_deg),
        ctypes.c_float(effective_yaw)
    )
    print(f"#   bgc_control_angles -> {rc}")
    return rc

def bgc_deinit():
    print("# bgc_deinit()")
    lib.bgc_deinit()


# ----------------------------------------------------------
# Simple test routine
# ----------------------------------------------------------
def main():

    if bgc_init() != 0:
        print("# ERROR: init failed.")
        return

    bgc_set_motors(True)
    time.sleep(1.0)

    print("\n# --- Sending test angles with FIRST YAW ZERO logic ---")

    # First command defines "yaw = 0"
    bgc_control_angles(roll_deg=0.0, pitch_deg=0.0, yaw_deg=20.0)
    time.sleep(3.0)

    # These are now relative to the first yaw (20°)
    bgc_control_angles(roll_deg=0.0, pitch_deg=0.0, yaw_deg=30.0)
    time.sleep(3.0)

    bgc_control_angles(roll_deg=0.0, pitch_deg=0.0, yaw_deg=10.0)
    time.sleep(3.0)

    # Return back toward reference
    bgc_control_angles(roll_deg=0.0, pitch_deg=0.0, yaw_deg=20.0)
    time.sleep(3.0)

    bgc_set_motors(False)
    bgc_deinit()

    print("# Test complete.")


if __name__ == "__main__":
    main()
