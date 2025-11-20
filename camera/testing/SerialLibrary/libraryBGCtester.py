#!/usr/bin/env python3
"""
libraryBGCtester.py

Simple tester for libsimplebgc.so + simplebgc_shim.c

Assumes directory layout:

    .../SerialLibrary/
        libraryBGCtester.py      <-- this file
        serialAPI/
            libsimplebgc.so      <-- built here

Build example (from inside serialAPI):

    gcc -fPIC -c sbgc32.c -o sbgc32.o
    gcc -fPIC -c simplebgc_shim.c -o simplebgc_shim.o
    gcc -shared -o libsimplebgc.so sbgc32.o simplebgc_shim.o -lpthread
"""

import os
import time
import ctypes

# ----------------------------------------------------------------------
# Locate the shared library
# ----------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
LIB_PATH = os.path.join(HERE, "serialAPI", "libsimplebgc.so")

print(f"# Using SBGC library at: {LIB_PATH}")

if not os.path.exists(LIB_PATH):
    print("# ERROR: libsimplebgc.so not found at that path.")
    raise SystemExit(1)

# ----------------------------------------------------------------------
# Load library and set up function prototypes
# ----------------------------------------------------------------------

lib = ctypes.CDLL(LIB_PATH)

# C shim signatures:
#   int bgc_init(void);
#   int bgc_set_motors(int on);
#   int bgc_control_angles(float roll_deg, float pitch_deg, float yaw_deg);
#   void bgc_deinit(void);

lib.bgc_init.argtypes = []
lib.bgc_init.restype  = ctypes.c_int

lib.bgc_set_motors.argtypes = [ctypes.c_int]
lib.bgc_set_motors.restype  = ctypes.c_int

lib.bgc_control_angles.argtypes = [
    ctypes.c_float,  # roll_deg
    ctypes.c_float,  # pitch_deg
    ctypes.c_float   # yaw_deg
]
lib.bgc_control_angles.restype = ctypes.c_int

lib.bgc_deinit.argtypes = []
lib.bgc_deinit.restype  = None

# ----------------------------------------------------------------------
# Helper wrappers
# ----------------------------------------------------------------------

def bgc_init():
    rc = lib.bgc_init()
    print(f"# bgc_init() -> {rc}")
    return rc

def bgc_set_motors(on: bool):
    rc = lib.bgc_set_motors(1 if on else 0)
    print(f"# bgc_set_motors({on}) -> {rc}")
    return rc

def bgc_control_angles(roll_deg: float, pitch_deg: float, yaw_deg: float):
    print(f"# bgc_control_angles(roll={roll_deg:.1f}, "
          f"pitch={pitch_deg:.1f}, yaw={yaw_deg:.1f})")
    rc = lib.bgc_control_angles(
        ctypes.c_float(roll_deg),
        ctypes.c_float(pitch_deg),
        ctypes.c_float(yaw_deg)
    )
    print(f"#   returned {rc}")
    return rc

def bgc_deinit():
    print("# bgc_deinit()")
    lib.bgc_deinit()

# ----------------------------------------------------------------------
# Simple test routine
# ----------------------------------------------------------------------

def main():
    # 1) Init
    if bgc_init() != 0:
        print("# ERROR: bgc_init failed, aborting.")
        return

    # 2) Motors ON
    bgc_set_motors(True)
    time.sleep(1.0)

    # 3) Move through a few poses (tweak angles as needed for safety)
    #    (roll, pitch, yaw) in degrees – these are ABSOLUTE targets.
    poses = [
        ( 0.0,   0.0,   0.0),
        ( 0.0, -10.0,   0.0),
        ( 0.0, +10.0,   0.0),
        ( 0.0,   0.0, -15.0),
        ( 0.0,   0.0, +15.0),
        (+5.0,   0.0,   0.0),
        (-5.0,   0.0,   0.0),
        ( 0.0,   0.0,   0.0),
    ]

    for roll, pitch, yaw in poses:
        bgc_control_angles(roll, pitch, yaw)
        time.sleep(1.0)  # give it time to move

    # 4) Motors OFF
    bgc_set_motors(False)

    # 5) De-init
    bgc_deinit()
    print("# Test complete.")

if __name__ == "__main__":
    main()
