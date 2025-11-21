#!/usr/bin/env python3
"""
libraryBGCtester.py  (0-based / relative version)

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
# Zero-based / relative state
# ----------------------------------------------------------------------
# These are in "gimbal degrees" relative to the pose when the script started.
# We never try to read the gimbal's internal zero; we just accumulate deltas.

acc_roll_deg  = 0.0
acc_pitch_deg = 0.0
acc_yaw_deg   = 0.0

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

def bgc_control_angles_absolute(roll_deg: float, pitch_deg: float, yaw_deg: float):
    """
    Call the shim with ABSOLUTE angles (in the gimbal's frame).
    Mostly for debugging; the tracking code should use the relative wrapper.
    """
    print(f"# ABSOLUTE: bgc_control_angles(roll={roll_deg:.1f}, "
          f"pitch={pitch_deg:.1f}, yaw={yaw_deg:.1f})")
    rc = lib.bgc_control_angles(
        ctypes.c_float(roll_deg),
        ctypes.c_float(pitch_deg),
        ctypes.c_float(yaw_deg)
    )
    print(f"#   returned {rc}")
    return rc

def bgc_control_angles_relative(d_roll: float, d_pitch: float, d_yaw: float):
    """
    Zero-based / relative API:
    - d_* are requested changes from the CURRENT pose (this run's logical zero).
    - We accumulate them into acc_* and send the accumulated ABSOLUTE angles
      down to the shim.

    Example:
      call bgc_control_angles_relative(0, 0, +20)  -> board sees yaw = +20
      call bgc_control_angles_relative(0, 0, -20)  -> board sees yaw =   0
    """
    global acc_roll_deg, acc_pitch_deg, acc_yaw_deg

    # Update our logical absolute angles
    acc_roll_deg  += d_roll
    acc_pitch_deg += d_pitch
    acc_yaw_deg   += d_yaw

    print(f"# RELATIVE command: dR={d_roll:+.1f}, dP={d_pitch:+.1f}, dY={d_yaw:+.1f}")
    print(f"#   -> accumulated (R,P,Y)=({acc_roll_deg:+.1f}, "
          f"{acc_pitch_deg:+.1f}, {acc_yaw_deg:+.1f})")

    rc = lib.bgc_control_angles(
        ctypes.c_float(acc_roll_deg),
        ctypes.c_float(acc_pitch_deg),
        ctypes.c_float(acc_yaw_deg)
    )
    print(f"#   bgc_control_angles returned {rc}")
    return rc

def bgc_reset_logical_zero():
    """
    Reset our Python-side logical zero WITHOUT moving the gimbal.
    After calling this, the *current* physical orientation is treated as (0,0,0)
    for subsequent relative commands.
    """
    global acc_roll_deg, acc_pitch_deg, acc_yaw_deg
    acc_roll_deg  = 0.0
    acc_pitch_deg = 0.0
    acc_yaw_deg   = 0.0
    print("# Logical zero reset: acc_roll=pitch=yaw = 0.0 (no command sent)")

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

    print("# ------------------------------------------------------------------")
    print("# TEST 1: Yaw +20 then -20 (should end roughly where it started)")
    print("# ------------------------------------------------------------------")
    bgc_control_angles_relative(0.0, 0.0, +20.0)
    time.sleep(3.0)
    bgc_control_angles_relative(0.0, 0.0, -20.0)
    time.sleep(3.0)

    print("# ------------------------------------------------------------------")
    print("# TEST 2: Roll +15 then -15")
    print("# ------------------------------------------------------------------")
    bgc_control_angles_relative(+15.0, 0.0, 0.0)
    time.sleep(3.0)
    bgc_control_angles_relative(-15.0, 0.0, 0.0)
    time.sleep(3.0)

    print("# ------------------------------------------------------------------")
    print("# TEST 3: Pitch -10 then +10")
    print("# ------------------------------------------------------------------")
    bgc_control_angles_relative(0.0, -10.0, 0.0)
    time.sleep(3.0)
    bgc_control_angles_relative(0.0, +10.0, 0.0)
    time.sleep(3.0)

    # 4) Motors OFF
    bgc_set_motors(False)

    # 5) De-init
    bgc_deinit()
    print("# Test complete.")

if __name__ == "__main__":
    main()
