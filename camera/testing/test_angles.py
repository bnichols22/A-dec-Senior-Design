#!/usr/bin/env python3
# ==============================================================
# File: test_angles.py
# Purpose:
#   Minimal "one-shot poll" of angles from libsimplebgc.so
#   - Calls bgc_init()
#   - Calls bgc_get_angles() once (with retries)
#   - Prints the angles in degrees
#
# Works with either shim signature:
#   (A) int bgc_get_angles(float *yaw, float *pitch, float *roll)
#   (B) int bgc_get_angles(float *pitch, float *yaw)
# ==============================================================

import os
import time
import ctypes
import sys

LIB_PATH = os.path.expanduser(
    "~/senior_design/A-dec-Senior-Design/camera/testing/SerialLibrary/serialAPI/libsimplebgc.so"
)

RETRIES = 30
DELAY_S = 0.05


def load_lib(path: str):
    if not os.path.exists(path):
        print(f"# ERROR: .so not found at: {path}")
        sys.exit(1)

    lib = ctypes.CDLL(path)

    # bgc_init
    if not hasattr(lib, "bgc_init"):
        print("# ERROR: bgc_init symbol not found in library.")
        sys.exit(1)
    lib.bgc_init.argtypes = []
    lib.bgc_init.restype = ctypes.c_int

    # bgc_deinit (optional)
    if hasattr(lib, "bgc_deinit"):
        lib.bgc_deinit.argtypes = []
        lib.bgc_deinit.restype = None

    # bgc_get_angles (required for this test)
    if not hasattr(lib, "bgc_get_angles"):
        print("# ERROR: bgc_get_angles symbol not found in library.")
        print("#        That means your shim was not compiled/linked into libsimplebgc.so.")
        sys.exit(1)

    return lib


def try_get_angles_3(lib):
    """
    Try signature: int bgc_get_angles(float *yaw, float *pitch, float *roll)
    Returns (rc, yaw, pitch, roll)
    """
    lib.bgc_get_angles.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.bgc_get_angles.restype = ctypes.c_int

    yaw = ctypes.c_float()
    pitch = ctypes.c_float()
    roll = ctypes.c_float()
    rc = lib.bgc_get_angles(ctypes.byref(yaw), ctypes.byref(pitch), ctypes.byref(roll))
    return rc, float(yaw.value), float(pitch.value), float(roll.value)


def try_get_angles_2(lib):
    """
    Try signature: int bgc_get_angles(float *pitch, float *yaw)
    Returns (rc, yaw, pitch, roll=0.0)
    """
    lib.bgc_get_angles.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.bgc_get_angles.restype = ctypes.c_int

    pitch = ctypes.c_float()
    yaw = ctypes.c_float()
    rc = lib.bgc_get_angles(ctypes.byref(pitch), ctypes.byref(yaw))
    return rc, float(yaw.value), float(pitch.value), 0.0


def get_angles_once(lib, retries=RETRIES, delay_s=DELAY_S):
    """
    Tries to read angles with retries.
    Attempts 3-float signature first; if it errors, falls back to 2-float.
    Returns tuple: (yaw, pitch, roll) or None if failed.
    """
    last_rc = None
    mode = None

    for attempt in range(1, retries + 1):
        try:
            rc, yaw, pitch, roll = try_get_angles_3(lib)
            mode = "3-float (yaw,pitch,roll)"
        except Exception:
            # Could be wrong arg signature; try 2-float
            try:
                rc, yaw, pitch, roll = try_get_angles_2(lib)
                mode = "2-float (pitch,yaw)"
            except Exception as e2:
                print(f"# get_angles: call failed (attempt {attempt}/{retries}): {e2}")
                time.sleep(delay_s)
                continue

        last_rc = rc
        if rc == 0:
            if attempt == 1:
                print(f"# bgc_get_angles OK on attempt {attempt} using {mode}")
            else:
                print(f"# bgc_get_angles OK on attempt {attempt} using {mode}")
            return (yaw, pitch, roll)

        # Print a few failures so you can see what rc is
        if attempt <= 5 or attempt == retries:
            print(f"# bgc_get_angles rc={rc} (attempt {attempt}/{retries}) using {mode}")
        time.sleep(delay_s)

    print(f"# get_angles: FAILED after {retries} attempts (last rc={last_rc})")
    return None


def main():
    print(f"# Loading: {LIB_PATH}")
    lib = load_lib(LIB_PATH)

    print("# Calling bgc_init() ...")
    rc = lib.bgc_init()
    print(f"# bgc_init rc={rc}")
    if rc != 0:
        print("# ERROR: bgc_init failed. Fix comm/config first before angle polling.")
        sys.exit(2)

    print("# Polling angles once ...")
    angles = get_angles_once(lib)
    if angles is None:
        print("# ERROR: Could not poll angles.")
        if hasattr(lib, "bgc_deinit"):
            lib.bgc_deinit()
        sys.exit(3)

    yaw, pitch, roll = angles
    print("# -------------------")
    print(f"# YAW   = {yaw:.2f} deg")
    print(f"# PITCH = {pitch:.2f} deg")
    print(f"# ROLL  = {roll:.2f} deg")
    print("# -------------------")

    if hasattr(lib, "bgc_deinit"):
        lib.bgc_deinit()
    print("# Done.")


if __name__ == "__main__":
    main()
