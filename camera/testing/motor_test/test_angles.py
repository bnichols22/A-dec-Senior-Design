#!/usr/bin/env python3
import os
import ctypes

LIB_PATH = os.path.expanduser(
    "~/senior_design/A-dec-Senior-Design/camera/testing/SerialLibrary/serialAPI/libsimplebgc.so"
)

def main():
    lib = ctypes.CDLL(LIB_PATH)

    lib.bgc_init.argtypes = []
    lib.bgc_init.restype = ctypes.c_int

    # bgc_get_angles(float* pitch, float* yaw, float* roll_optional)
    lib.bgc_get_angles.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.bgc_get_angles.restype = ctypes.c_int

    lib.bgc_deinit.argtypes = []
    lib.bgc_deinit.restype = None

    rc = lib.bgc_init()
    print(f"bgc_init rc={rc}")
    if rc != 0:
        return

    pitch = ctypes.c_float()
    yaw = ctypes.c_float()
    roll = ctypes.c_float()

    rc2 = lib.bgc_get_angles(ctypes.byref(pitch), ctypes.byref(yaw), ctypes.byref(roll))
    print(f"bgc_get_angles rc={rc2}")
    if rc2 == 0:
        print(f"POLLED (deg): roll={roll.value:.2f} pitch={pitch.value:.2f} yaw={yaw.value:.2f}")
    else:
        print("Failed to poll angles.")

    lib.bgc_deinit()

if __name__ == "__main__":
    main()
