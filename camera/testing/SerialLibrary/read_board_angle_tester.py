#!/usr/bin/env python3

import ctypes
import time
import os

LIB_PATH = os.path.expanduser(
    "~/senior_design/A-dec-Senior-Design/camera/testing/SerialLibrary/serialAPI/libsimplebgc.so"
)

motor_library = ctypes.CDLL(LIB_PATH)

# ----------- Function bindings -----------

motor_library.bgc_init.argtypes = []
motor_library.bgc_init.restype = ctypes.c_int

motor_library.bgc_set_motors.argtypes = [ctypes.c_int]
motor_library.bgc_set_motors.restype = ctypes.c_int

motor_library.bgc_control_angles.argtypes = [
    ctypes.c_float, ctypes.c_float, ctypes.c_float
]
motor_library.bgc_control_angles.restype = ctypes.c_int

# RAW angle readers (NEW)
motor_library.bgc_get_angles_raw.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
]
motor_library.bgc_get_angles_raw.restype = ctypes.c_int

motor_library.bgc_get_target_angles_raw.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
]
motor_library.bgc_get_target_angles_raw.restype = ctypes.c_int

motor_library.bgc_deinit.argtypes = []
motor_library.bgc_deinit.restype = None


# ----------- Helpers -----------

def get_angles_raw():
    roll = ctypes.c_float()
    pitch = ctypes.c_float()
    yaw = ctypes.c_float()

    rc = motor_library.bgc_get_angles_raw(
        ctypes.byref(roll),
        ctypes.byref(pitch),
        ctypes.byref(yaw),
    )
    if rc != 0:
        raise RuntimeError(f"bgc_get_angles_raw failed with rc={rc}")

    return roll.value, pitch.value, yaw.value


def get_target_angles_raw():
    roll = ctypes.c_float()
    pitch = ctypes.c_float()
    yaw = ctypes.c_float()

    rc = motor_library.bgc_get_target_angles_raw(
        ctypes.byref(roll),
        ctypes.byref(pitch),
        ctypes.byref(yaw),
    )
    if rc != 0:
        raise RuntimeError(f"bgc_get_target_angles_raw failed with rc={rc}")

    return roll.value, pitch.value, yaw.value


def command_angles(roll, pitch, yaw):
    rc = motor_library.bgc_control_angles(
        ctypes.c_float(roll),
        ctypes.c_float(pitch),
        ctypes.c_float(yaw),
    )
    if rc != 0:
        raise RuntimeError(f"bgc_control_angles failed with rc={rc}")


# ----------- Main -----------

def main():
    rc = motor_library.bgc_init()
    if rc != 0:
        raise RuntimeError(f"bgc_init failed with rc={rc}")

    try:
        rc = motor_library.bgc_set_motors(1)
        if rc != 0:
            raise RuntimeError(f"bgc_set_motors(1) failed with rc={rc}")

        tests = [
            (10.0,  -5.0,  15.0),
            (-20.0, 12.0, -30.0),
            (5.0,   18.0,  40.0),
        ]

        for idx, (cmd_r, cmd_p, cmd_y) in enumerate(tests, start=1):

            command_angles(cmd_r, cmd_p, cmd_y)

            time.sleep(2)

            tgt_r, tgt_p, tgt_y = get_target_angles_raw()
            cur_r, cur_p, cur_y = get_angles_raw()

            print(
                f"TEST {idx}\n"
                f"CMD (input) = ({cmd_r:+7.2f}, {cmd_p:+7.2f}, {cmd_y:+7.2f})\n"
                f"TARGET RAW  = ({tgt_r:+10.2f}, {tgt_p:+10.2f}, {tgt_y:+10.2f})\n"
                f"READ RAW    = ({cur_r:+10.2f}, {cur_p:+10.2f}, {cur_y:+10.2f})\n"
            )

    finally:
        try:
            motor_library.bgc_set_motors(0)
        except Exception:
            pass
        motor_library.bgc_deinit()


if __name__ == "__main__":
    main()
