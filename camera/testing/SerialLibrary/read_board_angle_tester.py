#!/usr/bin/env python3

import ctypes
import time
import os

LIB_PATH = os.path.expanduser(
    "~/senior_design/A-dec-Senior-Design/camera/testing/SerialLibrary/serialAPI/libsimplebgc.so"
)

motor_library = ctypes.CDLL(LIB_PATH)

motor_library.bgc_init.argtypes = []
motor_library.bgc_init.restype = ctypes.c_int

motor_library.bgc_set_motors.argtypes = [ctypes.c_int]
motor_library.bgc_set_motors.restype = ctypes.c_int

motor_library.bgc_control_angles.argtypes = [
    ctypes.c_float, ctypes.c_float, ctypes.c_float
]
motor_library.bgc_control_angles.restype = ctypes.c_int

motor_library.bgc_get_angles.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
]
motor_library.bgc_get_angles.restype = ctypes.c_int

motor_library.bgc_get_target_angles.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
]
motor_library.bgc_get_target_angles.restype = ctypes.c_int

motor_library.bgc_deinit.argtypes = []
motor_library.bgc_deinit.restype = None


def get_angles():
    roll = ctypes.c_float()
    pitch = ctypes.c_float()
    yaw = ctypes.c_float()

    rc = motor_library.bgc_get_angles(
        ctypes.byref(roll),
        ctypes.byref(pitch),
        ctypes.byref(yaw),
    )
    if rc != 0:
        raise RuntimeError(f"bgc_get_angles failed with rc={rc}")

    return roll.value, pitch.value, yaw.value


def get_target_angles():
    roll = ctypes.c_float()
    pitch = ctypes.c_float()
    yaw = ctypes.c_float()

    rc = motor_library.bgc_get_target_angles(
        ctypes.byref(roll),
        ctypes.byref(pitch),
        ctypes.byref(yaw),
    )
    if rc != 0:
        raise RuntimeError(f"bgc_get_target_angles failed with rc={rc}")

    return roll.value, pitch.value, yaw.value


def command_angles(roll_deg, pitch_deg, yaw_deg):
    rc = motor_library.bgc_control_angles(
        ctypes.c_float(roll_deg),
        ctypes.c_float(pitch_deg),
        ctypes.c_float(yaw_deg),
    )
    if rc != 0:
        raise RuntimeError(f"bgc_control_angles failed with rc={rc}")


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
            print(f"\n=== Test {idx} ===")
            print(f"Commanding angles: roll={cmd_r:.2f}, pitch={cmd_p:.2f}, yaw={cmd_y:.2f}")

            command_angles(cmd_r, cmd_p, cmd_y)

            # Give the controller some time to react
            time.sleep(1.5)

            for sample_idx in range(10):
                tgt_r, tgt_p, tgt_y = get_target_angles()
                cur_r, cur_p, cur_y = get_angles()

                print(
                    f"[{sample_idx:02d}] "
                    f"CMD    = ({cmd_r:+7.2f}, {cmd_p:+7.2f}, {cmd_y:+7.2f}) | "
                    f"TARGET = ({tgt_r:+7.2f}, {tgt_p:+7.2f}, {tgt_y:+7.2f}) | "
                    f"READ   = ({cur_r:+7.2f}, {cur_p:+7.2f}, {cur_y:+7.2f})"
                )
                time.sleep(0.2)

    finally:
        try:
            motor_library.bgc_set_motors(0)
        except Exception:
            pass
        motor_library.bgc_deinit()


if __name__ == "__main__":
    main()
