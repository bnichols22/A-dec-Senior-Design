import sys
import traceback

import board
import busio
from adafruit_ads1x15 import ADS1115, AnalogIn, ads1x15


I2C_ADDRESS = 0x48


def fail(stage, error):
    print(f"[FAIL] {stage}: {error}")
    traceback.print_exc()
    sys.exit(1)


def main():
    print("[STEP] Creating I2C bus")
    try:
        i2c = busio.I2C(board.SCL, board.SDA)
        print("[OK] I2C bus created")
    except Exception as error:
        fail("i2c_init", error)

    print(f"[STEP] Creating ADS1115 at address 0x{I2C_ADDRESS:02X}")
    try:
        ads = ADS1115(i2c, address=I2C_ADDRESS)
        ads.gain = 1
        print("[OK] ADS1115 initialized")
    except Exception as error:
        fail("ads_init", error)

    print("[STEP] Creating channel objects")
    try:
        ch0 = AnalogIn(ads, ads1x15.Pin.A0)
        ch1 = AnalogIn(ads, ads1x15.Pin.A1)
        ch2 = AnalogIn(ads, ads1x15.Pin.A2)
        ch3 = AnalogIn(ads, ads1x15.Pin.A3)
        print("[OK] Channel objects created")
    except Exception as error:
        fail("channel_init", error)

    print("[STEP] Reading voltages")
    try:
        print(
            f"A0: {ch0.voltage:.5f} V | "
            f"A1: {ch1.voltage:.5f} V | "
            f"A2: {ch2.voltage:.5f} V | "
            f"A3: {ch3.voltage:.5f} V"
        )
        print("[OK] Voltage read succeeded")
    except Exception as error:
        fail("voltage_read", error)

    print("[DONE] ADS1115 diagnostic completed successfully")


if __name__ == "__main__":
    main()
