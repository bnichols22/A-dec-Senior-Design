import time
import board
import busio

from adafruit_ads1x15 import ADS1115, AnalogIn, ads1x15

# ---------------- Settings ----------------
OFF_THRESHOLD = 2.0

# ---------------- ADC Setup ----------------
i2c = busio.I2C(board.SCL, board.SDA)
ads = ADS1115(i2c)

ads.gain = 1  # safe for ~1.4V signals

# Channels
ch0 = AnalogIn(ads, ads1x15.Pin.A0)
ch1 = AnalogIn(ads, ads1x15.Pin.A1)
ch2 = AnalogIn(ads, ads1x15.Pin.A2)
ch3 = AnalogIn(ads, ads1x15.Pin.A3)


def read_light_mode():
    v0 = ch0.voltage
    v1 = ch1.voltage
    v2 = ch2.voltage
    v3 = ch3.voltage

    if v0 < OFF_THRESHOLD:
        return "YELLOW_LIGHT"

    if v1 < OFF_THRESHOLD:
        return "LOWEST_LIGHT"

    if v2 < OFF_THRESHOLD:
        return "MEDIUM_LIGHT"

    if v3 < OFF_THRESHOLD:
        return "HIGHEST_LIGHT"

    return "LIGHT_OFF"


while True:

    mode = read_light_mode()

    print(
        f"MODE={mode}"
    )

    time.sleep(2)
