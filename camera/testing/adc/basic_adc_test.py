import time
import board
import busio

from adafruit_ads1x15 import ADS1115, AnalogIn, ads1x15

# Create I2C bus
i2c = busio.I2C(board.SCL, board.SDA)

# Create ADC object
ads = ADS1115(i2c)

# Set gain for small voltages
ads.gain = 1

# Create channels
ch0 = AnalogIn(ads, ads1x15.Pin.A0)
ch1 = AnalogIn(ads, ads1x15.Pin.A1)
ch2 = AnalogIn(ads, ads1x15.Pin.A2)
ch3 = AnalogIn(ads, ads1x15.Pin.A3)

while True:
    print(
        f"A0: {ch0.voltage:.5f} V | "
        f"A1: {ch1.voltage:.5f} V | "
        f"A2: {ch2.voltage:.5f} V | "
        f"A3: {ch3.voltage:.5f} V"
    )
    time.sleep(2)
