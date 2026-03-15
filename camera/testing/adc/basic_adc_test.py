import time
import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn

# Create I2C bus
i2c = busio.I2C(board.SCL, board.SDA)

# Create ADC object
ads = ADS.ADS1115(i2c)

# Set gain for small voltages
ads.gain = 16

# Create channels
ch0 = AnalogIn(ads, ADS.P0)
ch1 = AnalogIn(ads, ADS.P1)
ch2 = AnalogIn(ads, ADS.P2)
ch3 = AnalogIn(ads, ADS.P3)

while True:
    print(
        f"A0: {ch0.voltage:.5f} V | "
        f"A1: {ch1.voltage:.5f} V | "
        f"A2: {ch2.voltage:.5f} V | "
        f"A3: {ch3.voltage:.5f} V"
    )
    time.sleep(0.5)
