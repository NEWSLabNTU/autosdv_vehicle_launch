#!/usr/bin/env python3
"""
Direct motor stop script - ensures motor is stopped safely.

Usage: python3 stop_motor.py --car-profile car_a|car_b|car_c

The neutral and steering centre are the car's own (params/cars/<car>.param.yaml):
car A stops at 370, car B at 395, and 370 on car B is below neutral = brake/reverse.
"""

import time
import argparse
import os
import sys

# Run as a plain script: put the package source (a sibling of scripts/) on sys.path.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'autosdv_vehicle_interface'))
from autosdv_vehicle_interface import car_profile  # noqa: E402
from Adafruit_PCA9685 import PCA9685


def stop_motor(profile):
    """Stop the motor immediately by setting PWM to the car's neutral."""
    driver = PCA9685(address=profile['i2c_address'], busnum=profile['i2c_busnum'])
    driver.set_pwm_freq(profile['pwm_freq'])

    init_pwm = profile['init_pwm']      # neutral
    init_steer = profile['init_steer']  # steering centre

    print("Stopping motor...")

    # Set motor to stopped position
    driver.set_pwm(0, 0, init_pwm)
    print(f"Motor PWM set to {init_pwm} (stopped)")

    # Also center the steering for safety
    driver.set_pwm(1, 0, init_steer)
    print(f"Steering PWM set to {init_steer} (centered)")

    # Hold for a moment to ensure it takes effect
    time.sleep(0.5)

    print("Motor stopped successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    car_profile.add_argument(parser)
    args = parser.parse_args()
    stop_motor(car_profile.load(args.car_profile))
