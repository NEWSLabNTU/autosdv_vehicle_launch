#!/usr/bin/env python3
"""
Direct motor PWM test - set specific PWM value for testing.
Usage: python3 test_motor_pwm.py --car-profile car_a|car_b|car_c <pwm_value>
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


def test_motor_pwm(pwm_value, profile):
    """Set motor to specific PWM value, then back to the car's neutral."""
    driver = PCA9685(address=profile['i2c_address'], busnum=profile['i2c_busnum'])
    driver.set_pwm_freq(profile['pwm_freq'])

    init_steer = profile['init_steer']  # steering centre
    init_pwm = profile['init_pwm']      # neutral

    # Center the steering
    driver.set_pwm(1, 0, init_steer)
    print(f"Steering centered at PWM {init_steer}")

    # Set motor PWM
    print(f"Setting motor PWM to: {pwm_value}")
    driver.set_pwm(0, 0, pwm_value)

    # Run for 2 seconds
    print("Running for 2 seconds...")
    time.sleep(2)

    # Stop motor
    print(f"Stopping motor at PWM {init_pwm}")
    driver.set_pwm(0, 0, init_pwm)
    print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    car_profile.add_argument(parser)
    parser.add_argument('pwm_value', type=int)
    args = parser.parse_args()
    profile = car_profile.load(args.car_profile)
    pwm = args.pwm_value
    # Normal range: 140 below to 90 above the car's neutral (230-460 on car A).
    lo, hi = profile['init_pwm'] - 140, profile['init_pwm'] + 90
    if pwm < lo or pwm > hi:
        print(f"Warning: PWM {pwm} is outside normal range ({lo}-{hi})")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    test_motor_pwm(pwm, profile)
