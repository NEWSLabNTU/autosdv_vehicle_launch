#!/usr/bin/env python3
"""
Test motor PWM threshold - gradually increase PWM to find when motor starts moving.
Press Ctrl+C when you see the motor start moving.

Usage: python3 test_motor_threshold.py --car-profile car_a|car_b|car_c
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

def test_motor_threshold(profile):
    """Gradually increase motor PWM to find movement threshold."""
    # Initialize PCA9685
    driver = PCA9685(address=profile['i2c_address'], busnum=profile['i2c_busnum'])
    driver.set_pwm_freq(profile['pwm_freq'])
    
    # Motor control values from the car's profile
    init_pwm = profile['init_pwm']      # Stopped position (neutral)
    max_pwm = profile['max_pwm']        # Maximum forward
    init_steer = profile['init_steer']  # Center steering
    
    # Center the steering for safety
    driver.set_pwm(1, 0, init_steer)
    print(f"Steering centered at PWM {init_steer}")
    
    print("\n=== Motor Threshold Test ===")
    print(f"Starting from PWM {init_pwm} (stopped)")
    print("Will increase by 2 every 0.5 seconds")
    print("Press Ctrl+C when motor starts moving\n")
    
    current_pwm = init_pwm
    step = 2
    delay = 0.5  # seconds between steps
    
    try:
        while current_pwm <= max_pwm:
            print(f"Setting motor PWM to: {current_pwm}")
            driver.set_pwm(0, 0, current_pwm)
            time.sleep(delay)
            current_pwm += step
            
        print(f"\nReached maximum PWM ({max_pwm})")
        
    except KeyboardInterrupt:
        print(f"\n\nMotor started moving at approximately PWM: {current_pwm - step}")
        print("(Last value before Ctrl+C)")
        
    finally:
        # Always stop the motor when done
        print("\nStopping motor...")
        driver.set_pwm(0, 0, init_pwm)
        print(f"Motor stopped at PWM {init_pwm}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find the motor movement threshold')
    car_profile.add_argument(parser)
    args = parser.parse_args()
    test_motor_threshold(car_profile.load(args.car_profile))