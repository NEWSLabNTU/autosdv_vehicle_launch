#!/usr/bin/env python3
"""
Direct motor PWM test - set specific PWM value for testing.
Usage: python3 test_motor_pwm.py <pwm_value>
"""

import sys
import time
from Adafruit_PCA9685 import PCA9685

def test_motor_pwm(pwm_value):
    """Set motor to specific PWM value."""
    # Initialize PCA9685
    driver = PCA9685(address=64, busnum=7)
    driver.set_pwm_freq(60)
    
    init_steer = 400  # Center steering
    
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
    print("Stopping motor at PWM 340")
    driver.set_pwm(0, 0, 340)
    print("Done!")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python3 test_motor_pwm.py <pwm_value>")
        print("Example: python3 test_motor_pwm.py 345")
        sys.exit(1)
    
    try:
        pwm = int(sys.argv[1])
        if pwm < 230 or pwm > 460:
            print(f"Warning: PWM {pwm} is outside normal range (230-460)")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                sys.exit(0)
        test_motor_pwm(pwm)
    except ValueError:
        print(f"Error: '{sys.argv[1]}' is not a valid integer")
        sys.exit(1)