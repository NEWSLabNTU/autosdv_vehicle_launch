#!/usr/bin/env python3
"""
Direct motor stop script - ensures motor is stopped safely.
"""

import time
from Adafruit_PCA9685 import PCA9685

def stop_motor():
    """Stop the motor immediately by setting PWM to init value."""
    # Initialize PCA9685
    driver = PCA9685(address=64, busnum=7)
    driver.set_pwm_freq(60)
    
    # Motor control values from actuator.yaml
    init_pwm = 370  # Stopped position (updated)
    init_steer = 400  # Center steering
    
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
    stop_motor()