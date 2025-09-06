#!/usr/bin/env python3
"""
Test motor PWM threshold - gradually increase PWM to find when motor starts moving.
Press Ctrl+C when you see the motor start moving.
"""

import time
import sys
from Adafruit_PCA9685 import PCA9685

def test_motor_threshold():
    """Gradually increase motor PWM to find movement threshold."""
    # Initialize PCA9685
    driver = PCA9685(address=64, busnum=7)
    driver.set_pwm_freq(60)
    
    # Motor control values from actuator.yaml
    init_pwm = 340  # Stopped position
    max_pwm = 460   # Maximum forward
    init_steer = 400  # Center steering
    
    # Center the steering for safety
    driver.set_pwm(1, 0, init_steer)
    print(f"Steering centered at PWM {init_steer}")
    
    print("\n=== Motor Threshold Test ===")
    print("Starting from PWM 340 (stopped)")
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
    test_motor_threshold()