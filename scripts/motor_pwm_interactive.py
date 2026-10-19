#!/usr/bin/env python3
"""
Usage: python3 motor_pwm_interactive.py --car-profile car_a|car_b|car_c

Interactive motor PWM control - find the exact stop point and movement thresholds.
Keys respond immediately without pressing Enter.
"""

import termios
import tty
import signal
import argparse
import os
import sys

# Run as a plain script: put the package source (a sibling of scripts/) on sys.path.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'autosdv_vehicle_interface'))
from autosdv_vehicle_interface import car_profile  # noqa: E402
from Adafruit_PCA9685 import PCA9685

class MotorPWMController:
    def __init__(self, profile):
        # Initialize PCA9685
        self.driver = PCA9685(address=profile['i2c_address'],
                             busnum=profile['i2c_busnum'])
        self.driver.set_pwm_freq(profile['pwm_freq'])

        # PWM values from the car's profile
        self.neutral = profile['init_pwm']  # stop position
        self.brake = profile['brake_pwm']
        self.motor_pwm = self.neutral
        self.steer_pwm = profile['init_steer']  # Center steering

        # PWM limits: 140 below / 90 above neutral, never past the car's max_pwm
        self.min_pwm = self.neutral - 140
        self.max_pwm = min(self.neutral + 90, profile['max_pwm'])
        
        # Center the steering
        self.driver.set_pwm(1, 0, self.steer_pwm)
        
        # Save terminal settings
        self.old_settings = termios.tcgetattr(sys.stdin)
        
        # Setup signal handler for Ctrl-C
        signal.signal(signal.SIGINT, self.signal_handler)
        self.running = True
        
    def signal_handler(self, sig, frame):
        """Handle Ctrl-C signal."""
        self.running = False
        print("\n\nCtrl-C detected, stopping...")
        
    def set_motor_pwm(self, value):
        """Set motor PWM value with safety limits."""
        self.motor_pwm = max(self.min_pwm, min(self.max_pwm, value))
        self.driver.set_pwm(0, 0, self.motor_pwm)
        
    def get_single_char(self):
        """Get a single character from stdin without waiting for Enter."""
        try:
            tty.setraw(sys.stdin.fileno())
            char = sys.stdin.read(1)
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
        return char
        
    def run_interactive(self):
        """Run interactive control loop."""
        print("=== Interactive Motor PWM Control ===")
        n, b = self.neutral, self.brake
        print(f"\nIMPORTANT: Stop position is {n} (brake is {b})")
        print(f"Forward: PWM > {n}")
        print(f"Reverse: PWM < {n}")
        print(f"Brake from forward: Set to {b}, then back to {n}")
        print("\nControls (immediate response):")
        print("  w/W  : Increase by 1 / 10")
        print("  s/S  : Decrease by 1 / 10")
        print("  e/E  : Increase by 5 / 20")
        print("  d/D  : Decrease by 5 / 20")
        print(f"  r    : Reset to {n} (STOP)")
        print(f"  t    : Test brake ({b})")
        print(f"  1-9  : Jump to {n-40}, {n-30}, {n-20}... {n+40}")
        print(f"  0    : Jump to {n+50}")
        print(f"  f    : Forward test ({n+10})")
        print(f"  b    : Backward test ({n-10})")
        print("  q    : Quit")
        print("  Ctrl-C: Emergency stop")
        print("\n" + "="*40)
        
        # Set initial PWM
        self.set_motor_pwm(self.motor_pwm)
        print(f"\nStarting PWM: {self.motor_pwm}")
        print("\nPress any control key...")
        
        try:
            while self.running:
                # Get single character input
                char = self.get_single_char()
                
                if not self.running:  # Check if Ctrl-C was pressed
                    break
                
                if char == 'q' or char == 'Q':
                    print(f"\nQuitting... Final PWM was: {self.motor_pwm}")
                    break
                    
                # Fine adjustments
                elif char == 'w':
                    self.set_motor_pwm(self.motor_pwm + 1)
                    print(f"\rPWM: {self.motor_pwm} (+1)      ", end='', flush=True)
                    
                elif char == 's':
                    self.set_motor_pwm(self.motor_pwm - 1)
                    print(f"\rPWM: {self.motor_pwm} (-1)      ", end='', flush=True)
                    
                # Medium adjustments
                elif char == 'e':
                    self.set_motor_pwm(self.motor_pwm + 5)
                    print(f"\rPWM: {self.motor_pwm} (+5)      ", end='', flush=True)
                    
                elif char == 'd':
                    self.set_motor_pwm(self.motor_pwm - 5)
                    print(f"\rPWM: {self.motor_pwm} (-5)      ", end='', flush=True)
                    
                # Large adjustments
                elif char == 'W':
                    self.set_motor_pwm(self.motor_pwm + 10)
                    print(f"\rPWM: {self.motor_pwm} (+10)     ", end='', flush=True)
                    
                elif char == 'S':
                    self.set_motor_pwm(self.motor_pwm - 10)
                    print(f"\rPWM: {self.motor_pwm} (-10)     ", end='', flush=True)
                    
                # Extra large adjustments
                elif char == 'E':
                    self.set_motor_pwm(self.motor_pwm + 20)
                    print(f"\rPWM: {self.motor_pwm} (+20)     ", end='', flush=True)
                    
                elif char == 'D':
                    self.set_motor_pwm(self.motor_pwm - 20)
                    print(f"\rPWM: {self.motor_pwm} (-20)     ", end='', flush=True)
                    
                # Reset to stop (neutral)
                elif char == 'r' or char == 'R':
                    self.set_motor_pwm(self.neutral)
                    print(f"\rPWM: {self.motor_pwm} (STOP)    ", end='', flush=True)
                    
                # Test brake position
                elif char == 't' or char == 'T':
                    self.set_motor_pwm(self.brake)
                    print(f"\rPWM: {self.motor_pwm} (BRAKE)   ", end='', flush=True)
                    
                # Quick jumps to specific values (offsets from neutral: 330..420 on car A)
                elif char == '1':
                    self.set_motor_pwm(self.neutral-40)
                    print(f"\rPWM: {self.motor_pwm} (jump)    ", end='', flush=True)
                elif char == '2':
                    self.set_motor_pwm(self.neutral-30)
                    print(f"\rPWM: {self.motor_pwm} (jump)    ", end='', flush=True)
                elif char == '3':
                    self.set_motor_pwm(self.neutral-20)
                    print(f"\rPWM: {self.motor_pwm} (jump)    ", end='', flush=True)
                elif char == '4':
                    self.set_motor_pwm(self.neutral-10)
                    print(f"\rPWM: {self.motor_pwm} (jump)    ", end='', flush=True)
                elif char == '5':
                    self.set_motor_pwm(self.neutral)
                    print(f"\rPWM: {self.motor_pwm} (STOP)    ", end='', flush=True)
                elif char == '6':
                    self.set_motor_pwm(self.neutral+10)
                    print(f"\rPWM: {self.motor_pwm} (jump)    ", end='', flush=True)
                elif char == '7':
                    self.set_motor_pwm(self.neutral+20)
                    print(f"\rPWM: {self.motor_pwm} (jump)    ", end='', flush=True)
                elif char == '8':
                    self.set_motor_pwm(self.neutral+30)
                    print(f"\rPWM: {self.motor_pwm} (jump)    ", end='', flush=True)
                elif char == '9':
                    self.set_motor_pwm(self.neutral+40)
                    print(f"\rPWM: {self.motor_pwm} (jump)    ", end='', flush=True)
                elif char == '0':
                    self.set_motor_pwm(self.neutral+50)
                    print(f"\rPWM: {self.motor_pwm} (jump)    ", end='', flush=True)
                    
                # Test positions
                elif char == 'f' or char == 'F':
                    self.set_motor_pwm(self.neutral + 10)
                    print(f"\rPWM: {self.motor_pwm} (FWD test)", end='', flush=True)
                elif char == 'b' or char == 'B':
                    self.set_motor_pwm(self.neutral - 10)
                    print(f"\rPWM: {self.motor_pwm} (REV test)", end='', flush=True)
                    
                else:
                    print(f"\rPWM: {self.motor_pwm} (unknown: {char})", end='', flush=True)
                    
        except Exception as e:
            print(f"\n\nError: {e}")
            
        finally:
            # Restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
            
            # Always stop motor on exit (neutral is stop)
            print(f"\nStopping motor at PWM {self.neutral}...")
            self.driver.set_pwm(0, 0, self.neutral)
            print("Motor stopped.")

def main():
    parser = argparse.ArgumentParser(description='Interactive motor PWM control')
    car_profile.add_argument(parser)
    args = parser.parse_args()
    controller = MotorPWMController(car_profile.load(args.car_profile))
    controller.run_interactive()

if __name__ == '__main__':
    main()