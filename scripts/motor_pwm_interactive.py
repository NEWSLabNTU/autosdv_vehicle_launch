#!/usr/bin/env python3
"""
Interactive motor PWM control - find the exact stop point and movement thresholds.
Keys respond immediately without pressing Enter.
"""

import time
import sys
import termios
import tty
import signal
from Adafruit_PCA9685 import PCA9685

class MotorPWMController:
    def __init__(self):
        # Initialize PCA9685
        self.driver = PCA9685(address=64, busnum=7)
        self.driver.set_pwm_freq(60)
        
        # PWM values - updated based on findings
        self.motor_pwm = 370  # Actual stop position
        self.steer_pwm = 400  # Center steering
        
        # PWM limits
        self.min_pwm = 230
        self.max_pwm = 460
        
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
        print("\nIMPORTANT: Stop position is 370 (not 340)")
        print("Forward: PWM > 370")
        print("Reverse: PWM < 370") 
        print("Brake from forward: Set to 340, then back to 370")
        print("\nControls (immediate response):")
        print("  w/W  : Increase by 1 / 10")
        print("  s/S  : Decrease by 1 / 10")
        print("  e/E  : Increase by 5 / 20")
        print("  d/D  : Decrease by 5 / 20")
        print("  r    : Reset to 370 (STOP)")
        print("  t    : Test brake (340)")
        print("  1-9  : Jump to 330, 340, 350... 410")
        print("  0    : Jump to 420")
        print("  f    : Forward test (380)")
        print("  b    : Backward test (360)")
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
                    
                # Reset to stop (370)
                elif char == 'r' or char == 'R':
                    self.set_motor_pwm(370)
                    print(f"\rPWM: {self.motor_pwm} (STOP)    ", end='', flush=True)
                    
                # Test brake position
                elif char == 't' or char == 'T':
                    self.set_motor_pwm(340)
                    print(f"\rPWM: {self.motor_pwm} (BRAKE)   ", end='', flush=True)
                    
                # Quick jumps to specific values (adjusted for 370 center)
                elif char == '1':
                    self.set_motor_pwm(330)
                    print(f"\rPWM: {self.motor_pwm} (jump)    ", end='', flush=True)
                elif char == '2':
                    self.set_motor_pwm(340)
                    print(f"\rPWM: {self.motor_pwm} (jump)    ", end='', flush=True)
                elif char == '3':
                    self.set_motor_pwm(350)
                    print(f"\rPWM: {self.motor_pwm} (jump)    ", end='', flush=True)
                elif char == '4':
                    self.set_motor_pwm(360)
                    print(f"\rPWM: {self.motor_pwm} (jump)    ", end='', flush=True)
                elif char == '5':
                    self.set_motor_pwm(370)
                    print(f"\rPWM: {self.motor_pwm} (STOP)    ", end='', flush=True)
                elif char == '6':
                    self.set_motor_pwm(380)
                    print(f"\rPWM: {self.motor_pwm} (jump)    ", end='', flush=True)
                elif char == '7':
                    self.set_motor_pwm(390)
                    print(f"\rPWM: {self.motor_pwm} (jump)    ", end='', flush=True)
                elif char == '8':
                    self.set_motor_pwm(400)
                    print(f"\rPWM: {self.motor_pwm} (jump)    ", end='', flush=True)
                elif char == '9':
                    self.set_motor_pwm(410)
                    print(f"\rPWM: {self.motor_pwm} (jump)    ", end='', flush=True)
                elif char == '0':
                    self.set_motor_pwm(420)
                    print(f"\rPWM: {self.motor_pwm} (jump)    ", end='', flush=True)
                    
                # Test positions
                elif char == 'f' or char == 'F':
                    self.set_motor_pwm(380)
                    print(f"\rPWM: {self.motor_pwm} (FWD test)", end='', flush=True)
                elif char == 'b' or char == 'B':
                    self.set_motor_pwm(360)
                    print(f"\rPWM: {self.motor_pwm} (REV test)", end='', flush=True)
                    
                else:
                    print(f"\rPWM: {self.motor_pwm} (unknown: {char})", end='', flush=True)
                    
        except Exception as e:
            print(f"\n\nError: {e}")
            
        finally:
            # Restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
            
            # Always stop motor on exit (use 370 as stop)
            print(f"\nStopping motor at PWM 370...")
            self.driver.set_pwm(0, 0, 370)
            print("Motor stopped.")

def main():
    controller = MotorPWMController()
    controller.run_interactive()

if __name__ == '__main__':
    main()