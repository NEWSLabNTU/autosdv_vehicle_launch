#!/usr/bin/env python3
"""
Direct PWM control script for testing steering center position.
Usage: python3 test_steering_pwm.py <pwm_value>
"""

import sys
import time
import rclpy
from rclpy.node import Node
from Adafruit_PCA9685 import PCA9685
from std_msgs.msg import Int32

class DirectPWMController(Node):
    def __init__(self):
        super().__init__('direct_pwm_controller')
        
        # Initialize PCA9685
        self.driver = PCA9685(address=64, busnum=7)
        self.driver.set_pwm_freq(60)
        
        # Current PWM values
        self.motor_pwm = 340  # init_pwm value for stopped motor
        self.steer_pwm = 430  # Current test value
        
        # Create subscriber for PWM commands
        self.pwm_sub = self.create_subscription(
            Int32,
            '/test/steering_pwm',
            self.pwm_callback,
            10
        )
        
        # Create timer to update PWM
        self.timer = self.create_timer(0.1, self.update_pwm)
        
        self.get_logger().info(f'Direct PWM Controller started. Initial steering PWM: {self.steer_pwm}')
        self.get_logger().info('Publish to /test/steering_pwm to change steering PWM value')
        
    def pwm_callback(self, msg):
        """Handle incoming PWM commands."""
        new_pwm = msg.data
        # Clamp to valid range
        new_pwm = max(350, min(590, new_pwm))
        self.steer_pwm = new_pwm
        self.get_logger().info(f'Setting steering PWM to: {self.steer_pwm}')
        
    def update_pwm(self):
        """Update PWM outputs."""
        # Channel 0: Motor (keep stopped)
        self.driver.set_pwm(0, 0, self.motor_pwm)
        # Channel 1: Steering
        self.driver.set_pwm(1, 0, self.steer_pwm)
        
    def set_pwm_direct(self, pwm_value):
        """Set PWM directly from command line argument."""
        pwm_value = max(350, min(590, pwm_value))
        self.steer_pwm = pwm_value
        self.get_logger().info(f'Setting steering PWM to: {self.steer_pwm}')
        self.update_pwm()
        

def main(args=None):
    rclpy.init(args=args)
    
    controller = DirectPWMController()
    
    # If PWM value provided as argument, set it directly
    if len(sys.argv) > 1:
        try:
            pwm_value = int(sys.argv[1])
            controller.set_pwm_direct(pwm_value)
            # Run for a short time then exit
            for _ in range(20):  # Run for 2 seconds
                rclpy.spin_once(controller, timeout_sec=0.1)
            controller.get_logger().info(f'PWM {pwm_value} set. Exiting.')
        except ValueError:
            controller.get_logger().error(f'Invalid PWM value: {sys.argv[1]}')
    else:
        # Run continuously and listen for topic commands
        controller.get_logger().info('Running in continuous mode. Use topic /test/steering_pwm')
        rclpy.spin(controller)
    
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()