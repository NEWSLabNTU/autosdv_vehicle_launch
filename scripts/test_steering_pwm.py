#!/usr/bin/env python3
"""
Direct PWM control script for testing steering center position.
Usage: python3 test_steering_pwm.py --car-profile car_a|car_b|car_c [pwm_value]
"""

import time
import argparse
import os
import sys

# Run as a plain script: put the package source (a sibling of scripts/) on sys.path.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'autosdv_vehicle_interface'))
from autosdv_vehicle_interface import car_profile  # noqa: E402
import rclpy
from rclpy.node import Node
from Adafruit_PCA9685 import PCA9685
from std_msgs.msg import Int32

class DirectPWMController(Node):
    def __init__(self, profile):
        super().__init__('direct_pwm_controller')
        
        # Initialize PCA9685
        self.driver = PCA9685(address=profile['i2c_address'],
                             busnum=profile['i2c_busnum'])
        self.driver.set_pwm_freq(profile['pwm_freq'])
        # Steering is clamped to the car's own servo limits
        self.steer_min = profile['min_steer']
        self.steer_max = profile['max_steer']
        
        # Current PWM values
        self.motor_pwm = profile['init_pwm']  # neutral = stopped motor
        self.steer_pwm = profile['init_steer']  # start centred
        
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
        new_pwm = max(self.steer_min, min(self.steer_max, new_pwm))
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
        pwm_value = max(self.steer_min, min(self.steer_max, pwm_value))
        self.steer_pwm = pwm_value
        self.get_logger().info(f'Setting steering PWM to: {self.steer_pwm}')
        self.update_pwm()
        

def main(args=None):
    parser = argparse.ArgumentParser(description='Direct steering PWM test')
    car_profile.add_argument(parser)
    parser.add_argument('pwm_value', nargs='?', type=int,
                        help='set once and exit; omit to follow /test/steering_pwm')
    opts, ros_args = parser.parse_known_args(args)
    rclpy.init(args=ros_args)

    controller = DirectPWMController(car_profile.load(opts.car_profile))

    # If PWM value provided as argument, set it directly
    if opts.pwm_value is not None:
        try:
            pwm_value = opts.pwm_value
            controller.set_pwm_direct(pwm_value)
            # Run for a short time then exit
            for _ in range(20):  # Run for 2 seconds
                rclpy.spin_once(controller, timeout_sec=0.1)
            controller.get_logger().info(f'PWM {pwm_value} set. Exiting.')
        except ValueError:
            controller.get_logger().error(f'Invalid PWM value: {opts.pwm_value}')
    else:
        # Run continuously and listen for topic commands
        controller.get_logger().info('Running in continuous mode. Use topic /test/steering_pwm')
        rclpy.spin(controller)
    
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()