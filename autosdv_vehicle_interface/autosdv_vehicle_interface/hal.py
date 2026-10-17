#!/usr/bin/env python3
"""
The vehicle interface's two hardware touchpoints, and their simulated backends.

On the car the actuator writes PWM through an Adafruit PCA9685 and
velocity_report times hall-sensor edges through Jetson.GPIO. With
``hardware: sim`` the same nodes run against the autosdv_rc_sim plant
instead, and nothing else in them changes:

    actuator ── set_pwm ──> SimPCA9685 ── /rc_sim/pwm ──> plant
    plant ── /rc_sim/hall_edges ──> SimGPIO ── callback(pin, stamp_ns) ──> velocity_report

Each simulated edge carries its true time on the simulation clock, so
message latency does not become speed noise. The hardware libraries are
imported only on the real path, so the package imports on a laptop.
"""
from typing import Callable

HARDWARE_REAL = 'real'
HARDWARE_SIM = 'sim'
PWM_TOPIC = '/rc_sim/pwm'
EDGE_TOPIC = '/rc_sim/hall_edges'


def check_hardware(value: str) -> str:
    """Return ``value`` if it names a backend, else raise ValueError."""
    if value not in (HARDWARE_REAL, HARDWARE_SIM):
        raise ValueError(
            f"hardware must be '{HARDWARE_REAL}' or '{HARDWARE_SIM}', got {value!r}")
    return value


class SimPCA9685:
    """Stand-in for ``Adafruit_PCA9685.PCA9685``: the two methods the actuator calls."""

    def __init__(self, publish: Callable[[int, int], None]):
        self._publish = publish
        self.freq = None

    def set_pwm_freq(self, freq) -> None:
        self.freq = int(freq)

    def set_pwm(self, channel, on, off) -> None:
        self._publish(int(channel), int(off))


class SimGPIO:
    """Stand-in for the ``Jetson.GPIO`` module: the three calls velocity_report makes."""

    BOARD = 'BOARD'
    IN = 'IN'
    RISING = 'RISING'

    def __init__(self, node, topic: str = EDGE_TOPIC):
        self._node = node
        self._topic = topic
        self._subscriptions = []

    def setmode(self, mode) -> None:
        pass

    def setup(self, pin, direction) -> None:
        pass

    def add_event_detect(self, pin, edge, callback) -> None:
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from std_msgs.msg import Int64MultiArray

        def on_edges(msg):
            for stamp_ns in msg.data:
                callback(pin, int(stamp_ns))

        qos = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE)
        self._subscriptions.append(
            self._node.create_subscription(Int64MultiArray, self._topic, on_edges, qos))


def make_pwm_driver(params: dict, node=None, pca_class=None):
    """Return the PWM driver for ``params["hardware"]`` (dry runs are the caller's business)."""
    hardware = check_hardware(params.get('hardware', HARDWARE_REAL))
    if hardware == HARDWARE_SIM:
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from std_msgs.msg import Int32MultiArray
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        publisher = node.create_publisher(Int32MultiArray, PWM_TOPIC, qos)

        def publish(channel: int, value: int) -> None:
            publisher.publish(Int32MultiArray(data=[channel, value]))

        driver = SimPCA9685(publish)
    else:
        if pca_class is None:
            from Adafruit_PCA9685 import PCA9685 as pca_class
        driver = pca_class(address=params['i2c_address'], busnum=params['i2c_busnum'])
    driver.set_pwm_freq(params['pwm_freq'])
    return driver


def make_gpio(hardware: str, node=None):
    """Return the GPIO module for ``hardware``: Jetson.GPIO, or a SimGPIO on ``node``."""
    if check_hardware(hardware) == HARDWARE_SIM:
        return SimGPIO(node)
    import Jetson.GPIO as GPIO
    return GPIO
