# Copyright 2026 NEWSLab, National Taiwan University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Hardware backends: real and simulated PCA9685 and GPIO. No I2C, no GPIO, no ROS graph."""
import sys

import pytest

from autosdv_vehicle_interface import hal


class FakePub:

    def __init__(self, topic):
        self.topic = topic
        self.sent = []

    def publish(self, msg):
        self.sent.append(list(msg.data))


class FakeNode:

    def __init__(self):
        self.pubs = []
        self.subs = []

    def create_publisher(self, msg_type, topic, qos):
        pub = FakePub(topic)
        self.pubs.append(pub)
        return pub

    def create_subscription(self, msg_type, topic, callback, qos):
        self.subs.append((topic, callback))
        return object()


def test_check_hardware():
    assert hal.check_hardware('real') == 'real'
    assert hal.check_hardware('sim') == 'sim'
    with pytest.raises(ValueError, match='hardware'):
        hal.check_hardware('simulation')


def test_sim_pca9685_forwards_each_write():
    sent = []
    drv = hal.SimPCA9685(lambda ch, value: sent.append((ch, value)))
    drv.set_pwm_freq(60)
    drv.set_pwm(0, 0, 425)
    drv.set_pwm(1, 0, 489)
    assert drv.freq == 60
    assert sent == [(0, 425), (1, 489)]


def test_make_pwm_driver_real_uses_the_given_class(fake_pca):
    params = {'hardware': 'real', 'i2c_address': 64, 'i2c_busnum': 7, 'pwm_freq': 60}
    drv = hal.make_pwm_driver(params, pca_class=fake_pca)
    assert (drv.address, drv.busnum, drv.freq) == (64, 7, 60)


def test_make_pwm_driver_sim_publishes_on_the_pwm_topic():
    pytest.importorskip('std_msgs')
    node = FakeNode()
    params = {'hardware': 'sim', 'i2c_address': 64, 'i2c_busnum': 7, 'pwm_freq': 60}
    drv = hal.make_pwm_driver(params, node=node)
    drv.set_pwm(0, 0, 400)
    assert isinstance(drv, hal.SimPCA9685)
    assert node.pubs[0].topic == hal.PWM_TOPIC
    assert node.pubs[0].sent == [[0, 400]]


def test_sim_gpio_delivers_every_edge_with_its_stamp():
    pytest.importorskip('std_msgs')
    from std_msgs.msg import Int64MultiArray
    node = FakeNode()
    gpio = hal.make_gpio('sim', node=node)
    got = []
    gpio.setmode(gpio.BOARD)
    gpio.setup(15, gpio.IN)
    gpio.add_event_detect(15, gpio.RISING,
                          callback=lambda ch, stamp_ns: got.append((ch, stamp_ns)))
    topic, callback = node.subs[0]
    assert topic == hal.EDGE_TOPIC
    callback(Int64MultiArray(data=[1_000, 2_500]))
    assert got == [(15, 1_000), (15, 2_500)]


def test_real_gpio_is_imported_only_when_asked(monkeypatch):
    monkeypatch.setitem(sys.modules, 'Jetson', None)       # no Jetson.GPIO on this machine
    monkeypatch.delitem(sys.modules, 'Jetson.GPIO', raising=False)
    with pytest.raises(ImportError):
        hal.make_gpio('real')
    assert isinstance(hal.make_gpio('sim', node=FakeNode()), hal.SimGPIO)
