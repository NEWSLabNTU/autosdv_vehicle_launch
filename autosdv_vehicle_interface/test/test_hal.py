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

from autosdv_vehicle_interface import hal
import pytest


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


def test_sim_without_a_node_is_a_clear_error():
    with pytest.raises(ValueError, match='pass the node'):
        hal.make_pwm_driver({'hardware': 'sim'})
    with pytest.raises(ValueError, match='pass the node'):
        hal.make_gpio('sim')


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


# ---------------------------------------------------------------- lockstep (sim only)
def test_ack_round_trip():
    text = hal.format_ack('actuator', 12_050_000_000, 10_000_000, report=12_000_000_000, pwm=400)
    assert text == 'actuator 12 50000000 10000000 pwm=400 report=12000000000'
    assert hal.parse_ack(text) == ('actuator', 12_050_000_000, 10_000_000,
                                   {'pwm': 400, 'report': 12_000_000_000})
    assert hal.parse_ack('velocity_report 0 0 50000000') == ('velocity_report', 0, 50_000_000, {})


@pytest.mark.parametrize('text', ['', 'actuator 1 2', 'actuator 1 2 0', 'actuator a 2 3',
                                  'actuator 1 2 3 report'])
def test_malformed_ack(text):
    with pytest.raises(ValueError):
        hal.parse_ack(text)


class FakeTimer:

    def __init__(self, period_ns, until_ns):
        self.timer_period_ns = period_ns
        self._until = until_ns

    def time_until_next_call(self):
        return self._until


def test_timer_deadline_is_the_latest_at_or_before_now():
    # deadline at 100 ms, the callback runs at 103 ms: next call at 110 ms
    assert hal.timer_deadline_ns(FakeTimer(10_000_000, 7_000_000), 103_000_000) == 100_000_000
    assert hal.timer_deadline_ns(FakeTimer(10_000_000, 10_000_000), 100_000_000) == 100_000_000
    assert hal.timer_deadline_ns(FakeTimer(10_000_000, None), 5) == 5     # canceled


def test_report_gate_holds_a_tick_until_its_report():
    gate = hal.ReportGate()
    assert not gate.awaiting(0)                         # velocity_report not known yet
    gate.on_ack('actuator', 0, 10_000_000)              # other participants ignored
    assert not gate.awaiting(0)
    gate.on_ack('velocity_report', 1_000_000_000, 50_000_000)
    assert gate.expected_ns(1_040_000_000) == 1_000_000_000
    assert gate.expected_ns(1_050_000_000) == 1_050_000_000   # predicted, not yet acked
    assert gate.awaiting(1_000_000_000)                 # no report received at all
    gate.on_report(1_000_000_000)
    assert not gate.awaiting(1_000_000_000)
    assert not gate.awaiting(1_040_000_000)
    assert gate.awaiting(1_050_000_000)                 # equal instant: the report goes first
    gate.on_report(1_050_000_000)
    assert not gate.awaiting(1_050_000_000)
    gate.on_report(1_000_000_000)                       # an old stamp never moves it back
    assert gate.last_report_ns == 1_050_000_000


def test_real_hardware_never_takes_part():
    assert hal.make_lockstep_participant('real', FakeNode(), hal.ACTUATOR) is None


def test_sim_participant_acks_and_follows_status():
    import json
    node = FakeNode()
    gate = hal.ReportGate()
    seen = []
    part = hal.make_lockstep_participant('sim', node, hal.ACTUATOR, gate, seen.append)
    [ack_pub] = node.pubs
    assert ack_pub.topic == hal.LOCKSTEP_ACK_TOPIC
    topics = dict(node.subs)
    assert set(topics) == {hal.LOCKSTEP_STATUS_TOPIC, hal.LOCKSTEP_ACK_TOPIC}

    class Msg:

        def __init__(self, data):
            self.data = data

    assert not part.active
    topics[hal.LOCKSTEP_STATUS_TOPIC](Msg(json.dumps({'ok': True, 'enabled': True})))
    assert part.active and seen == [True]
    topics[hal.LOCKSTEP_STATUS_TOPIC](Msg(json.dumps({'ok': False, 'enabled': True})))
    assert not part.active and seen == [True, False]
    topics[hal.LOCKSTEP_ACK_TOPIC](Msg('velocity_report 2 0 50000000'))
    assert gate.report_deadline_ns == 2_000_000_000
    topics[hal.LOCKSTEP_ACK_TOPIC](Msg('garbage'))      # ignored
    part.ack(5, 10, report=1)
    assert [''.join(m) for m in ack_pub.sent] == ['actuator 0 5 10 report=1']


def test_sim_gpio_tracks_the_plants_complete_up_to_mark():
    pytest.importorskip('std_msgs')
    from std_msgs.msg import Int64MultiArray, MultiArrayDimension
    node = FakeNode()
    gpio = hal.make_gpio('sim', node=node)
    got, marks = [], []
    gpio.on_until = marks.append
    gpio.add_event_detect(15, gpio.RISING, callback=lambda ch, stamp_ns: got.append(stamp_ns))
    _, callback = node.subs[0]
    callback(Int64MultiArray(data=[1_000]))            # unmarked: lockstep off
    assert gpio.until_ns is None and not gpio.awaiting_edges(10**9)
    msg = Int64MultiArray(data=[2_000, 3_000])
    msg.layout.dim = [MultiArrayDimension(label=hal.EDGES_UNTIL + '5000', size=2, stride=1)]
    assert hal.edges_until_ns(msg) == 5_000
    callback(msg)
    assert got == [1_000, 2_000, 3_000] and marks == [5_000]
    assert not gpio.awaiting_edges(5_000) and gpio.awaiting_edges(5_001)
    empty = Int64MultiArray(data=[])
    empty.layout.dim = [MultiArrayDimension(label=hal.EDGES_UNTIL + '10000', size=0, stride=1)]
    callback(empty)
    assert gpio.until_ns == 10_000 and marks == [5_000, 10_000]


def test_report_gate_on_schedule():
    gate = hal.ReportGate(hal.PATH_TRACKER)
    assert not gate.on_schedule(1_000_000_000)              # schedule unknown
    gate.on_ack('path_tracker', 3_866_666_628, 33_333_333)
    assert gate.on_schedule(3_900_000_000)                  # ceil of 3.899999961
    assert gate.on_schedule(3_866_666_628)
    assert not gate.on_schedule(3_880_000_000)              # e.g. vehicle_cmd_gate's own stamp
