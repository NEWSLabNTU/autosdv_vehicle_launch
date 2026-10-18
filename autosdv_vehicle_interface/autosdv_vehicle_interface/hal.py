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

With ``hardware: sim`` the two nodes also take part in the plant's clock
handshake (lockstep): each acknowledges every timer deadline on
``/rc_sim/lockstep/ack`` once its work for that deadline is done, and the
plant does not advance ``/clock`` past a deadline until it has the ack. See
``LockstepParticipant``. The real path never publishes or waits.
"""
import json
from typing import Callable, Dict, Optional, Tuple

HARDWARE_REAL = 'real'
HARDWARE_SIM = 'sim'
PWM_TOPIC = '/rc_sim/pwm'
EDGE_TOPIC = '/rc_sim/hall_edges'
LOCKSTEP_ACK_TOPIC = '/rc_sim/lockstep/ack'
LOCKSTEP_STATUS_TOPIC = '/rc_sim/lockstep/status'
VELOCITY_REPORT = 'velocity_report'     # the participants' names in an ack
# Under lockstep every edge message also says up to when it is complete: its layout's one
# dimension is labelled "until_ns=<ns>" (data stays the edge stamps). See SimGPIO.
EDGES_UNTIL = 'until_ns='
ACTUATOR = 'actuator'
PATH_TRACKER = 'path_tracker'           # Lab 2 Part 1: the target's source, through the gate
EMERGENCY = 'emergency'                 # Lab 2 Part 1: the e-stop's source (lab2_part1_scenario)


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


def edges_until_ns(msg) -> Optional[int]:
    """Return the time up to which an edge message is complete, or None if it does not say."""
    for dim in msg.layout.dim:
        if dim.label.startswith(EDGES_UNTIL):
            return int(dim.label[len(EDGES_UNTIL):])
    return None


class SimGPIO:
    """
    Stand-in for the ``Jetson.GPIO`` module: the three calls velocity_report makes.

    ``until_ns`` is the latest "complete up to" mark the plant has sent with
    the edges (lockstep only, None otherwise): every edge at or before it has
    been delivered. ``on_until`` is called with it after each such message.
    """

    BOARD = 'BOARD'
    IN = 'IN'
    RISING = 'RISING'

    def __init__(self, node, topic: str = EDGE_TOPIC):
        self._node = node
        self._topic = topic
        self._subscriptions = []
        self.until_ns: Optional[int] = None
        self.on_until: Optional[Callable[[int], None]] = None

    def awaiting_edges(self, t_ns: int) -> bool:
        """Return True if the plant has marked its edges complete, but not yet up to ``t_ns``."""
        return self.until_ns is not None and self.until_ns < t_ns

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
            until = edges_until_ns(msg)
            if until is not None and (self.until_ns is None or until > self.until_ns):
                self.until_ns = until
                if self.on_until is not None:
                    self.on_until(until)

        qos = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE)
        self._subscriptions.append(
            self._node.create_subscription(Int64MultiArray, self._topic, on_edges, qos))


def make_pwm_driver(params: dict, node=None, pca_class=None):
    """Return the PWM driver for ``params["hardware"]``; sim needs ``node``; dry runs skip it."""
    hardware = check_hardware(params.get('hardware', HARDWARE_REAL))
    if hardware == HARDWARE_SIM:
        if node is None:
            raise ValueError("hardware 'sim' publishes PWM on a ROS topic: pass the node")
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
        if node is None:
            raise ValueError("hardware 'sim' takes hall edges from a ROS topic: pass the node")
        return SimGPIO(node)
    import Jetson.GPIO as GPIO
    return GPIO


# ---------------------------------------------------------------- lockstep (sim only)
def format_ack(name: str, deadline_ns: int, period_ns: int, **extra: int) -> str:
    """
    Return the ack text: ``"<name> <sec> <nanosec> <period_ns> [key=value ...]"``.

    ``deadline_ns`` is the timer deadline the work was for (not the time it
    ran), ``period_ns`` the timer's period, and each extra value an integer.
    """
    sec, nanosec = divmod(int(deadline_ns), 1_000_000_000)
    words = [name, str(sec), str(nanosec), str(int(period_ns))]
    words += [f'{key}={int(value)}' for key, value in sorted(extra.items())]
    return ' '.join(words)


def parse_ack(text: str) -> Tuple[str, int, int, Dict[str, int]]:
    """Return ``(name, deadline_ns, period_ns, extra)`` from an ack; ValueError if malformed."""
    words = text.split()
    if len(words) < 4:
        raise ValueError(f'lockstep ack needs "<name> <sec> <nanosec> <period_ns>": {text!r}')
    name = words[0]
    deadline_ns = int(words[1]) * 1_000_000_000 + int(words[2])
    period_ns = int(words[3])
    if period_ns <= 0:
        raise ValueError(f'lockstep ack with a period of {period_ns} ns: {text!r}')
    extra = {}
    for word in words[4:]:
        key, sep, value = word.partition('=')
        if not sep:
            raise ValueError(f'lockstep ack field {word!r} is not key=value: {text!r}')
        extra[key] = int(value)
    return name, deadline_ns, period_ns, extra


def timer_deadline_ns(timer, now_ns: int) -> int:
    """
    Return the deadline a timer callback is running for: the latest one at or before ``now_ns``.

    rcl has already moved the timer's next call one period (or more, if the
    clock jumped) past ``now_ns`` when the callback runs.
    """
    until = timer.time_until_next_call()
    if until is None:
        return int(now_ns)
    return int(now_ns) + int(until) - int(timer.timer_period_ns)


class ReportGate:
    """
    The actuator's half of "velocity_report goes first at equal instants".

    The plant holds ``/clock`` at an instant until every participant due then
    has acknowledged it, but both nodes see that instant at once. A control
    tick at ``tick_ns`` must use the report of the latest velocity_report
    deadline at or before ``tick_ns``; ``awaiting`` says whether it has not
    arrived yet, so the actuator holds the tick (and its ack) until it does.
    velocity_report's period and phase come from its own acks.

    ``participant`` names another source the same way: with ``PATH_TRACKER``
    the gate holds a tick until a control command stamped at or after the
    tracker's latest deadline has arrived (the forwarder and vehicle_cmd_gate
    keep the tracker's stamp); with ``EMERGENCY``, until the emergency message
    of the scenario's latest tick has. ``on_report`` takes the message's stamp,
    and ``on_schedule`` tells a message of the participant's own (stamped within
    one physics step after one of its deadlines) from any other on that topic.
    """

    def __init__(self, participant: str = VELOCITY_REPORT):
        self.participant = participant
        self.report_deadline_ns: Optional[int] = None
        self.report_period_ns: Optional[int] = None
        self.last_report_ns: Optional[int] = None

    def on_ack(self, name: str, deadline_ns: int, period_ns: int) -> None:
        """Learn the participant's schedule from its ack; other participants are ignored."""
        if name == self.participant:
            self.report_deadline_ns, self.report_period_ns = int(deadline_ns), int(period_ns)

    def on_report(self, stamp_ns: int) -> None:
        """Take the stamp of a message from the participant's stream."""
        if self.last_report_ns is None or stamp_ns > self.last_report_ns:
            self.last_report_ns = int(stamp_ns)

    def expected_ns(self, tick_ns: int) -> Optional[int]:
        """Return the participant's latest deadline at or before ``tick_ns`` (None: unknown)."""
        if self.report_deadline_ns is None:
            return None
        period = self.report_period_ns
        return self.report_deadline_ns + ((tick_ns - self.report_deadline_ns) // period) * period

    def on_schedule(self, stamp_ns: int, tolerance_ns: int = 1_000_000) -> bool:
        """Return True if ``stamp_ns`` is one of the participant's deadlines (to the step)."""
        expected = self.expected_ns(stamp_ns)
        return expected is not None and stamp_ns - expected < tolerance_ns

    def awaiting(self, tick_ns: int) -> bool:
        """Return True while the message a tick at ``tick_ns`` must use has not arrived."""
        expected = self.expected_ns(tick_ns)
        if expected is None:
            return False
        return self.last_report_ns is None or self.last_report_ns < expected


class LockstepParticipant:
    """
    A node's side of the plant's clock handshake: publish acks, follow the plant's status.

    ``active`` is True while the plant says lockstep is enabled and no
    participant has timed out; a node may hold work for the handshake only
    then, so a run without lockstep (or after a timeout) never stalls.
    """

    def __init__(self, node, name: str, gate=None,
                 on_status: Optional[Callable[[bool], None]] = None):
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import String
        self.name = name
        # one ReportGate, or several (the actuator: velocity_report and path_tracker)
        if gate is None:
            self.gates = []
        else:
            self.gates = list(gate) if isinstance(gate, (list, tuple)) else [gate]
        self.active = False
        self._on_status = on_status
        self._string = String
        reliable = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE)
        self._publisher = node.create_publisher(String, LOCKSTEP_ACK_TOPIC, reliable)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._subscriptions = [node.create_subscription(
            String, LOCKSTEP_STATUS_TOPIC, self._status_callback, latched)]
        if self.gates:
            self._subscriptions.append(
                node.create_subscription(String, LOCKSTEP_ACK_TOPIC, self._ack_callback, reliable))

    def ack(self, deadline_ns: int, period_ns: int, **extra: int) -> None:
        """Acknowledge the work for ``deadline_ns``."""
        self._publisher.publish(self._string(data=format_ack(self.name, deadline_ns, period_ns,
                                                             **extra)))

    def _ack_callback(self, msg) -> None:
        try:
            name, deadline_ns, period_ns, _ = parse_ack(msg.data)
        except ValueError:
            return
        for gate in self.gates:
            gate.on_ack(name, deadline_ns, period_ns)

    def _status_callback(self, msg) -> None:
        try:
            status = json.loads(msg.data)
        except ValueError:
            return
        self.active = bool(status.get('enabled', False)) and bool(status.get('ok', False))
        if self._on_status is not None:
            self._on_status(self.active)


def make_lockstep_participant(hardware: str, node, name: str,
                              gate=None,
                              on_status: Optional[Callable[[bool], None]] = None):
    """Return a LockstepParticipant for ``hardware: sim``, else None (the car never acks)."""
    if check_hardware(hardware) != HARDWARE_SIM:
        return None
    return LockstepParticipant(node, name, gate, on_status)
