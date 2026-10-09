#!/usr/bin/env python3
"""
Safety helpers for the AutoSDV actuator, free of ROS so they can be tested.

- :class:`InputWatchdog` -- has an input arrived recently enough to act on?
- :func:`classify_stamp_age` -- is a command's header stamp usable?
- :func:`mrm_requires_stop` -- does an Autoware MRM state call for a stop?
- :func:`write_neutral` -- put the PCA9685 outputs in neutral without hanging.

Why these exist: the PCA9685 latches its last duty cycle. Without them, a
control stack that stops publishing, or an actuator process that exits, leaves
the ESC receiving whatever throttle was last written.
"""
from enum import Enum
import logging
import math
import threading
from typing import Iterable, Optional, Tuple


class InputState(Enum):
    """Freshness of a periodic input."""

    WAITING = "waiting"      # never received since start
    FRESH = "fresh"          # received within the timeout
    TIMED_OUT = "timed_out"  # received before, but not within the timeout


class InputWatchdog:
    """
    Track whether a periodic input is fresh, and log each lapse exactly once.

    The caller feeds it a monotonic time on every accepted message and asks for
    the state once per control tick. A lapse logs one warning when it starts and
    one info line when the input returns, however long it lasts.

    A ``timeout`` of zero or less disables the check: once the first message has
    arrived the input is always FRESH.
    """

    def __init__(self, name: str, timeout: float, logger=None):
        self.name = name
        self.timeout = float(timeout)
        self.logger = logger if logger is not None else logging.getLogger(__name__)

        self.last_feed_time: Optional[float] = None
        self.state = InputState.WAITING
        self.episodes = 0  # number of lapses so far, for diagnostics and tests

    def feed(self, now: float) -> None:
        """Record that a usable message arrived at monotonic time ``now``."""
        self.last_feed_time = now

    def check(self, now: float) -> InputState:
        """Return the state at monotonic time ``now``, logging transitions."""
        if self.last_feed_time is None:
            new_state = InputState.WAITING
        elif self.timeout > 0.0 and now - self.last_feed_time > self.timeout:
            new_state = InputState.TIMED_OUT
        else:
            new_state = InputState.FRESH

        if new_state != self.state:
            if new_state == InputState.TIMED_OUT:
                self.episodes += 1
                self.logger.warning(
                    f"No {self.name} for {now - self.last_feed_time:.3f} s "
                    f"(timeout {self.timeout:.3f} s): stopping the motor"
                )
            elif new_state == InputState.FRESH and self.state == InputState.TIMED_OUT:
                self.logger.info(
                    f"{self.name[:1].upper() + self.name[1:]} is back: resuming control"
                )
            self.state = new_state

        return new_state


class StampVerdict(Enum):
    """How a command's header stamp compares with the receiving node's clock."""

    OK = "ok"
    STALE = "stale"                    # older than max_age
    FUTURE = "future"                  # newer than now + max_age
    CLOCK_MISMATCH = "clock_mismatch"  # so far off that the clocks disagree


def classify_stamp_age(
    age: Optional[float],
    max_age: float,
    clock_mismatch_threshold: float = 10.0,
) -> StampVerdict:
    """
    Classify a message by the age of its header stamp.

    Args:
        age: ``now - stamp`` in seconds on the receiving node's clock, or None
            for an unstamped message (stamp zero), which is accepted: some
            publishers never stamp and the reception watchdog still covers them.
        max_age: Largest acceptable ``|age|``. Zero or less disables the check.
        clock_mismatch_threshold: Beyond this, the publisher and this node are
            not on the same clock (wall time vs. sim time), which is a
            configuration error rather than a late message.
    """
    if age is None or max_age <= 0.0:
        return StampVerdict.OK
    if not math.isfinite(age) or abs(age) > clock_mismatch_threshold:
        return StampVerdict.CLOCK_MISMATCH
    if age > max_age:
        return StampVerdict.STALE
    if age < -max_age:
        return StampVerdict.FUTURE
    return StampVerdict.OK


# autoware_adapi_v1_msgs/msg/MrmState constants (Autoware 1.5.0). Mirrored here
# so this module does not import ROS messages; test_safety.py checks them
# against the real message when it is available.
MRM_STATE_UNKNOWN = 0
MRM_STATE_NORMAL = 1
MRM_STATE_OPERATING = 2
MRM_STATE_SUCCEEDED = 3
MRM_STATE_FAILED = 4
MRM_BEHAVIOR_EMERGENCY_STOP = 2


def mrm_requires_stop(state: int, behavior: int, stop_on_any_mrm: bool) -> bool:
    """
    Decide whether an MRM state means the actuator must brake to a stop.

    A failed MRM always does: nothing downstream is going to stop the vehicle.
    An operating or succeeded MRM does when its behaviour is an emergency stop
    (the condition ``vehicle_cmd_gate`` uses for its own emergency output), or
    for any behaviour when ``stop_on_any_mrm`` is set -- needed when the planner
    in use ignores the velocity limit through which a comfortable stop acts.
    """
    if state == MRM_STATE_FAILED:
        return True
    if state in (MRM_STATE_OPERATING, MRM_STATE_SUCCEEDED):
        return stop_on_any_mrm or behavior == MRM_BEHAVIOR_EMERGENCY_STOP
    return False


def write_neutral(
    driver,
    outputs: Iterable[Tuple[int, int]],
    timeout: float = 0.5,
    logger=None,
) -> bool:
    """
    Write ``(channel, pwm)`` pairs to a PCA9685, giving up after ``timeout``.

    The writes run on a daemon thread, so a wedged I2C bus can delay process
    exit by at most ``timeout`` and never blocks it. Errors are logged, not
    raised: this runs on the way out, often from a ``finally`` or ``atexit``.

    Returns:
        bool: True when every write completed.
    """
    logger = logger if logger is not None else logging.getLogger(__name__)
    outputs = list(outputs)
    if driver is None:
        logger.warning("No PWM driver: cannot write neutral outputs")
        return False

    result = {"ok": False}

    def _write():
        try:
            for channel, value in outputs:
                driver.set_pwm(channel, 0, int(value))
            result["ok"] = True
        except Exception as e:  # noqa: B902 - any I2C failure must be survivable here
            logger.error(f"Failed to write neutral PWM: {e}")

    thread = threading.Thread(target=_write, name="pwm-neutral", daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        logger.error(f"Neutral PWM write did not finish within {timeout:.2f} s (I2C bus hung?)")
        return False
    return result["ok"]
