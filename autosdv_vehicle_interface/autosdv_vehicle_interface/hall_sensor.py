#!/usr/bin/env python3
"""
One count per wheel magnet from the hall sensor's chattering edges.

The KY-003's output switches twice per magnet: LOW as a magnet arrives, HIGH
as it leaves. Each switch is a BURST of edges microseconds apart, faster than
the kernel can record every one (rising and falling counts differ), so single
edges cannot be counted. Bursts can: a burst is edges less than ``burst_gap``
apart, and once the line has been quiet for ``burst_gap`` the level it settled
at says which switch it was. Only the burst that settles at ``active_level``
(the magnet arriving) is counted, timed at its first edge.

    BurstDetector     the rule, without hardware (unit-tested)
    GpioBurstReader   a thread on the pin's kernel event line, both edges,
                      feeding a BurstDetector; Jetson.GPIO only maps the pin

Measured on car A (2026-10-28, KY-003 at 5 V on BOARD pin 16): one burst per
switch, two per magnet, with no stray bursts between magnets.
"""
import ctypes
import fcntl
import os
import select
import threading
import time
from typing import Callable, Optional

DEFAULT_BURST_GAP = 0.001   # [s] quiet time that ends a burst
# the KY-003 output is LOW while a magnet is under it
DEFAULT_ACTIVE_LEVEL = 0


class BurstDetector:
    """Group edge times into bursts; report those settling at active level."""

    def __init__(self, burst_gap: float = DEFAULT_BURST_GAP,
                 active_level: int = DEFAULT_ACTIVE_LEVEL):
        """Set up the detector; ``burst_gap`` is in seconds."""
        if burst_gap <= 0.0:
            raise ValueError(f'burst_gap must be positive, got {burst_gap}')
        self.burst_gap_ns = int(burst_gap * 1e9)
        self.active_level = int(active_level)
        self.burst_start_ns: Optional[int] = None
        self.last_edge_ns: Optional[int] = None
        self.bursts = 0           # every burst, both switches
        self.markers = 0          # the counted ones

    def edge(self, t_ns: int) -> None:
        """Record an edge of either direction at ``t_ns``."""
        if self.burst_start_ns is None:
            self.burst_start_ns = t_ns
            self.bursts += 1
        self.last_edge_ns = t_ns

    def quiet_until(self) -> Optional[int]:
        """
        Return when the open burst ends if no edge comes first.

        None means there is no open burst.
        """
        if self.burst_start_ns is None:
            return None
        return self.last_edge_ns + self.burst_gap_ns

    def settle(self, now_ns: int, level: int) -> Optional[int]:
        """
        Close the open burst if the line has been quiet for burst_gap.

        Quiet is measured at ``now_ns``. ``level`` is the line's value read at
        ``now_ns``. Returns the burst's first-edge time if it settled at the
        active level (a magnet), else None.
        """
        end = self.quiet_until()
        if end is None or now_ns < end:
            return None
        start = self.burst_start_ns
        self.burst_start_ns = None
        self.last_edge_ns = None
        if int(level) != self.active_level:
            return None
        self.markers += 1
        return start


class GpioBurstReader:
    """
    Call ``on_marker(stamp_ns)`` once per magnet, from a daemon thread.

    Reads BOARD ``pin``'s edges from its kernel event line.

    ``clock_ns`` is the clock the stamps are wanted on (a ROS clock's
    ``now().nanoseconds``); kernel timestamps (CLOCK_MONOTONIC) are moved onto
    it by the offset between the two read when the burst is reported.
    """

    def __init__(self, pin: int, on_marker: Callable[[int], None],
                 clock_ns: Callable[[], int],
                 burst_gap: float = DEFAULT_BURST_GAP,
                 active_level: int = DEFAULT_ACTIVE_LEVEL,
                 consumer: str = 'velocity_report'):
        """Open the pin's event line and start the reader thread."""
        import Jetson.GPIO as GPIO
        from Jetson.GPIO import gpio as jgpio
        from Jetson.GPIO import gpio_cdev as cdev

        self._cdev = cdev
        self._event_size = ctypes.sizeof(cdev.gpioevent_data)
        self.detector = BurstDetector(burst_gap, active_level)
        self._on_marker = on_marker
        self._clock_ns = clock_ns
        GPIO.setmode(GPIO.BOARD)
        info = jgpio._channel_to_info(pin, need_gpio=True)
        self._chip_fd = cdev.chip_open_by_label(info.gpio_chip)
        request = cdev.request_event(
            info.line_offset, cdev.GPIOEVENT_REQUEST_BOTH_EDGES, consumer)
        try:
            fcntl.ioctl(self._chip_fd, cdev.GPIO_GET_LINEEVENT_IOCTL, request)
        except OSError as e:
            os.close(self._chip_fd)
            raise RuntimeError(
                f'BOARD pin {pin} ({info.gpio_chip} line {info.line_offset}): '
                f'{e.strerror}; is another reader holding it?') from None
        self._fd = request.fd
        self.description = (
            f'BOARD pin {pin} ({info.gpio_name}, '
            f'{info.gpio_chip} line {info.line_offset})')
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name='hall_bursts', daemon=True)
        self._thread.start()

    def level(self) -> int:
        """Return the line's value now."""
        return self._cdev.get_value(self._fd)

    def _run(self) -> None:
        cdev, d = self._cdev, self.detector
        while not self._stop.is_set():
            end = d.quiet_until()
            mono = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
            timeout = 0.2 if end is None else max(0.0, (end - mono) / 1e9)
            try:
                ready, _, _ = select.select([self._fd], [], [], timeout)
                if ready:
                    data = os.read(self._fd, self._event_size * 64)
                    for i in range(
                            0, len(data) - self._event_size + 1,
                            self._event_size):
                        event = cdev.gpioevent_data.from_buffer_copy(data, i)
                        d.edge(event.timestamp)
                    continue
                mono = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
                start = d.settle(mono, self.level())
            except (OSError, ValueError):
                if self._stop.is_set():
                    return
                raise
            if start is not None:
                self._on_marker(self._clock_ns() - (mono - start))

    def close(self) -> None:
        """Stop the reader thread and release both file descriptors."""
        self._stop.set()
        self._thread.join(timeout=1.0)
        for fd in (self._fd, self._chip_fd):
            try:
                os.close(fd)
            except OSError:
                pass
