#!/usr/bin/env python3
"""
Wheel speed from hall-sensor edges: velocity_report's arithmetic, without ROS.

Shared by the velocity_report node and the rc_sim lockstep grader, so a
graded run measures speed with the car's code. Times are integer
nanoseconds, the unit of the ROS clock, which keeps the arithmetic identical
to the node's ``(now - last).nanoseconds / 10**9``.

- Each edge gives an instantaneous speed ``distance_per_marker / dt``
  (edges closer than 1 ms are ignored), smoothed by an EMA (alpha 0.3).
- ``gap_bound``: no edge for ``since`` seconds means the wheel rolled less
  than one marker spacing in that time, so the estimate is capped at
  ``overdue_slack * distance_per_marker / since``. Without it the estimate
  holds its last value until the timeout after the wheel stops.
- No edge for ``timeout`` seconds: zero.
- ``gap_median`` (1 = off): the gap is the median of the last N edge gaps.
  The node times an edge when its GPIO callback runs, so one late callback
  makes a long gap followed by a short one, and the short one reads far too
  fast (car C: 0.5 -> 0.88 m/s at a steady PWM, which the controller answers
  by slowing the car). A median of 3 rejects that pair.

The sensor has no direction: the result is never negative.
"""
from collections import deque
import statistics
from typing import Optional


class VelocityEstimator:
    """Edge-to-edge wheel speed, as published by velocity_report."""

    MIN_EDGE_DT = 0.001  # [s] shorter gaps are noise (about 27 m/s for this wheel)

    def __init__(self, distance_per_marker: float, alpha: float = 0.3,
                 timeout: float = 0.5, overdue_slack: float = 1.5,
                 gap_median: int = 1):
        if gap_median < 1:
            raise ValueError(f'gap_median must be at least 1, got {gap_median}')
        self.distance_per_marker = distance_per_marker
        self.gaps: deque = deque(maxlen=gap_median)
        self.alpha = alpha
        self.timeout = timeout
        self.overdue_slack = overdue_slack
        self.last_edge_ns: Optional[int] = None
        self.current_velocity = 0.0

    def gap_bound(self, now_ns: int) -> float:
        """Cap the estimate at what the time since the last edge allows."""
        if self.last_edge_ns is None:
            return self.current_velocity
        since = (now_ns - self.last_edge_ns) / (10**9)
        if since <= 0.0:
            return self.current_velocity
        return min(self.current_velocity,
                   self.overdue_slack * self.distance_per_marker / since)

    def on_edge(self, t_ns: int) -> None:
        """Register a marker edge at ``t_ns``."""
        if self.last_edge_ns is not None:
            dt = (t_ns - self.last_edge_ns) / (10**9)
            if dt > self.MIN_EDGE_DT:
                self.gaps.append(dt)
                instantaneous = self.distance_per_marker / statistics.median(self.gaps)
                # Start from the capped estimate, so a slowdown the cap already
                # showed does not bounce back up through the EMA
                self.current_velocity = self.gap_bound(t_ns)
                if self.current_velocity == 0.0:
                    self.current_velocity = instantaneous
                else:
                    self.current_velocity = (
                        self.alpha * instantaneous
                        + (1.0 - self.alpha) * self.current_velocity
                    )
        self.last_edge_ns = t_ns

    def report(self, now_ns: int) -> float:
        """Return the speed published at ``now_ns`` (applies the timeout, then the cap)."""
        if self.last_edge_ns is not None:
            if (now_ns - self.last_edge_ns) / (10**9) > self.timeout:
                self.current_velocity = 0.0
                # a stop: the next run starts from fresh gaps
                self.gaps.clear()
        return self.gap_bound(now_ns)
