# SPDX-License-Identifier: MulanPSL-2.0
"""Pose stability over the whole RGB acquisition interval (not motion permission)."""
from __future__ import annotations

import math

from .geometry import pose_to_transform
from .workflow import pose_distance


class CaptureStability:
    """Check every feedback packet against the first actual flange pose.

    Instantaneous velocity estimates can fluctuate on an idle controller. For
    image acquisition, require idle flags and bounded pose excursion instead.
    The bounded click test also uses this window with an enabled/idle check;
    normal execution retains the adapter's strict stationary/ready gates.
    """

    min_duration_s = 1.0
    min_samples = 10
    max_gap_s = 0.25
    position_limit_mm = 0.2
    rotation_limit_deg = 0.1

    def __init__(self, first, check_idle):
        self._check_idle = check_idle
        self.first = first
        self.last = None
        self.anchor = pose_to_transform(first["flange_pose_mm_deg"])
        self.last_transform = self.anchor
        self.samples = 0
        self.largest_gap_s = 0.0
        self.max_translation_mm = 0.0
        self.max_rotation_deg = 0.0
        self.add([first])

    def add(self, observations):
        for observation in observations:
            sequence = observation["sequence"]
            received_at = observation["received_at"]
            controller_timestamp = observation["controller_timestamp"]
            if (type(sequence) is not int or sequence < 0
                    or not isinstance(received_at, (int, float)) or not math.isfinite(received_at)
                    or type(controller_timestamp) is not int or controller_timestamp < 0):
                raise ValueError("Invalid capture feedback sequence or timestamp; check robot feedback")
            if self.last is not None:
                gap = received_at - self.last["received_at"]
                if (sequence != self.last["sequence"] + 1 or gap <= 0
                        or controller_timestamp <= self.last["controller_timestamp"]):
                    raise ValueError("Capture feedback is repeated, missing or has an unchanged timestamp; check the feedback connection and retry")
                if gap > self.max_gap_s:
                    raise ValueError(f"Feedback interrupted for {gap:.3f} s during capture; check the connection and retry")
                self.largest_gap_s = max(self.largest_gap_s, gap)
            self._check_idle(observation["state"])
            transform = pose_to_transform(observation["flange_pose_mm_deg"])
            moved, rotated = pose_distance(self.anchor, transform)
            if moved > self.position_limit_mm or rotated > self.rotation_limit_deg:
                raise ValueError(
                    "Robot moved during capture: actual pose was unstable; "
                    f"offset from start {moved:.3f} mm / {rotated:.3f} deg; stop and retry"
                )
            self.max_translation_mm = max(self.max_translation_mm, moved)
            self.max_rotation_deg = max(self.max_rotation_deg, rotated)
            self.last_transform = transform
            self.last = observation
            self.samples += 1

    def check_fresh(self, now):
        age = now - self.last["received_at"]
        if age < 0 or age > self.max_gap_s:
            raise ValueError("Capture feedback is stale; check the robot feedback connection and retry")

    def covers(self, frame_received_at):
        return (self.last["received_at"] >= frame_received_at
                and self.last["received_at"] - self.first["received_at"] >= self.min_duration_s
                and self.samples >= self.min_samples)

    def report(self):
        return {
            "duration_s": self.last["received_at"] - self.first["received_at"],
            "feedback_samples": self.samples,
            "largest_feedback_gap_s": self.largest_gap_s,
            "max_translation_from_start_mm": self.max_translation_mm,
            "max_rotation_from_start_deg": self.max_rotation_deg,
            "position_limit_mm": self.position_limit_mm,
            "rotation_limit_deg": self.rotation_limit_deg,
        }
