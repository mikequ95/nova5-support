# SPDX-License-Identifier: MulanPSL-2.0
"""Reject incompatible gripper feedback without opening a physical serial port."""
from types import SimpleNamespace

import pytest

from pgi_hand.motion import GuardedPort, MotionRejected, check_closing, check_opening


def feedback(status=1, position=500, target=500):
    return {"initialization_raw": 1, "io_mode_raw": 0, "grip_status_raw": status,
            "position_raw": position,
            "configuration": {"force_setpoint_percent": 20, "speed_setpoint_percent": 1,
                              "target_position_raw": target}}


def test_closing_requires_idle_matching_target_and_bounded_step():
    check_closing(feedback(), 20)
    for state, step in ((feedback(status=2), 10), (feedback(status=3), 10),
                        (feedback(target=480), 10), (feedback(), 21)):
        with pytest.raises(MotionRejected):
            check_closing(state, step)


def test_opening_from_contact_uses_actual_position_and_rejects_dropped_state():
    check_opening(feedback(status=2, target=480), 520)
    for state, target in ((feedback(status=3), 510), (feedback(status=0), 510),
                          (feedback(status=2, target=501), 510), (feedback(), 521)):
        with pytest.raises(MotionRejected):
            check_opening(state, target)


def test_cancel_gate_runs_at_actual_write_boundary():
    writes = []

    def canceled():
        raise MotionRejected("Canceled before write")

    guarded = GuardedPort(SimpleNamespace(write=writes.append), canceled)
    with pytest.raises(MotionRejected, match="Canceled"):
        guarded.write(bytes.fromhex("0106010301f40000"))
    with pytest.raises(MotionRejected, match="Only one FC06"):
        guarded.write(bytes.fromhex("0106010000010000"))
    assert writes == []
