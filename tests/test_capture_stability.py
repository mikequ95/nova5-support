# SPDX-License-Identifier: MulanPSL-2.0
"""Capture windows must tolerate noisy speed estimates but reject real motion."""
import pytest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from handeye.capture import CaptureStability
from handeye.cli import _require_capture_idle, _stationary_snapshot


def observation(index, **updates):
    result = {
        "sequence": index + 1, "received_at": 100.0 + index * 0.1,
        "controller_timestamp": 100000 + index * 100,
        "flange_pose_mm_deg": [100, 0, 500, 180, 0, 0],
        "state": {"capture_idle": True, "stationary": False, "ready": False,
                  "mode": 5, "error": False, "source": "feedback",
                  "stationarity_details": {"max_tcp_speed_component": 0.71,
                                           "max_joint_speed_deg_s": 0.11}},
    }
    result.update(updates)
    return result


def window():
    return CaptureStability(observation(0), _require_capture_idle)


def test_stable_pose_with_noisy_speed_passes_only_after_full_window():
    stable = window()
    samples = [observation(i, flange_pose_mm_deg=[100 + .002 * (i % 2), 0, 500, 180, 0, 0])
               for i in range(1, 12)]
    for sample in samples[:9]:
        stable.add([sample])
    assert not stable.covers(100.9)
    stable.add(samples[9:])
    stable.check_fresh(101.11)
    assert stable.covers(101.05)
    assert not stable.covers(101.2)  # No feedback after this frame yet.
    assert stable.report()["max_translation_from_start_mm"] == pytest.approx(.002)
    assert stable.report()["feedback_samples"] == 12
    assert all(not s["state"]["ready"] and not s["state"]["stationary"] for s in samples)


@pytest.mark.parametrize("changed_pose", [[101, 0, 500, 180, 0, 0], [100, 0, 500, 180.2, 0, 0]])
def test_mid_window_motion_cannot_be_hidden_by_returning_to_start(changed_pose):
    stable = window()
    samples = [observation(i) for i in range(1, 12)]
    samples[4]["flange_pose_mm_deg"] = changed_pose
    with pytest.raises(ValueError, match="moved during capture"):
        stable.add(samples)


@pytest.mark.parametrize("mode,flag,message", [(6, "drag_status", "drag mode"),
                                            (8, "jog_status", "jog"),
                                            (10, "pause_cmd_flag", "Paused"),
                                            (7, "run_queued_cmd", "queue"),
                                            (9, "error_status", "alarm")])
def test_transient_controller_activity_between_idle_endpoints_is_rejected(mode, flag, message):
    samples = [observation(i) for i in range(1, 12)]
    state = samples[4]["state"]
    state.update(capture_idle=False, mode=mode, error=mode == 9)
    state["stationarity_details"][flag] = 1
    with pytest.raises(ValueError, match=message):
        window().add(samples)


@pytest.mark.parametrize("update", [{"sequence": 1}, {"sequence": 3},
                                    {"controller_timestamp": 100000},
                                    {"controller_timestamp": 99999},
                                    {"received_at": 100.0}])
def test_missing_duplicate_or_frozen_feedback_rejected(update):
    with pytest.raises(ValueError, match="repeated, missing or has an unchanged timestamp"):
        window().add([observation(1, **update)])


def test_gap_and_old_feedback_refused():
    with pytest.raises(ValueError, match="interrupted"):
        window().add([observation(1, received_at=100.3)])
    with pytest.raises(ValueError, match="stale"):
        window().check_fresh(100.3)


def test_equivalent_euler_wrap_is_not_motion():
    stable = CaptureStability(observation(0, flange_pose_mm_deg=[100, 0, 500, 0, 0, 180]),
                              _require_capture_idle)
    stable.add([observation(1, flange_pose_mm_deg=[100, 0, 500, 0, 0, -180])])
    assert stable.max_rotation_deg < 1e-10


def test_capture_deadline_prevents_starting_another_device_read():
    camera = SimpleNamespace(backend="orbbec", read=Mock())
    robot = Mock()
    with patch("handeye.cli.time.monotonic", return_value=10), \
            pytest.raises(TimeoutError, match="deadline"):
        _stationary_snapshot(camera, robot, deadline=10)
    robot.capture_feedback.assert_not_called()
    camera.read.assert_not_called()


def test_capture_deadline_includes_feedback_wait():
    camera = SimpleNamespace(backend="orbbec", read=Mock())
    robot = Mock()
    robot.capture_feedback.return_value = [observation(0)]
    with patch("handeye.cli.time.monotonic", side_effect=[9, 10]), \
            pytest.raises(TimeoutError, match="deadline"):
        _stationary_snapshot(camera, robot, deadline=10)
    robot.capture_feedback.assert_called_once()
    camera.read.assert_not_called()
