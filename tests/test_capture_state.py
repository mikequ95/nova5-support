# SPDX-License-Identifier: MulanPSL-2.0
"""Capture state and command routing regressions; no device connections."""
from types import SimpleNamespace

import numpy as np
import pytest

from handeye import cli


@pytest.mark.parametrize("mode,details,error,expected", [
    (6, {"drag_status": 1}, False, "exit drag mode"),
    (8, {"jog_status": 1}, False, "release the jog button"),
    (10, {"pause_cmd_flag": 1}, False, "Paused"),
    (7, {"run_queued_cmd": 1}, False, "queue"),
    (9, {"error_status": True}, True, "alarm"),
    (5, {}, False, "idle conditions"),
])
def test_refused_capture_explains_state_before_reading_image(mode, details, error, expected):
    robot = SimpleNamespace(capture_feedback=lambda: [{
        "sequence": 1, "received_at": 100.0, "controller_timestamp": 100000,
        "flange_pose_mm_deg": [0, 0, 500, 180, 0, 0],
        "state": {"capture_idle": False, "stationary": False, "mode": mode,
                  "error": error, "stationarity_details": details},
    }])
    camera = SimpleNamespace(backend="orbbec")  # No read method: it must not capture.
    with pytest.raises(ValueError, match=expected) as exc:
        cli._stationary_snapshot(camera, robot)
    assert f'"mode": {mode}' in str(exc.value)


@pytest.mark.parametrize("command,execute,expected_feedback_only", [
    ("capture-handeye", False, True),
    ("capture-plane", False, True),
    ("click", False, True),
    ("click", True, False),
])
def test_capture_and_preview_never_request_command_connection(
        tmp_path, monkeypatch, command, execute, expected_feedback_only):
    import handeye.robot as adapter

    class ConnectionReached(Exception):
        pass

    class FakeCamera:
        def __init__(self, config):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    requested = []
    def constructor(config, *, feedback_only):
        requested.append(feedback_only)
        raise ConnectionReached

    monkeypatch.setattr(adapter, "DobotRobot", constructor)
    monkeypatch.setattr(cli, "Camera", FakeCamera)
    monkeypatch.setattr(cli, "require_board", lambda config: None)
    monkeypatch.setattr(cli, "load_intrinsics", lambda config: {"image_size": [1280, 720]})
    monkeypatch.setattr(cli, "load_handeye", lambda config: {})
    monkeypatch.setattr(cli, "load_plane", lambda config: {})
    monkeypatch.setattr(cli, "tcp_transform", lambda config: np.eye(4))
    config = {"_root": str(tmp_path), "robot": {}, "camera": {}}
    args = SimpleNamespace(command=command, execute=execute,
                           session="samples", surface_offset_mm=0)
    run = {"capture-handeye": cli.capture, "capture-plane": cli.plane_capture,
           "click": cli.click}[command]
    with pytest.raises(ConnectionReached):
        run(args, config)
    assert requested == [expected_feedback_only]
