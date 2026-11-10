# SPDX-License-Identifier: MulanPSL-2.0
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from handeye.cli import main
from handeye.geometry import pose_to_transform
from handeye.test_motion import require_test_motion
from handeye.workflow import (digest, load_handeye, load_plane, plan_click, read_json,
                              require_motion, save_json, setup_signature)


@pytest.fixture
def setup(tmp_path):
    config = read_json(Path(__file__).resolve().parents[1] / "config.example.json")
    config["_root"] = str(tmp_path)
    config["_path"] = str(tmp_path / "config.json")
    config["board"]["print_size_verified"] = True
    config["tcp"] = {"flange_to_tcp_mm_deg": [30, 10, 100, 0, 0, 0], "measured_and_verified": True}
    config["robot"]["compatibility_verified"] = True
    config["validation"]["independent_validation_passed"] = True
    config["motion"].update(enabled=True, max_step_mm=1000,
                            workspace_min_mm=[-1000, -1000, -1000], workspace_max_mm=[1000, 1000, 1000])
    intrinsic = {"K": [[800, 0, 640], [0, 800, 360], [0, 0, 1]], "dist": [0]*5,
                 "image_size": [1280, 720], "rms": .2, "board": config["board"]}
    save_json(tmp_path / config["files"]["intrinsics"], intrinsic)
    handeye = {"T_flange_camera": np.eye(4).tolist(), "quality_passed": True,
               "metrics": {"board_position_rms_mm": .2, "board_rotation_rms_deg": .1},
               "views": [{"reprojection_rms_px": .2}],
               "intrinsics_sha256": digest(tmp_path / config["files"]["intrinsics"])}
    save_json(tmp_path / config["files"]["handeye"], handeye)
    plane = {"normal": [0, 0, 1], "offset_mm": 0, "quality": {"reprojection_rms_px": .2},
             "handeye_sha256": digest(tmp_path / config["files"]["handeye"])}
    save_json(tmp_path / config["files"]["plane"], plane)
    save_json(tmp_path / "calibrations/validation.json", {"setup_sha256": setup_signature(config)})
    return config, intrinsic, handeye, plane


def test_click_intersects_surface_then_changes_height_and_compensates_rotated_tcp(setup):
    config, intrinsic, handeye, plane = setup
    current = pose_to_transform([0, 0, 500, 180, 0, 0])
    result = plan_click([800, 360], 100, current, config, intrinsic, handeye, plane)
    np.testing.assert_allclose(result["surface_point_mm"], [100, 0, 0], atol=1e-8)
    np.testing.assert_allclose(result["tcp_pose_mm_deg"][:3], [100, 0, 100], atol=1e-8)
    np.testing.assert_allclose(result["flange_pose_mm_deg"][:3], [70, 10, 200], atol=1e-8)
    require_motion(config, current, result)


@pytest.mark.parametrize("section,key", [("motion", "enabled"), ("robot", "compatibility_verified"),
                                         ("tcp", "measured_and_verified"),
                                         ("validation", "independent_validation_passed"),
                                         ("board", "print_size_verified")])
def test_string_false_never_enables_motion(setup, section, key):
    config, intrinsic, handeye, plane = setup
    current = pose_to_transform([0, 0, 500, 180, 0, 0])
    result = plan_click([800, 360], 100, current, config, intrinsic, handeye, plane)
    config[section][key] = "false"
    with pytest.raises(ValueError):
        require_motion(config, current, result)


def test_workspace_and_step_and_orientation_rejected(setup):
    config, intrinsic, handeye, plane = setup
    current = pose_to_transform([0, 0, 500, 180, 0, 0])
    result = plan_click([800, 360], 100, current, config, intrinsic, handeye, plane)
    config["motion"]["max_step_mm"] = 10
    with pytest.raises(ValueError, match="distance"):
        require_motion(config, current, result)
    config["motion"]["max_step_mm"] = 1000
    config["motion"]["workspace_max_mm"][2] = 250
    with pytest.raises(ValueError, match="outside"):
        require_motion(config, current, result)
    config["motion"]["workspace_max_mm"][2] = 1000
    changed = pose_to_transform([0, 0, 500, 180, 0, 10])
    with pytest.raises(ValueError, match="orientation"):
        require_motion(config, changed, result)


@pytest.mark.parametrize("low,high,error", [
    (None, None, None),
    (None, [1000] * 3, "both specified or both absent"),
    ([-1000] * 3, None, "both specified or both absent"),
    ([-1000] * 2, [1000] * 3, "Invalid workspace bounds"),
    ([-1000, -1000, float("nan")], [1000] * 3, "Invalid workspace bounds"),
    ([1000] * 3, [1000] * 3, "Invalid workspace bounds"),
    ([-1000] * 3, [1000] * 3, None),
])
def test_click_test_workspace_bounds_remain_optional(setup, low, high, error):
    config, intrinsic, handeye, plane = setup
    current = pose_to_transform([0, 0, 500, 180, 0, 0])
    result = plan_click([800, 360], 150, current, config, intrinsic, handeye, plane)
    config["motion"].update(workspace_min_mm=low, workspace_max_mm=high)
    if error:
        with pytest.raises(ValueError, match=error):
            require_test_motion(config, current, result, current, plane, execute=True)
    else:
        require_test_motion(config, current, result, current, plane, execute=True)
    if low is None or high is None:
        with pytest.raises(ValueError, match="Set measured workspace bounds before motion"):
            require_motion(config, current, result)


def test_unknown_tcp_and_below_surface_rejected(setup):
    config, intrinsic, handeye, plane = setup
    current = pose_to_transform([0, 0, 500, 180, 0, 0])
    with pytest.raises(ValueError, match="above"):
        plan_click([800, 360], -1, current, config, intrinsic, handeye, plane)
    config["tcp"]["flange_to_tcp_mm_deg"] = None
    with pytest.raises(ValueError, match="unknown"):
        plan_click([800, 360], 100, current, config, intrinsic, handeye, plane)


def test_validation_bound_to_tcp_and_calibration(setup):
    config, intrinsic, handeye, plane = setup
    current = pose_to_transform([0, 0, 500, 180, 0, 0])
    config["tcp"]["flange_to_tcp_mm_deg"][2] = 101
    result = plan_click([800, 360], 100, current, config, intrinsic, handeye, plane)
    with pytest.raises(ValueError, match="stale"):
        require_motion(config, current, result)


def test_recalibration_invalidates_downstream_files(setup):
    config, intrinsic, handeye, plane = setup
    assert load_handeye(config) == handeye
    assert load_plane(config) == plane
    intrinsic["rms"] = .3
    save_json(Path(config["_root"]) / config["files"]["intrinsics"], intrinsic)
    with pytest.raises(ValueError, match="different intrinsics"):
        load_handeye(config)
    handeye["metrics"]["board_position_rms_mm"] = .3
    save_json(Path(config["_root"]) / config["files"]["handeye"], handeye)
    with pytest.raises(ValueError, match="different hand-eye"):
        load_plane(config)


def test_stricter_quality_limit_rechecked_on_load(setup):
    config, *_ = setup
    config["validation"]["handeye_translation_rms_mm"] = .1
    with pytest.raises(ValueError, match="current quality"):
        load_handeye(config)


def test_tcp_cli_offline(capsys):
    main(["tcp-target", "--tcp-pose", "100", "200", "300", "180", "0", "0",
          "--offset", "0", "0", "100", "0", "0", "0"])
    value = json.loads(capsys.readouterr().out)
    np.testing.assert_allclose(value["flange_pose_mm_deg"][:3], [100, 200, 400], atol=1e-8)


def test_default_doctor_does_not_construct_hardware(monkeypatch, capsys, tmp_path):
    import handeye.cli as cli
    def fail(*args):
        raise AssertionError("Offline command opened hardware")
    monkeypatch.setattr(cli, "_robot", fail)
    monkeypatch.setattr(cli, "Camera", fail)
    config = read_json(Path(__file__).resolve().parents[1] / "config.example.json")
    config["tcp"]["flange_to_tcp_mm_deg"] = None
    config["motion"]["enabled"] = False
    path = tmp_path / "config.json"
    save_json(path, config)
    main(["--config", str(path), "doctor"])
    result = json.loads(capsys.readouterr().out)
    assert result["motion_enabled"] is False
    assert result["tcp_known"] is False


@pytest.mark.parametrize("moved", [False, True])
def test_snapshot_drains_old_frames_and_rejects_robot_motion(monkeypatch, moved):
    import handeye.cli as cli
    clock = [100.0]
    monkeypatch.setattr(cli.time, "monotonic", lambda: clock[0])

    class FakeCamera:
        backend = "orbbec"
        last_frame_metadata = {"sdk_system_timestamp_us": 123}
        count = 0
        def read(self):
            self.count += 1
            clock[0] += .1
            return np.full((4, 4, 3), 0 if self.count <= 10 else 255, np.uint8), 123.0

    class FakeRobot:
        count = 0
        def capture_feedback(self, after_sequence=None):
            self.count += 1
            return [{"sequence": self.count, "received_at": clock[0],
                     "controller_timestamp": self.count * 100,
                     "flange_pose_mm_deg": [1 if moved and self.count > 1 else 0, 0, 500, 180, 0, 0],
                     "state": {"capture_idle": True, "stationary": False, "source": "feedback"}}]

    camera = FakeCamera()
    if moved:
        with pytest.raises(ValueError, match="moved during capture"):
            cli._stationary_snapshot(camera, FakeRobot())
    else:
        frame, record = cli._stationary_snapshot(camera, FakeRobot())
        assert np.all(frame == 255)
        assert record["camera_timestamp_diagnostics"] == camera.last_frame_metadata
        assert record["capture_stability"]["duration_s"] >= 1
        assert record["capture_stability"]["feedback_samples"] >= 10
        assert camera.count >= 15 and clock[0] >= 101
