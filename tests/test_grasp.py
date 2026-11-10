# SPDX-License-Identifier: MulanPSL-2.0
import copy
import unittest

from handeye.geometry import pose_to_transform, target_flange
from handeye import grasp


class MotionBoundsTest(unittest.TestCase):
    def setUp(self):
        self.config = {
            "motion": {"enabled": True, "speed_mm_s": 50,
                       "workspace_min_mm": [220, -90, -90],
                       "workspace_max_mm": [630, 0, 620]},
            "tcp": {"flange_to_tcp_mm_deg": [0, 0, 174, 0, 0, 0]},
            "robot": {},
        }

    def test_recorded_home_and_grasp_positions_fit(self):
        for pose in ([235.4507, -74.7852, 603.2738, 170.8342, 25.7119, -171.0346],
                     [614.1188, -15.5272, 256.656, -180, 0, 111.3784],
                     [605.4342, -20.9022, 91.5235, 179.9998, .0001, 111.3786]):
            with self.subTest(pose=pose):
                grasp.require_grasp_motion(self.config, current=pose_to_transform(pose))

    def test_start_and_target_check_flange_and_rotated_tcp(self):
        cases = [([219.9, -20, 260, 180, 0, 0], "flange X"),
                 ([600, .1, 260, 180, 0, 0], "flange Y"),
                 ([600, -20, 621, 180, 0, 0], "flange Z"),
                 ([600, -20, 83.9, 180, 0, 0], "TCP Z"),
                 ([600, -20, 200, 0, 90, 0], "TCP X")]
        for label in ("start", "target", "current"):
            for pose, point in cases:
                with self.subTest(label=label, pose=pose):
                    with self.assertRaisesRegex(ValueError, f"{label} {point}.*outside workspace"):
                        grasp.require_grasp_motion(self.config, **{label: pose_to_transform(pose)})

    def test_missing_or_invalid_bounds_are_rejected(self):
        for low, high in ((None, [630, 0, 620]), ([220, -90], [630, 0, 620]),
                          ([220, -90, float("nan")], [630, 0, 620]),
                          ([630, -90, -90], [220, 0, 620]),
                          ([220, -90, -90], [630, 0, float("inf")])):
            with self.subTest(low=low, high=high):
                config = copy.deepcopy(self.config)
                config["motion"].update(workspace_min_mm=low, workspace_max_mm=high)
                with self.assertRaisesRegex(ValueError, "finite XYZ"):
                    grasp.require_grasp_motion(config)

    def test_permission_and_speed_must_be_explicit(self):
        for enabled, speed in ((False, 50), (1, 50), (True, 5),
                               (True, "50"), (True, float("nan"))):
            with self.subTest(enabled=enabled, speed=speed):
                config = copy.deepcopy(self.config)
                config["motion"].update(enabled=enabled, speed_mm_s=speed)
                with self.assertRaises(ValueError):
                    grasp.require_grasp_motion(config)

    def stage_poses(self, start_height, end_height, rx=180, ry=0, rz=112.673347):
        tcp = grasp.tcp_transform(self.config)
        return tuple(target_flange(pose_to_transform(
            [595, -37, height - 109, rx, ry, rz]), tcp)
            for height in (start_height, end_height))

    def test_recorded_tilt_allows_vertical_grasp_stages(self):
        stages = {"Pregrasp descent": (200, 75), "Grasp descent": (75, 30),
                  "Lift object": (30, 60), "Place object": (60, 30),
                  "Postplace retreat": (30, 75), "Grasp clearance retreat": (75, 200)}
        for purpose, heights in stages.items():
            with self.subTest(purpose=purpose):
                start, end = self.stage_poses(*heights, rx=179.997955, ry=.000286)
                grasp.require_stage(self.config, {"normal": [0, 0, 1], "offset_mm": 109},
                                    purpose, start, end)

    def test_downward_tolerance_is_angular_and_independent_of_yaw(self):
        for rx, ry in ((180, 0), (-180, 0), (179.901, 0), (180, .099)):
            for yaw in (-180, 0, 112.673347):
                with self.subTest(rx=rx, ry=ry, yaw=yaw):
                    for purpose, heights in (("Pregrasp descent", (200, 75)),
                                             ("Grasp alignment", (200, 200))):
                        grasp.require_stage(
                            self.config, {"normal": [0, 0, 1], "offset_mm": 109}, purpose,
                            *self.stage_poses(*heights, rx=rx, ry=ry, rz=yaw))

    def test_excessive_absolute_tilt_is_rejected_without_rotation_during_move(self):
        for rx, ry in ((179.899, 0), (180, .101), (179.92, .08), (175, 0)):
            for purpose, heights in (("Pregrasp descent", (200, 75)),
                                     ("Grasp alignment", (200, 200))):
                with self.subTest(rx=rx, ry=ry, purpose=purpose):
                    with self.assertRaisesRegex(ValueError, "downward"):
                        grasp.require_stage(
                            self.config, {"normal": [0, 0, 1], "offset_mm": 109}, purpose,
                            *self.stage_poses(*heights, rx=rx, ry=ry))

    def test_small_tilt_does_not_relax_xy_or_clearance_bounds(self):
        plane = {"normal": [0, 0, 1], "offset_mm": 109}
        start, end = self.stage_poses(200, 75, rx=179.997955, ry=.000286)
        end[0, 3] += .201
        with self.assertRaisesRegex(ValueError, "vertical"):
            grasp.require_stage(self.config, plane, "Pregrasp descent", start, end)
        start, end = self.stage_poses(200, 69, rx=179.997955, ry=.000286)
        with self.assertRaisesRegex(ValueError, "clearance bounds"):
            grasp.require_stage(self.config, plane, "Pregrasp descent", start, end)


if __name__ == "__main__":
    unittest.main()
