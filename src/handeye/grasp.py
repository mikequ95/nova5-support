# SPDX-License-Identifier: MulanPSL-2.0
"""Pure endpoint and stage checks for the demonstrated Nova5 grasp."""
import numpy as np

from .geometry import validate_transform
from .test_motion import table_z
from .workflow import pose_distance, tcp_transform, workspace_bounds

DOWNWARD_TOLERANCE_DEG = 0.1


def _is_downward(flange):
    """Allow small feedback tilt from base -Z, independently of wrist yaw."""
    axis = flange[:3, 2]
    tilt_deg = np.degrees(np.arctan2(np.linalg.norm(axis[:2]), -axis[2]))
    return tilt_deg <= DOWNWARD_TOLERANCE_DEG


def require_grasp_motion(config, **flanges):
    """Require explicit permission, 50 mm/s and finite flange/TCP endpoints."""
    motion = config["motion"]
    if motion.get("enabled") is not True:
        raise ValueError("Set motion.enabled=true after verifying the installation")
    speed = motion.get("speed_mm_s")
    if type(speed) not in (int, float) or speed != 50:
        raise ValueError("This grasp requires motion.speed_mm_s=50")
    low, high = workspace_bounds(config)
    tcp = tcp_transform(config)
    for label, transform in flanges.items():
        flange = validate_transform(transform)
        for point_name, point in (("flange", flange[:3, 3]),
                                  ("TCP", (flange @ tcp)[:3, 3])):
            outside = np.flatnonzero((point < low) | (point > high))
            if outside.size:
                axis = int(outside[0])
                raise ValueError(
                    f"{label} {point_name} {'XYZ'[axis]}={point[axis]:.3f} mm "
                    f"outside workspace [{low[axis]:.3f}, {high[axis]:.3f}] mm")


def require_stage(config, plane, purpose, expected_flange, target_flange):
    """Preserve the task's distance, orientation and table-clearance bounds."""
    expected = validate_transform(expected_flange)
    target = validate_transform(target_flange)
    require_grasp_motion(config, start=expected, target=target)
    distance, angle = pose_distance(expected, target)
    max_distance = {
        "Object hover observation": 600, "Grasp alignment": 100,
        "Wrist transition": 100, "Grasp centering": 15,
        "Pregrasp descent": 140, "Grasp descent": 65, "Lift object": 50,
        "Place object": 50, "Postplace retreat": 50,
        "Grasp clearance retreat": 140, "Return alignment": 100,
        "Return home": 600,
    }.get(purpose)
    if max_distance is None:
        raise ValueError("Unsupported motion purpose")
    alignment = purpose in ("Grasp alignment", "Wrist transition", "Return alignment")
    if distance > max_distance or angle > (70 if alignment else .1):
        raise ValueError("Motion exceeds the planned distance or rotation bound")
    tcp = tcp_transform(config)
    start_tcp, target_tcp = expected @ tcp, target @ tcp
    if alignment:
        if np.linalg.norm(start_tcp[:3, 3] - target_tcp[:3, 3]) > .2:
            raise ValueError("Alignment must preserve TCP position")
        if purpose == "Grasp alignment" and not _is_downward(target):
            raise ValueError("Grasp alignment must end vertically downward")
    bounds = {
        "Pregrasp descent": (199, 70, 80), "Grasp descent": (70, 25, 35),
        "Lift object": (25, 50, 85), "Place object": (50, 25, 35),
        "Postplace retreat": (25, 70, 80), "Grasp clearance retreat": (70, 199, 205),
    }
    if purpose in bounds:
        if (np.linalg.norm(start_tcp[:2, 3] - target_tcp[:2, 3]) > .2
                or not _is_downward(target)):
            raise ValueError("Grasp moves must be vertical with the gripper downward")
        start_height = start_tcp[2, 3] - table_z(plane, *start_tcp[:2, 3])
        end_height = target_tcp[2, 3] - table_z(plane, *target_tcp[:2, 3])
        min_start, min_end, max_end = bounds[purpose]
        if start_height < min_start - .5 or not min_end - .5 <= end_height <= max_end + .5:
            raise ValueError("Grasp height outside planned clearance bounds")
        if (purpose in ("Lift object", "Postplace retreat", "Grasp clearance retreat")
                and end_height <= start_height):
            raise ValueError("Retreat must move upward")
        if purpose == "Place object" and end_height >= start_height:
            raise ValueError("Place must move downward")
    elif any(t[2, 3] - table_z(plane, *t[:2, 3]) < 199 for t in (start_tcp, target_tcp)):
        raise ValueError("Observation motion requires at least 200 mm table clearance")
