# SPDX-License-Identifier: MulanPSL-2.0
"""Operator-confirmed click tests; no calibration verification flags are altered."""
import numpy as np

from .geometry import validate_transform, target_flange
from .workflow import pose_distance, tcp_transform, workspace_bounds

HEIGHT_MM = 150.0
SPEED_MM_S = 50
TIMEOUT_S = 30.0


def table_z(plane, x, y):
    normal = np.asarray(plane["normal"], dtype=float)
    offset = float(plane["offset_mm"])
    if (normal.shape != (3,) or not np.all(np.isfinite(normal))
            or not np.isfinite(offset) or normal[2] < 0.95):
        raise ValueError("The click test requires a calibrated, nearly horizontal work plane with an upward normal")
    return float(-(normal[0] * x + normal[1] * y + offset) / normal[2])


def require_test_motion(config, current_flange, plan, anchor_flange, plane, *, execute):
    """Check target consistency without requiring manual pre-positioning."""
    if execute is not True:
        raise ValueError("Preview only; explicitly enable execution and press M to move")
    current = validate_transform(current_flange)
    anchor = validate_transform(anchor_flange)
    target = validate_transform(plan["T_base_flange"])
    target_tcp = validate_transform(plan["T_base_tcp"])
    tcp = tcp_transform(config)
    point = np.asarray(plan["surface_point_mm"], dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError("Invalid clicked surface point")
    if abs(point[2] - table_z(plane, *point[:2])) > 1e-5:
        raise ValueError("Clicked point is not on the calibrated table")
    if (not np.allclose(target_tcp[:2, 3], point[:2], atol=1e-6, rtol=0)
            or abs(target_tcp[2, 3] - point[2] - HEIGHT_MM) > 1e-6):
        raise ValueError("The test target must be 150 mm vertically above the clicked position")
    if not np.allclose(target, target_flange(target_tcp, tcp), atol=1e-6, rtol=0):
        raise ValueError("TCP and compensated flange targets disagree")
    for a, b in ((current, target), (anchor, current), (anchor, target)):
        _, angle = pose_distance(a, b)
        if angle > 0.1:
            raise ValueError("Keep the end-effector orientation recorded at startup; restart after changing orientation")
    low = config["motion"].get("workspace_min_mm")
    high = config["motion"].get("workspace_max_mm")
    if (low is None) != (high is None):
        raise ValueError("Workspace bounds must be both specified or both absent")
    if low is not None:
        low, high = workspace_bounds(config, error_message="Invalid workspace bounds")
    for label, f in (("current", current), ("target", target)):
        t = f @ tcp
        if low is not None:
            for endpoint in (f[:3, 3], t[:3, 3]):
                if np.any(endpoint < low) or np.any(endpoint > high):
                    raise ValueError(f"{label} flange/TCP exceeds configured workspace")


def main(argv=None):
    import argparse
    from .cli import click
    from .workflow import load_config

    parser = argparse.ArgumentParser(description="Click the table to target the gripper TCP 150 mm above it; click to preview and M to confirm motion")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--execute", action="store_true", help="Allow execution with M; preview only by default")
    args = parser.parse_args(argv)
    args.z_mm = None
    args.height_mm = HEIGHT_MM
    args.test_motion = True
    try:
        config = load_config(args.config)
        print("Target: 150 mm above the clicked surface along base Z, keeping the current orientation.")
        print("Click to preview; M confirms motion (requires --execute); R refreshes; Q/ESC exits or requests stop.")
        print("Test speed is 50 mm/s; move linearly from the current pose to the target without pre-positioning.")
        print("Ensure the entire straight path is clear; this program does not plan around obstacles.")
        print("Execution requires TCP/IP control mode, an enabled arm and User0/Tool0; this script does not switch modes, enable the arm or clear alarms.")
        click(args, config)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Test exited.\n")
