# SPDX-License-Identifier: MulanPSL-2.0
"""Configuration, provenance and click-target validation; no hardware imports."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import (intersect_pixel_plane, pose_to_transform, target_flange,
                       transform_to_pose, validate_transform)


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_config(path):
    path = Path(path).resolve()
    config = read_json(path)
    config["_root"] = str(path.parent)
    config["_path"] = str(path)
    sdk = config["robot"].get("sdk_path")
    if sdk:
        config["robot"]["sdk_path"] = str((path.parent / sdk).resolve())
    return config


def resolve(config, path):
    return (Path(config["_root"]) / path).resolve()


def board_spec(board):
    return {key: board[key] for key in ("inner_corners", "square_size_mm", "top_left_black")}


def require_board(config):
    if config["board"].get("print_size_verified") is not True:
        raise ValueError("Measure the printed board, set square_size_mm and print_size_verified=true first")


def tcp_transform(config):
    pose = config["tcp"].get("flange_to_tcp_mm_deg")
    if pose is None:
        raise ValueError("TCP is unknown. Set tcp.flange_to_tcp_mm_deg to six measured mm/degree values")
    return pose_to_transform(pose)


def load_intrinsics(config):
    path = resolve(config, config["files"]["intrinsics"])
    data = read_json(path)
    if board_spec(data["board"]) != board_spec(config["board"]):
        raise ValueError("Printed board geometry changed; recalibrate")
    if data["image_size"] != [config["camera"]["width"], config["camera"]["height"]]:
        raise ValueError("Camera resolution does not match intrinsics")
    rms = float(data["rms"])
    if not np.isfinite(rms) or rms > config["validation"]["intrinsics_max_rms_px"]:
        raise ValueError("Intrinsics reprojection error exceeds configured limit")
    return data


def load_handeye(config):
    data = read_json(resolve(config, config["files"]["handeye"]))
    if data.get("intrinsics_sha256") != digest(resolve(config, config["files"]["intrinsics"])):
        raise ValueError("Hand-eye result belongs to different intrinsics; recalibrate")
    validate_transform(np.asarray(data["T_flange_camera"]))
    if data.get("quality_passed") is not True:
        raise ValueError("Hand-eye quality gates did not pass")
    for metric, limit in (("board_position_rms_mm", "handeye_translation_rms_mm"),
                          ("board_rotation_rms_deg", "handeye_rotation_rms_deg")):
        value = float(data["metrics"][metric])
        threshold = float(config["validation"][limit])
        if not np.isfinite(value) or not np.isfinite(threshold) or threshold <= 0 or value > threshold:
            raise ValueError(f"Hand-eye {metric} exceeds current quality limit")
    if not data.get("views") or any(not np.isfinite(v["reprojection_rms_px"]) or
            v["reprojection_rms_px"] > config["validation"]["board_max_rms_px"] for v in data["views"]):
        raise ValueError("Hand-eye chessboard reprojection error exceeds current limit")
    return data


def load_plane(config):
    data = read_json(resolve(config, config["files"]["plane"]))
    if data.get("handeye_sha256") != digest(resolve(config, config["files"]["handeye"])):
        raise ValueError("Work plane belongs to a different hand-eye calibration")
    n = np.asarray(data["normal"], dtype=float)
    if n.shape != (3,) or not np.all(np.isfinite(n)) or not np.isclose(np.linalg.norm(n), 1):
        raise ValueError("Work plane must have a finite unit normal")
    if not np.isfinite(data["offset_mm"]):
        raise ValueError("Invalid work plane offset")
    error = data["quality"]["reprojection_rms_px"]
    if not np.isfinite(error) or error > config["validation"]["board_max_rms_px"]:
        raise ValueError("Plane reprojection error exceeds current limit")
    return data


def setup_signature(config):
    """Bind physical validation to this exact camera, TCP and calibration set."""
    setup = {"files": {k: digest(resolve(config, config["files"][k])) for k in ("intrinsics", "handeye", "plane")},
             "tcp": tcp_transform(config).tolist(), "board": board_spec(config["board"]),
             "camera": config["camera"]}
    return hashlib.sha256(json.dumps(setup, sort_keys=True, allow_nan=False).encode()).hexdigest()


def check_camera_identity(camera, intrinsics):
    if camera.identity != intrinsics.get("camera_identity"):
        raise ValueError("Camera identity/profile differs from calibration; do not reuse these intrinsics")


def pose_distance(a, b):
    a, b = np.asarray(a), np.asarray(b)
    translation = float(np.linalg.norm(a[:3, 3] - b[:3, 3]))
    angle = float(np.rad2deg(Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude()))
    return translation, angle


def workspace_bounds(config, *, error_message="Set finite XYZ workspace_min_mm < workspace_max_mm"):
    motion = config["motion"]
    low = np.asarray(motion.get("workspace_min_mm"), dtype=float)
    high = np.asarray(motion.get("workspace_max_mm"), dtype=float)
    if (low.shape != (3,) or high.shape != (3,)
            or not np.all(np.isfinite([low, high])) or not np.all(low < high)):
        raise ValueError(error_message)
    return low, high


def plan_click(pixel, z_mm, T_base_flange, config, intrinsics, handeye, plane, orientation=None,
               *, height_mm=None):
    """Intersect the clicked surface; use absolute Z or vertical height above it."""
    T_base_flange = np.asarray(T_base_flange, dtype=float)
    validate_transform(T_base_flange)
    if (z_mm is None) == (height_mm is None):
        raise ValueError("Specify exactly one of base Z and height above surface")
    if height_mm is not None:
        if not np.isfinite(height_mm) or height_mm <= 0:
            raise ValueError("Height above surface must be finite and positive")
    elif not np.isfinite(z_mm):
        raise ValueError("Target base Z must be finite")
    u, v = pixel
    width, height = intrinsics["image_size"]
    if not (np.isfinite(u) and np.isfinite(v) and 0 <= u < width and 0 <= v < height):
        raise ValueError("Pixel is outside the original calibrated image")
    T_flange_tcp = tcp_transform(config)
    T_base_camera = T_base_flange @ np.asarray(handeye["T_flange_camera"])
    point = intersect_pixel_plane(pixel, intrinsics["K"], intrinsics["dist"],
                                  T_base_camera, plane["normal"], plane["offset_mm"])
    if height_mm is not None:
        z_mm = float(point[2] + height_mm)
    T_current_tcp = T_base_flange @ T_flange_tcp
    target = T_current_tcp.copy()
    if orientation is not None:
        target[:3, :3] = np.asarray(orientation)
    target[:3, 3] = [point[0], point[1], z_mm]
    validate_transform(target)
    if z_mm <= point[2]:
        raise ValueError("Target TCP must hover above the clicked surface in base Z")
    flange = target_flange(target, T_flange_tcp)
    return {"pixel": [float(u), float(v)], "surface_point_mm": point.tolist(),
            "height_above_surface_mm": float(z_mm - point[2]),
            "tcp_pose_mm_deg": transform_to_pose(target),
            "flange_pose_mm_deg": transform_to_pose(flange),
            "T_base_tcp": target.tolist(), "T_base_flange": flange.tolist()}


def require_motion(config, current_flange, plan):
    """First version: short, fixed-orientation linear moves inside measured bounds."""
    if config["motion"].get("enabled") is not True:
        raise ValueError("motion.enabled must be explicitly set true after validation")
    if config["robot"].get("compatibility_verified") is not True:
        raise ValueError("Verify SDK/controller compatibility before enabling motion")
    if config["tcp"].get("measured_and_verified") is not True:
        raise ValueError("TCP must be physically verified before motion")
    if config["validation"].get("independent_validation_passed") is not True:
        raise ValueError("Independent physical calibration validation is required")
    validation = read_json(resolve(config, "calibrations/validation.json"))
    if validation.get("setup_sha256") != setup_signature(config):
        raise ValueError("Physical validation is stale after calibration/TCP/camera changes")
    require_board(config)
    motion = config["motion"]
    if motion["workspace_min_mm"] is None or motion["workspace_max_mm"] is None:
        raise ValueError("Set measured workspace bounds before motion")
    low, high = workspace_bounds(config, error_message="Invalid workspace bounds")
    target = np.asarray(plan["T_base_flange"])
    current = np.asarray(current_flange)
    distance, angle = pose_distance(current, target)
    max_step = float(motion["max_step_mm"])
    if not np.isfinite(max_step) or max_step <= 0 or distance > max_step:
        raise ValueError(f"Move distance {distance:.2f} mm exceeds max_step_mm")
    if angle > 0.1:
        raise ValueError("First version requires unchanged tool orientation; reacquire image")
    tcp = tcp_transform(config)
    # Fixed orientation makes TCP and flange paths straight; this bounds endpoints,
    # not the arm links, fingers, camera or obstacles. This is not collision checking.
    for label, matrix in (("current flange", current), ("target flange", target),
                          ("current TCP", current @ tcp), ("target TCP", np.asarray(plan["T_base_tcp"]))):
        if np.any(matrix[:3, 3] < low) or np.any(matrix[:3, 3] > high):
            raise ValueError(f"{label} is outside configured workspace")
    if type(motion["speed_mm_s"]) is not int or not 1 <= motion["speed_mm_s"] <= 20:
        raise ValueError("First version speed_mm_s must be an integer from 1 to 20")
    if not np.isfinite(motion["timeout_s"]) or motion["timeout_s"] <= 0:
        raise ValueError("timeout_s must be finite and positive")
