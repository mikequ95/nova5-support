# SPDX-License-Identifier: MulanPSL-2.0
"""Rigid transforms for the Nova RGB hand-eye workflow.

Every translation is in millimetres. ``T_A_B`` maps coordinates expressed in
frame B into frame A. Dobot poses are [x, y, z, rx, ry, rz] with degrees and
fixed-axis XYZ Euler angles: R = Rz(rz) @ Ry(ry) @ Rx(rx).
"""

from __future__ import annotations

import warnings

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def validate_transform(transform) -> np.ndarray:
    """Return a validated copy of a finite, proper 4 by 4 rigid transform."""
    matrix = np.asarray(transform, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("A transform must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8, rtol=0):
        raise ValueError("Transform last row must be [0, 0, 0, 1]")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0):
        raise ValueError("Transform rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1, atol=1e-6, rtol=0):
        raise ValueError("Transform rotation must have determinant +1")
    return matrix.copy()


def pose_to_transform(pose6) -> np.ndarray:
    """Convert Dobot x/y/z millimetres, fixed XYZ rx/ry/rz degrees to T."""
    pose = np.asarray(pose6, dtype=float)
    if pose.shape != (6,) or not np.all(np.isfinite(pose)):
        raise ValueError("Pose must contain six finite values: x,y,z,rx,ry,rz")
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler("xyz", pose[3:], degrees=True).as_matrix()
    result[:3, 3] = pose[:3]
    return result


def transform_to_pose(transform) -> list[float]:
    """Return one equivalent Dobot Euler pose; angles need not match input.

    At an Euler singularity SciPy chooses rz=0, preserving the represented
    rotation. Cartesian command paths must still obey the robot's own limits.
    """
    matrix = validate_transform(transform)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Gimbal lock detected")
        angles = Rotation.from_matrix(matrix[:3, :3]).as_euler("xyz", degrees=True)
    return np.concatenate((matrix[:3, 3], angles)).tolist()


def target_flange(T_base_tcp, T_flange_tcp) -> np.ndarray:
    """Compute a flange target for an external TCP (controller Tool=0).

    Do not also configure this same TCP offset in the controller: doing both
    would compensate twice. The tool translation rotates with the flange.
    """
    tcp = validate_transform(T_base_tcp)
    offset = validate_transform(T_flange_tcp)
    return validate_transform(tcp @ np.linalg.inv(offset))


def validate_camera(K, dist) -> tuple[np.ndarray, np.ndarray]:
    """Validate the standard OpenCV pinhole camera calibration."""
    matrix = np.asarray(K, dtype=float)
    distortion = np.asarray(dist, dtype=float).reshape(-1)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("K must be a finite 3x3 camera matrix")
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise ValueError("Camera focal lengths must be positive")
    if not np.allclose(matrix[2], [0, 0, 1], atol=1e-9, rtol=0):
        raise ValueError("K last row must be [0,0,1]")
    if not np.allclose([matrix[0, 1], matrix[1, 0]], 0, atol=1e-9):
        raise ValueError("OpenCV pinhole model requires zero skew")
    if distortion.size not in (4, 5, 8, 12, 14) or not np.all(np.isfinite(distortion)):
        raise ValueError("dist must contain 4, 5, 8, 12, or 14 finite coefficients")
    return matrix, distortion


def intersect_pixel_plane(pixel, K, dist, T_base_camera, plane_normal, plane_offset_mm) -> np.ndarray:
    """Intersect an undistorted pixel ray with n dot X + d = 0 in base.

    This plane is the physical clicked surface, not the requested hover plane.
    The returned xyz is in millimetres. Rays parallel to the plane or whose
    intersection is behind the camera are rejected.
    """
    point = np.asarray(pixel, dtype=float)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise ValueError("Pixel must be two finite values")
    matrix, distortion = validate_camera(K, dist)
    camera = validate_transform(T_base_camera)
    normal = np.asarray(plane_normal, dtype=float)
    if normal.shape != (3,) or not np.all(np.isfinite(normal)):
        raise ValueError("Plane normal must be three finite values")
    length = np.linalg.norm(normal)
    if length < 1e-12 or not np.isfinite(plane_offset_mm):
        raise ValueError("Plane normal must be nonzero and plane offset finite")
    normal = normal / length
    offset = float(plane_offset_mm) / length
    normalized = cv2.undistortPoints(point.reshape(1, 1, 2), matrix, distortion).reshape(2)
    direction = camera[:3, :3] @ np.array([normalized[0], normalized[1], 1.0])
    direction = direction / np.linalg.norm(direction)
    origin = camera[:3, 3]
    denominator = float(normal @ direction)
    if abs(denominator) < 1e-8:
        raise ValueError("Pixel ray is parallel to the work plane")
    distance = -float(normal @ origin + offset) / denominator
    if not np.isfinite(distance) or distance <= 0:
        raise ValueError("Plane intersection is behind or at the camera")
    return origin + distance * direction


def fit_tcp_pivot(T_base_flange_samples) -> dict:
    """Fit a TCP translation from many orientations touching one fixed point.

    Only the translation is observable; the operator must define the TCP axes
    separately. The contact point must remain the SAME physical/virtual point.
    """
    transforms = [validate_transform(t) for t in T_base_flange_samples]
    if len(transforms) < 6:
        raise ValueError("Pivot calibration requires at least six different orientations")
    A = np.vstack([np.hstack((t[:3, :3], -np.eye(3))) for t in transforms])
    b = np.concatenate([-t[:3, 3] for t in transforms])
    singular = np.linalg.svd(A, compute_uv=False)
    if singular[-1] < singular[0] * 0.02:
        raise ValueError("Pivot poses lack sufficient rotation about different axes")
    solution, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    if rank != 6:
        raise ValueError("Pivot calibration is underconstrained")
    residuals = (A @ solution - b).reshape(-1, 3)
    errors = np.linalg.norm(residuals, axis=1)
    return {
        "tcp_translation_mm": solution[:3].tolist(),
        "fixed_point_base_mm": solution[3:].tolist(),
        "rms_mm": float(np.sqrt(np.mean(errors ** 2))),
        "max_error_mm": float(errors.max()),
        "condition_number": float(singular[0] / singular[-1]),
        "sample_count": len(transforms),
        "note": "Translation only; TCP axis orientation must be configured separately.",
    }
