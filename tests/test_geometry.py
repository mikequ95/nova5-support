# SPDX-License-Identifier: MulanPSL-2.0
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from handeye.geometry import (
    fit_tcp_pivot,
    intersect_pixel_plane,
    pose_to_transform,
    target_flange,
    transform_to_pose,
    validate_transform,
)


def test_fixed_xyz_euler_and_roundtrip():
    pose = [100, -20, 30, 25, -35, 70]
    T = pose_to_transform(pose)
    rx = Rotation.from_euler("x", 25, degrees=True).as_matrix()
    ry = Rotation.from_euler("y", -35, degrees=True).as_matrix()
    rz = Rotation.from_euler("z", 70, degrees=True).as_matrix()
    np.testing.assert_allclose(T[:3, :3], rz @ ry @ rx, atol=1e-12)
    np.testing.assert_allclose(pose_to_transform(transform_to_pose(T)), T, atol=1e-10)


def test_tcp_full_rigid_inverse_rotates_translation():
    offset = pose_to_transform([10, -15, 180, 5, 0, 20])
    desired = pose_to_transform([300, 50, 200, 180, 30, -60])
    flange = target_flange(desired, offset)
    np.testing.assert_allclose(flange @ offset, desired, atol=1e-10)
    assert not np.allclose(flange[:3, 3], desired[:3, 3] - offset[:3, 3])


@pytest.mark.parametrize("bad", [np.zeros((4, 4)), np.diag([-1, 1, 1, 1]), np.eye(3), np.full((4, 4), np.nan)])
def test_invalid_transforms_rejected(bad):
    with pytest.raises(ValueError):
        validate_transform(bad)


def test_ray_plane_and_target_height_are_different_steps():
    K = [[600, 0, 320], [0, 600, 240], [0, 0, 1]]
    # Camera at z=500 with optical axis pointing toward table z=0.
    camera = pose_to_transform([100, 50, 500, 180, 0, 0])
    point = intersect_pixel_plane([440, 300], K, np.zeros(5), camera, [0, 0, 1], 0)
    np.testing.assert_allclose(point, [200, 0, 0], atol=1e-9)
    wrong_hover_plane = intersect_pixel_plane([440, 300], K, np.zeros(5), camera, [0, 0, 1], -100)
    assert not np.allclose(point[:2], wrong_hover_plane[:2])


def test_ray_undistortion_recovers_projected_point():
    import cv2
    K = np.array([[700, 0, 320], [0, 710, 240], [0, 0, 1]], dtype=float)
    dist = np.array([0.1, -0.08, 0.001, -0.002, 0.01])
    xyz = np.array([100, 70, 500], dtype=float)
    pixel, _ = cv2.projectPoints(xyz[None], np.zeros(3), np.zeros(3), K, dist)
    observed = intersect_pixel_plane(pixel.reshape(2), K, dist, np.eye(4), [0, 0, 1], -500)
    np.testing.assert_allclose(observed, xyz, atol=1e-5)


def test_parallel_and_behind_camera_rays_rejected():
    K = [[600, 0, 320], [0, 600, 240], [0, 0, 1]]
    with pytest.raises(ValueError, match="parallel"):
        intersect_pixel_plane([320, 240], K, np.zeros(5), np.eye(4), [1, 0, 0], -20)
    with pytest.raises(ValueError, match="behind"):
        intersect_pixel_plane([320, 240], K, np.zeros(5), np.eye(4), [0, 0, 1], 20)


def test_pivot_recovers_tool_translation_and_rejects_no_rotation():
    tool = np.array([13, -4, 160])
    pivot = np.array([300, 10, 100])
    rotations = Rotation.from_euler("xyz", [[0, 0, 0], [30, 0, 0], [-25, 10, 0], [0, 30, 5], [0, -30, 0], [20, 20, 30]], degrees=True)
    poses = []
    for rotation in rotations.as_matrix():
        T = np.eye(4)
        T[:3, :3] = rotation
        T[:3, 3] = pivot - rotation @ tool
        poses.append(T)
    result = fit_tcp_pivot(poses)
    np.testing.assert_allclose(result["tcp_translation_mm"], tool, atol=1e-9)
    np.testing.assert_allclose(result["fixed_point_base_mm"], pivot, atol=1e-9)
    assert result["rms_mm"] < 1e-9
    with pytest.raises(ValueError, match="rotation"):
        fit_tcp_pivot([np.eye(4)] * 6)
