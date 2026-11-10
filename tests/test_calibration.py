# SPDX-License-Identifier: MulanPSL-2.0
import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from handeye import calibration
from handeye.geometry import pose_to_transform


BOARD = {"inner_corners": [9, 6], "square_size_mm": 25.0, "top_left_black": True}


def flat_board():
    image = np.full((380, 500), 255, dtype=np.uint8)
    for row in range(7):
        for col in range(10):
            image[50 + row * 40:50 + (row + 1) * 40, 50 + col * 40:50 + (col + 1) * 40] = 0 if (row + col) % 2 == 0 else 255
    return image


def render_board(T_camera_board, K, size=(1024, 768)):
    """Render the physical black-top-left board through a pinhole camera."""
    scale = 4
    texture = flat_board()[50:330, 50:450]
    outer = np.array([[-25, -25, 0], [225, -25, 0], [225, 150, 0], [-25, 150, 0]], dtype=float)
    rvec, _ = cv2.Rodrigues(T_camera_board[:3, :3])
    target, _ = cv2.projectPoints(outer, rvec, T_camera_board[:3, 3], K, np.zeros(5))
    # Image edges represent the square boundaries; half-pixel edges avoid bias.
    source = np.array([[-.5, -.5], [399.5, -.5], [399.5, 279.5], [-.5, 279.5]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(source, (target.reshape(4, 2) * scale).astype(np.float32))
    result = cv2.warpPerspective(texture, H, (size[0] * scale, size[1] * scale), flags=cv2.INTER_LINEAR, borderValue=220)
    return cv2.resize(result, size, interpolation=cv2.INTER_AREA)


def test_object_points_units_and_shape():
    points = calibration.object_points(BOARD)
    assert points.shape == (54, 3)
    np.testing.assert_allclose(points[1], [25, 0, 0])
    np.testing.assert_allclose(points[9], [0, 25, 0])
    with pytest.raises(ValueError, match="opposite-parity"):
        calibration.object_points({"inner_corners": [9, 7], "square_size_mm": 25})


def test_checkerboard_order_stays_physical_after_180_degree_image_rotation():
    image = flat_board()
    original = calibration.detect_corners(image, BOARD).reshape(-1, 2)
    rotated = calibration.detect_corners(cv2.rotate(image, cv2.ROTATE_180), BOARD).reshape(-1, 2)
    restored = np.array([image.shape[1] - 1, image.shape[0] - 1]) - rotated
    np.testing.assert_allclose(original, restored, atol=.2)
    np.testing.assert_allclose(original[0], [89.5, 89.5], atol=.3)


def test_board_pose_from_rendered_tilted_image():
    K = np.array([[1100, 0, 512], [0, 1100, 384], [0, 0, 1]], dtype=float)
    expected = pose_to_transform([-110, -70, 700, 25, -15, 8])
    image = render_board(expected, K)
    actual, quality = calibration.solve_board_pose(image, K, np.zeros(5), BOARD)
    np.testing.assert_allclose(actual[:3, 3], expected[:3, 3], atol=.7)
    angle = np.rad2deg(Rotation.from_matrix(actual[:3, :3].T @ expected[:3, :3]).magnitude())
    assert angle < .25
    assert quality["reprojection_rms_px"] < .5


def make_synthetic_samples(tmp_path):
    flange_camera = pose_to_transform([40, -25, 65, 12, -7, 25])
    base_board = pose_to_transform([320, 70, 20, 4, -7, 20])
    rng = np.random.default_rng(52)
    samples = []
    camera_boards = []
    for i in range(16):
        base_flange = pose_to_transform([*rng.uniform([100, -80, 250], [400, 150, 500]), *rng.uniform([-40, -40, -70], [40, 40, 70])])
        path = tmp_path / f"{i:02}.png"
        # Encoded index lets the monkeypatched image pose reader select a pose.
        assert cv2.imwrite(str(path), np.full((60, 80, 3), i, dtype=np.uint8))
        samples.append({"image": str(path), "T_base_flange": base_flange.tolist()})
        camera_boards.append(np.linalg.inv(flange_camera) @ np.linalg.inv(base_flange) @ base_board)
    intrinsic = {"K": [[600, 0, 40], [0, 600, 30], [0, 0, 1]], "dist": [0] * 5, "image_size": [80, 60], "board": BOARD}
    return flange_camera, base_board, samples, camera_boards, intrinsic


def test_handeye_recovers_nontrivial_camera_transform(tmp_path, monkeypatch):
    expected, base_board, samples, camera_boards, intrinsics = make_synthetic_samples(tmp_path)
    def board_pose(image, K, dist, board, *, max_rms_px=2.0):
        return camera_boards[int(image[0, 0, 0])], {"reprojection_rms_px": 0.0}
    monkeypatch.setattr(calibration, "solve_board_pose", board_pose)
    result = calibration.solve_handeye(samples, BOARD, intrinsics)
    np.testing.assert_allclose(result["T_flange_camera"], expected, atol=1e-8)
    np.testing.assert_allclose(result["T_base_board"], base_board, atol=1e-8)
    assert result["metrics"]["board_position_rms_mm"] < 1e-8


def test_handeye_rejects_translation_only_before_loading_images():
    samples = [{"T_base_flange": pose_to_transform([i * 20, 0, 100, 0, 0, 0]), "image": "missing.png"} for i in range(10)]
    with pytest.raises(ValueError, match="orientation diversity"):
        calibration.solve_handeye(samples, BOARD, {"K": np.eye(3), "dist": np.zeros(5), "image_size": [640, 480]})


def test_handeye_rejects_inconsistent_pose_image_pairs(tmp_path, monkeypatch):
    _, _, samples, camera_boards, intrinsics = make_synthetic_samples(tmp_path)
    def wrong_board_pose(image, K, dist, board, *, max_rms_px=2.0):
        return camera_boards[(int(image[0, 0, 0]) + 1) % len(camera_boards)], {"reprojection_rms_px": 0.0}
    monkeypatch.setattr(calibration, "solve_board_pose", wrong_board_pose)
    with pytest.raises(ValueError, match="inconsistent|orthonormal|finite"):
        calibration.solve_handeye(samples, BOARD, intrinsics)


def test_intrinsics_rejects_small_image_collection():
    with pytest.raises(ValueError, match="at least ten"):
        calibration.calibrate_intrinsics(["unused.png"] * 7, BOARD)


def test_handeye_passes_board_reprojection_limit_and_rejects_bad_limits(tmp_path, monkeypatch):
    _, _, samples, camera_boards, intrinsics = make_synthetic_samples(tmp_path)
    observed_limits = []
    def board_pose(image, K, dist, board, *, max_rms_px=2.0):
        observed_limits.append(max_rms_px)
        simulated_rms = .8
        if simulated_rms > max_rms_px:
            raise ValueError("Chessboard reprojection RMS exceeds requested limit")
        return camera_boards[int(image[0, 0, 0])], {"reprojection_rms_px": simulated_rms}
    monkeypatch.setattr(calibration, "solve_board_pose", board_pose)
    result = calibration.solve_handeye(samples, BOARD, intrinsics, max_board_rms_px=1.0)
    assert observed_limits == [1.0] * len(samples)
    assert result["quality_limits"]["max_board_rms_px"] == 1.0
    with pytest.raises(ValueError, match="reprojection RMS"):
        calibration.solve_handeye(samples, BOARD, intrinsics, max_board_rms_px=.5)
    for invalid in (0, -1, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="quality limits"):
            calibration.solve_handeye(samples, BOARD, intrinsics, max_board_rms_px=invalid)


def rendered_dataset(tmp_path):
    K = np.array([[1100, 0, 512], [0, 1120, 384], [0, 0, 1]], dtype=float)
    rng = np.random.default_rng(84)
    paths, poses = [], []
    for index in range(14):
        angles = rng.uniform([-35, -30, -35], [35, 30, 35])
        pose = pose_to_transform([0, 0, 0, *angles])
        center = rng.uniform([-65, -45, 650], [65, 45, 850])
        pose[:3, 3] = center - pose[:3, :3] @ np.array([100, 62.5, 0])
        path = tmp_path / f"rendered_{index:02}.png"
        assert cv2.imwrite(str(path), render_board(pose, K))
        paths.append(path)
        poses.append(pose)
    return K, paths, poses


def test_rendered_rgb_intrinsics_recover_camera(tmp_path):
    K, paths, _ = rendered_dataset(tmp_path)
    result = calibration.calibrate_intrinsics(paths, BOARD)
    measured = np.array(result["K"])
    np.testing.assert_allclose(np.diag(measured)[:2], np.diag(K)[:2], rtol=.01)
    np.testing.assert_allclose(measured[:2, 2], K[:2, 2], atol=3)
    assert result["rms"] < .3


def test_rendered_rgb_handeye_end_to_end(tmp_path):
    K, paths, camera_boards = rendered_dataset(tmp_path)
    expected = pose_to_transform([35, -27, 68, 12, -8, 24])
    base_board = pose_to_transform([350, 30, 50, 3, -8, 12])
    samples = [{"image": str(path), "T_base_flange": base_board @ np.linalg.inv(camera_board) @ np.linalg.inv(expected)} for path, camera_board in zip(paths, camera_boards)]
    result = calibration.solve_handeye(samples, BOARD, {"K": K, "dist": np.zeros(5), "image_size": [1024, 768], "board": BOARD})
    measured = np.array(result["T_flange_camera"])
    np.testing.assert_allclose(measured[:3, 3], expected[:3, 3], atol=1.5)
    orientation_error = np.rad2deg(Rotation.from_matrix(expected[:3, :3].T @ measured[:3, :3]).magnitude())
    assert orientation_error < .15
    assert result["metrics"]["board_position_rms_mm"] < .5
