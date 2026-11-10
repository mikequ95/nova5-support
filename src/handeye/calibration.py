# SPDX-License-Identifier: MulanPSL-2.0
"""RGB chessboard calibration with explicit transform directions.

Lengths are mm. T_A_B maps B-frame coordinates to A. The hand-eye input is
actual T_base_flange (Tool=0, User=0), paired with a stationary RGB exposure.
OpenCV returns T_flange_camera. The stationary board is never moved during a
hand-eye capture session. No depth stream is used.

The board has opposite-parity inner-corner counts (default 9 by 6), and its
top-left OUTER square is black. This makes the ordinary chessboard's 180-degree
ordering identifiable by square polarity. Both printed board and full corner
grid must be visible. Same-parity boards are rejected because polarity cannot
remove their 180-degree ambiguity.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import validate_camera, validate_transform


def _board_shape(board: dict) -> tuple[int, int, float]:
    inner = board.get("inner_corners", [9, 6])
    if len(inner) != 2 or any(isinstance(v, bool) or int(v) != v for v in inner):
        raise ValueError("inner_corners must be [columns, rows] of integer corner counts")
    columns, rows = map(int, inner)
    square = float(board.get("square_size_mm", 25.0))
    if min(columns, rows) < 3 or not np.isfinite(square) or square <= 0:
        raise ValueError("Board needs at least 3x3 inner corners and positive measured square size")
    if columns % 2 == rows % 2:
        raise ValueError("Use opposite-parity inner corner counts, e.g. [9,6], to resolve 180-degree ordering")
    return columns, rows, square


def object_points(board: dict) -> np.ndarray:
    """Return Nx3 float32 corners: origin at first inner corner, x right, y down."""
    columns, rows, square = _board_shape(board)
    points = np.zeros((columns * rows, 3), dtype=np.float32)
    points[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * square
    return points


def _gray_image(image) -> np.ndarray:
    if image is None:
        raise ValueError("Image is empty or could not be read")
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim not in (2, 3):
        raise ValueError("Expected an 8-bit grayscale or BGR image")
    if image.ndim == 3:
        if image.shape[2] not in (3, 4):
            raise ValueError("Expected an 8-bit grayscale, BGR or BGRA image")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY if image.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)
    return image


def _canonical_corners(gray, corners, board) -> tuple[np.ndarray, float]:
    """Use dark/light cell polarity to remove detector order reversals."""
    columns, rows, _ = _board_shape(board)
    grid = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2).astype(np.float32)
    homography, _ = cv2.findHomography(grid, corners.reshape(-1, 2), method=0)
    if homography is None:
        raise ValueError("Chessboard homography could not be fitted")
    cells = np.array([(x - 0.5, y - 0.5) for y in range(rows + 1) for x in range(columns + 1)], dtype=np.float32)
    parity = np.array([(x + y) % 2 for y in range(rows + 1) for x in range(columns + 1)])
    centers = cv2.perspectiveTransform(cells.reshape(-1, 1, 2), homography).reshape(-1, 2)
    height, width = gray.shape
    valid = (centers[:, 0] >= 2) & (centers[:, 0] < width - 2) & (centers[:, 1] >= 2) & (centers[:, 1] < height - 2)
    if valid.sum() < 0.9 * len(valid):
        raise ValueError("Show the complete chessboard, including its outer squares")
    values = np.array([float(cv2.getRectSubPix(gray, (3, 3), tuple(map(float, point))).mean()) for point in centers[valid]])
    parity = parity[valid]
    # The first outer square is index (0,0); its expected color is configurable.
    contrast = float(np.median(values[parity == 1]) - np.median(values[parity == 0]))
    if not board.get("top_left_black", True):
        contrast = -contrast
    if abs(contrast) < 25:
        raise ValueError("Chessboard black/white polarity is unreliable; improve focus, exposure and lighting")
    if contrast < 0:
        corners = corners[::-1].copy()
    return corners.astype(np.float32), abs(contrast)


def _detect(image, board) -> tuple[np.ndarray, float]:
    columns, rows, _ = _board_shape(board)
    gray = _gray_image(image)
    found, corners = cv2.findChessboardCornersSB(gray, (columns, rows), flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE)
    if not found:
        found, corners = cv2.findChessboardCorners(gray, (columns, rows), flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if found:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001))
    if not found:
        raise ValueError(f"Could not detect all {columns}x{rows} inner chessboard corners")
    return _canonical_corners(gray, corners, board)


def detect_corners(image, board: dict) -> np.ndarray:
    """Return Nx1x2 canonical float32 image corners or raise ValueError."""
    return _detect(image, board)[0]


def calibrate_intrinsics(image_paths, board: dict, *, max_rms_px: float = 2.0) -> dict:
    """Calibrate RGB K/dist from at least ten distinct full-board images.

    A failed image is reported, never silently dropped. The caller should
    capture different tilts, scales and image positions at one resolution.
    """
    paths = list(image_paths)
    if len(paths) < 10:
        raise ValueError("RGB intrinsics require at least ten calibration images")
    observations = []
    size = None
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot read calibration image: {path}")
        current_size = (image.shape[1], image.shape[0])
        if size is not None and size != current_size:
            raise ValueError("All calibration images must use exactly the same resolution")
        size = current_size
        try:
            observations.append(detect_corners(image, board))
        except ValueError as exc:
            raise ValueError(f"{path}: {exc}") from exc
    if not np.isfinite(max_rms_px) or max_rms_px <= 0:
        raise ValueError("max_rms_px must be finite and positive")
    if max(float(np.linalg.norm(view - observations[0], axis=2).mean()) for view in observations[1:]) < 2:
        raise ValueError("Intrinsic calibration images are effectively identical; capture different board tilts and positions")
    obj = object_points(board)
    rms, K, dist, rotations, translations = cv2.calibrateCamera([obj] * len(paths), observations, size, None, None)
    validate_camera(K, dist)
    per_view = []
    for path, measured, rotation, translation in zip(paths, observations, rotations, translations):
        predicted, _ = cv2.projectPoints(obj, rotation, translation, K, dist)
        error = np.linalg.norm(predicted.reshape(-1, 2) - measured.reshape(-1, 2), axis=1)
        per_view.append({"image": str(Path(path).resolve()), "rms_px": float(np.sqrt(np.mean(error ** 2)))})
    if not np.isfinite(rms) or rms > max_rms_px:
        raise ValueError(f"RGB reprojection RMS {rms:.3f} px exceeds {max_rms_px:g} px; recapture/verify the board")
    normals = np.array([cv2.Rodrigues(rotation)[0][:, 2] for rotation in rotations])
    normal_spread = float(np.rad2deg(np.arccos(np.clip((normals @ normals.T).min(), -1, 1))))
    if normal_spread < 10:
        raise ValueError("Intrinsic views lack board tilt diversity; rotate the board about different axes")
    return {"K": K.tolist(), "dist": dist.reshape(-1).tolist(), "image_size": list(size), "rms": float(rms), "per_view_errors": per_view, "board": dict(board), "units": "mm", "board_normal_spread_deg": normal_spread, "quality_limits": {"max_rms_px": max_rms_px}}


def solve_board_pose(image, K, dist, board: dict, *, max_rms_px: float = 2.0) -> tuple[np.ndarray, dict]:
    """Return (T_camera_board, quality) using canonical chessboard ordering.

    Near-frontal tiny planar targets can have ambiguous pose estimates; collect
    substantial tilts and verify hand-eye consistency on independent poses.
    """
    K, dist = validate_camera(K, dist)
    corners, contrast = _detect(image, board)
    points = object_points(board)
    found, rvec, tvec = cv2.solvePnP(points, corners, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not found:
        raise ValueError("Chessboard pose could not be solved")
    rotation, _ = cv2.Rodrigues(rvec)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = tvec.reshape(3)
    validate_transform(transform)
    camera_points = (rotation @ points.T + tvec).T
    if np.any(camera_points[:, 2] <= 0):
        raise ValueError("Chessboard pose lies behind the RGB camera")
    projected, _ = cv2.projectPoints(points, rvec, tvec, K, dist)
    errors = np.linalg.norm(projected.reshape(-1, 2) - corners.reshape(-1, 2), axis=1)
    rms = float(np.sqrt(np.mean(errors ** 2)))
    if not np.isfinite(max_rms_px) or max_rms_px <= 0:
        raise ValueError("max_rms_px must be finite and positive")
    if not np.isfinite(rms) or rms > max_rms_px:
        raise ValueError(f"Chessboard reprojection RMS {rms:.3f} px exceeds {max_rms_px:g} px")
    quality = {"reprojection_rms_px": rms, "max_reprojection_px": float(errors.max()), "corner_polarity": "canonical", "polarity_contrast_gray": contrast}
    return transform, quality


def _rotation_diversity(transforms) -> dict:
    reference = transforms[0][:3, :3]
    vectors = np.array([Rotation.from_matrix(reference.T @ t[:3, :3]).as_rotvec() for t in transforms[1:]])
    singular = np.linalg.svd(vectors, compute_uv=False)
    max_angle = float(np.rad2deg(np.max(np.linalg.norm(vectors, axis=1))))
    if max_angle < 15 or singular[1] < max(0.1 * singular[0], np.deg2rad(8)):
        raise ValueError("Insufficient orientation diversity: rotate at least 15 degrees around at least two different axes")
    return {"max_relative_rotation_deg": max_angle, "rotation_singular_values_deg": np.rad2deg(singular).tolist()}


def solve_handeye(samples, board: dict, intrinsics: dict, *, max_board_position_rms_mm: float = 10.0, max_board_rotation_rms_deg: float = 3.0, max_board_rms_px: float = 2.0) -> dict:
    """Return T_flange_camera and stationary-board consistency diagnostics.

    Each sample contains ``image`` and ``T_base_flange``. For compatibility,
    ``base_to_flange`` is also accepted but has the SAME meaning: it maps
    flange-frame coordinates to base, not the reverse. All poses must be
    actual feedback in the robot's base with Tool=0/User=0.
    """
    samples = list(samples)
    if len(samples) < 8:
        raise ValueError("Hand-eye calibration requires at least eight paired poses/images")
    _board_shape(board)
    if any(not np.isfinite(value) or value <= 0 for value in (max_board_position_rms_mm, max_board_rotation_rms_deg, max_board_rms_px)):
        raise ValueError("Hand-eye quality limits must be finite and positive")
    K, dist = validate_camera(intrinsics["K"], intrinsics["dist"])
    expected_size = tuple(intrinsics["image_size"])
    if "board" in intrinsics:
        old = intrinsics["board"]
        if _board_shape(old) != _board_shape(board) or old.get("top_left_black", True) != board.get("top_left_black", True):
            raise ValueError("Hand-eye board configuration differs from the RGB intrinsic calibration")
    flange_poses = []
    board_poses = []
    views = []
    for sample in samples:
        pose = sample.get("T_base_flange", sample.get("base_to_flange"))
        if pose is None:
            raise ValueError("Each sample must provide T_base_flange")
        flange_poses.append(validate_transform(pose))
    diversity = _rotation_diversity(flange_poses)
    for sample in samples:
        image = cv2.imread(str(sample["image"]), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot read sample image: {sample['image']}")
        if (image.shape[1], image.shape[0]) != expected_size:
            raise ValueError("Sample image size differs from the calibrated RGB stream")
        transform, quality = solve_board_pose(image, K, dist, board, max_rms_px=max_board_rms_px)
        board_poses.append(transform)
        views.append({"image": str(sample["image"]), **quality})
    R, t = cv2.calibrateHandEye(
        [p[:3, :3] for p in flange_poses], [p[:3, 3] for p in flange_poses],
        [p[:3, :3] for p in board_poses], [p[:3, 3] for p in board_poses],
        method=cv2.CALIB_HAND_EYE_PARK,
    )
    T_flange_camera = np.eye(4)
    T_flange_camera[:3, :3] = R
    T_flange_camera[:3, 3] = np.asarray(t).reshape(3)
    try:
        validate_transform(T_flange_camera)
    except ValueError as exc:
        raise ValueError(f"Hand-eye solver produced an invalid transform; check inconsistent samples: {exc}") from exc
    fixed_boards = np.array([flange @ T_flange_camera @ board_pose for flange, board_pose in zip(flange_poses, board_poses)])
    mean = np.eye(4)
    mean[:3, :3] = Rotation.from_matrix(fixed_boards[:, :3, :3]).mean().as_matrix()
    mean[:3, 3] = fixed_boards[:, :3, 3].mean(axis=0)
    translation_error = np.linalg.norm(fixed_boards[:, :3, 3] - mean[:3, 3], axis=1)
    rotation_error = np.rad2deg(Rotation.from_matrix(mean[:3, :3].T @ fixed_boards[:, :3, :3]).magnitude())
    translation_rms = float(np.sqrt(np.mean(translation_error ** 2)))
    rotation_rms = float(np.sqrt(np.mean(rotation_error ** 2)))
    if translation_rms > max_board_position_rms_mm or rotation_rms > max_board_rotation_rms_deg:
        raise ValueError(f"Hand-eye data are inconsistent: board position RMS={translation_rms:.2f} mm, orientation RMS={rotation_rms:.2f} deg; check pairing, Euler convention, board motion and corner order")
    for quality, position_error, angle_error in zip(views, translation_error, rotation_error):
        quality.update({"board_position_error_mm": float(position_error), "board_rotation_error_deg": float(angle_error)})
    metrics = {"sample_count": len(samples), "board_position_rms_mm": translation_rms, "board_position_max_mm": float(translation_error.max()), "board_rotation_rms_deg": rotation_rms, "board_rotation_max_deg": float(rotation_error.max()), **diversity}
    return {"T_flange_camera": T_flange_camera.tolist(), "T_base_board": mean.tolist(), "metrics": metrics, "views": views, "board": dict(board), "units": "mm", "method": "OpenCV PARK", "quality_limits": {"max_board_position_rms_mm": max_board_position_rms_mm, "max_board_rotation_rms_deg": max_board_rotation_rms_deg, "max_board_rms_px": max_board_rms_px}, "validation_note": "In-sample consistency is not independent validation. Verify held-out poses and physical targets before motion."}
