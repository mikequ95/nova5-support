# SPDX-License-Identifier: MulanPSL-2.0
"""Small command-line workflow. Camera UI uses unscaled, frozen RGB frames."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import queue
import threading
import time

import cv2
import numpy as np

from .calibration import calibrate_intrinsics, detect_corners, solve_board_pose, solve_handeye
from .camera import Camera
from .capture import CaptureStability
from .geometry import pose_to_transform, target_flange, transform_to_pose
from .workflow import (board_spec, check_camera_identity, digest, load_config,
                       load_handeye, load_intrinsics, load_plane, plan_click,
                       pose_distance, read_json, require_board, require_motion,
                       resolve, save_json, setup_signature, tcp_transform)


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _print(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), flush=True)


def _robot(config, *, feedback_only=True):
    from .robot import DobotRobot
    return DobotRobot(config["robot"], feedback_only=feedback_only)


def _save_image(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"Cannot save {path}")


def _require_capture_idle(state):
    if state.get("capture_idle") is True:
        return state
    details = state.get("stationarity_details", {})
    mode = state.get("mode")
    if mode == 6 or details.get("drag_status"):
        reason = "Drag mode is active; exit drag mode and remain still for at least 1 second before pressing S"
    elif details.get("jog_status") or mode == 8:
        reason = "Jog or single motion is active; release the jog button and wait until motion finishes before capture"
    elif details.get("pause_cmd_flag") or mode == 10:
        reason = "Paused state is active; finish the current task in the official software and return to idle before capture"
    elif state.get("error"):
        reason = "Controller reports an alarm or collision; inspect it in the official software first"
    elif details.get("running_status") or details.get("run_queued_cmd") or mode == 7:
        reason = "A program or motion queue is running; wait for completion and rest before capture"
    else:
        reason = "Capture idle conditions are not met; ensure the arm is idle and stationary for at least 1 second"
    diagnostic = {"mode": mode, **details}
    raise ValueError(f"{reason}. State: {json.dumps(diagnostic, ensure_ascii=False)}")


def _stationary_snapshot(camera, robot, *, deadline=None):
    """Check actual pose stability throughout drained RGB acquisition.

    These are host timestamps, not synchronized hardware timestamps. No moving
    arm sampling is accepted. Operator must hold still throughout acquisition.
    """
    if camera.backend != "orbbec":
        raise ValueError("Robot-paired capture currently requires Orbbec backend; OpenCV is for intrinsics only")
    def check_deadline():
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Stationary RGB capture exceeded its deadline")

    check_deadline()
    first = robot.capture_feedback()[0]
    check_deadline()
    stability = CaptureStability(first, _require_capture_idle)
    stability.check_fresh(time.monotonic())
    started = time.time()
    # Official Orbbec pipeline may queue ten frames. Read beyond the queue and
    # spend at least one second acquiring while the robot remains stationary.
    count = 0
    until = time.monotonic() + stability.min_duration_s
    while count < 15 or time.monotonic() < until:
        check_deadline()
        frame, stamp = camera.read()
        check_deadline()
        frame_received_at = time.monotonic()
        count += 1
        stability.add(robot.capture_feedback(after_sequence=stability.last["sequence"]))
        check_deadline()
        stability.check_fresh(time.monotonic())
    # Include a feedback packet received after the final RGB frame, even if the
    # most recent callback was just before that frame. Never reuse a frozen pose.
    feedback_deadline = frame_received_at + stability.max_gap_s
    while not stability.covers(frame_received_at):
        check_deadline()
        if time.monotonic() >= feedback_deadline:
            raise ValueError("Feedback did not cover the complete capture interval; check the connection and remain still before retrying")
        time.sleep(0.005)
        stability.add(robot.capture_feedback(after_sequence=stability.last["sequence"]))
        check_deadline()
        stability.check_fresh(time.monotonic())
    after = stability.last_transform
    moved, rotated = pose_distance(stability.anchor, after)
    return frame, {"T_base_flange": after.tolist(), "captured_at": _utc(),
                   "host_frame_received_at": stamp, "capture_started_at": started,
                   "camera_timestamp_diagnostics": dict(camera.last_frame_metadata),
                   "robot_pose_source": "feedback",
                   "capture_stability": stability.report(),
                   "capture_pose_difference_mm": moved, "capture_pose_difference_deg": rotated}


def _same_size(frame, intrinsics):
    if [frame.shape[1], frame.shape[0]] != intrinsics["image_size"]:
        raise ValueError("Image dimensions do not match intrinsic calibration")


def capture(args, config):
    require_board(config)
    folder = resolve(config, args.session)
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / "samples.json"
    intrinsic = load_intrinsics(config) if args.command == "capture-handeye" else None
    with ExitStack() as stack:
        camera = stack.enter_context(Camera(config["camera"]))
        robot = stack.enter_context(_robot(config)) if intrinsic is not None else None
        if intrinsic:
            check_camera_identity(camera, intrinsic)
        manifest = {"kind": args.command, "board": board_spec(config["board"]),
                    "camera_identity": camera.identity, "samples": []}
        if manifest_path.exists():
            previous = read_json(manifest_path)
            for key in ("kind", "board", "camera_identity"):
                if previous[key] != manifest[key]:
                    raise ValueError(f"Cannot mix {key} in existing capture folder")
            manifest = previous
        title = "RGB capture: S save | Q quit"
        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
        print("Press S to save and Q to quit. Keep the board fixed during hand-eye capture; stop jogging, exit drag mode and remain still for at least 1 second before S.")
        if robot:
            print("Robot poses are read only from the feedback port. Use the official software to adjust poses; capture does not require TCP/IP control mode.")
        try:
            while True:
                frame, stamp = camera.read()
                preview = frame.copy()
                try:
                    corners = detect_corners(frame, config["board"])
                    cv2.drawChessboardCorners(preview, tuple(config["board"]["inner_corners"]), corners, True)
                    status = f"Board found. Saved {len(manifest['samples'])}"
                except ValueError:
                    status = "Board not found"
                cv2.putText(preview, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)
                cv2.imshow(title, preview)
                key = cv2.waitKey(20) & 0xFF
                if key in (ord("q"), 27) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == ord("s"):
                    try:
                        if robot:
                            frame, record = _stationary_snapshot(camera, robot)
                        else:
                            frame, stamp = camera.read()
                            record = {"captured_at": _utc(), "host_frame_received_at": stamp}
                        detect_corners(frame, config["board"])
                        filename = f"{len(manifest['samples']):04d}.png"
                        if (folder / filename).exists():
                            raise ValueError("Untracked image exists; use a new session folder")
                        _save_image(folder / filename, frame)
                        record["image"] = filename
                        manifest["samples"].append(record)
                        save_json(manifest_path, manifest)
                        print(f"Saved {filename}", flush=True)
                    except (ValueError, RuntimeError) as exc:
                        print(f"Capture refused: {exc}", flush=True)
        finally:
            cv2.destroyAllWindows()


def calibrate(args, config):
    require_board(config)
    folder = resolve(config, args.session)
    manifest = read_json(folder / "samples.json")
    expected = "capture-intrinsics" if args.command == "calibrate-intrinsics" else "capture-handeye"
    if manifest["kind"] != expected or manifest["board"] != board_spec(config["board"]):
        raise ValueError("Session kind/board geometry mismatch; recapture after changing measured size")
    samples = [{**s, "image": str((folder / s["image"]).resolve())} for s in manifest["samples"]]
    if args.command == "calibrate-intrinsics":
        result = calibrate_intrinsics([s["image"] for s in samples], config["board"],
                                      max_rms_px=config["validation"]["intrinsics_max_rms_px"])
        result["camera_identity"] = manifest["camera_identity"]
        key = "intrinsics"
    else:
        intrinsic = load_intrinsics(config)
        if manifest["camera_identity"] != intrinsic["camera_identity"]:
            raise ValueError("Hand-eye session uses a different camera/profile")
        result = solve_handeye(samples, config["board"], intrinsic,
                               max_board_position_rms_mm=config["validation"]["handeye_translation_rms_mm"],
                               max_board_rotation_rms_deg=config["validation"]["handeye_rotation_rms_deg"],
                               max_board_rms_px=config["validation"]["board_max_rms_px"])
        result["intrinsics_sha256"] = digest(resolve(config, config["files"]["intrinsics"]))
        result["quality_passed"] = True  # solver rejects configured thresholds
        key = "handeye"
    result["created_at"] = _utc()
    result["source_session"] = str(folder)
    out = resolve(config, config["files"][key])
    save_json(out, result)
    _print({"saved": str(out), "quality": result.get("metrics", {"rms_px": result.get("rms")})})


def plane_capture(args, config):
    require_board(config)
    intrinsic, handeye = load_intrinsics(config), load_handeye(config)
    offset = args.surface_offset_mm
    if not np.isfinite(offset) or offset < 0:
        raise ValueError("surface-offset-mm must be the nonnegative board-face height above the clicked plane")
    with Camera(config["camera"]) as camera, _robot(config) as robot:
        check_camera_identity(camera, intrinsic)
        input("Keep the board on the work plane and arm stationary; press Enter to capture: ")
        frame, sample = _stationary_snapshot(camera, robot)
    _same_size(frame, intrinsic)
    T_camera_board, quality = solve_board_pose(frame, intrinsic["K"], intrinsic["dist"], config["board"],
                                              max_rms_px=config["validation"]["board_max_rms_px"])
    T_base_board = np.asarray(sample["T_base_flange"]) @ np.asarray(handeye["T_flange_camera"]) @ T_camera_board
    normal = T_base_board[:3, 2].copy()
    if normal[2] < 0:
        normal *= -1
    if normal[2] < 0.95:
        raise ValueError("First version expects a near-horizontal tabletop in base coordinates")
    point = T_base_board[:3, 3] - offset * normal
    output = resolve(config, config["files"]["plane"])
    picture = output.with_suffix(".png")
    _save_image(picture, frame)
    result = {"normal": normal.tolist(), "offset_mm": float(-normal @ point),
              "surface_offset_mm": offset, "T_base_board": T_base_board.tolist(),
              "handeye_sha256": digest(resolve(config, config["files"]["handeye"])),
              "image": str(picture), "sample": sample, "quality": quality, "created_at": _utc()}
    save_json(output, result)
    _print({"saved": str(output), "normal": result["normal"], "offset_mm": result["offset_mm"]})


def click(args, config):
    require_board(config)
    intrinsic, handeye, plane = load_intrinsics(config), load_handeye(config), load_plane(config)
    T_flange_tcp = tcp_transform(config)
    results = queue.Queue()
    cancel = threading.Event()
    worker = None
    pending_pixel = None
    plan = None
    snapshot = None
    snapshot_monotonic = 0
    orientation = None
    frame = None
    test_mode = getattr(args, "test_motion", False)
    anchor = None
    session_signature = None
    session_config_digest = None
    needs_refresh = True
    title = "Click RGB | R fresh snapshot | M execute | ESC stop/quit"

    def select(event, x, y, flags, param):
        nonlocal pending_pixel
        if event == cv2.EVENT_LBUTTONDOWN and worker is None and not needs_refresh:
            pending_pixel = (x, y)

    def move(robot, target):
        try:
            if time.monotonic() - snapshot_monotonic > config["motion"]["max_snapshot_age_s"]:
                raise ValueError("Snapshot expired before motion submission; press R")
            if test_mode:
                from .test_motion import require_test_motion, SPEED_MM_S, TIMEOUT_S
                current = robot.wait_enabled_still(cancel_event=cancel)
                distance, angle = pose_distance(current, np.asarray(snapshot["T_base_flange"]))
                if distance > .2 or angle > .1:
                    raise ValueError("Arm moved while confirming; press R")
                require_test_motion(config, current, target, anchor, plane, execute=args.execute)
                if (setup_signature(config) != session_signature
                        or digest(config["_path"]) != session_config_digest):
                    raise ValueError("Calibration/configuration changed; restart the test")
                robot.verify_feedback_pose()
                travel_mm, _ = pose_distance(current, np.asarray(target["T_base_flange"]))
                timeout_s = max(TIMEOUT_S, 2 * travel_mm / SPEED_MM_S + 10)
                robot.move_flange(target["flange_pose_mm_deg"], SPEED_MM_S,
                                  timeout_s=timeout_s, cancel_event=cancel, pose_stability=True,
                                  expected_start=transform_to_pose(np.asarray(snapshot["T_base_flange"])))
                from .test_motion import table_z
                actual_tcp = pose_to_transform(robot.read_flange_pose()) @ T_flange_tcp
                achieved = {"controller_tcp_mm_deg": transform_to_pose(actual_tcp),
                            "height_above_calibrated_table_mm":
                                float(actual_tcp[2, 3] - table_z(plane, *actual_tcp[:2, 3])),
                            "note": "Controller feedback only; check physical alignment with the clicked point."}
                _print(achieved)
                save_json(resolve(config, f"logs/test-result-{time.time_ns()}.json"),
                          {"created_at": _utc(), "target": target, **achieved})
            else:
                robot.move_flange(target["flange_pose_mm_deg"], config["motion"]["speed_mm_s"],
                                  timeout_s=config["motion"]["timeout_s"], cancel_event=cancel)
            results.put((True, "Motion complete. Press R for a new snapshot."))
        except Exception as exc:
            results.put((False, f"Motion stopped/failed: {exc}"))

    with Camera(config["camera"]) as camera, _robot(config, feedback_only=not args.execute) as robot:
        check_camera_identity(camera, intrinsic)
        if test_mode:
            session_signature = setup_signature(config)
            session_config_digest = digest(config["_path"])
            if args.execute:
                # Read-only command/feedback cross-check; no enable or mode switch.
                robot.verify_feedback_pose()
        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(title, select)
        message = "PREVIEW ONLY" if not args.execute else "Execution requested; M checks all motion gates"
        request_refresh = True
        try:
            while True:
                if worker is not None and not worker.is_alive():
                    worker.join()
                    worker = None
                    _, message = results.get_nowait()
                    needs_refresh = True
                    plan = None
                    print(message, flush=True)
                if request_refresh and worker is None:
                    frame, snapshot = _stationary_snapshot(camera, robot)
                    _same_size(frame, intrinsic)
                    snapshot_monotonic = time.monotonic()
                    if anchor is None:
                        anchor = np.asarray(snapshot["T_base_flange"])
                    orientation = (np.asarray(snapshot["T_base_flange"]) @ T_flange_tcp)[:3, :3]
                    plan = None
                    pending_pixel = None
                    needs_refresh = False
                    request_refresh = False
                    message = "Frozen RGB: click a point on the calibrated surface"
                if pending_pixel is not None:
                    try:
                        plan = plan_click(pending_pixel, args.z_mm, snapshot["T_base_flange"], config,
                                          intrinsic, handeye, plane, orientation,
                                          height_mm=getattr(args, "height_mm", None))
                        _print(plan)
                        message = "Target preview. M moves only with --execute. R reacquires."
                    except (ValueError, RuntimeError) as exc:
                        plan = None
                        message = f"Rejected: {exc}"
                    pending_pixel = None
                preview = frame.copy()
                if test_mode and anchor is not None:
                    from .test_motion import table_z
                    start = (anchor @ T_flange_tcp)[:3, 3]
                    ground = np.array([start[0], start[1], table_z(plane, *start[:2]), 1.0])
                    camera_point = np.linalg.inv(np.asarray(snapshot["T_base_flange"]) @
                                                np.asarray(handeye["T_flange_camera"])) @ ground
                    if camera_point[2] > 0:
                        projected, _ = cv2.projectPoints(camera_point[:3].reshape(1, 3), np.zeros(3),
                                                        np.zeros(3), np.asarray(intrinsic["K"]),
                                                        np.asarray(intrinsic["dist"]))
                        uv = tuple(int(round(v)) for v in projected.reshape(2))
                        if 0 <= uv[0] < preview.shape[1] and 0 <= uv[1] < preview.shape[0]:
                            cv2.drawMarker(preview, uv, (0, 255, 0), cv2.MARKER_CROSS, 24, 2)
                            cv2.putText(preview, "START TCP", (uv[0]+8, uv[1]-8),
                                        cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)
                if plan:
                    cv2.drawMarker(preview, tuple(int(v) for v in plan["pixel"]), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
                    xyz = plan["tcp_pose_mm_deg"][:3]
                    cv2.putText(preview, f"TCP mm: {xyz[0]:.1f} {xyz[1]:.1f} {xyz[2]:.1f}",
                                (12, 56), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 0, 255), 2)
                cv2.putText(preview, "MOVING - ESC cancels" if worker else message[:105],
                            (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 255), 2)
                cv2.imshow(title, preview)
                key = cv2.waitKey(30) & 0xFF
                if key in (27, ord("q")) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                    cancel.set()
                    break
                if key == ord("r") and worker is None:
                    request_refresh = True
                if key == ord("m") and worker is None and plan is not None and not needs_refresh:
                    try:
                        if not args.execute:
                            raise ValueError("Preview only; restart the click test with --execute"
                                             if test_mode else
                                             "Preview only; restart with --execute after physical validation")
                        max_age = config["motion"]["max_snapshot_age_s"]
                        if not np.isfinite(max_age) or max_age <= 0 or time.monotonic() - snapshot_monotonic > max_age:
                            raise ValueError("Snapshot expired; press R")
                        current = pose_to_transform(robot.read_flange_pose())
                        distance, angle = pose_distance(current, np.asarray(snapshot["T_base_flange"]))
                        if distance > .2 or angle > .1:
                            raise ValueError("Arm moved since image; press R")
                        if test_mode:
                            from .test_motion import require_test_motion
                            require_test_motion(config, current, plan, anchor, plane, execute=args.execute)
                            if (setup_signature(config) != session_signature
                                    or digest(config["_path"]) != session_config_digest):
                                raise ValueError("Calibration/configuration changed; restart the test")
                        else:
                            require_motion(config, current, plan)
                            if not robot.status()["ready"]:
                                raise ValueError("Robot must already be enabled, stationary and error-free")
                        log = {"created_at": _utc(), "target": plan, "snapshot": snapshot,
                               "test_motion": test_mode, "session_anchor": anchor.tolist(),
                               "configuration_sha256": digest(config["_path"]),
                               "intrinsics_sha256": digest(resolve(config, config["files"]["intrinsics"])),
                               "handeye_sha256": digest(resolve(config, config["files"]["handeye"])),
                               "plane_sha256": digest(resolve(config, config["files"]["plane"]))}
                        save_json(resolve(config, f"logs/target-{time.time_ns()}.json"), log)
                        cancel.clear()
                        worker = threading.Thread(target=move, args=(robot, plan), daemon=False)
                        worker.start()
                        needs_refresh = True
                    except (ValueError, RuntimeError) as exc:
                        message = f"Move refused: {exc}"
                        print(message, flush=True)
        finally:
            cancel.set()
            if worker is not None:
                # Do not close the SDK while its movement thread is sending Stop.
                worker.join()
            cv2.destroyAllWindows()


def main(argv=None):
    parser = argparse.ArgumentParser(description="RGB checkerboard hand-eye calibration and software TCP compensation; lengths in mm and angles in deg")
    parser.add_argument("--config", default="config.json")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Check dependencies and incomplete parameters offline; do not connect to devices")
    cmd = sub.add_parser("robot-status", help="Read state, zero tool coordinates and actual pose from real-time feedback")
    cmd.add_argument("--dashboard", action="store_true",
                     help="Also inspect the command interface (requires TCP/IP control mode); do not change settings or move")
    for name, default in (("capture-intrinsics", "sessions/intrinsics"), ("calibrate-intrinsics", "sessions/intrinsics"),
                          ("capture-handeye", "sessions/handeye"), ("calibrate-handeye", "sessions/handeye")):
        cmd = sub.add_parser(name)
        cmd.add_argument("--session", default=default, help="Sample directory relative to the configuration file")
    cmd = sub.add_parser("capture-plane", help="Capture the work plane using the board-to-surface normal offset")
    cmd.add_argument("--surface-offset-mm", type=float, required=True,
                     help="Distance from the clicked surface to the board surface along its normal; zero for the board itself")
    cmd = sub.add_parser("click", help="Freeze the RGB frame and click; preview targets by default")
    cmd.add_argument("--z-mm", type=float, required=True, help="Absolute target TCP Z in the robot base frame, in mm")
    cmd.add_argument("--execute", action="store_true", help="Allow execution with M after configuration and runtime checks")
    cmd = sub.add_parser("project", help="Calculate pixel-to-TCP/flange targets offline without device connections")
    cmd.add_argument("--pixel", type=float, nargs=2, required=True)
    cmd.add_argument("--flange-pose", type=float, nargs=6, required=True)
    cmd.add_argument("--z-mm", type=float, required=True)
    cmd = sub.add_parser("tcp-target", help="Calculate the target TCP-to-flange conversion without device connections")
    cmd.add_argument("--tcp-pose", type=float, nargs=6, required=True)
    cmd.add_argument("--offset", type=float, nargs=6, required=True,
                     help="Flange-to-TCP XYZ (mm) and fixed XYZ Euler angles (deg)")
    cmd = sub.add_parser("record-validation", help="Record completed independent physical validation without commanding motion")
    cmd.add_argument("--max-error-mm", type=float, required=True)
    cmd.add_argument("--tolerance-mm", type=float, required=True)
    cmd.add_argument("--points", type=int, required=True)
    cmd.add_argument("--note", required=True, help="Measurement method, locations and result notes")
    args = parser.parse_args(argv)
    try:
        if args.command == "tcp-target":
            _print({"flange_pose_mm_deg": transform_to_pose(target_flange(pose_to_transform(args.tcp_pose),
                                                                                      pose_to_transform(args.offset)))})
            return
        config = load_config(args.config)
        if args.command == "doctor":
            _print({"config": config["_path"], "dependencies": {m: importlib.util.find_spec(m) is not None
                     for m in ("numpy", "scipy", "cv2", "pyorbbecsdk")},
                    "official_sdk_found": ((Path(config["robot"]["sdk_path"]) / "dobot_sdk/__init__.py").is_file()
                        if config["robot"].get("sdk_path") else importlib.util.find_spec("dobot_sdk") is not None),
                    "board_print_measured": config["board"]["print_size_verified"],
                    "tcp_known": config["tcp"]["flange_to_tcp_mm_deg"] is not None,
                    "motion_enabled": config["motion"]["enabled"],
                    "calibration_files": {k: resolve(config, v).is_file() for k, v in config["files"].items()},
                    "note": "Offline check only. No device was opened."})
        elif args.command == "robot-status":
            with _robot(config, feedback_only=not args.dashboard) as robot:
                _print({"status": robot.status(), "flange_pose_mm_deg": robot.read_flange_pose()})
        elif args.command.startswith("capture-") and args.command != "capture-plane":
            capture(args, config)
        elif args.command.startswith("calibrate-"):
            calibrate(args, config)
        elif args.command == "capture-plane":
            plane_capture(args, config)
        elif args.command == "click":
            click(args, config)
        elif args.command == "project":
            require_board(config)
            _print(plan_click(args.pixel, args.z_mm, pose_to_transform(args.flange_pose), config,
                              load_intrinsics(config), load_handeye(config), load_plane(config)))
        elif args.command == "record-validation":
            load_intrinsics(config), load_handeye(config), load_plane(config)
            if (not np.isfinite(args.max_error_mm) or not np.isfinite(args.tolerance_mm)
                    or args.max_error_mm < 0 or args.tolerance_mm <= 0
                    or args.max_error_mm > args.tolerance_mm or args.points < 3 or not args.note.strip()):
                raise ValueError("Need >=3 independently measured points, finite max error within tolerance, and a note")
            report = {"created_at": _utc(), "setup_sha256": setup_signature(config),
                      "max_error_mm": args.max_error_mm, "tolerance_mm": args.tolerance_mm,
                      "points": args.points, "operator_note": args.note,
                      "source": "Operator-provided physical measurements; not an automatic accuracy assessment"}
            path = resolve(config, "calibrations/validation.json")
            save_json(path, report)
            updated = read_json(config["_path"])
            updated["validation"]["independent_validation_passed"] = True
            save_json(config["_path"], updated)
            _print({"recorded": str(path), "motion_enabled": updated["motion"]["enabled"]})
    except (ValueError, RuntimeError, OSError, KeyError, ImportError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
