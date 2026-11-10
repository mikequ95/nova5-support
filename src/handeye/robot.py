# SPDX-License-Identifier: MulanPSL-2.0
"""Conservative adapter for the locally inspected official ``dobot_sdk`` V4 API.

All poses are [x, y, z, rx, ry, rz] in mm / degrees, expressed in User0 with
Tool0. TCP compensation belongs to the caller. No enable, alarm-clear, tool,
user, load or control-mode settings are changed here. The V4 SDK's published
model list does not explicitly confirm this Nova5: verify controller/firmware
compatibility before permitting motion. This adapter is not a collision planner.
"""
from __future__ import annotations

import importlib
import copy
from collections import deque
import json
import math
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any, Sequence


class RobotError(RuntimeError):
    """Unexpected protocol, controller state, or motion completion failure."""


def _load_sdk(sdk_path: str | None):
    """Import only when a real adapter is constructed; offline tools stay offline."""
    if sdk_path:
        path = Path(sdk_path).expanduser().resolve()
        if not (path / "dobot_sdk" / "__init__.py").is_file():
            raise RobotError("sdk_path must be the directory containing dobot_sdk/")
        loaded = sys.modules.get("dobot_sdk")
        if loaded is not None:
            origin = getattr(loaded, "__file__", None)
            if not origin or Path(origin).resolve().parent != path / "dobot_sdk":
                raise RobotError("Another dobot_sdk is already imported; start a new process")
        sys.path.insert(0, str(path))
        try:
            sdk = importlib.import_module("dobot_sdk")
        finally:
            sys.path.remove(str(path))
    else:
        sdk = importlib.import_module("dobot_sdk")
    if not hasattr(sdk, "DobotRobot") or not hasattr(sdk, "CoordinateType"):
        raise RobotError("Unsupported SDK: expected DobotRobot and CoordinateType")
    return sdk


def _reply(raw: str, command: str) -> str:
    """Parse the result braces, never numbers in the echoed command arguments."""
    if not isinstance(raw, str):
        raise RobotError(f"{command}: expected a text response")
    if raw.strip().lower() == "control mode is not tcp":
        raise RobotError(
            f"{command}: robot is not in TCP/IP control mode; the controller rejected the command. "
            "TCP/IP here means the control mode, not the gripper TCP tool center point. "
            "Use read-only feedback for capture; select the control mode on site before motion. The program never switches it automatically."
        )
    prefix = re.match(r"\s*([+-]?\d+)\s*,\s*\{", raw)
    if prefix is None:
        raise RobotError(f"{command}: malformed response {raw!r}")
    start, depth, end = prefix.end(), 1, None
    for i in range(start, len(raw)):
        if raw[i] == "{":
            depth += 1
        elif raw[i] == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end is None or not re.fullmatch(
        rf"\s*,\s*{re.escape(command)}\([^;]*\)\s*;\s*", raw[end + 1:]
    ):
        raise RobotError(f"{command}: malformed or mismatched response {raw!r}")
    code = int(prefix.group(1))
    if code:
        raise RobotError(f"{command}: controller rejected command (ErrorID={code})")
    return raw[start:end].strip()


def _numbers(payload: str, count: int, command: str) -> list[float]:
    try:
        values = json.loads("[" + payload + "]")
        if len(values) != count or any(type(v) not in (int, float) for v in values):
            raise ValueError("wrong shape")
        result = [float(v) for v in values]
        if not all(math.isfinite(v) for v in result):
            raise ValueError("non-finite")
        return result
    except (ValueError, TypeError, OverflowError) as exc:
        raise RobotError(f"{command}: expected {count} finite numeric values") from exc


def _integer(payload: str, command: str) -> int:
    # Do not pass queue IDs through float: the controller uses uint64 IDs.
    if not re.fullmatch(r"\d+", payload):
        raise RobotError(f"{command}: expected one non-negative integer")
    return int(payload)


def _errors(payload: str) -> list[int]:
    try:
        values = json.loads("[" + payload + "]")
    except ValueError as exc:
        raise RobotError("GetErrorID: malformed alarm list") from exc
    result: list[int] = []

    def visit(item):
        if isinstance(item, list):
            for entry in item:
                visit(entry)
        elif type(item) is int:
            if item:
                result.append(item)
        else:
            raise RobotError("GetErrorID: expected integer alarm codes")

    visit(values)
    return result


def _pose(values: Sequence[float], label: str) -> list[float]:
    try:
        if len(values) != 6 or any(isinstance(v, (bool, str)) for v in values):
            raise ValueError("wrong shape")
        result = [float(v) for v in values]
        if not all(math.isfinite(v) for v in result):
            raise ValueError("non-finite")
        return result
    except (ValueError, TypeError, OverflowError) as exc:
        raise RobotError(f"{label}: expected six finite numbers") from exc


class DobotRobot:
    """Connect to official V4 dashboard + feedback, without enabling the arm.

    feedback_only=True opens only the feedback port and never sends commands.
    This is suitable for capture while the operator controls poses in Studio.

    Config: host, sdk_path (SDK root), port=29999, feedback_port=30004,
    connect_timeout_s=3, receive_timeout_s=1, feedback_timeout_s=3,
    feedback_max_age_s=0.5, poll_interval_s=0.05,
    position_tolerance_mm=0.5, orientation_tolerance_deg=0.5.

    The inspected SDK lacks GetUser/GetTool. Fresh feedback must therefore show
    User0/Tool0 selected with zero offsets, including during motion. Select them
    on the pendant before using this adapter. Never share motion ownership with
    another client/program. Timeout includes polling but may overrun by pending
    socket receive timeouts; Stop is a best-effort controller command, not an E-stop.
    """

    def __init__(self, config: dict[str, Any], *, feedback_only: bool = False):
        if type(feedback_only) is not bool:
            raise RobotError("feedback_only must be a boolean")
        self._feedback_only = feedback_only
        self._feedback_connection = None
        self._feedback_monitor_started = False
        host = config.get("host")
        if not isinstance(host, str) or not host.strip():
            raise RobotError("robot.host must be configured")
        self._max_age = self._positive(config, "feedback_max_age_s", 0.5)
        self._poll = self._positive(config, "poll_interval_s", 0.05)
        self._position_tolerance = self._positive(config, "position_tolerance_mm", 0.5)
        self._angle_tolerance = self._positive(config, "orientation_tolerance_deg", 0.5)
        self._feedback = None
        self._feedback_sequence = 0
        self._feedback_history = deque(maxlen=2048)
        self._received_at = 0.0
        self._feedback_lock = threading.Lock()
        self._move_lock = threading.Lock()
        self._closed = False
        self._motion_active = False
        sdk = _load_sdk(config.get("sdk_path"))
        self._cartesian = sdk.CoordinateType.CARTESIAN
        connect_timeout = self._positive(config, "connect_timeout_s", 3.0)
        self._sdk = sdk.DobotRobot(
            host,
            dashboard_port=int(config.get("port", 29999)),
            feedback_port=int(config.get("feedback_port", 30004)),
            connect_timeout=connect_timeout,
            receive_timeout=self._positive(config, "receive_timeout_s", 1.0),
        )
        try:
            if self._feedback_only:
                # The inspected, pinned V4 SDK exposes only a combined Connect().
                # Reuse its feedback connection/parser without opening Dashboard.
                connection = getattr(self._sdk, "_feedback_conn", None)
                if (connection is None
                        or not callable(getattr(connection, "connect", None))
                        or not callable(getattr(connection, "disconnect", None))
                        or not callable(getattr(self._sdk, "StartFeedbackMonitor", None))
                        or not callable(getattr(self._sdk, "StopFeedbackMonitor", None))):
                    raise RobotError("SDK lacks the inspected feedback-only connection API; "
                                     "refusing to open Dashboard as a fallback")
                self._feedback_connection = connection
                connection.connect(connect_timeout)
            else:
                self._sdk.Connect()
            self._feedback_monitor_started = True
            self._sdk.StartFeedbackMonitor(callback=self._on_feedback)
            deadline = time.monotonic() + self._positive(config, "feedback_timeout_s", 3.0)
            while self._feedback is None and time.monotonic() < deadline:
                time.sleep(self._poll)
            self._verify_frames()
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _positive(config, key, default):
        value = config.get(key, default)
        if isinstance(value, bool):
            raise RobotError(f"{key} must be a positive finite number")
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise RobotError(f"{key} must be a positive finite number") from exc
        if not math.isfinite(result) or result <= 0:
            raise RobotError(f"{key} must be a positive finite number")
        return result

    def _on_feedback(self, feedback):
        with self._feedback_lock:
            self._received_at = time.monotonic()
            try:
                self._feedback = copy.deepcopy(feedback)
            except Exception as exc:
                # Retain a failed sample too, so a later good packet cannot hide
                # an invalid interval from capture_feedback().
                self._feedback = RobotError(f"Cannot copy V4 feedback sample: {exc}")
            self._feedback_sequence += 1
            self._feedback_history.append(
                (self._feedback_sequence, self._received_at, self._feedback)
            )

    def _snapshot(self):
        if self._closed:
            raise RobotError("Robot connection is closed")
        with self._feedback_lock:
            feedback, received_at = self._feedback, self._received_at
        if feedback is None or time.monotonic() - received_at > self._max_age:
            raise RobotError("No fresh V4 feedback; check controller/SDK compatibility and port")
        if isinstance(feedback, RobotError):
            raise feedback
        return feedback, received_at

    def _verify_frames(self, feedback=None, *, feedback_only=False):
        if feedback is None:
            feedback, _ = self._snapshot()
        control = None if self._feedback_only or feedback_only else self._sdk.robot_control
        for kind in ("User", "Tool"):
            getter = getattr(control, "Get" + kind, None)
            if callable(getter):
                values = _numbers(_reply(getter(0), "Get" + kind), 6, "Get" + kind)
            else:
                index = getattr(feedback, kind.lower() + "_coordinate", None)
                if index != 0:
                    raise RobotError(f"Select {kind}0 on pendant; cannot verify identity otherwise")
                values = _pose(getattr(feedback, kind.lower() + "_value", None), kind + "0")
            if any(abs(v) > 1e-6 for v in values):
                raise RobotError(f"{kind}0 is not identity; refusing base/flange assumptions")

    def read_flange_pose(self) -> list[float]:
        if self._feedback_only:
            feedback, _ = self._snapshot()
            self._verify_frames(feedback)
            try:
                return _pose(feedback.tool_vector_actual.to_list(), "actual feedback flange pose")
            except AttributeError as exc:
                raise RobotError("Incomplete V4 feedback: missing actual TCP pose") from exc
        self._verify_frames()
        raw = self._sdk.robot_control.GetPose(user=0, tool=0)
        return _numbers(_reply(raw, "GetPose"), 6, "GetPose")

    def read_joints(self) -> list[float]:
        feedback, _ = self._snapshot()
        if self._feedback_only:
            self._verify_frames(feedback)
            try:
                return _pose(feedback.joint_state.q_actual, "actual feedback joints")
            except AttributeError as exc:
                raise RobotError("Incomplete V4 feedback: missing actual joint positions") from exc
        return _numbers(_reply(self._sdk.robot_control.GetAngle(), "GetAngle"), 6, "GetAngle")

    def status(self) -> dict[str, Any]:
        feedback, received_at = self._snapshot()
        if self._feedback_only:
            self._verify_frames(feedback)
            mode = None
            errors = None  # Detailed alarm IDs require Dashboard GetErrorID().
        else:
            control = self._sdk.robot_control
            mode = _integer(_reply(control.RobotMode(), "RobotMode"), "RobotMode")
            errors = _errors(_reply(control.GetErrorID(), "GetErrorID"))
            feedback, received_at = self._snapshot()
        return self._feedback_state(feedback, received_at, mode=mode, errors=errors,
                                    feedback_only=self._feedback_only)

    @staticmethod
    def _feedback_state(feedback, received_at, *, mode=None, errors=None, feedback_only=True):
        """Derive state from one packet; never send a command or fetch another packet."""
        try:
            joint_speed = _pose(feedback.joint_state.qd_actual, "joint speeds")
            tcp_speed = _pose(feedback.tcp_speed_actual.to_list(), "TCP speeds")
            if isinstance(feedback.robot_mode, bool) or not isinstance(feedback.robot_mode, int):
                raise RobotError("Invalid V4 feedback robot mode")
            feedback_mode = int(feedback.robot_mode)
            if feedback_only:
                mode = feedback_mode
            activity = {key: getattr(feedback, key) for key in
                        ("running_status", "run_queued_cmd", "pause_cmd_flag", "drag_status", "jog_status")}
            flags = {**activity, "error_status": feedback.error_status,
                     "collision_state": feedback.collision_state, "enable_status": feedback.enable_status}
            if any(not isinstance(v, int) or v < 0 for v in flags.values()):
                raise RobotError("Invalid V4 feedback state flag")
            no_activity = not any(activity.values())
            fault = bool(errors or feedback.error_status or feedback.collision_state or mode in (9, 11))
            max_joint_speed = max(abs(v) for v in joint_speed)
            max_tcp_speed = max(abs(v) for v in tcp_speed)
            capture_idle = mode in (4, 5) and feedback_mode == mode and no_activity and not fault
            stationary = capture_idle and max_joint_speed <= 0.05 and max_tcp_speed <= 0.1
            enabled = bool(feedback.enable_status)
        except AttributeError as exc:
            raise RobotError("Incomplete V4 feedback; refusing to infer motion status") from exc
        return {
            "mode": mode, "feedback_mode": feedback_mode, "errors": errors,
            "error": fault, "enabled": enabled, "stationary": stationary,
            "capture_idle": capture_idle,
            "ready": not feedback_only and mode == 5 and enabled and stationary and not fault,
            "source": "feedback" if feedback_only else "dashboard+feedback",
            "feedback_only": feedback_only,
            "stationarity_details": {
                "mode": mode, "feedback_mode": feedback_mode, **activity,
                "error_status": bool(feedback.error_status), "collision_state": feedback.collision_state,
                "max_joint_speed_deg_s": max_joint_speed,
                "max_tcp_speed_component": max_tcp_speed,
                "joint_speed_limit_deg_s": 0.05, "tcp_speed_component_limit": 0.1,
            },
            "feedback_age_s": time.monotonic() - received_at,
            "feedback_received_at": received_at,
        }

    def capture_feedback(self, after_sequence: int | None = None) -> list[dict[str, Any]]:
        """Return copied, validated feedback records for continuous capture checks.

        None selects only the newest packet; a sequence selects every newer
        packet. No packets may be silently lost through history overflow. Idle
        flags alone are insufficient for capture: the caller must also check
        pose stability and timestamp progression across the returned interval.
        """
        if after_sequence is not None and (type(after_sequence) is not int or after_sequence < 0):
            raise RobotError("after_sequence must be a non-negative integer or None")
        if self._closed:
            raise RobotError("Robot connection is closed")
        with self._feedback_lock:
            if not self._feedback_history or time.monotonic() - self._received_at > self._max_age:
                raise RobotError("No fresh V4 feedback; check controller/SDK compatibility and port")
            newest_sequence = self._feedback_history[-1][0]
            if after_sequence is None:
                selected = [self._feedback_history[-1]]
            else:
                if after_sequence < self._feedback_history[0][0] - 1:
                    raise RobotError("Feedback history overflow; restart capture from a fresh packet")
                if after_sequence > newest_sequence:
                    raise RobotError("after_sequence is newer than the latest feedback packet")
                selected = [entry for entry in self._feedback_history if entry[0] > after_sequence]
        records = []
        for sequence, received_at, feedback in selected:
            try:
                if isinstance(feedback, RobotError):
                    raise feedback
                # Even a command-capable adapter must validate this exact packet
                # through feedback fields; GetUser/GetTool could describe another instant.
                self._verify_frames(feedback, feedback_only=True)
                pose = _pose(feedback.tool_vector_actual.to_list(), "actual feedback flange pose")
                timestamp = getattr(feedback, "timestamp", None)
                if type(timestamp) is not int or timestamp < 0:
                    raise RobotError("Missing or invalid V4 feedback controller timestamp")
                state = self._feedback_state(feedback, received_at)
            except (RobotError, AttributeError) as exc:
                raise RobotError(f"Invalid capture feedback sample {sequence}: {exc}") from exc
            records.append({
                "sequence": sequence, "received_at": received_at,
                "controller_timestamp": timestamp, "flange_pose_mm_deg": pose,
                "state": state,
            })
        return records

    def _at_target(self, actual, target):
        distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(actual[:3], target[:3])))
        # Conservative Euler check: equivalent alternative Euler representations
        # may time out, but cannot be incorrectly accepted as a different pose.
        angle = max(abs((a - b + 180.0) % 360.0 - 180.0)
                    for a, b in zip(actual[3:], target[3:]))
        return distance <= self._position_tolerance and angle <= self._angle_tolerance

    def verify_feedback_pose(self):
        """Cross-check the command interface against actual feedback, without writes."""
        if self._feedback_only:
            raise RobotError("Command interface verification needs a Dashboard connection")
        state = self.status()
        if not state["capture_idle"]:
            raise RobotError("Stop jogging, exit drag mode and remain stationary; the controller must have no alarms")
        actual = self.read_flange_pose()
        observed = self.capture_feedback()[0]["flange_pose_mm_deg"]
        from .geometry import pose_to_transform
        from .workflow import pose_distance
        distance, angle = pose_distance(pose_to_transform(actual), pose_to_transform(observed))
        if distance > 0.5 or angle > 0.2:
            raise RobotError(f"Dashboard/feedback pose mismatch: {distance:.3f} mm / {angle:.3f} deg")
        return actual

    @staticmethod
    def _enabled_idle(state):
        if not (state["capture_idle"] and state["enabled"] and state["mode"] == 5):
            raise RobotError("Robot must be enabled, idle, outside drag mode and free of alarms; complete these settings on site")

    def wait_enabled_still(self, timeout_s=3.0, cancel_event=None):
        """Use a continuous pose window for the explicitly requested first-use test.

        This handles measured idle velocity-estimate noise without accepting
        active motion, stale feedback, transient faults or moving actual poses.
        """
        from .capture import CaptureStability
        timeout = self._positive({"timeout": timeout_s}, "timeout", 3.0)
        deadline = time.monotonic() + timeout
        self._enabled_idle(self.status())
        window = CaptureStability(self.capture_feedback()[0], self._enabled_idle)
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise RobotError("Motion cancelled while checking stability")
            self._enabled_idle(self.status())
            window.add(self.capture_feedback(after_sequence=window.last["sequence"]))
            window.check_fresh(time.monotonic())
            if window.covers(window.first["received_at"] + window.min_duration_s):
                return window.last_transform
            if time.monotonic() >= deadline:
                raise RobotError("Timed out waiting for a complete stationary feedback window")
            time.sleep(self._poll)

    def move_flange(self, pose: Sequence[float], speed_mm_s: float, timeout_s: float = 30,
                    cancel_event: threading.Event | None = None, *, pose_stability: bool = False,
                    expected_start: Sequence[float] | None = None) -> None:
        if self._feedback_only:
            raise RobotError("Feedback-only connection cannot send motion commands")
        target = _pose(pose, "flange target")
        if type(pose_stability) is not bool:
            raise RobotError("pose_stability must be a boolean")
        expected = _pose(expected_start, "expected start") if expected_start is not None else None
        speed = self._positive({"speed": speed_mm_s}, "speed", 1)
        timeout = self._positive({"timeout": timeout_s}, "timeout", 30)
        # The official method annotates speed as int (mm/s). Reject fractions
        # instead of silently rounding a requested speed.
        if speed != int(speed):
            raise RobotError("speed_mm_s must be a whole positive number for this SDK")
        if not self._move_lock.acquire(blocking=False):
            raise RobotError("Another motion is already active on this adapter")
        try:
            if cancel_event is not None and cancel_event.is_set():
                raise RobotError("Motion cancelled before submission")
            self._verify_frames()
            if pose_stability:
                settled = self.wait_enabled_still(cancel_event=cancel_event)
                if expected is not None:
                    from .geometry import pose_to_transform
                    from .workflow import pose_distance
                    distance, angle = pose_distance(settled, pose_to_transform(expected))
                    if distance > 0.2 or angle > 0.1:
                        raise RobotError("Arm moved since the selected image; refresh before executing")
            else:
                initial = self.status()
                if not initial["ready"]:
                    raise RobotError(f"Robot is not enabled and idle: {initial}")
            # Verify queue query support before submitting any motion.
            _integer(_reply(self._sdk.motion.GetCurrentCommandID(), "GetCurrentCommandID"),
                     "GetCurrentCommandID")
            if cancel_event is not None and cancel_event.is_set():
                raise RobotError("Motion cancelled before submission")
            deadline = time.monotonic() + timeout
            sent_at = time.monotonic()
            self._motion_active = True
            try:
                raw = self._sdk.motion.MovL(target, self._cartesian, user=0, tool=0,
                                           speed=int(speed), cp=0)
                command_id = _integer(_reply(raw, "MovL"), "MovL")
                settled_count = 0
                last_settled_feedback = 0.0
                while time.monotonic() < deadline:
                    if cancel_event is not None and cancel_event.is_set():
                        raise RobotError("Motion cancelled")
                    current_id = _integer(
                        _reply(self._sdk.motion.GetCurrentCommandID(), "GetCurrentCommandID"),
                        "GetCurrentCommandID",
                    )
                    state = self.status()
                    self._verify_frames()
                    if cancel_event is not None and cancel_event.is_set():
                        raise RobotError("Motion cancelled")
                    if state["error"] or state["mode"] not in (5, 7) or not state["enabled"]:
                        raise RobotError(f"Motion interrupted by controller state: {state}")
                    if current_id > command_id:
                        raise RobotError("Unexpected later queue ID; another client may be commanding motion")
                    settled_state = (state["capture_idle"] and state["enabled"] and state["mode"] == 5
                                     if pose_stability else state["ready"])
                    if (current_id == command_id and settled_state
                            and state["feedback_received_at"] > sent_at):
                        actual = self.read_flange_pose()
                        if self._at_target(actual, target):
                            if pose_stability:
                                remaining = deadline - time.monotonic()
                                if remaining <= 0:
                                    raise RobotError("Motion timed out before stable arrival")
                                self.wait_enabled_still(timeout_s=min(3.0, remaining), cancel_event=cancel_event)
                                final_id = _integer(_reply(self._sdk.motion.GetCurrentCommandID(),
                                                          "GetCurrentCommandID"), "GetCurrentCommandID")
                                if final_id != command_id:
                                    raise RobotError("Queue ID changed during arrival verification")
                                if self._at_target(self.read_flange_pose(), target):
                                    return
                                raise RobotError("Actual pose changed during arrival verification")
                            if state["feedback_received_at"] > last_settled_feedback:
                                settled_count += 1
                                last_settled_feedback = state["feedback_received_at"]
                            if settled_count >= 2:
                                return
                        else:
                            settled_count = 0
                    else:
                        settled_count = 0
                    time.sleep(self._poll)
                raise RobotError(f"Motion timed out after {timeout:g}s before verified arrival")
            except BaseException as exc:
                try:
                    self.stop()
                except Exception as stop_exc:
                    if hasattr(exc, "add_note"):
                        exc.add_note(f"Stop also failed: {stop_exc}; use the physical stop if needed")
                raise
            finally:
                self._motion_active = False
        finally:
            self._move_lock.release()

    def stop(self) -> None:
        if self._feedback_only:
            raise RobotError("Feedback-only connection cannot send Stop commands")
        if self._closed:
            raise RobotError("Robot connection is closed")
        _reply(self._sdk.robot_control.Stop(), "Stop")

    def close(self) -> None:
        if not self._closed:
            if self._feedback_only:
                self._closed = True
                try:
                    if self._feedback_monitor_started:
                        self._sdk.StopFeedbackMonitor()
                finally:
                    if self._feedback_connection is not None:
                        self._feedback_connection.disconnect()
                return
            try:
                if self._motion_active:
                    self.stop()
            finally:
                self._closed = True
                self._sdk.Disconnect()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
