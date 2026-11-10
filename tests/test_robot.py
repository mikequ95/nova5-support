# SPDX-License-Identifier: MulanPSL-2.0
"""Offline protocol / state-machine tests. Never import or connect real hardware."""
from types import SimpleNamespace
import copy
import threading
import unittest
from unittest.mock import patch

from handeye.robot import DobotRobot, RobotError, _reply


class FakePose:
    def __init__(self, values):
        self.values = list(values)

    def to_list(self):
        return list(self.values)


class FakeSdkRobot:
    def __init__(self, *args, **kwargs):
        self.calls = []
        self.pose = [100, 200, 300, 180, 0, 0]
        self.target = None
        self.mode = 5
        self.errors = "[]"
        self.command_id = 12
        self.motion_reply = None
        self.complete = True
        self.on_move = lambda: None
        self.callback = None
        self.feedback = SimpleNamespace(
            timestamp=1000,
            robot_mode=5, enable_status=True, error_status=False, collision_state=0,
            running_status=0, run_queued_cmd=0, pause_cmd_flag=0, drag_status=0, jog_status=0,
            user_coordinate=0, tool_coordinate=0, user_value=[0] * 6, tool_value=[0] * 6,
            joint_state=SimpleNamespace(qd_actual=[0] * 6, q_actual=[1, 2, 3, 4, 5, 6]),
            tcp_speed_actual=FakePose([0] * 6),
            tool_vector_actual=FakePose(self.pose),
        )
        self._feedback_conn = SimpleNamespace(
            connect=lambda timeout: self.calls.append(("feedback_connect", timeout)),
            disconnect=lambda: self.calls.append("feedback_disconnect"),
        )
        self.robot_control = SimpleNamespace(
            GetPose=self.GetPose, GetAngle=self.GetAngle, RobotMode=self.RobotMode,
            GetErrorID=self.GetErrorID, Stop=self.Stop,
        )
        self.motion = SimpleNamespace(MovL=self.MovL, GetCurrentCommandID=self.GetCurrentCommandID)

    def Connect(self):
        self.calls.append("Connect")

    def Disconnect(self):
        self.calls.append("Disconnect")

    def StartFeedbackMonitor(self, callback):
        self.callback = callback
        self.tick()

    def StopFeedbackMonitor(self):
        self.calls.append("StopFeedbackMonitor")

    def tick(self):
        if self.callback:
            self.feedback.timestamp += 8
            self.feedback.tool_vector_actual = FakePose(self.pose)
            self.callback(self.feedback)

    def GetPose(self, **kwargs):
        self.calls.append(("GetPose", kwargs))
        return "0,{" + ",".join(str(x) for x in self.pose) + "},GetPose(user=0,tool=0);"

    def GetAngle(self):
        return "0,{1,2,3,4,5,6},GetAngle();"

    def RobotMode(self):
        self.feedback.robot_mode = self.mode
        self.tick()
        return f"0,{{{self.mode}}},RobotMode();"

    def GetErrorID(self):
        return f"0,{{{self.errors}}},GetErrorID();"

    def GetCurrentCommandID(self):
        if self.target is not None and self.complete:
            self.pose = list(self.target)
            self.mode = 5
        self.tick()
        return f"0,{{{self.command_id}}},GetCurrentCommandID();"

    def MovL(self, pose, coord_type, **kwargs):
        self.calls.append(("MovL", list(pose), coord_type, kwargs))
        self.target = pose
        self.command_id += 1
        self.mode = 7
        self.on_move()
        return self.motion_reply or f"0,{{{self.command_id}}},MovL(pose={{1,2,3,4,5,6}},user=0,tool=0);"

    def Stop(self):
        self.calls.append("Stop")
        return "0,{},Stop();"


class RobotTests(unittest.TestCase):
    def setUp(self):
        self.fake = FakeSdkRobot()
        self.sdk = SimpleNamespace(DobotRobot=lambda *args, **kwargs: self.fake,
                                   CoordinateType=SimpleNamespace(CARTESIAN=1))
        self.loader = patch("handeye.robot._load_sdk", return_value=self.sdk)
        self.loader.start()
        self.addCleanup(self.loader.stop)

    def robot(self, *, feedback_only=False):
        robot = DobotRobot({"host": "fake-only", "poll_interval_s": 0.001},
                           feedback_only=feedback_only)
        self.addCleanup(robot.close)
        return robot

    def test_reply_does_not_confuse_echoed_pose_with_queue_id(self):
        self.assertEqual(_reply("0,{41},MovL(pose={1,2,3,4,5,6},tool=0);", "MovL"), "41")

    def test_rejects_truncated_wrong_and_error_replies(self):
        for raw in ("0,{1},GetAngle();", "0,{1},MovL(", "-1,{},MovL();",
                    "0,{1},MovL();0,{2},MovL();", "nan,{},MovL();"):
            with self.subTest(raw=raw), self.assertRaises(RobotError):
                _reply(raw, "MovL")

    def test_non_tcp_mode_has_specific_actionable_error(self):
        with self.assertRaisesRegex(RobotError, "TCP/IP control mode") as caught:
            _reply("Control Mode Is Not Tcp", "RobotMode")
        self.assertIn("gripper TCP", str(caught.exception))

    def test_reads_use_explicit_flange_and_do_not_mutate_settings(self):
        robot = self.robot()
        self.assertEqual(robot.read_flange_pose(), self.fake.pose)
        self.assertEqual(robot.read_joints(), [1, 2, 3, 4, 5, 6])
        self.assertTrue(robot.status()["ready"])
        self.assertEqual(self.fake.calls, ["Connect", ("GetPose", {"user": 0, "tool": 0})])

    def test_nonidentity_tool_rejected_without_writing_it(self):
        self.fake.feedback.tool_value[2] = 100
        with self.assertRaisesRegex(RobotError, "Tool0 is not identity"):
            self.robot()
        self.assertEqual(self.fake.calls, ["Connect", "Disconnect"])

    def test_other_selected_tool_rejected_without_getter(self):
        self.fake.feedback.tool_coordinate = 1
        with self.assertRaisesRegex(RobotError, "Select Tool0"):
            self.robot()

    def test_optional_frame_getters_verified(self):
        self.fake.feedback.tool_coordinate = 1
        self.fake.robot_control.GetTool = lambda i: "0,{0,0,0,0,0,0},GetTool(0);"
        self.robot()

    def test_stale_feedback_blocks_read(self):
        robot = self.robot()
        robot._received_at -= 20
        with self.assertRaisesRegex(RobotError, "fresh V4 feedback"):
            robot.read_flange_pose()

    def test_invalid_pose_reply_rejected(self):
        robot = self.robot()
        self.fake.robot_control.GetPose = lambda **kw: "0,{1,2,3,4,5,NaN},GetPose();"
        with self.assertRaises(RobotError):
            robot.read_flange_pose()

    def test_alarms_and_moving_or_disabled_state_block_submission(self):
        robot = self.robot()
        for mode, errors in ((4, "[]"), (7, "[]"), (5, "[1537]"), (9, "[]")):
            self.fake.mode = mode
            self.fake.errors = errors
            with self.subTest(mode=mode, errors=errors), self.assertRaises(RobotError):
                robot.move_flange([1, 2, 3, 4, 5, 6], 10)
        self.assertFalse(any(isinstance(c, tuple) and c[0] == "MovL" for c in self.fake.calls))
        self.assertNotIn("Stop", self.fake.calls)

    def test_success_waits_and_uses_mm_speed_and_zero_frames(self):
        robot = self.robot()
        target = [105, 200, 300, 180, 0, 0]
        robot.move_flange(target, 10, timeout_s=1)
        moves = [c for c in self.fake.calls if isinstance(c, tuple) and c[0] == "MovL"]
        self.assertEqual(moves, [("MovL", target, 1, {"user": 0, "tool": 0, "speed": 10, "cp": 0})])
        self.assertEqual(robot.read_flange_pose(), target)
        self.assertNotIn("Stop", self.fake.calls)

    def simulated_clock(self):
        clock = [100.0]
        def now():
            clock[0] += .002
            return clock[0]
        def sleep(seconds):
            clock[0] += seconds
        for name, replacement in (("monotonic", now), ("sleep", sleep)):
            mock = patch("handeye.robot.time." + name, replacement)
            mock.start()
            self.addCleanup(mock.stop)

    def test_test_motion_uses_actual_pose_stability_despite_velocity_noise(self):
        self.simulated_clock()
        self.fake.feedback.joint_state.qd_actual = [.11] * 6
        self.fake.feedback.tcp_speed_actual = FakePose([.71] * 6)
        robot = self.robot()
        self.assertFalse(robot.status()["ready"])
        robot.verify_feedback_pose()
        target = [105, 200, 300, 180, 0, 0]
        robot.move_flange(target, 5, pose_stability=True, expected_start=self.fake.pose)
        self.assertEqual(robot.read_flange_pose(), target)
        self.assertNotIn("Stop", self.fake.calls)

    def test_test_motion_rejects_image_pose_mismatch_before_send(self):
        self.simulated_clock()
        robot = self.robot()
        with self.assertRaisesRegex(RobotError, "Arm moved since"):
            robot.move_flange([105, 200, 300, 180, 0, 0], 5, pose_stability=True,
                              expected_start=[101, 200, 300, 180, 0, 0])
        self.assertFalse(any(isinstance(c, tuple) and c[0] == "MovL" for c in self.fake.calls))

    def test_stability_detects_actual_motion_before_submission(self):
        self.simulated_clock()
        robot = self.robot()
        original = self.fake.robot_control.RobotMode
        def drifting():
            self.fake.pose[0] += .06
            return original()
        self.fake.robot_control.RobotMode = drifting
        with self.assertRaisesRegex(ValueError, "actual pose was unstable"):
            robot.move_flange([105, 200, 300, 180, 0, 0], 5, pose_stability=True)
        self.assertFalse(any(isinstance(c, tuple) and c[0] == "MovL" for c in self.fake.calls))

    def test_stability_rejects_frozen_feedback(self):
        self.simulated_clock()
        robot = self.robot()
        self.fake.tick = lambda: None
        with self.assertRaisesRegex((ValueError, RobotError), "stale|fresh"):
            robot.wait_enabled_still()

    def test_test_motion_cancellation_after_send_stops(self):
        self.simulated_clock()
        cancel = threading.Event()
        self.fake.on_move = cancel.set
        with self.assertRaisesRegex(RobotError, "cancelled"):
            self.robot().move_flange([105, 200, 300, 180, 0, 0], 5,
                                    cancel_event=cancel, pose_stability=True)
        self.assertEqual(self.fake.calls.count("Stop"), 1)

    def test_dashboard_feedback_pose_disagreement_is_rejected(self):
        robot = self.robot()
        self.fake.robot_control.GetPose = lambda **kw: "0,{999,200,300,180,0,0},GetPose();"
        with self.assertRaisesRegex(RobotError, "pose mismatch"):
            robot.verify_feedback_pose()

    def test_large_queue_id_preserves_integer_precision(self):
        self.fake.command_id = 2 ** 60
        self.robot().move_flange([105, 200, 300, 180, 0, 0], 10, timeout_s=1)

    def test_rejected_motion_requests_stop_and_does_not_retry(self):
        self.fake.motion_reply = "-2,{},MovL();"
        with self.assertRaisesRegex(RobotError, "ErrorID=-2"):
            self.robot().move_flange([105, 200, 300, 180, 0, 0], 10)
        self.assertEqual(self.fake.calls.count("Stop"), 1)
        self.assertEqual(sum(isinstance(c, tuple) and c[0] == "MovL" for c in self.fake.calls), 1)

    def test_timeout_stops_instead_of_returning_on_accepted_command(self):
        self.fake.complete = False
        with self.assertRaisesRegex(RobotError, "timed out"):
            self.robot().move_flange([105, 200, 300, 180, 0, 0], 10, timeout_s=0.01)
        self.assertIn("Stop", self.fake.calls)

    def test_cancel_before_send_has_no_motion_or_stop(self):
        robot = self.robot()
        cancel = threading.Event()
        cancel.set()
        with self.assertRaisesRegex(RobotError, "cancelled before submission"):
            robot.move_flange([105, 200, 300, 180, 0, 0], 10, cancel_event=cancel)
        self.assertEqual(self.fake.calls, ["Connect"])

    def test_cancel_after_send_stops_from_motion_thread(self):
        cancel = threading.Event()
        self.fake.on_move = cancel.set
        with self.assertRaisesRegex(RobotError, "cancelled"):
            self.robot().move_flange([105, 200, 300, 180, 0, 0], 10, cancel_event=cancel)
        self.assertEqual(self.fake.calls.count("Stop"), 1)

    def test_in_motion_alarm_stops(self):
        self.fake.on_move = lambda: setattr(self.fake, "errors", "[1537]")
        with self.assertRaisesRegex(RobotError, "interrupted"):
            self.robot().move_flange([105, 200, 300, 180, 0, 0], 10)
        self.assertEqual(self.fake.calls.count("Stop"), 1)

    def test_in_motion_frame_change_stops(self):
        self.fake.on_move = lambda: setattr(self.fake.feedback, "tool_value", [0, 0, 30, 0, 0, 0])
        with self.assertRaisesRegex(RobotError, "not identity"):
            self.robot().move_flange([105, 200, 300, 180, 0, 0], 10)
        self.assertEqual(self.fake.calls.count("Stop"), 1)

    def test_bad_speed_or_target_never_submitted(self):
        robot = self.robot()
        for speed in (0, -1, float("nan"), 1.5, True):
            with self.subTest(speed=speed), self.assertRaises(RobotError):
                robot.move_flange([0] * 6, speed)
        with self.assertRaises(RobotError):
            robot.move_flange([0] * 5, 10)
        self.assertEqual(self.fake.calls, ["Connect"])

    def test_closed_robot_refuses_read(self):
        robot = self.robot()
        robot.close()
        with self.assertRaisesRegex(RobotError, "closed"):
            robot.read_flange_pose()

    def test_feedback_only_never_opens_dashboard_or_sends_commands(self):
        def forbidden(*args, **kwargs):
            self.fail("Feedback-only adapter accessed a Dashboard command")

        self.fake.Connect = forbidden
        self.fake.robot_control = SimpleNamespace(
            GetPose=forbidden, GetAngle=forbidden, RobotMode=forbidden,
            GetErrorID=forbidden, GetUser=forbidden, GetTool=forbidden, Stop=forbidden,
        )
        self.fake.motion = SimpleNamespace(MovL=forbidden, GetCurrentCommandID=forbidden)
        robot = self.robot(feedback_only=True)
        self.assertEqual(robot.read_flange_pose(), self.fake.pose)
        self.assertEqual(robot.read_joints(), [1, 2, 3, 4, 5, 6])
        state = robot.status()
        self.assertTrue(state["stationary"])
        self.assertFalse(state["ready"])
        self.assertIsNone(state["errors"])
        self.assertEqual(state["source"], "feedback")
        self.assertTrue(state["feedback_only"])
        robot.close()
        self.assertEqual(self.fake.calls, [("feedback_connect", 3.0), "StopFeedbackMonitor", "feedback_disconnect"])

    def test_feedback_only_missing_private_api_has_no_dashboard_fallback(self):
        del self.fake._feedback_conn
        with self.assertRaisesRegex(RobotError, "feedback-only connection API"):
            self.robot(feedback_only=True)
        self.assertEqual(self.fake.calls, [])

    def test_feedback_only_blocks_move_stop_and_close_never_stops(self):
        robot = self.robot(feedback_only=True)
        with self.assertRaisesRegex(RobotError, "cannot send motion"):
            robot.move_flange([0] * 6, 1)
        with self.assertRaisesRegex(RobotError, "cannot send Stop"):
            robot.stop()
        # Cleanup must not gain command authority, even if internal state is bad.
        robot._motion_active = True
        robot.close()
        self.assertNotIn("Stop", self.fake.calls)
        self.assertNotIn("Connect", self.fake.calls)

    def test_feedback_only_stale_data_blocks_all_read_methods(self):
        robot = self.robot(feedback_only=True)
        robot._received_at -= 20
        for method in (robot.read_flange_pose, robot.read_joints, robot.status):
            with self.subTest(method=method.__name__), self.assertRaisesRegex(RobotError, "fresh V4 feedback"):
                method()

    def test_feedback_only_identity_uses_feedback_even_when_getters_exist(self):
        robot = self.robot(feedback_only=True)
        self.fake.robot_control.GetTool = lambda i: "0,{0,0,0,0,0,0},GetTool(0);"
        for field, invalid in (("tool_coordinate", 1), ("user_coordinate", 1),
                               ("tool_value", [0, 0, 10, 0, 0, 0]),
                               ("user_value", [0, 0, 0, 0, 0, 1])):
            original = getattr(self.fake.feedback, field)
            setattr(self.fake.feedback, field, invalid)
            self.fake.tick()
            for method in (robot.read_flange_pose, robot.read_joints, robot.status):
                with self.subTest(field=field, method=method.__name__), self.assertRaises(RobotError):
                    method()
            setattr(self.fake.feedback, field, original)

    def test_feedback_pose_and_identity_are_from_exact_same_packet(self):
        robot = self.robot(feedback_only=True)
        first = copy.deepcopy(self.fake.feedback)
        first.tool_vector_actual = SimpleNamespace(to_list=lambda: [10, 20, 30, 40, 50, 60])
        later = copy.deepcopy(first)
        later.tool_coordinate = 1
        later.tool_vector_actual = SimpleNamespace(to_list=lambda: [999] * 6)
        with patch.object(robot, "_snapshot", side_effect=[(first, 1.0), (later, 2.0)]) as snapshot:
            self.assertEqual(robot.read_flange_pose(), [10, 20, 30, 40, 50, 60])
            snapshot.assert_called_once_with()
        with patch.object(robot, "_snapshot", side_effect=[(later, 1.0), (first, 2.0)]) as snapshot:
            with self.assertRaisesRegex(RobotError, "Select Tool0"):
                robot.read_flange_pose()
            snapshot.assert_called_once_with()

    def test_feedback_only_invalid_pose_joints_and_speeds_are_rejected(self):
        robot = self.robot(feedback_only=True)
        self.fake.pose[-1] = float("nan")
        self.fake.tick()
        with self.assertRaisesRegex(RobotError, "finite"):
            robot.read_flange_pose()
        self.fake.feedback.joint_state.q_actual[-1] = float("inf")
        self.fake.tick()
        with self.assertRaisesRegex(RobotError, "finite"):
            robot.read_joints()
        self.fake.feedback.joint_state.qd_actual[-1] = float("nan")
        self.fake.tick()
        with self.assertRaisesRegex(RobotError, "finite"):
            robot.status()
        self.fake.feedback.joint_state.qd_actual[-1] = 0
        self.fake.feedback.tcp_speed_actual = SimpleNamespace(to_list=lambda: [float("nan")] * 6)
        self.fake.tick()
        with self.assertRaisesRegex(RobotError, "finite"):
            robot.status()

    def test_feedback_only_activity_modes_and_fault_flags_block_capture(self):
        robot = self.robot(feedback_only=True)
        for field, invalid in (("running_status", 1), ("run_queued_cmd", 1),
                               ("pause_cmd_flag", 1), ("drag_status", 2), ("jog_status", 1),
                               ("robot_mode", 6), ("robot_mode", 7), ("robot_mode", 9),
                               ("robot_mode", 11), ("error_status", True), ("collision_state", 1)):
            original = getattr(self.fake.feedback, field)
            setattr(self.fake.feedback, field, invalid)
            self.fake.tick()
            with self.subTest(field=field, invalid=invalid):
                state = robot.status()
                self.assertFalse(state["stationary"])
                self.assertFalse(state["ready"])
            setattr(self.fake.feedback, field, original)
        self.fake.feedback.joint_state.qd_actual[0] = 0.06
        self.fake.tick()
        self.assertFalse(robot.status()["stationary"])
        self.fake.feedback.joint_state.qd_actual[0] = 0
        self.fake.feedback.tcp_speed_actual = SimpleNamespace(to_list=lambda: [0.11] + [0] * 5)
        self.fake.tick()
        self.assertFalse(robot.status()["stationary"])

    def test_stationarity_diagnostics_in_both_modes(self):
        for feedback_only in (False, True):
            with self.subTest(feedback_only=feedback_only):
                robot = self.robot(feedback_only=feedback_only)
                self.fake.feedback.drag_status = 1
                self.fake.feedback.joint_state.qd_actual[0] = 0.2
                self.fake.tick()
                state = robot.status()
                details = state["stationarity_details"]
                self.assertFalse(state["stationary"])
                self.assertEqual(details["drag_status"], 1)
                self.assertEqual(details["max_joint_speed_deg_s"], 0.2)
                self.assertEqual(details["joint_speed_limit_deg_s"], 0.05)
                self.assertEqual(details["mode"], 5)
                robot.close()

    def test_capture_history_copies_packets_and_does_not_expose_mutable_storage(self):
        robot = self.robot(feedback_only=True)
        first = robot.capture_feedback()[0]
        original_pose = list(self.fake.pose)
        self.fake.pose[0] += 10
        self.fake.feedback.tool_value[2] = 100
        self.fake.feedback.robot_mode = 6
        retained = robot.capture_feedback()[0]
        self.assertEqual(retained["flange_pose_mm_deg"], original_pose)
        self.assertEqual(retained["state"]["mode"], 5)
        self.assertEqual(set(retained), {"sequence", "received_at", "controller_timestamp",
                                         "flange_pose_mm_deg", "state"})
        first["flange_pose_mm_deg"][0] = 999
        first["state"]["stationarity_details"]["drag_status"] = 1
        reread = robot.capture_feedback()[0]
        self.assertEqual(reread["flange_pose_mm_deg"], retained["flange_pose_mm_deg"])
        self.assertEqual(reread["state"]["stationarity_details"], retained["state"]["stationarity_details"])

    def test_capture_history_never_reads_dashboard_even_on_command_capable_adapter(self):
        robot = self.robot()

        def forbidden(*args, **kwargs):
            self.fail("Capture history must never send Dashboard commands")

        self.fake.robot_control = SimpleNamespace(GetPose=forbidden, GetUser=forbidden,
                                                GetTool=forbidden, RobotMode=forbidden,
                                                GetErrorID=forbidden)
        record = robot.capture_feedback()[0]
        self.assertEqual(record["state"]["source"], "feedback")
        self.assertFalse(record["state"]["ready"])
        self.assertIsNone(record["state"]["errors"])
        self.fake.feedback.tool_coordinate = 1
        self.fake.tick()
        with self.assertRaisesRegex(RobotError, "Select Tool0"):
            robot.capture_feedback()

    def test_capture_history_returns_all_intermediate_activity_and_timestamp_samples(self):
        robot = self.robot(feedback_only=True)
        first = robot.capture_feedback()[0]
        self.fake.tick()
        self.fake.feedback.drag_status = 1
        self.fake.tick()
        self.fake.feedback.drag_status = 0
        self.fake.tick()
        records = robot.capture_feedback(first["sequence"])
        self.assertEqual([r["sequence"] for r in records], [2, 3, 4])
        self.assertEqual([r["controller_timestamp"] for r in records], [1016, 1024, 1032])
        self.assertEqual([r["state"]["capture_idle"] for r in records], [True, False, True])
        self.assertTrue(all(r["received_at"] >= first["received_at"] for r in records))
        self.assertEqual(robot.capture_feedback(records[-1]["sequence"]), [])
        self.assertEqual(len(robot.capture_feedback()), 1)

    def test_capture_history_rejects_invalid_intermediate_frames_and_state_data(self):
        robot = self.robot(feedback_only=True)
        for field, bad_value in (("tool_value", [0, 0, 100, 0, 0, 0]),
                                 ("running_status", float("nan")), ("robot_mode", "5")):
            marker = robot.capture_feedback()[0]["sequence"]
            original = getattr(self.fake.feedback, field)
            setattr(self.fake.feedback, field, bad_value)
            self.fake.tick()
            setattr(self.fake.feedback, field, original)
            self.fake.tick()
            with self.subTest(field=field), self.assertRaisesRegex(RobotError, "Invalid capture feedback sample"):
                robot.capture_feedback(marker)
            self.assertTrue(robot.capture_feedback()[0]["state"]["capture_idle"])

    def test_capture_history_rejects_copy_failure_even_after_good_packet(self):
        class BadCopy:
            def __deepcopy__(self, memo):
                raise ValueError("cannot copy")

        robot = self.robot(feedback_only=True)
        marker = robot.capture_feedback()[0]["sequence"]
        robot._on_feedback(BadCopy())
        self.fake.tick()
        with self.assertRaisesRegex(RobotError, "Cannot copy V4 feedback sample"):
            robot.capture_feedback(marker)

    def test_capture_history_requires_explicit_integer_controller_timestamp(self):
        robot = self.robot(feedback_only=True)
        del self.fake.feedback.timestamp
        robot._on_feedback(self.fake.feedback)
        with self.assertRaisesRegex(RobotError, "controller timestamp"):
            robot.capture_feedback()
        for value in (None, True, "1000", 1000.5, -1):
            self.fake.feedback.timestamp = value
            robot._on_feedback(self.fake.feedback)
            with self.subTest(value=value), self.assertRaisesRegex(RobotError, "controller timestamp"):
                robot.capture_feedback()

    def test_capture_history_stale_closed_and_overflow_are_rejected(self):
        robot = self.robot(feedback_only=True)
        initial = robot.capture_feedback()[0]
        with patch("handeye.robot.time.monotonic", return_value=initial["received_at"] + 10):
            for marker in (None, initial["sequence"]):
                with self.subTest(marker=marker), self.assertRaisesRegex(RobotError, "fresh V4 feedback"):
                    robot.capture_feedback(marker)
        robot._max_age = 60
        for _ in range(2048):
            self.fake.tick()
        self.assertEqual(len(robot._feedback_history), 2048)
        with self.assertRaisesRegex(RobotError, "history overflow"):
            robot.capture_feedback(0)
        self.assertEqual(len(robot.capture_feedback(initial["sequence"])), 2048)
        with self.assertRaisesRegex(RobotError, "newer than"):
            robot.capture_feedback(9999)
        for marker in (-1, True, 1.5):
            with self.subTest(marker=marker), self.assertRaisesRegex(RobotError, "after_sequence"):
                robot.capture_feedback(marker)
        robot.close()
        with self.assertRaisesRegex(RobotError, "closed"):
            robot.capture_feedback()

    def test_capture_idle_allows_velocity_noise_but_motion_readiness_stays_strict(self):
        robot = self.robot()
        self.assertTrue(robot.status()["ready"])
        self.fake.feedback.joint_state.qd_actual[0] = 0.11
        self.fake.feedback.tcp_speed_actual = FakePose([0.708, 0, 0, 0, 0, 0])
        self.fake.tick()
        state = robot.status()
        self.assertTrue(state["capture_idle"])
        self.assertFalse(state["stationary"])
        self.assertFalse(state["ready"])
        capture_state = robot.capture_feedback()[0]["state"]
        self.assertTrue(capture_state["capture_idle"])
        self.assertFalse(capture_state["stationary"])
        self.assertFalse(capture_state["ready"])


if __name__ == "__main__":
    unittest.main()
