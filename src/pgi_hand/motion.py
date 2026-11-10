# SPDX-License-Identifier: MulanPSL-2.0
"""One explicitly enabled commissioning command; no queues of physical actions."""
import math

from pgi_hand.protocol import write_position_once, read_before_deadline

AXIS_NAME = "pgi_opening"


class MotionRejected(RuntimeError):
    """The request was not dispatched."""


class MotionUnknown(RuntimeError):
    """Dispatch was committed; transmission may have started, so never retry or release."""


def target_from_request(targets):
    """Require one known normalized axis at an exactly representable device tick."""
    if len(targets) != 1 or targets[0].name != AXIS_NAME:
        raise MotionRejected("Exactly one target named pgi_opening is required; duplicates are rejected")
    value = targets[0].value
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise MotionRejected("Target must be finite and within [0,1]; it is never clamped")
    raw = round(value * 1000)
    if not math.isclose(value * 1000, raw, rel_tol=0, abs_tol=1e-9):
        raise MotionRejected("Target must be an exact 0.001 normalized position increment")
    return raw


def check_closing(feedback, step_raw):
    """Require stopped feedback without a detected obstruction before closing."""
    if type(step_raw) is not int or not 1 <= step_raw <= 20:
        raise MotionRejected("step_raw must be an integer between 1 and 20")
    configuration = feedback["configuration"]
    required = (("initialization", feedback["initialization_raw"], 1),
                ("IO mode", feedback["io_mode_raw"], 0),
                ("grip status (stopped, no object detected)", feedback["grip_status_raw"], 1),
                ("force setpoint percent", configuration["force_setpoint_percent"], 20),
                ("speed setpoint percent", configuration["speed_setpoint_percent"], 1))
    for name, actual, expected in required:
        if type(actual) is not int or actual != expected:
            raise MotionRejected(f"{name} must be {expected}; observed {actual!r}")
    current, target = feedback["position_raw"], configuration["target_position_raw"]
    if any(type(value) is not int or not 0 <= value <= 1000 for value in (current, target)):
        raise MotionRejected("Current and configured target position must both be within 0..1000")
    if current != target:
        raise MotionRejected("Current position does not equal the configured target")
    if current < step_raw:
        raise MotionRejected("This step would move the target below zero; no clamping is performed")


def check_opening(feedback, requested_raw):
    """Require a small explicit release step from a stopped, direction-consistent state.

    State 2 can stop before a closing target is reached. That target/actual
    difference is allowed only when the old target is not above actual.
    These checks cannot establish that the box has been supported on a table.
    """
    settings = feedback["configuration"]
    required = (("initialization", feedback["initialization_raw"], 1),
                ("IO mode", feedback["io_mode_raw"], 0),
                ("force setpoint percent", settings["force_setpoint_percent"], 20),
                ("speed setpoint percent", settings["speed_setpoint_percent"], 1))
    for name, value, expected in required:
        if type(value) is not int or value != expected:
            raise MotionRejected(f"{name} must be {expected}; observed {value!r}")
    current, old_target = feedback["position_raw"], settings["target_position_raw"]
    if any(type(value) is not int or not 0 <= value <= 1000
           for value in (current, old_target, requested_raw)):
        raise MotionRejected("Actual, previous target and requested position must be within 0..1000")
    if not 1 <= requested_raw - current <= 20:
        raise MotionRejected("Opening must increase actual position by 1..20 raw units")
    status = feedback["grip_status_raw"]
    if type(status) is not int or status not in (1, 2):
        raise MotionRejected("Opening requires stopped status 1 or 2; moving, dropped or unknown is rejected")
    if status == 1 and old_target != current:
        raise MotionRejected("Status 1 opening requires matching actual and previous target")
    if status == 2 and old_target > current:
        raise MotionRejected("Status 2 opening requires previous target <= actual position")


class GuardedPort:
    """Check cancellation at the actual FC06 write boundary, after helper preparation."""

    def __init__(self, port, before_write):
        self._port = port
        self._before_write = before_write

    def __getattr__(self, name):
        return getattr(self._port, name)

    @property
    def write_timeout(self):
        return self._port.write_timeout

    @write_timeout.setter
    def write_timeout(self, value):
        self._port.write_timeout = value

    @property
    def timeout(self):
        return self._port.timeout

    @timeout.setter
    def timeout(self, value):
        self._port.timeout = value

    def write(self, request):
        """Allow only the helper's single position-write function and register."""
        if len(request) != 8 or request[1:4] != b"\x06\x01\x03":
            raise MotionRejected("Only one FC06 position-register write is supported")
        self._before_write()
        return self._port.write(request)


def execute_command(port, target_raw, deadline, config, before_write):
    """Freshly preflight, send once, and read back the target; do not wait for a grasp."""
    def read(address):
        return read_before_deadline(port, config["slave_id"], address,
                                    config["timeout_s"], deadline)

    values = {address: read(address) for address in
              (0x0200, 0x0201, 0x0202, 0x0402, 0x0101, 0x0103, 0x0104)}
    feedback = {"initialization_raw": values[0x0200], "grip_status_raw": values[0x0201],
                "position_raw": values[0x0202], "io_mode_raw": values[0x0402],
                "configuration": {"force_setpoint_percent": values[0x0101],
                                  "target_position_raw": values[0x0103],
                                  "speed_setpoint_percent": values[0x0104]}}
    delta = target_raw - values[0x0202]
    if delta < 0:
        check_closing(feedback, -delta)
    elif delta > 0:
        if not config["enable_opening"]:
            raise MotionRejected("Opening is disabled; configure enable_opening=true explicitly")
        check_opening(feedback, target_raw)
    else:
        raise MotionRejected("No-op position requests are rejected")
    guarded = GuardedPort(port, before_write)
    write_position_once(guarded, config["slave_id"], target_raw, config["timeout_s"])
    if read(0x0103) != target_raw:
        raise MotionUnknown("Target readback differs after FC06; outcome unknown, do not retry")
