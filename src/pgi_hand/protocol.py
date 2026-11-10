# SPDX-License-Identifier: MulanPSL-2.0
"""Bounded PGI Modbus RTU exchanges on an already owned serial port."""
import math
import time


STATUS_REGISTERS = (0x0200, 0x0201, 0x0202, 0x0402)
CONFIGURATION_REGISTERS = (0x0101, 0x0103, 0x0104)
READABLE_REGISTERS = STATUS_REGISTERS + CONFIGURATION_REGISTERS


class ModbusError(Exception):
    """A malformed response or a device-reported Modbus exception."""


class OutcomeUnknown(RuntimeError):
    """A write was attempted, but its acknowledgement is uncertain."""


class CommandRejected(RuntimeError):
    """The controller returned a valid FC06 exception; no retry is allowed."""


def crc16(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0xA001 if crc & 1 else 0)
    return crc


def read_request(slave_id: int, register: int) -> bytes:
    if not 1 <= slave_id <= 247:
        raise ValueError("slave ID must be between 1 and 247")
    if register not in READABLE_REGISTERS:
        raise ValueError("only the seven explicitly allowed PGI read registers are permitted")
    body = bytes((slave_id, 3)) + register.to_bytes(2, "big") + b"\x00\x01"
    return body + crc16(body).to_bytes(2, "little")


def read_exact(port, size: int, deadline: float) -> bytes:
    data = bytearray()
    while len(data) < size:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("timed out waiting for a complete Modbus response")
        port.timeout = remaining
        chunk = port.read(size - len(data))
        if not chunk:
            raise TimeoutError("timed out waiting for a complete Modbus response")
        data.extend(chunk)
    return bytes(data)


def read_register(port, slave_id: int, register: int, timeout: float) -> int:
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    request = read_request(slave_id, register)
    # Keep consecutive RTU requests separate at 115200 baud.
    time.sleep(0.002)
    port.reset_input_buffer()
    if port.write(request) != len(request):
        raise ModbusError("incomplete FC03 request transmission")
    deadline = time.monotonic() + timeout
    header = read_exact(port, 3, deadline)
    if header[0] != slave_id:
        raise ModbusError("response slave ID does not match the request")
    if header[1] == 0x83:
        frame = header + read_exact(port, 2, deadline)
    elif header[1] == 3:
        if header[2] != 2:
            raise ModbusError("FC03 response must contain exactly one register")
        frame = header + read_exact(port, 4, deadline)
    else:
        raise ModbusError(f"unexpected response function 0x{header[1]:02x}")
    if crc16(frame[:-2]) != int.from_bytes(frame[-2:], "little"):
        raise ModbusError("response CRC mismatch")
    if header[1] == 0x83:
        raise ModbusError(f"device returned Modbus exception 0x{header[2]:02x}")
    return int.from_bytes(frame[3:5], "big")


def read_status(port, slave_id: int, timeout: float) -> dict:
    """Read four raw status registers sequentially, not as an atomic snapshot."""
    values = {register: read_register(port, slave_id, register, timeout)
              for register in STATUS_REGISTERS}
    return {"initialization_raw": values[0x0200], "grip_status_raw": values[0x0201],
            "position_raw": values[0x0202], "io_mode_raw": values[0x0402]}


def write_position_once(port, slave_id, target_raw, response_timeout):
    """Require the complete exact FC06 echo; an uncertain write is never retried."""
    if type(slave_id) is not int or not 1 <= slave_id <= 247:
        raise ValueError("Invalid slave_id")
    if type(target_raw) is not int or not 0 <= target_raw <= 1000:
        raise ValueError("Invalid target_raw")
    body = bytes((slave_id, 6)) + b"\x01\x03" + target_raw.to_bytes(2, "big")
    request = body + crc16(body).to_bytes(2, "little")
    time.sleep(0.002)
    port.reset_input_buffer()
    port.write_timeout = response_timeout
    deadline = time.monotonic() + response_timeout
    try:
        if port.write(request) != len(request):
            raise OutcomeUnknown("Incomplete FC06 transmission; outcome uncertain, do not resend")
        header = read_exact(port, 2, deadline)
        if header[0] != slave_id:
            raise OutcomeUnknown("FC06 echo has the wrong slave ID")
        if header[1] == 0x86:
            frame = header + read_exact(port, 3, deadline)
        elif header[1] == 6:
            frame = header + read_exact(port, 6, deadline)
        else:
            raise OutcomeUnknown("Unexpected FC06 response function")
        if crc16(frame[:-2]) != int.from_bytes(frame[-2:], "little"):
            raise OutcomeUnknown("FC06 response CRC mismatch")
        if header[1] == 0x86:
            raise CommandRejected(f"Controller rejected FC06: exception 0x{frame[2]:02x}")
        if frame != request:
            raise OutcomeUnknown("FC06 echo does not exactly match the requested register/value")
    except (OutcomeUnknown, CommandRejected):
        raise
    except Exception as error:
        raise OutcomeUnknown(f"FC06 acknowledgement uncertain: {type(error).__name__}: {error}") from error


def read_before_deadline(port, slave_id, register, response_timeout, deadline):
    """Bound request transmission and response by the remaining overall budget."""
    remaining = deadline - time.monotonic()
    if remaining <= 0.004:
        raise TimeoutError("Post-command observation deadline expired")
    timeout = min(response_timeout, (remaining - 0.002) / 2)
    port.write_timeout = timeout
    value = read_register(port, slave_id, register, timeout)
    if time.monotonic() > deadline:
        raise TimeoutError("Post-command observation deadline expired")
    return value
