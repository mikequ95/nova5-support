# SPDX-License-Identifier: MulanPSL-2.0
"""Offline frame and deadline tests; no serial device is opened."""
import time
import unittest
from unittest.mock import patch

from pgi_hand import protocol


class FakePort:
    def __init__(self, response=b"", write_result=None, write_error=None):
        self.response = response
        self.written = []
        self.timeout = self.write_timeout = 0.1
        self.write_result = write_result
        self.write_error = write_error

    def reset_input_buffer(self):
        pass

    def write(self, request):
        self.written.append(request)
        if self.write_error is not None:
            raise self.write_error
        return len(request) if self.write_result is None else self.write_result

    def read(self, size):
        size = min(size, 1)
        result, self.response = self.response[:size], self.response[size:]
        return result


class ProtocolTests(unittest.TestCase):
    def test_known_crc_and_fragmented_fc03_response(self):
        self.assertEqual(protocol.crc16(b"123456789"), 0x4B37)
        port = FakePort(bytes.fromhex("01030203e8b8fa"))
        self.assertEqual(protocol.read_register(port, 1, 0x0202, 0.1), 1000)
        self.assertEqual(port.written, [bytes.fromhex("0103020200012472")])

    def test_bad_fc03_identity_size_crc_and_exception(self):
        cases = (("020302", "slave ID"), ("010304", "one register"),
                 ("010602", "function"), ("01030203e8b8fb", "CRC"),
                 ("018302c0f1", "exception 0x02"))
        for response, message in cases:
            with self.subTest(response=response), self.assertRaisesRegex(protocol.ModbusError, message):
                protocol.read_register(FakePort(bytes.fromhex(response)), 1, 0x0202, 0.1)

    def test_incomplete_fc03_exchange_never_retries(self):
        for response in (b"", bytes.fromhex("01030203")):
            port = FakePort(response)
            with self.subTest(response=response), self.assertRaises(TimeoutError):
                protocol.read_register(port, 1, 0x0202, 0.1)
            self.assertEqual(len(port.written), 1)
        port = FakePort(write_result=3)
        with self.assertRaisesRegex(protocol.ModbusError, "incomplete"):
            protocol.read_register(port, 1, 0x0202, 0.1)
        self.assertEqual(len(port.written), 1)

    def test_only_seven_allowed_registers_use_fc03(self):
        registers = {0x0101, 0x0103, 0x0104, 0x0200, 0x0201, 0x0202, 0x0402}
        self.assertEqual(set(protocol.READABLE_REGISTERS), registers)
        for register in registers:
            request = protocol.read_request(1, register)
            self.assertEqual(request[:2], b"\x01\x03")
            self.assertEqual(request[2:4], register.to_bytes(2, "big"))
            self.assertEqual(request[4:6], b"\x00\x01")
            self.assertEqual(len(request), 8)
        port = FakePort()
        for register in (0x0100, 0x0102, 0x0105, 0x0203, 0x0300, 0x0401, 0x0504):
            with self.subTest(register=register), self.assertRaises(ValueError):
                protocol.read_register(port, 1, register, 0.1)
        for slave_id in (0, 248):
            with self.assertRaises(ValueError):
                protocol.read_request(slave_id, 0x0200)
        self.assertEqual(port.written, [])

    def test_status_reads_only_four_raw_registers(self):
        port = FakePort(bytes.fromhex("01030200017984" "01030200017984"
                                      "01030203e8b8fa" "0103020000b844"))
        self.assertEqual(protocol.read_status(port, 1, 0.1), {
            "initialization_raw": 1, "grip_status_raw": 1,
            "position_raw": 1000, "io_mode_raw": 0})
        self.assertEqual([int.from_bytes(frame[2:4], "big") for frame in port.written],
                         [0x0200, 0x0201, 0x0202, 0x0402])
        self.assertTrue(all(frame[1] == 3 for frame in port.written))

    def test_fc06_requires_one_exact_fragmented_echo(self):
        body = bytes.fromhex("0106010303de")
        request = body + protocol.crc16(body).to_bytes(2, "little")
        port = FakePort(request)
        protocol.write_position_once(port, 1, 990, 0.1)
        self.assertEqual(port.written, [request])

    def test_uncertain_fc06_echo_or_write_never_retries(self):
        body = bytes.fromhex("0106010303de")
        request = body + protocol.crc16(body).to_bytes(2, "little")
        wrong_body = bytes.fromhex("0106010303dd")
        wrong_echo = wrong_body + protocol.crc16(wrong_body).to_bytes(2, "little")
        cases = (FakePort(b""), FakePort(request[:5]), FakePort(b"\x02\x06"),
                 FakePort(b"\x01\x03"), FakePort(request[:-1] + bytes((request[-1] ^ 1,))),
                 FakePort(wrong_echo), FakePort(request, write_result=3),
                 FakePort(write_error=OSError("fake disconnect")))
        for port in cases:
            with self.subTest(port=port), self.assertRaises(protocol.OutcomeUnknown):
                protocol.write_position_once(port, 1, 990, 0.1)
            self.assertEqual(port.written, [request])

    def test_fc06_exception_is_rejected_without_retry(self):
        body = b"\x01\x86\x02"
        port = FakePort(body + protocol.crc16(body).to_bytes(2, "little"))
        with self.assertRaisesRegex(protocol.CommandRejected, "exception 0x02"):
            protocol.write_position_once(port, 1, 990, 0.1)
        self.assertEqual(len(port.written), 1)

    def test_invalid_target_or_expired_read_budget_sends_nothing(self):
        port = FakePort()
        for slave, target in ((0, 990), (248, 990), (1, -1), (1, 1001), (1, True)):
            with self.subTest(slave=slave, target=target), self.assertRaises(ValueError):
                protocol.write_position_once(port, slave, target, 0.1)
        with self.assertRaises(TimeoutError):
            protocol.read_before_deadline(port, 1, 0x0202, 0.1, time.monotonic() - 1)
        with self.assertRaises(TimeoutError):
            protocol.read_exact(port, 1, time.monotonic() - 1)
        self.assertEqual(port.written, [])

    def test_overall_deadline_bounds_read_and_rejects_late_result(self):
        port = FakePort()
        with patch.object(protocol.time, "monotonic", side_effect=[10.0, 10.05]), \
                patch.object(protocol, "read_register", return_value=990) as read:
            self.assertEqual(protocol.read_before_deadline(port, 1, 0x0103, 1, 10.1), 990)
        self.assertAlmostEqual(port.write_timeout, 0.049)
        self.assertAlmostEqual(read.call_args.args[3], 0.049)
        with patch.object(protocol.time, "monotonic", side_effect=[10.0, 10.2]), \
                patch.object(protocol, "read_register", return_value=990):
            with self.assertRaises(TimeoutError):
                protocol.read_before_deadline(port, 1, 0x0103, 1, 10.1)


if __name__ == "__main__":
    unittest.main()
