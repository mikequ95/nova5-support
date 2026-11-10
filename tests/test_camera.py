# SPDX-License-Identifier: MulanPSL-2.0
"""Camera contracts checked without importing the SDK or accessing hardware."""
import sys
from types import ModuleType, SimpleNamespace

import cv2
import numpy as np
import pytest

from handeye.camera import Camera


FORMATS = SimpleNamespace(MJPG=1, RGB=2, BGR=3, YUYV=4)


@pytest.fixture
def fake_sdk(monkeypatch):
    sdk = ModuleType("pyorbbecsdk")

    class OBError(Exception):
        pass

    sdk.OBError = OBError
    sdk.OBFormat = FORMATS
    sdk.OBSensorType = SimpleNamespace(COLOR_SENSOR="color")
    state = SimpleNamespace(opened=[], sensors=[], selected=[], enabled=[], stopped=False,
                            open_error=None, profile_error=None, started=False, enumerated=[],
                            available_profiles=[(1280, 800, "MJPG", 30), (640, 400, "YUYV", 15)])
    info = SimpleNamespace(get_name=lambda: "Orbbec Gemini 336L", get_serial_number=lambda: "wrist-rgb")
    target = SimpleNamespace(get_device_info=lambda: info)

    def open_device(index):
        state.opened.append(index)
        if state.open_error is not None:
            raise state.open_error
        if index != 1:
            raise RuntimeError("Unrelated camera is exclusively owned by another process")
        return target

    devices = SimpleNamespace(
        get_count=lambda: 2,
        get_device_serial_number_by_index=lambda index: ["unrelated-busy", "wrist-rgb"][index],
        get_device_by_index=open_device,
    )
    sdk.Context = lambda: SimpleNamespace(query_devices=lambda: devices,
                                          get_timestamp_clock_type=lambda: "REALTIME")

    def select_profile(width, height, fmt, fps):
        profile = (width, height, fmt, fps)
        state.selected.append(profile)
        if state.profile_error is not None:
            raise state.profile_error
        return profile

    def profile_by_index(index):
        state.enumerated.append(index)
        width, height, format_name, fps = state.available_profiles[index]
        video_profile = SimpleNamespace(
            get_width=lambda: width, get_height=lambda: height,
            get_format=lambda: SimpleNamespace(name=format_name), get_fps=lambda: fps,
        )
        return SimpleNamespace(as_video_stream_profile=lambda: video_profile)

    def profiles(sensor):
        state.sensors.append(sensor)
        assert sensor == "color", "Depth or IR stream was requested"
        return SimpleNamespace(get_video_stream_profile=select_profile,
                               get_count=lambda: len(state.available_profiles),
                               get_stream_profile_by_index=profile_by_index)

    def start(config):
        state.started = True
        state.enabled = config.enabled.copy()

    def stop():
        state.stopped = True

    def pipeline(device):
        assert device is target
        return SimpleNamespace(get_stream_profile_list=profiles, start=start, stop=stop)

    class Config:
        def __init__(self):
            self.enabled = []

        def enable_stream(self, profile):
            self.enabled.append(profile)

    sdk.Config = Config
    sdk.Pipeline = pipeline
    monkeypatch.setitem(sys.modules, "pyorbbecsdk", sdk)
    return state


def test_serial_selection_ignores_busy_device_and_starts_only_requested_rgb_profile(fake_sdk):
    config = {"backend": "orbbec", "serial": "wrist-rgb", "expected_name": "Gemini 336L",
              "width": 1280, "height": 720, "fps": 30, "format": "MJPG"}
    with Camera(config) as camera:
        assert fake_sdk.opened == [1]
        assert fake_sdk.sensors == ["color"]
        assert fake_sdk.selected == [(1280, 720, FORMATS.MJPG, 30)]
        assert fake_sdk.enabled == fake_sdk.selected
        assert camera.identity == {"backend": "orbbec", "serial": "wrist-rgb",
                                   "name": "Orbbec Gemini 336L", "format": "MJPG",
                                   "width": 1280, "height": 720, "fps": 30}
    assert fake_sdk.stopped


def test_ambiguous_serial_selection_does_not_open_any_camera(fake_sdk):
    with pytest.raises(RuntimeError, match="exactly one"):
        Camera({"backend": "orbbec", "serial": None, "width": 1280,
                "height": 720, "fps": 30, "format": "MJPG"})
    assert fake_sdk.opened == []


def test_sdk_open_failure_has_actionable_message_and_preserves_original_error(fake_sdk):
    original = sys.modules["pyorbbecsdk"].OBError("usbEnumerator openUsbDevice failed!")
    fake_sdk.open_error = original
    with pytest.raises(RuntimeError) as raised:
        Camera({"backend": "orbbec", "serial": "wrist-rgb", "width": 1280,
                "height": 720, "fps": 30, "format": "MJPG"})
    message = str(raised.value)
    assert "USB access permissions" in message
    assert "using the camera" in message
    assert "USB connection" in message
    assert str(original) in message
    assert raised.value.__cause__ is original
    assert fake_sdk.opened == [1]
    assert fake_sdk.sensors == []
    assert fake_sdk.enabled == []


def test_non_sdk_open_failure_is_not_relabelled_as_a_device_error(fake_sdk):
    original = ValueError("unexpected application bug")
    fake_sdk.open_error = original
    with pytest.raises(ValueError) as raised:
        Camera({"backend": "orbbec", "serial": "wrist-rgb", "width": 1280,
                "height": 720, "fps": 30, "format": "MJPG"})
    assert raised.value is original


def test_unsupported_rgb_profile_lists_available_modes_without_starting_or_falling_back(fake_sdk):
    original = sys.modules["pyorbbecsdk"].OBError("Invalid input, No matched video stream profile found!")
    fake_sdk.profile_error = original
    config = {"backend": "orbbec", "serial": "wrist-rgb", "width": 1280,
              "height": 720, "fps": 30, "format": "MJPG"}
    with pytest.raises(RuntimeError) as raised:
        Camera(config)
    message = str(raised.value)
    assert "1280x720 MJPG 30 fps" in message
    assert "1280x800 MJPG 30 fps; 640x400 YUYV 15 fps" in message
    assert "config.json" in message
    assert "recapture and recalibrate" in message
    assert str(original) in message
    assert raised.value.__cause__ is original
    assert fake_sdk.selected == [(1280, 720, FORMATS.MJPG, 30)]
    assert fake_sdk.enumerated == [0, 1]
    assert fake_sdk.sensors == ["color"]
    assert not fake_sdk.started
    assert fake_sdk.enabled == []
    assert config["height"] == 720


def test_non_sdk_profile_failure_is_not_relabelled_as_an_unsupported_mode(fake_sdk):
    original = ValueError("unexpected application bug")
    fake_sdk.profile_error = original
    with pytest.raises(ValueError) as raised:
        Camera({"backend": "orbbec", "serial": "wrist-rgb", "width": 1280,
                "height": 720, "fps": 30, "format": "MJPG"})
    assert raised.value is original
    assert fake_sdk.enumerated == []
    assert fake_sdk.sensors == ["color"]
    assert not fake_sdk.started
    assert fake_sdk.enabled == []


def test_available_rgb_mode_message_is_bounded(fake_sdk):
    fake_sdk.profile_error = sys.modules["pyorbbecsdk"].OBError("No matched video stream profile found!")
    fake_sdk.available_profiles = [(1280, 800, "MJPG", fps) for fps in range(1, 15)]
    with pytest.raises(RuntimeError) as raised:
        Camera({"backend": "orbbec", "serial": "wrist-rgb", "width": 1280,
                "height": 720, "fps": 30, "format": "MJPG"})
    assert fake_sdk.enumerated == list(range(12))
    assert "1280x800 MJPG 12 fps" in str(raised.value)
    assert "1280x800 MJPG 13 fps" not in str(raised.value)
    assert "2 additional profiles not listed" in str(raised.value)
    assert not fake_sdk.started


def camera_with_frame(fmt, data, *, expected_width=4):
    color = SimpleNamespace(
        get_format=lambda: fmt, get_data=lambda: data,
        get_width=lambda: 4, get_height=lambda: 2,
        get_system_timestamp_us=lambda: 1770000000123456,
        get_timestamp_us=lambda: 123456789,
    )
    camera = Camera.__new__(Camera)
    camera.backend = "orbbec"
    camera.config = {"width": expected_width, "height": 2}
    camera.clock_type = "REALTIME"
    camera.ob = SimpleNamespace(OBFormat=FORMATS)
    camera.handle = SimpleNamespace(wait_for_frames=lambda timeout: SimpleNamespace(get_color_frame=lambda: color))
    return camera


@pytest.mark.parametrize("format_name", ["MJPG", "RGB", "BGR", "YUYV"])
def test_color_conversion_keeps_bgr_channels_and_sdk_metadata(format_name):
    expected = np.full((2, 4, 3), [20, 70, 150], dtype=np.uint8)
    if format_name == "MJPG":
        success, data = cv2.imencode(".jpg", expected)
        assert success
        tolerance = 3  # JPEG is lossy; verify channel order rather than encoded bytes.
    elif format_name == "RGB":
        data = expected[:, :, ::-1].copy().ravel()
        tolerance = 0
    elif format_name == "BGR":
        data = expected.ravel()
        tolerance = 0
    else:
        # BT.601 limited-range red: Y=81, U=90, V=240, packed Y0-U-Y1-V.
        data = np.tile(np.array([81, 90, 81, 240], dtype=np.uint8), 4)
        expected = np.full((2, 4, 3), [0, 0, 255], dtype=np.uint8)
        tolerance = 1
    camera = camera_with_frame(getattr(FORMATS, format_name), data)
    frame, received_at = camera.read()
    assert frame.shape == (2, 4, 3)
    np.testing.assert_allclose(frame, expected, rtol=0, atol=tolerance)
    assert isinstance(received_at, float)
    assert camera.last_frame_metadata == {
        "sdk_system_timestamp_us": 1770000000123456,
        "sdk_device_timestamp_us": 123456789,
        "sdk_clock_type": "REALTIME",
    }


def test_actual_frame_dimensions_must_match_calibration_profile():
    camera = camera_with_frame(FORMATS.BGR, np.zeros(2 * 4 * 3, dtype=np.uint8), expected_width=8)
    with pytest.raises(RuntimeError, match="resolution changed"):
        camera.read()
