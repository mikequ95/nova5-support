# SPDX-License-Identifier: MulanPSL-2.0
"""RGB-only acquisition. No depth stream, alignment or point cloud is enabled."""
from __future__ import annotations

import time
import cv2
import numpy as np


class Camera:
    def __init__(self, config):
        self.config = dict(config)
        self.backend = config.get("backend", "orbbec")
        self.handle = None
        self.identity = {}
        self.last_frame_metadata = {}
        width, height, fps = (int(config[k]) for k in ("width", "height", "fps"))
        if min(width, height, fps) <= 0:
            raise ValueError("Camera width/height/fps must be positive")
        if self.backend == "opencv":
            source = config.get("source")
            if not isinstance(source, str) or not source.startswith("/dev/v4l/by-id/"):
                raise ValueError("Set camera.source to the stable /dev/v4l/by-id/... RGB device path")
            self.handle = cv2.VideoCapture(source)
            if not self.handle.isOpened():
                self.close()
                raise RuntimeError(f"Cannot open RGB camera {source!r}")
            self.handle.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.handle.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.handle.set(cv2.CAP_PROP_FPS, fps)
            self.identity = {"backend": "opencv", "source": source}
        elif self.backend == "orbbec":
            try:
                import pyorbbecsdk as ob
            except ImportError as exc:
                raise RuntimeError("Install nova5-support[camera] for Orbbec RGB acquisition") from exc
            self.ob = ob
            self.context = ob.Context()
            self.clock_type = str(self.context.get_timestamp_clock_type()) if hasattr(self.context, "get_timestamp_clock_type") else "unknown"
            devices = self.context.query_devices()
            wanted = config.get("serial")
            matches = []
            for i in range(devices.get_count()):
                serial = devices.get_device_serial_number_by_index(i)
                if wanted is None or serial == wanted:
                    matches.append(i)
            if len(matches) != 1:
                raise RuntimeError("Connect exactly one Orbbec camera or set camera.serial")
            try:
                device = devices.get_device_by_index(matches[0])
            except ob.OBError as exc:
                raise RuntimeError(
                    "Cannot open the Orbbec RGB camera. Check USB access permissions for this user"
                    " (active udev rules), whether another program is using the camera, and the USB connection. "
                    f"SDK error: {exc}"
                ) from exc
            info = device.get_device_info()
            expected = config.get("expected_name")
            if expected and expected not in info.get_name():
                raise RuntimeError(f"Unexpected camera model: {info.get_name()}")
            fmt = getattr(ob.OBFormat, config.get("format", "MJPG"), None)
            if fmt is None:
                raise ValueError("Unknown Orbbec RGB format; use MJPG, RGB, BGR or YUYV")
            pipeline = ob.Pipeline(device)
            profiles = pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
            # Deliberately fail if this profile is unavailable: never silently change calibration resolution.
            try:
                profile = profiles.get_video_stream_profile(width, height, fmt, fps)
            except ob.OBError as exc:
                available = []
                profile_count = profiles.get_count()
                for i in range(min(profile_count, 12)):
                    candidate = profiles.get_stream_profile_by_index(i).as_video_stream_profile()
                    available.append(
                        f"{candidate.get_width()}x{candidate.get_height()} "
                        f"{candidate.get_format().name} {candidate.get_fps()} fps"
                    )
                if profile_count > len(available):
                    available.append(f"{profile_count - len(available)} additional profiles not listed")
                raise RuntimeError(
                    "Cannot select the requested RGB profile: "
                    f"{width}x{height} {config.get('format', 'MJPG')} {fps} fps. "
                    "Available RGB profiles: "
                    + ("; ".join(available) if available else "none")
                    + ". Set a supported width, height, format and fps in config.json; "
                    "the profile is never changed automatically. Keep it consistent with calibration; "
                    "recapture and recalibrate after changing the profile. "
                    f"SDK error: {exc}"
                ) from exc
            stream_config = ob.Config()
            stream_config.enable_stream(profile)  # COLOR ONLY
            pipeline.start(stream_config)
            self.handle = pipeline
            self.identity = {"backend": "orbbec", "serial": info.get_serial_number(),
                             "name": info.get_name(), "format": config.get("format", "MJPG")}
        else:
            raise ValueError("camera.backend must be orbbec or opencv")
        self.identity.update(width=width, height=height, fps=fps)

    def read(self):
        """Return BGR image and host receipt time; samples must be taken at rest."""
        if self.backend == "opencv":
            ok, frame = self.handle.read()
            if not ok:
                raise RuntimeError("RGB frame unavailable")
        else:
            frames = self.handle.wait_for_frames(2000)
            color = frames.get_color_frame() if frames is not None else None
            if color is None:
                raise RuntimeError("RGB frame unavailable")
            self.last_frame_metadata = {
                "sdk_system_timestamp_us": int(color.get_system_timestamp_us()),
                "sdk_device_timestamp_us": int(color.get_timestamp_us()),
                "sdk_clock_type": self.clock_type,
            }
            fmt = color.get_format()
            data = np.asanyarray(color.get_data(), dtype=np.uint8)
            w, h = color.get_width(), color.get_height()
            ob = self.ob
            if fmt == ob.OBFormat.MJPG:
                frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
            elif fmt == ob.OBFormat.RGB:
                frame = cv2.cvtColor(data.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
            elif fmt == ob.OBFormat.BGR:
                frame = data.reshape(h, w, 3).copy()
            elif fmt == ob.OBFormat.YUYV:
                frame = cv2.cvtColor(data.reshape(h, w, 2), cv2.COLOR_YUV2BGR_YUY2)
            else:
                raise RuntimeError(f"Unsupported color format {fmt}")
        if frame is None or frame.shape[:2] != (self.config["height"], self.config["width"]):
            raise RuntimeError("Camera frame resolution changed; recalibration required")
        return frame, time.time()

    def close(self):
        if self.handle is not None:
            if self.backend == "opencv":
                self.handle.release()
            else:
                self.handle.stop()
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
