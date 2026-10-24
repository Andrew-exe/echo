"""Persistent webcam wrapper with Windows-friendly backend fallback."""

from __future__ import annotations

import platform
import time
from typing import Any

import cv2
import numpy as np

from config import AppConfig


class CameraOpenError(RuntimeError):
    pass


class Camera:
    def __init__(self, config: AppConfig):
        self.config = config
        self.cap: cv2.VideoCapture | None = None
        self.consecutive_failures = 0
        self.last_frame: np.ndarray | None = None

    def open(self) -> None:
        if self.cap is not None:
            return
        platform_name = platform.system()
        attempts: list[tuple[str, Any]] = []
        if self.config.camera_backend == "dshow":
            attempts = [("DirectShow", getattr(cv2, "CAP_DSHOW", 700))]
        elif self.config.camera_backend == "msmf":
            attempts = [("Media Foundation", getattr(cv2, "CAP_MSMF", 1400))]
        elif platform_name == "Windows":
            # DirectShow tends to open faster in OpenCV on Windows. Fall back to
            # the platform default if it fails.
            attempts = [
                ("DirectShow", getattr(cv2, "CAP_DSHOW", 700)),
                ("default", cv2.CAP_ANY),
            ]
        else:
            attempts = [("default", cv2.CAP_ANY)]

        errors: list[str] = []
        for label, backend in attempts:
            cap = cv2.VideoCapture(int(self.config.camera_index), backend)
            if not cap.isOpened():
                cap.release()
                errors.append(f"{label}: could not open")
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.config.frame_width))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.config.frame_height))
            cap.set(cv2.CAP_PROP_FPS, 30)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            # Some cameras return black frames for a few hundred milliseconds.
            deadline = time.monotonic() + 2.0
            got_frame = False
            while time.monotonic() < deadline:
                ok, frame = cap.read()
                if ok and frame is not None and frame.size:
                    got_frame = True
                    self.last_frame = frame
                    break
                time.sleep(0.05)
            if got_frame:
                self.cap = cap
                self.consecutive_failures = 0
                actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(
                    f"[camera] Opened index {self.config.camera_index} via {label} "
                    f"({actual_w}x{actual_h}).",
                    flush=True,
                )
                return
            cap.release()
            errors.append(f"{label}: opened but returned no frames")

        raise CameraOpenError(
            f"Could not read from camera index {self.config.camera_index} "
            f"({' ; '.join(errors)}).\n"
            "Check Windows Settings > Privacy & security > Camera, close Teams/Zoom, "
            "try `--camera 1`, or test with `python main.py --camera-test`."
        )

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.cap is None:
            raise CameraOpenError("Camera is not open")
        ok, frame = self.cap.read()
        if ok and frame is not None and frame.size:
            self.consecutive_failures = 0
            self.last_frame = frame
            return True, frame
        self.consecutive_failures += 1
        return False, None

    def reopen(self) -> None:
        self.close()
        self.open()

    def close(self) -> None:
        if self.cap is not None:
            try:
                self.cap.release()
            finally:
                self.cap = None
