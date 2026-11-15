from __future__ import annotations

import logging
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import cv2
import mediapipe as mp

from ..config import (
    CAPTURE_HEIGHT,
    CAPTURE_PREFER_MJPG,
    CAPTURE_WIDTH,
    TARGET_FPS,
)

logger = logging.getLogger(__name__)


class CaptureDevice(Protocol):
    def isOpened(self) -> bool:
        """Return whether the device is ready to read frames."""

    def read(self) -> tuple[bool, Any]:
        """Read a frame from the capture device."""

    def release(self) -> None:
        """Release the device handle."""


class HandTracker(Protocol):
    def process(self, frame: Any) -> Any:
        """Run hand tracking on a frame."""

    def close(self) -> None:
        """Release model resources."""


CaptureFactory = Callable[[], CaptureDevice | None]
HandTrackerFactory = Callable[[], HandTracker]


def _preferred_backends() -> list[int | None]:
    """Capture backends to try, best first.

    On Windows the default MSMF backend is slow to open and often refuses the
    requested resolution; DirectShow honours it and starts faster. ``None``
    means "let OpenCV choose", which is the only option elsewhere.
    """
    if sys.platform != "win32":
        return [None]
    return [getattr(cv2, "CAP_DSHOW", None), getattr(cv2, "CAP_MSMF", None), None]


def configure_capture(capture: CaptureDevice) -> tuple[int, int]:
    """Ask the device for the configured stream. Returns the size granted.

    Every ``set`` is advisory: drivers may ignore any of them, so the caller
    uses the values read back rather than the values requested.
    """
    setter = getattr(capture, "set", None)
    if setter is None:
        return (0, 0)
    if CAPTURE_PREFER_MJPG and hasattr(cv2, "VideoWriter_fourcc"):
        # Before the size: the pixel format constrains which sizes are offered.
        setter(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    if hasattr(cv2, "CAP_PROP_FRAME_WIDTH"):
        setter(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
        setter(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    setter(cv2.CAP_PROP_FPS, TARGET_FPS)
    if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
        # Keep the queue shallow so the pointer tracks the hand, not the past.
        setter(cv2.CAP_PROP_BUFFERSIZE, 1)

    getter = getattr(capture, "get", None)
    if getter is None or not hasattr(cv2, "CAP_PROP_FRAME_WIDTH"):
        return (0, 0)
    try:
        return int(getter(cv2.CAP_PROP_FRAME_WIDTH)), int(
            getter(cv2.CAP_PROP_FRAME_HEIGHT)
        )
    except (TypeError, ValueError):
        return (0, 0)


def create_default_capture() -> CaptureDevice | None:
    required_attrs = ("VideoCapture", "CAP_PROP_FPS")
    if not all(hasattr(cv2, attr) for attr in required_attrs):
        logger.warning("OpenCV capture bindings are unavailable in this environment.")
        return None

    for backend in _preferred_backends():
        try:
            capture = (
                cv2.VideoCapture(0) if backend is None else cv2.VideoCapture(0, backend)
            )
        except Exception as exc:  # noqa: BLE001 - backend availability varies
            logger.debug("Capture backend %s unavailable: %s", backend, exc)
            continue
        if not capture.isOpened():
            capture.release()
            continue
        width, height = configure_capture(capture)
        logger.info(
            "Camera opened (backend=%s) at %sx%s",
            "default" if backend is None else backend,
            width or "?",
            height or "?",
        )
        if width and width < CAPTURE_WIDTH:
            logger.info(
                "Camera granted %sx%s; requested %sx%s. Set ARIA_CAMERA_WIDTH/"
                "ARIA_CAMERA_HEIGHT to match a mode this device supports.",
                width,
                height,
                CAPTURE_WIDTH,
                CAPTURE_HEIGHT,
            )
        return capture
    return None


def create_default_hand_tracker() -> HandTracker:
    if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "hands"):
        raise RuntimeError("MediaPipe Hands is unavailable in this environment.")

    mp_hands = mp.solutions.hands
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )


@dataclass(slots=True)
class RuntimeDiagnostics:
    frames_processed: int = 0
    capture_failures: int = 0
    hand_tracking_failures: int = 0
    last_frame_latency_ms: float = 0.0
    last_status: str | None = None
    last_error: str | None = None
    last_error_at: float | None = None

    def record_frame(self, elapsed_seconds: float, status_text: str) -> None:
        self.frames_processed += 1
        self.last_frame_latency_ms = elapsed_seconds * 1000.0
        self.last_status = status_text

    def record_capture_failure(self, message: str) -> None:
        self.capture_failures += 1
        self.record_error(message)

    def record_hand_tracking_failure(self, message: str) -> None:
        self.hand_tracking_failures += 1
        self.record_error(message)

    def record_error(self, message: str) -> None:
        self.last_error = message
        self.last_error_at = time.time()
        logger.warning("Runtime diagnostic event: %s", message)
