from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

import cv2
import mediapipe as mp

from ..config import TARGET_FPS


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


def create_default_capture() -> CaptureDevice | None:
    required_attrs = ("VideoCapture", "CAP_PROP_FPS")
    if not all(hasattr(cv2, attr) for attr in required_attrs):
        logger.warning("OpenCV capture bindings are unavailable in this environment.")
        return None

    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    if not capture.isOpened():
        capture.release()
        return None
    return capture


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
