"""Non-blocking, bounded, time-based notifications (toasts)."""

from __future__ import annotations

import threading
from dataclasses import dataclass

from .. import config

TOAST_KINDS = ("info", "success", "warn", "error")


@dataclass(slots=True)
class Toast:
    text: str
    kind: str
    created_at: float
    expires_at: float
    key: str | None = None


class Notifier:
    """Keeps at most ``max_items`` live toasts.

    A toast pushed with a ``key`` replaces an existing toast with that key, so
    repeated feedback (for example "Nothing to undo") refreshes instead of
    stacking up.
    """

    def __init__(self, max_items: int = config.MAX_TOASTS) -> None:
        self.max_items = max(1, max_items)
        self._items: list[Toast] = []
        self._lock = threading.Lock()

    def push(
        self,
        text: str,
        kind: str = "info",
        now: float = 0.0,
        duration: float | None = None,
        key: str | None = None,
    ) -> Toast:
        if kind not in TOAST_KINDS:
            kind = "info"
        if duration is None:
            duration = (
                config.TOAST_ERROR_SECONDS
                if kind in {"warn", "error"}
                else config.TOAST_SECONDS
            )
        toast = Toast(text, kind, now, now + duration, key or text)
        with self._lock:
            self._items = [item for item in self._items if item.key != toast.key]
            self._items.append(toast)
            if len(self._items) > self.max_items:
                self._items = self._items[-self.max_items :]
        return toast

    def active(self, now: float) -> list[Toast]:
        with self._lock:
            self._items = [item for item in self._items if item.expires_at > now]
            return list(self._items)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
