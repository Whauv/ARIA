"""Shared command handling for every input method.

Gestures (toolbar/palette dwell), keyboard, mouse, and voice all produce a
``Command``. The render thread executes commands through ``CommandDispatcher``
so equivalent actions behave identically and UI state is only mutated on its
owning thread. Background threads (voice) submit through ``CommandQueue`` and
wait for the real result, so feedback such as "Saved" is only reported after
the action actually succeeded.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable, Mapping
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from enum import Enum

from .. import config

logger = logging.getLogger(__name__)


class Action(str, Enum):
    MODE_DRAW = "mode.draw"
    MODE_SELECT = "mode.select"
    BRUSH = "brush"
    UNDO = "edit.undo"
    REDO = "edit.redo"
    CLEAR = "canvas.clear"
    DELETE = "sprite.delete"
    SCALE = "sprite.scale"
    DESELECT = "sprite.deselect"
    EXPORT_PNG = "file.export_png"
    SAVE_PROJECT = "file.save_project"
    LOAD_PROJECT = "file.load_project"
    TOGGLE_HELP = "view.help"
    TOGGLE_STATUS = "view.status"
    CANCEL = "view.cancel"


@dataclass(frozen=True, slots=True)
class Command:
    action: Action
    arg: object = None
    source: str = "keyboard"


@dataclass(frozen=True, slots=True)
class CommandResult:
    ok: bool
    message: str = ""
    kind: str = "info"

    @classmethod
    def success(cls, message: str = "") -> CommandResult:
        return cls(True, message, "success")

    @classmethod
    def info(cls, message: str = "") -> CommandResult:
        return cls(True, message, "info")

    @classmethod
    def unavailable(cls, message: str) -> CommandResult:
        return cls(False, message, "warn")

    @classmethod
    def failure(cls, message: str) -> CommandResult:
        return cls(False, message, "error")


Handler = Callable[[Command], CommandResult]


class CommandDispatcher:
    def __init__(
        self,
        handlers: Mapping[Action, Handler],
        on_result: Callable[[Command, CommandResult], None] | None = None,
    ) -> None:
        self._handlers = dict(handlers)
        self._on_result = on_result

    def dispatch(self, command: Command) -> CommandResult:
        handler = self._handlers.get(command.action)
        if handler is None:
            result = CommandResult.unavailable("That action is not available.")
        else:
            try:
                result = handler(command)
            except Exception as exc:  # keep the render loop alive
                logger.exception("Command %s failed", command.action.value)
                result = CommandResult.failure(
                    f"{command.action.value} failed: {type(exc).__name__}"
                )
        if self._on_result is not None:
            self._on_result(command, result)
        return result


class CommandQueue:
    """Bounded hand-off from background threads to the render thread."""

    def __init__(self, maxsize: int = config.VOICE_COMMAND_QUEUE_SIZE) -> None:
        self._queue: queue.Queue[tuple[Command, Future[CommandResult]]] = queue.Queue(
            maxsize=maxsize
        )
        self._closed = threading.Event()

    def submit(self, command: Command) -> Future[CommandResult]:
        future: Future[CommandResult] = Future()
        if self._closed.is_set():
            future.set_result(CommandResult.unavailable("ARIA is shutting down."))
            return future
        try:
            self._queue.put_nowait((command, future))
        except queue.Full:
            future.set_result(
                CommandResult.unavailable("ARIA is busy. Try that again.")
            )
        return future

    def drain(self, dispatcher: CommandDispatcher, max_items: int = 4) -> int:
        handled = 0
        while handled < max_items:
            try:
                command, future = self._queue.get_nowait()
            except queue.Empty:
                break
            handled += 1
            if not future.set_running_or_notify_cancel():
                continue
            future.set_result(dispatcher.dispatch(command))
        return handled

    def close(self) -> None:
        """Reject new work and resolve anything still waiting."""
        self._closed.set()
        while True:
            try:
                _command, future = self._queue.get_nowait()
            except queue.Empty:
                break
            if future.set_running_or_notify_cancel():
                future.set_result(CommandResult.unavailable("ARIA is shutting down."))


def wait_for_result(future: Future[CommandResult], timeout: float) -> CommandResult:
    try:
        return future.result(timeout=timeout)
    except FutureTimeoutError:
        future.cancel()
        return CommandResult.unavailable("ARIA did not respond in time.")


HELP_CLOSE_ITEM_ID = "overlay:close_help"

# Every toolbar button maps to the same Action the keyboard and voice use, so
# pointing at a control and pressing its key are the same command.
UI_ITEM_COMMANDS: dict[str, Action] = {
    "toolbar:draw_mode": Action.MODE_DRAW,
    "toolbar:select_mode": Action.MODE_SELECT,
    "toolbar:undo": Action.UNDO,
    "toolbar:redo": Action.REDO,
    "toolbar:clear": Action.CLEAR,
    "toolbar:delete": Action.DELETE,
    "toolbar:export": Action.EXPORT_PNG,
    "toolbar:save_project": Action.SAVE_PROJECT,
    "toolbar:load_project": Action.LOAD_PROJECT,
    "toolbar:help": Action.TOGGLE_HELP,
    "toolbar:status": Action.TOGGLE_STATUS,
    # Help suspends every other gesture, so its Close button is the one
    # dwell target that stays live while the overlay is open.
    HELP_CLOSE_ITEM_ID: Action.TOGGLE_HELP,
}


def command_for_ui_item(item_id: str, source: str) -> Command | None:
    if item_id.startswith("palette:"):
        return Command(Action.BRUSH, item_id.split(":", 1)[1], source)
    action = UI_ITEM_COMMANDS.get(item_id)
    return Command(action, None, source) if action is not None else None


KEY_ESCAPE = 27
KEY_BACKSPACE = 8

_KEY_ACTIONS: dict[str, tuple[Action, object]] = {
    "d": (Action.MODE_DRAW, None),
    "v": (Action.MODE_SELECT, None),
    "u": (Action.UNDO, None),
    "r": (Action.REDO, None),
    "c": (Action.CLEAR, None),
    "x": (Action.DELETE, None),
    "+": (Action.SCALE, config.SPRITE_SCALE_STEP_UP),
    "=": (Action.SCALE, config.SPRITE_SCALE_STEP_UP),
    "-": (Action.SCALE, config.SPRITE_SCALE_STEP_DOWN),
    "_": (Action.SCALE, config.SPRITE_SCALE_STEP_DOWN),
    "e": (Action.EXPORT_PNG, None),
    "p": (Action.SAVE_PROJECT, None),
    "l": (Action.LOAD_PROJECT, None),
    "h": (Action.TOGGLE_HELP, None),
    "s": (Action.TOGGLE_STATUS, None),
}

# Shown in the in-app help and kept in sync with the README/website.
KEYBOARD_REFERENCE: tuple[tuple[str, str], ...] = (
    ("D / V", "Draw mode / Select mode"),
    ("1-8", "Brush colors"),
    ("U / R", "Undo / redo"),
    ("C", "Clear drawing (undoable)"),
    ("X / Bksp", "Delete sprite (undoable)"),
    ("+ / -", "Resize selected sprite"),
    ("E", "Export PNG snapshot"),
    ("P / L", "Save / load project"),
    ("H / S", "Help / status panel"),
    ("Esc", "Close help, cancel, deselect"),
    ("Q", "Quit"),
)


def command_for_key(key: int) -> Command | None:
    """Map an OpenCV ``waitKey`` code (already masked to 0-255) to a command."""
    if key in (-1, 255):
        return None
    if key == KEY_ESCAPE:
        return Command(Action.CANCEL)
    if key == KEY_BACKSPACE:
        return Command(Action.DELETE)
    try:
        char = chr(key).lower()
    except ValueError:
        return None
    if char.isdigit() and char != "0":
        index = int(char) - 1
        if index < len(config.PALETTE_ORDER):
            return Command(Action.BRUSH, config.PALETTE_ORDER[index])
        return None
    mapped = _KEY_ACTIONS.get(char)
    if mapped is None:
        return None
    action, arg = mapped
    return Command(action, arg)


def is_quit_key(key: int) -> bool:
    return key in (ord("q"), ord("Q"))
