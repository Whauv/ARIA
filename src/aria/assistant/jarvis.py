from __future__ import annotations

import logging
import os
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .. import config
from ..ai.ai_utils import describe_image_with_gemini
from .command_intents import CommandIntent, parse_command

try:
    import openwakeword
    from openwakeword.model import Model
except ImportError:  # pragma: no cover - optional dependency
    openwakeword = None
    Model = None

try:
    import speech_recognition as sr
except ImportError:  # pragma: no cover - optional dependency
    sr = None

try:
    import pyttsx3
except ImportError:  # pragma: no cover - optional dependency
    pyttsx3 = None


logger = logging.getLogger(__name__)


@dataclass
class JarvisContext:
    """Callbacks the assistant uses to act on the workspace.

    In the app these submit commands to the render thread and block until the
    real result is known, so replies such as "Saved!" are only spoken after
    the action succeeded.
    """

    get_current_frame: Callable[[], np.ndarray | None]
    clear_canvas: Callable[[], None]
    delete_selected_sprite: Callable[[], bool]
    scale_selected_sprite: Callable[[float], bool]
    undo_last_stroke: Callable[[], bool]
    save_snapshot: Callable[[str], bool]
    set_brush_color: Callable[[str], bool]
    set_interaction_mode: Callable[[str], bool]
    redo_last_scene: Callable[[], bool] | None = None


def command_handler(command: str, context: JarvisContext) -> str:
    parsed_command = parse_command(command)

    if parsed_command.intent == CommandIntent.DESCRIBE_SCENE:
        frame = context.get_current_frame()
        if frame is None:
            return "I do not have a frame to inspect yet."
        return describe_image_with_gemini(
            frame,
            "Describe what is drawn or placed in this image in one sentence.",
        )

    if parsed_command.intent == CommandIntent.CHANGE_BRUSH_COLOR:
        if parsed_command.color_name and context.set_brush_color(
            parsed_command.color_name
        ):
            return f"Changed the brush color to {parsed_command.color_name}."
        supported = ", ".join(sorted(config.BRUSH_COLORS))
        return f"I do not know that color yet. Try one of: {supported}."

    if parsed_command.intent == CommandIntent.CLEAR_CANVAS:
        result = context.clear_canvas()
        if result is False:
            return "I could not clear the canvas right now."
        return "Canvas cleared."

    if parsed_command.intent == CommandIntent.DELETE_SELECTION:
        if context.delete_selected_sprite():
            return "Removed the selected sprite."
        return "There is no selected sprite to remove."

    if (
        parsed_command.intent == CommandIntent.SCALE_SELECTION
        and parsed_command.scale_factor
        and parsed_command.scale_factor > 1.0
    ):
        if context.scale_selected_sprite(parsed_command.scale_factor):
            return "Made the selected sprite bigger."
        return "Select a sprite first."

    if (
        parsed_command.intent == CommandIntent.SCALE_SELECTION
        and parsed_command.scale_factor
        and parsed_command.scale_factor < 1.0
    ):
        if context.scale_selected_sprite(parsed_command.scale_factor):
            return "Made the selected sprite smaller."
        return "Select a sprite first."

    if parsed_command.intent == CommandIntent.UNDO:
        if context.undo_last_stroke():
            return "Undone."
        return "There is nothing to undo."

    if parsed_command.intent == CommandIntent.REDO:
        if context.redo_last_scene is not None and context.redo_last_scene():
            return "Redone."
        return "There is nothing to redo."

    if parsed_command.intent == CommandIntent.SAVE:
        if context.save_snapshot(config.DEFAULT_SAVE_PATH):
            return "Saved!"
        return "I could not save the snapshot."

    if parsed_command.intent == CommandIntent.SWITCH_MODE and parsed_command.mode_name:
        if context.set_interaction_mode(parsed_command.mode_name):
            if parsed_command.mode_name == "select_mode":
                return "Switched to select mode."
            return "Switched to draw mode."
        return "I could not switch mode right now."

    return "I heard the command, but I do not know how to handle it yet."


class JarvisAssistant:
    """Wake word + command loop on a background thread, TTS on another.

    Thread model: ``aria-jarvis`` owns the microphone and handles one command
    at a time (no overlapping requests). ``aria-tts`` owns the pyttsx3 engine
    and reads from a small bounded queue; when replies arrive faster than they
    can be spoken, the oldest pending reply is dropped.
    """

    def __init__(self, context: JarvisContext) -> None:
        self.context = context
        self.listening_event = threading.Event()
        self.speaking_event = threading.Event()
        self.stop_event = threading.Event()
        self.voice_available = True
        self.wake_word_ready = False
        self.transcription_enabled = bool(config.ALLOW_CLOUD_SPEECH_RECOGNITION)
        self.wake_word_label = self._wake_word_label()
        self.last_error: str | None = None
        self.last_transcript: str | None = None
        self.last_response: str | None = None
        self.last_update_at: float | None = None
        self._tts_queue: queue.Queue[str] = queue.Queue(maxsize=config.TTS_QUEUE_SIZE)
        self.thread = threading.Thread(
            target=self._run_loop, daemon=True, name="aria-jarvis"
        )
        self._tts_thread = threading.Thread(
            target=self._tts_loop, daemon=True, name="aria-tts"
        )

    @staticmethod
    def _wake_word_label() -> str:
        model_path = os.getenv("OPENWAKEWORD_MODEL_PATH")
        if model_path:
            return os.path.splitext(os.path.basename(model_path))[0].replace("_", " ")
        return os.getenv("OPENWAKEWORD_BUILTIN", "hey_jarvis").replace("_", " ")

    def start(self) -> None:
        self.thread.start()
        self._tts_thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        for thread in (self.thread, self._tts_thread):
            if thread.is_alive():
                thread.join(timeout=2.0)

    def _set_unavailable(self, message: str) -> None:
        self.voice_available = False
        self.wake_word_ready = False
        self.last_error = message

    def _run_loop(self) -> None:
        if openwakeword is None or Model is None or sr is None:
            self._set_unavailable(
                "Voice disabled: missing openWakeWord or SpeechRecognition dependency."
            )
            return

        wakeword_model_path = os.getenv("OPENWAKEWORD_MODEL_PATH")
        built_in_wakeword = os.getenv("OPENWAKEWORD_BUILTIN", "hey_jarvis")

        wake_model = None
        try:
            if wakeword_model_path:
                wake_model = Model(wakeword_models=[wakeword_model_path])
                wakeword_key = os.path.splitext(os.path.basename(wakeword_model_path))[
                    0
                ]
            else:
                try:
                    openwakeword.utils.download_models()
                except Exception as exc:  # noqa: BLE001 - optional model download
                    logger.warning(
                        "openWakeWord model download failed, using local cache if present: %s",
                        exc,
                    )
                wake_model = Model()
                wakeword_key = built_in_wakeword

            recognizer = sr.Recognizer()
            try:
                with sr.Microphone(sample_rate=config.VOICE_SAMPLE_RATE) as source:
                    recognizer.adjust_for_ambient_noise(
                        source, duration=config.VOICE_WAKEWORD_CALIBRATION_SECONDS
                    )
                    chunk_size = (
                        int(config.VOICE_SAMPLE_RATE * config.VOICE_CHUNK_SECONDS) * 2
                    )
                    self.wake_word_ready = True

                    while not self.stop_event.is_set():
                        audio_chunk = source.stream.read(
                            chunk_size, exception_on_overflow=False
                        )
                        pcm = np.frombuffer(audio_chunk, dtype=np.int16)
                        prediction = wake_model.predict(pcm)
                        if (
                            prediction.get(wakeword_key, 0.0)
                            >= config.OPENWAKEWORD_THRESHOLD
                        ):
                            self._handle_wake_word()
                            self.stop_event.wait(config.VOICE_WAKEWORD_COOLDOWN_SECONDS)
            except Exception as exc:  # noqa: BLE001 - microphone/backend boundary
                self._set_unavailable(f"Voice disabled: {type(exc).__name__}")
                logger.warning("Voice input loop disabled: %s", exc)
                return
        except Exception as exc:  # noqa: BLE001 - voice initialization boundary
            self._set_unavailable(f"Voice disabled: {type(exc).__name__}")
            logger.warning("Voice initialization disabled: %s", exc)
        finally:
            self.wake_word_ready = False
            if wake_model is not None:
                del wake_model

    def _record_exchange(self, transcript: str | None, response: str | None) -> None:
        self.last_transcript = transcript
        self.last_response = response
        self.last_update_at = time.monotonic()

    def _handle_wake_word(self) -> None:
        if not self.transcription_enabled:
            self._record_exchange(
                None,
                "Voice commands are off: cloud transcription is disabled "
                "(ALLOW_CLOUD_SPEECH_RECOGNITION=0).",
            )
            return

        self.listening_event.set()
        try:
            command = self._capture_command()
        finally:
            self.listening_event.clear()

        if not command:
            self._record_exchange("No speech detected", "I did not catch that.")
            self._speak_async("I did not catch that.")
            return

        self._record_exchange(command, "Working...")
        response = command_handler(command, self.context)
        if response:
            self._record_exchange(command, response)
            self._speak_async(response)

    def _capture_command(self) -> str | None:
        if sr is None:
            return None
        if not config.ALLOW_CLOUD_SPEECH_RECOGNITION:
            self.last_error = "Voice command transcription disabled by configuration."
            logger.info("Cloud speech recognition disabled by configuration.")
            return None

        recognizer = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(
                    source, duration=config.VOICE_AMBIENT_CALIBRATION_SECONDS
                )
                audio = recognizer.listen(
                    source,
                    timeout=config.VOICE_COMMAND_TIMEOUT_SECONDS,
                    phrase_time_limit=config.VOICE_COMMAND_PHRASE_LIMIT_SECONDS,
                )
            return recognizer.recognize_google(audio)
        except sr.WaitTimeoutError:
            logger.info("Voice command timed out waiting for speech.")
            return None
        except sr.UnknownValueError:
            logger.info("Voice command could not be understood.")
            return None
        except sr.RequestError as exc:
            self.last_error = "Speech recognition request failed (network?)."
            logger.warning("Speech recognition request failed: %s", exc)
            return None
        except Exception as exc:  # noqa: BLE001 - speech backend boundary
            self.last_error = f"Voice capture failed: {type(exc).__name__}"
            logger.warning("Voice capture failed: %s", exc)
            return None

    def _speak_async(self, text: str) -> None:
        """Queue a reply for the TTS worker without ever blocking."""
        try:
            self._tts_queue.put_nowait(text)
        except queue.Full:
            try:
                self._tts_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._tts_queue.put_nowait(text)
            except queue.Full:
                logger.info("Dropping TTS reply; queue is full.")

    def _tts_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                text = self._tts_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            self._speak(text)

    def _speak(self, text: str) -> None:
        self.speaking_event.set()
        try:
            self._synthesize_tts(text)
        finally:
            self.speaking_event.clear()

    def _synthesize_tts(self, text: str) -> None:
        if pyttsx3 is None:
            return
        try:
            engine = pyttsx3.init()
            voices = engine.getProperty("voices")
            if voices:
                engine.setProperty("voice", voices[0].id)
            engine.setProperty("rate", 175)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as exc:  # noqa: BLE001 - TTS backend boundary
            logger.warning("TTS synthesis failed: %s", exc)
            return
