from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .. import config
from ..ai.ai_utils import describe_image_with_gemini
from .command_intents import CommandIntent, parse_command

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    def load_dotenv() -> bool:
        return False

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
    get_current_frame: Callable[[], Optional[np.ndarray]]
    clear_canvas: Callable[[], None]
    delete_selected_sprite: Callable[[], bool]
    scale_selected_sprite: Callable[[float], bool]
    undo_last_stroke: Callable[[], bool]
    save_snapshot: Callable[[str], bool]
    set_brush_color: Callable[[str], bool]
    set_interaction_mode: Callable[[str], bool]


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
        if parsed_command.color_name and context.set_brush_color(parsed_command.color_name):
            return f"Changed the brush color to {parsed_command.color_name}."
        supported = ", ".join(sorted(config.BRUSH_COLORS))
        return f"I do not know that color yet. Try one of: {supported}."

    if parsed_command.intent == CommandIntent.CLEAR_CANVAS:
        context.clear_canvas()
        return "Canvas cleared."

    if parsed_command.intent == CommandIntent.DELETE_SELECTION:
        if context.delete_selected_sprite():
            return "Removed the selected sprite."
        return "There is no selected sprite to remove."

    if parsed_command.intent == CommandIntent.SCALE_SELECTION and parsed_command.scale_factor and parsed_command.scale_factor > 1.0:
        if context.scale_selected_sprite(parsed_command.scale_factor):
            return "Made the selected sprite bigger."
        return "Select a sprite first."

    if parsed_command.intent == CommandIntent.SCALE_SELECTION and parsed_command.scale_factor and parsed_command.scale_factor < 1.0:
        if context.scale_selected_sprite(parsed_command.scale_factor):
            return "Made the selected sprite smaller."
        return "Select a sprite first."

    if parsed_command.intent == CommandIntent.UNDO:
        if context.undo_last_stroke():
            return "Undid the last stroke."
        return "There is nothing to undo."

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
    def __init__(self, context: JarvisContext) -> None:
        load_dotenv()
        self.context = context
        self.listening_event = threading.Event()
        self.speaking_event = threading.Event()
        self.stop_event = threading.Event()
        self.voice_available = True
        self.last_error: str | None = None
        self.thread = threading.Thread(target=self._run_loop, daemon=True, name="aria-jarvis")

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)

    def _run_loop(self) -> None:
        if openwakeword is None or Model is None or sr is None:
            self.voice_available = False
            self.last_error = "Voice disabled: missing openWakeWord or SpeechRecognition dependency."
            return

        wakeword_model_path = os.getenv("OPENWAKEWORD_MODEL_PATH")
        built_in_wakeword = os.getenv("OPENWAKEWORD_BUILTIN", "hey_jarvis")

        wake_model = None
        try:
            if wakeword_model_path:
                wake_model = Model(wakeword_models=[wakeword_model_path])
                wakeword_key = os.path.splitext(os.path.basename(wakeword_model_path))[0]
            else:
                try:
                    openwakeword.utils.download_models()
                except Exception as exc:  # pragma: no cover - environment dependent
                    logger.warning("openWakeWord model download failed, using local cache if present: %s", exc)
                wake_model = Model()
                wakeword_key = built_in_wakeword

            recognizer = sr.Recognizer()
            try:
                with sr.Microphone(sample_rate=config.VOICE_SAMPLE_RATE) as source:
                    recognizer.adjust_for_ambient_noise(source, duration=config.VOICE_WAKEWORD_CALIBRATION_SECONDS)
                    chunk_size = int(config.VOICE_SAMPLE_RATE * config.VOICE_CHUNK_SECONDS) * 2

                    while not self.stop_event.is_set():
                        audio_chunk = source.stream.read(chunk_size, exception_on_overflow=False)
                        pcm = np.frombuffer(audio_chunk, dtype=np.int16)
                        prediction = wake_model.predict(pcm)
                        if prediction.get(wakeword_key, 0.0) >= config.OPENWAKEWORD_THRESHOLD:
                            self._handle_wake_word()
                            time.sleep(config.VOICE_WAKEWORD_COOLDOWN_SECONDS)
            except Exception as exc:
                self.voice_available = False
                self.last_error = f"Voice disabled: {exc}"
                logger.warning("Voice input loop disabled: %s", exc)
                return
        except Exception as exc:
            self.voice_available = False
            self.last_error = f"Voice disabled: {exc}"
            logger.warning("Voice initialization disabled: %s", exc)
        finally:
            if wake_model is not None:
                del wake_model

    def _handle_wake_word(self) -> None:
        self.listening_event.set()
        try:
            command = self._capture_command()
            if not command:
                self._speak_async("I did not catch that.")
                return

            response = command_handler(command, self.context)
            if response:
                self._speak_async(response)
        finally:
            self.listening_event.clear()

    def _capture_command(self) -> Optional[str]:
        if sr is None:
            return None
        if not config.ALLOW_CLOUD_SPEECH_RECOGNITION:
            self.last_error = "Voice command transcription disabled by configuration."
            logger.info("Cloud speech recognition disabled by configuration.")
            return None

        recognizer = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=config.VOICE_AMBIENT_CALIBRATION_SECONDS)
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
            self.last_error = f"Speech recognition request failed: {exc}"
            logger.warning("Speech recognition request failed: %s", exc)
            return None
        except Exception as exc:
            self.last_error = f"Voice capture failed: {exc}"
            logger.warning("Voice capture failed: %s", exc)
            return None

    def _speak_async(self, text: str) -> None:
        threading.Thread(target=self._speak, args=(text,), daemon=True, name="aria-tts").start()

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
        except Exception as exc:
            logger.warning("TTS synthesis failed: %s", exc)
            return
