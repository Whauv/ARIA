from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from dotenv import load_dotenv

import config
from ai_utils import describe_image_with_gemini

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


@dataclass
class JarvisContext:
    get_current_frame: Callable[[], Optional[np.ndarray]]
    clear_canvas: Callable[[], None]
    delete_selected_sprite: Callable[[], bool]
    scale_selected_sprite: Callable[[float], bool]
    undo_last_stroke: Callable[[], bool]
    save_snapshot: Callable[[str], bool]
    set_brush_color: Callable[[str], bool]


def command_handler(command: str, context: JarvisContext) -> str:
    normalized = command.lower().strip()

    if "what did i draw" in normalized or "what is this" in normalized:
        frame = context.get_current_frame()
        if frame is None:
            return "I do not have a frame to inspect yet."
        return describe_image_with_gemini(
            frame,
            "Describe what is drawn or placed in this image in one sentence.",
        )

    if "change color to" in normalized:
        color_name = normalized.split("change color to", 1)[1].strip()
        for supported_color in config.BRUSH_COLORS:
            if supported_color in color_name:
                if context.set_brush_color(supported_color):
                    return f"Changed the brush color to {supported_color}."
        supported = ", ".join(sorted(config.BRUSH_COLORS))
        return f"I do not know that color yet. Try one of: {supported}."

    if normalized in {"clear", "clear canvas"}:
        context.clear_canvas()
        return "Canvas cleared."

    if normalized in {"delete", "remove it"}:
        if context.delete_selected_sprite():
            return "Removed the selected sprite."
        return "There is no selected sprite to remove."

    if normalized in {"make it bigger", "enlarge"}:
        if context.scale_selected_sprite(1.5):
            return "Made the selected sprite bigger."
        return "Select a sprite first."

    if normalized in {"make it smaller", "shrink"}:
        if context.scale_selected_sprite(0.5):
            return "Made the selected sprite smaller."
        return "Select a sprite first."

    if normalized == "undo":
        if context.undo_last_stroke():
            return "Undid the last stroke."
        return "There is nothing to undo."

    if normalized == "save":
        if context.save_snapshot("aria_snapshot.png"):
            return "Saved!"
        return "I could not save the snapshot."

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
            openwakeword.utils.download_models()
            if wakeword_model_path:
                wake_model = Model(wakeword_models=[wakeword_model_path])
                wakeword_key = os.path.splitext(os.path.basename(wakeword_model_path))[0]
            else:
                wake_model = Model()
                wakeword_key = built_in_wakeword

            recognizer = sr.Recognizer()
            try:
                with sr.Microphone(sample_rate=16000) as source:
                    recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    chunk_size = int(16000 * 0.08) * 2

                    while not self.stop_event.is_set():
                        audio_chunk = source.stream.read(chunk_size, exception_on_overflow=False)
                        pcm = np.frombuffer(audio_chunk, dtype=np.int16)
                        prediction = wake_model.predict(pcm)
                        if prediction.get(wakeword_key, 0.0) >= 0.5:
                            self._handle_wake_word()
                            time.sleep(0.5)
            except Exception as exc:
                self.voice_available = False
                self.last_error = f"Voice disabled: {exc}"
                return
        except Exception as exc:
            self.voice_available = False
            self.last_error = f"Voice disabled: {exc}"
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

        recognizer = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.3)
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=6)
            return recognizer.recognize_google(audio)
        except Exception:
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
        except Exception:
            return
