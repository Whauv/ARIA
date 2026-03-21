from __future__ import annotations

import base64
import os
import time
from typing import Optional

import cv2
import numpy as np

from config import GEMINI_CACHE_SECONDS

try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover - optional dependency
    genai = None


_CACHE = {
    "key": None,
    "response": None,
    "timestamp": 0.0,
}


def frame_to_base64_png(frame: np.ndarray) -> Optional[str]:
    success, encoded = cv2.imencode(".png", frame)
    if not success:
        return None

    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def gemini_enabled() -> bool:
    return genai is not None and bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))


def describe_image_with_gemini(frame: np.ndarray, prompt: str) -> str:
    if genai is None:
        return "Gemini vision is unavailable because google-generativeai is not installed."

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Gemini vision is unavailable because the API key is missing."

    encoded_image = frame_to_base64_png(frame)
    if encoded_image is None:
        return "I could not capture the current frame."

    cache_key = f"{prompt}:{hash(encoded_image)}"
    now = time.time()
    if _CACHE["key"] == cache_key and now - _CACHE["timestamp"] <= GEMINI_CACHE_SECONDS:
        return _CACHE["response"]  # type: ignore[return-value]

    genai.configure(api_key=api_key)
    image_bytes = base64.b64decode(encoded_image)
    model = genai.GenerativeModel("gemini-1.5-flash")

    try:
        response = model.generate_content(
            [
                prompt,
                {
                    "mime_type": "image/png",
                    "data": image_bytes,
                },
            ]
        )
    except Exception as exc:  # pragma: no cover - network/API behavior
        return f"I could not analyze the image right now: {exc}"

    text = getattr(response, "text", "") or ""
    final_text = text.strip() or "I could not identify the drawing."
    _CACHE["key"] = cache_key
    _CACHE["response"] = final_text
    _CACHE["timestamp"] = now
    return final_text
