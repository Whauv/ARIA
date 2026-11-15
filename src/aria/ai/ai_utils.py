from __future__ import annotations

import base64
import hashlib
import logging
import os
import time

import cv2
import numpy as np

from .. import config
from ..config import GEMINI_CACHE_SECONDS

try:
    from google import genai as google_genai
except ImportError:  # pragma: no cover - optional dependency
    google_genai = None

legacy_genai = None
if google_genai is None:  # only fall back to the deprecated SDK when needed
    try:
        import google.generativeai as legacy_genai
    except ImportError:  # pragma: no cover - optional dependency
        legacy_genai = None


logger = logging.getLogger(__name__)


_CACHE = {
    "key": None,
    "response": None,
    "timestamp": 0.0,
}


def frame_to_base64_png(frame: np.ndarray) -> str | None:
    success, encoded = cv2.imencode(".png", frame)
    if not success:
        return None

    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def gemini_enabled() -> bool:
    has_key = bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
    return has_key and (google_genai is not None or legacy_genai is not None)


def describe_image_with_gemini(frame: np.ndarray, prompt: str) -> str:
    if frame is None or frame.size == 0:
        return "I could not capture the current frame."

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Gemini vision is unavailable because the API key is missing."

    if google_genai is None and legacy_genai is None:
        return (
            "Gemini vision is unavailable because no supported Gemini SDK is installed."
        )

    encoded_image = frame_to_base64_png(frame)
    if encoded_image is None:
        return "I could not capture the current frame."

    cache_key = f"{prompt}:{hashlib.sha256(encoded_image.encode('utf-8')).hexdigest()}"
    now = time.time()
    if _CACHE["key"] == cache_key and now - _CACHE["timestamp"] <= GEMINI_CACHE_SECONDS:
        return _CACHE["response"]  # type: ignore[return-value]

    image_bytes = base64.b64decode(encoded_image)

    try:
        if google_genai is not None:
            client = google_genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=[
                    {
                        "role": "user",
                        "parts": [
                            {"text": prompt},
                            {
                                "inline_data": {
                                    "mime_type": "image/png",
                                    "data": encoded_image,
                                }
                            },
                        ],
                    }
                ],
            )
            text = getattr(response, "text", "") or ""
        else:
            legacy_genai.configure(api_key=api_key)
            model = legacy_genai.GenerativeModel(config.GEMINI_MODEL)
            response = model.generate_content(
                [
                    prompt,
                    {
                        "mime_type": "image/png",
                        "data": image_bytes,
                    },
                ]
            )
            text = getattr(response, "text", "") or ""
    except Exception as exc:
        # Log locally for debugging; never speak or display raw API errors.
        logger.warning("Gemini image description failed: %s", type(exc).__name__)
        logger.debug("Gemini failure detail", exc_info=True)
        return "I could not reach Gemini right now. Check your connection or API key."

    final_text = text.strip() or "I could not identify the drawing."
    _CACHE["key"] = cache_key
    _CACHE["response"] = final_text
    _CACHE["timestamp"] = now
    return final_text
