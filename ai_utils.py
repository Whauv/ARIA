from __future__ import annotations

import base64
import os
from typing import Optional

import cv2
import numpy as np

try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover - optional dependency
    genai = None


def frame_to_base64_png(frame: np.ndarray) -> Optional[str]:
    success, encoded = cv2.imencode(".png", frame)
    if not success:
        return None

    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def describe_image_with_gemini(frame: np.ndarray, prompt: str) -> str:
    if genai is None:
        return "Gemini vision is unavailable because google-generativeai is not installed."

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Gemini vision is unavailable because the API key is missing."

    encoded_image = frame_to_base64_png(frame)
    if encoded_image is None:
        return "I could not capture the current frame."

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
    return text.strip() or "I could not identify the drawing."
