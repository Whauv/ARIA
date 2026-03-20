# ARIA

ARIA stands for Augmented Reality Intelligence Assistant.

This project is a Python prototype for real-time webcam-based air drawing with OpenCV and MediaPipe Hands.

The current assistant stack is designed to stay free-tier friendly:

- TTS uses offline `pyttsx3` with no API key
- Speech-to-text uses `speech_recognition`
- Gemini image description can use the Google free tier
- Porcupine wake word can use a free Picovoice developer key

## Current Features

- Live webcam capture with `cv2.VideoCapture(0)`
- Mirrored webcam feed for natural interaction
- Real-time hand landmark detection with MediaPipe Hands
- Index fingertip tracking with exponential smoothing
- Air drawing onto a separate canvas
- Gesture-based draw and pause controls
- Finish gesture that converts a drawing into a transparent sprite
- Multi-sprite rendering with alpha compositing and `z_index` ordering
- Voice assistant controls for canvas and sprite operations

## Files

```text
.
├── main.py
├── canvas.py
├── gestures.py
├── config.py
├── sprite.py
├── ai_utils.py
├── jarvis.py
└── README.md
```

## Gestures

- Index finger only up: draw
- Index and middle fingers up: pause drawing
- Closed fist held for 1.5 seconds: convert the current drawing into a sprite
- Press `q`: quit

## Install

```bash
pip install opencv-python mediapipe numpy speechrecognition pyttsx3 python-dotenv pvporcupine pvrecorder google-generativeai
```

## Run

```bash
python main.py
```

## Free-Tier Setup

- No key needed for offline TTS with `pyttsx3`
- Optional `GOOGLE_API_KEY` or `GEMINI_API_KEY` for Gemini image description
- Optional `PICOVOICE_ACCESS_KEY` and `PORCUPINE_KEYWORD_PATH` for the wake word
- Store keys in `.env` and never hardcode them
