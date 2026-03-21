# ARIA

Augmented Reality Intelligence Assistant is a Python computer-vision drawing workspace that combines:

- Air drawing with MediaPipe Hands
- Sprite creation, selection, dragging, and resizing
- Voice-triggered assistant controls
- Visual AI scene description through Gemini

## Demo GIF

`Demo GIF coming soon`

## Installation

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root.
4. Run ARIA:

```bash
python main.py
```

## .env Format

```env
# Optional Gemini vision
GOOGLE_API_KEY=your_google_api_key

# Optional custom openWakeWord model
OPENWAKEWORD_MODEL_PATH=C:\full\path\to\hey_aria.onnx

# Optional built-in wake word override
OPENWAKEWORD_BUILTIN=hey_jarvis
```

Notes:

- `GOOGLE_API_KEY` is optional. Without it, Gemini scene description is disabled and ARIA shows a warning overlay.
- `OPENWAKEWORD_MODEL_PATH` is optional. If omitted, ARIA uses the built-in `hey_jarvis` wake word.
- `OPENWAKEWORD_BUILTIN` is optional if you want a different built-in openWakeWord model name.
- TTS uses offline `pyttsx3`, so no TTS API key is required.

## Gesture Reference

| Gesture | Effect |
| --- | --- |
| Index finger only up | Draw in draw mode |
| Index + middle finger up | Lift pen / pause stroke |
| Closed fist for 1.5 seconds | Convert current drawing into a sprite |
| Single-hand pinch on sprite | Drag selected sprite in select mode |
| Two hands present | Resize selected sprite in select mode |
| Hover index over palette for 1 second | Change brush color |
| Hover index over toolbar for 1 second | Trigger toolbar action |
| Pinch over thumbnail | Bring that sprite to front |
| Double pinch on sprite within 0.5s | Delete sprite |
| Voice: "what did I draw" | Describe current scene with Gemini |
| Voice: "change color to red" | Change brush color |
| Voice: "clear canvas" | Clear canvas |
| Voice: "delete" | Remove selected sprite |
| Voice: "make it bigger" | Scale selected sprite up |
| Voice: "make it smaller" | Scale selected sprite down |
| Voice: "undo" | Undo last canvas snapshot |
| Voice: "save" | Save `aria_snapshot.png` |

## Product Features

- Top palette with dwell selection for colors and eraser
- Bottom toolbar with dwell actions for draw/select/clear/save/undo
- Brush preview near the fingertip
- Sprite thumbnail strip on the right edge
- FPS counter and active mode indicator
- Voice status overlays for idle, listening, and speaking
- Free-tier friendly assistant stack with offline TTS

## Architecture

```text
Webcam -> OpenCV Frame -> MediaPipe Hands -> Gesture Router
                                      |-> Draw Mode -> canvas.py
                                      |-> Select Mode -> sprite.py
                                      |-> Dwell UI -> ui.py
                                      |-> Voice State Overlay -> jarvis.py

jarvis.py -> openWakeWord + SpeechRecognition
jarvis.py -> ai_utils.py -> Gemini Vision
jarvis.py -> pyttsx3 offline speech

main.py -> render pipeline -> palette + toolbar + thumbnails + fps + warnings
```

## File Layout

```text
.
├── main.py
├── canvas.py
├── sprite.py
├── gestures.py
├── jarvis.py
├── ai_utils.py
├── ui.py
├── config.py
├── requirements.txt
└── README.md
```

## Notes

- The main OpenCV loop keeps AI and voice work off the render thread.
- MediaPipe processing uses a resized input capped at `640x480` for better performance.
- Gemini responses are cached for 5 seconds to avoid redundant calls.
- Canvas undo uses up to 10 stored snapshots.
- Wake-word detection uses `openWakeWord`, with `hey_jarvis` as the default built-in model.
