# ARIA

Augmented Reality Intelligence Assistant is a Python computer-vision workspace for air drawing, sprite manipulation, and voice-assisted scene understanding.

## Core Capabilities

- Air drawing with MediaPipe Hands
- Sprite creation, selection, dragging, and resizing
- Voice-triggered assistant controls
- Visual AI scene description through Gemini
- Offline-friendly TTS with `pyttsx3`

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

5. Run the automated checks:

```bash
python -m unittest discover -s tests
python scripts/run_checks.py
```

## .env Format

```env
# Optional Gemini vision
GOOGLE_API_KEY=your_google_api_key

# Optional custom openWakeWord model
OPENWAKEWORD_MODEL_PATH=C:\full\path\to\hey_aria.onnx

# Optional built-in wake word override
OPENWAKEWORD_BUILTIN=hey_jarvis

# Optional privacy control for cloud transcription
ALLOW_CLOUD_SPEECH_RECOGNITION=1
```

Notes:

- `GOOGLE_API_KEY` is optional. Without it, Gemini scene description is disabled and ARIA shows a warning overlay.
- `OPENWAKEWORD_MODEL_PATH` is optional. If omitted, ARIA uses the built-in `hey_jarvis` wake word.
- `OPENWAKEWORD_BUILTIN` is optional if you want a different built-in `openWakeWord` model name.
- `ALLOW_CLOUD_SPEECH_RECOGNITION=0` disables cloud speech transcription for voice commands while leaving the local wake-word flow intact.
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

## Architecture

```text
main.py -> src/aria/__main__.py -> app_runner.py
app_runner.py -> runtime/state.py
app_runner.py -> runtime/controllers.py
app_runner.py -> runtime/services.py
app_runner.py -> drawing/canvas.py
app_runner.py -> drawing/sprite.py
app_runner.py -> ui/ui.py
app_runner.py -> assistant/jarvis.py
assistant/jarvis.py -> assistant/command_intents.py
assistant/jarvis.py -> ai/ai_utils.py
```

## File Layout

```text
.
|-- main.py
|-- pyproject.toml
|-- requirements.txt
|-- README.md
|-- AGENTS.md
|-- CONTRIBUTING.md
|-- LICENSE
|-- .env.example
|-- src/
|   |-- README.md
|   `-- aria/
|       |-- __main__.py
|       |-- app_runner.py
|       |-- config.py
|       |-- README.md
|       |-- ai/
|       |-- assistant/
|       |-- drawing/
|       |-- runtime/
|       |-- ui/
|       `-- vision/
|-- scripts/
|   |-- README.md
|   `-- run_checks.py
`-- tests/
    |-- README.md
    |-- unit/
    `-- integration/
```

## Notes

- The main OpenCV loop keeps AI and voice work off the render thread.
- MediaPipe processing uses a resized input capped at `640x480` for better performance.
- Gemini responses are cached for 5 seconds to avoid redundant calls.
- Canvas undo uses up to 10 stored snapshots.
- Wake-word detection uses `openWakeWord`, with `hey_jarvis` as the default built-in model.
- Voice command parsing is separated into intent parsing to keep the assistant command surface testable and easier to extend.
- ARIA supports both `google-genai` and the legacy `google-generativeai` package during the Gemini transition.
- Interaction state and workflow controllers are separated from the OpenCV loop to improve testability and maintainability.
- `scripts/run_checks.py` provides a single CI-friendly command for syntax and unit test verification.
