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
python scripts/run_checks.py   # also runs landing/ checks when its node_modules exist
```

## .env Format

```env
# Optional Gemini vision
GOOGLE_API_KEY=your_google_api_key
# Optional model override (default gemini-2.5-flash)
GEMINI_MODEL=gemini-2.5-flash

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

## Using ARIA

ARIA opens with a small **Quick start** card (draw, make a sprite, move it)
that checks itself off and disappears. The current mode is always shown by the
top-left chip and the workspace border: **amber = Draw**, **cyan = Select**.
Every action reports its real outcome in a short notification; destructive
actions (clear, delete) are undoable.

**Every action can be run by pointing at it.** The bottom toolbar carries a
button for each command, grouped as *modes | edit | file | view* — Draw,
Select, Undo, Redo, Clear, Delete, Export, Save, Load, Help, Status — plus the
colour swatches down the left. Hold your fingertip over a button for 0.8 s in
Draw mode (0.55 s in Select mode) and it fires once; a progress bar fills as
you hold. Buttons that cannot act right now (Redo with no redo, Delete with
nothing selected, Load with no saved project) are greyed out and ignore the
dwell. The help overlay has its own **Close** button that accepts a dwell, so
opening help by pointing never leaves you stranded. The keyboard and mouse do
exactly the same things and remain available.

### Gestures

| Mode | Gesture | Effect |
| --- | --- | --- |
| Draw | Index finger only up | Draw |
| Draw | Index + middle finger up | Lift pen |
| Draw | Closed fist held 1.5 s | Turn the drawing into a sprite (fires once per hold) |
| Draw | Pinch held 0.6 s, away from controls | Toggle quick color green / white |
| Select | Pinch a sprite | Select it (topmost wins when overlapping) |
| Select | Keep pinching ~0.2 s | Drag; release to place |
| Select | Pinch empty space | Deselect |
| Select | Pinch a layer thumbnail | Select it and bring it to the front |
| Select | Two quick pinches, then a third within 1.5 s | Delete the sprite (undoable) |
| Select | Both hands pinching ~0.3 s | Resize the selected sprite around its center |
| Any | Point at a toolbar button / swatch: 0.8 s (Draw), 0.55 s (Select) | Press it; fires once until the fingertip leaves |
| Any | Point at **Close** while help is open | Dismiss help without the keyboard |

Gesture priority is explicit: help open > two-hand resize > control dwell >
mode gestures; in Draw mode a fist beats a pinch. When the hand leaves the
frame, every pending hold or drag is cancelled and ARIA says so.

### Voice

Say the wake word (default **"hey jarvis"**), then:
"what did I draw" / "what is this", "change color to red", "clear" / "clear canvas",
"delete" / "remove it", "make it bigger" / "enlarge", "make it smaller" / "shrink",
"undo", "redo", "save" (exports `aria_snapshot.png`), "draw mode" / "select mode".

### Keyboard and mouse

| Key | Action |
| --- | --- |
| D / V | Draw mode / Select mode |
| 1-8 | Brush colors (palette order) |
| U / R | Undo / redo |
| C | Clear drawing |
| X / Backspace | Delete selected sprite |
| + / - | Resize selected sprite |
| E | Export PNG (`aria_snapshot.png`, without UI overlays) |
| P / L | Save / load project (`aria_project.npz`) |
| H / S | Help / status panel |
| Esc | Close help, cancel pending delete, hide quick start, deselect |
| Q or window close | Quit |

Mouse: click toolbar buttons, swatches, and layer thumbnails. In Select mode,
click a sprite to select, drag to move, click empty space to deselect, and use
the wheel to resize. Keyboard, mouse and pointing are three routes to the same
commands, so each one behaves identically. ARIA does not claim screen-reader
accessibility for the OpenCV window.

### Camera

ARIA asks the camera for 1280x720 (MJPG, shallow buffer) rather than accepting
the 640x480 most webcams default to, so the workspace shows real pixels instead
of an upscaled frame. If your camera does not offer that mode, set
`ARIA_CAMERA_WIDTH` / `ARIA_CAMERA_HEIGHT` in `.env` to a size it supports; the
startup log prints the size actually granted. `ARIA_CAMERA_MJPG=0` disables the
MJPG request for devices that misbehave with it.

## Local vs cloud

- **Local:** camera capture, MediaPipe hand tracking, drawing, sprites,
  history, project files, wake-word detection (openWakeWord), and TTS (pyttsx3).
  On first run openWakeWord may download its built-in models.
- **Cloud, optional:** spoken commands after the wake word are transcribed by
  Google's web speech recognition (on by default; set
  `ALLOW_CLOUD_SPEECH_RECOGNITION=0` to disable). "What did I draw" sends one
  frame (camera + drawing, without UI chrome) to Gemini, only when an API key is
  set. No telemetry; no recordings are stored.

## Website

`landing/` contains the project website (React + TypeScript + Vite). See
`landing/README.md`; it builds with a relative base so it can be hosted from
any subpath.

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
runtime/commands.py   shared Command/Dispatcher/Queue for every input method
runtime/notifications.py  bounded toasts
ui/theme.py, ui/layout.py, ui/help_content.py  tokens, geometry, help text
```

Input flow: gesture dwell, keyboard, mouse and voice all produce a `Command`
executed by one `CommandDispatcher` on the render thread. Voice runs on its own
thread and submits through a bounded `CommandQueue`, waiting for the real
result before replying. TTS runs on a single worker with a small bounded queue.

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
- Scene history (undo/redo) keeps up to 20 content snapshots; selection, mode, and brush changes are not history steps.
- Camera frames below 720 px tall are upscaled (aspect preserved) so overlays stay legible; the window is sized to the frame aspect.
- Wake-word detection uses `openWakeWord`, with `hey_jarvis` as the default built-in model.
- Voice command parsing is separated into intent parsing to keep the assistant command surface testable and easier to extend.
- ARIA supports both `google-genai` and the legacy `google-generativeai` package during the Gemini transition.
- Interaction state and workflow controllers are separated from the OpenCV loop to improve testability and maintainability.
- `scripts/run_checks.py` runs compile, Ruff, unit tests, and (when installed) the landing typecheck/tests/build.
