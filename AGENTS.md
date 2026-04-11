# AGENTS

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

## Checks

```powershell
python scripts\run_checks.py
```

## Folder Map

- `src/aria/`: application package
- `src/aria/ai/`: Gemini image analysis helpers
- `src/aria/assistant/`: Jarvis voice assistant and command parsing
- `src/aria/drawing/`: drawing canvas and sprite primitives
- `src/aria/runtime/`: state, controllers, capture, and hand-tracking services
- `src/aria/ui/`: UI layout and rendering helpers
- `src/aria/vision/`: gesture utilities
- `tests/unit/`: unit tests
- `tests/integration/`: integration placeholders for hardware-backed flows
- `scripts/`: developer automation and checks

## Code Style

- Prefer small modules with explicit responsibilities
- Keep business logic out of entrypoints
- Use package-relative imports inside `src/aria`
- Avoid hardcoding secrets; use `.env` and `.env.example`

## Test Commands

```powershell
python -m unittest discover -s tests
python scripts\run_checks.py
```
