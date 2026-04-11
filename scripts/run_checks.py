from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_step(label: str, command: list[str]) -> None:
    print(f"[ARIA checks] {label}: {' '.join(command)}")
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    src_path = str(PROJECT_ROOT / "src")
    env["PYTHONPATH"] = src_path if not existing_pythonpath else f"{src_path}{os.pathsep}{existing_pythonpath}"
    subprocess.run(command, cwd=PROJECT_ROOT, check=True, env=env)


def main() -> None:
    python = sys.executable
    run_step(
        "py_compile",
        [
            python,
            "-m",
            "py_compile",
            "main.py",
            "src\\aria\\__main__.py",
            "src\\aria\\app_runner.py",
            "src\\aria\\config.py",
            "src\\aria\\ai\\ai_utils.py",
            "src\\aria\\assistant\\command_intents.py",
            "src\\aria\\assistant\\jarvis.py",
            "src\\aria\\drawing\\canvas.py",
            "src\\aria\\drawing\\sprite.py",
            "src\\aria\\runtime\\controllers.py",
            "src\\aria\\runtime\\services.py",
            "src\\aria\\runtime\\state.py",
            "src\\aria\\ui\\ui.py",
            "src\\aria\\vision\\gestures.py",
            "tests\\unit\\test_core.py",
            "tests\\unit\\test_app_runner.py",
            "tests\\unit\\test_commands.py",
            "tests\\unit\\test_runtime.py",
            "tests\\integration\\test_camera_placeholder.py",
            "tests\\integration\\test_hand_tracking_placeholder.py",
            "tests\\integration\\test_voice_placeholder.py",
        ],
    )
    run_step("unit_tests", [python, "-m", "unittest", "discover", "-s", "tests"])
    print("[ARIA checks] all checks passed")


if __name__ == "__main__":
    main()
