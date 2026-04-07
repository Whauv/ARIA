from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_step(label: str, command: list[str]) -> None:
    print(f"[ARIA checks] {label}: {' '.join(command)}")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    python = sys.executable
    run_step(
        "py_compile",
        [
            python,
            "-m",
            "py_compile",
            "main.py",
            "app_runner.py",
            "runtime_state.py",
            "runtime_controllers.py",
            "ui.py",
            "canvas.py",
            "config.py",
            "gestures.py",
            "sprite.py",
            "ai_utils.py",
            "jarvis.py",
            "tests\\test_core.py",
            "tests\\test_runtime.py",
        ],
    )
    run_step("unit_tests", [python, "-m", "unittest", "discover", "-s", "tests"])
    print("[ARIA checks] all checks passed")


if __name__ == "__main__":
    main()
