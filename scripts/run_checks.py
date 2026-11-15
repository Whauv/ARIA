from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LANDING = PROJECT_ROOT / "landing"


def run_step(label: str, command: list[str], cwd: Path = PROJECT_ROOT) -> None:
    print(f"[ARIA checks] {label}: {' '.join(command)}")
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    src_path = str(PROJECT_ROOT / "src")
    env["PYTHONPATH"] = (
        src_path
        if not existing_pythonpath
        else f"{src_path}{os.pathsep}{existing_pythonpath}"
    )
    subprocess.run(command, cwd=cwd, check=True, env=env)


def main() -> None:
    python = sys.executable
    # compileall covers every module, so new files can't be silently skipped.
    run_step("compile", [python, "-m", "compileall", "-q", "main.py", "src", "tests"])
    if importlib.util.find_spec("ruff") is not None:
        run_step("ruff_check", [python, "-m", "ruff", "check", "."])
        run_step("ruff_format", [python, "-m", "ruff", "format", "--check", "."])
    else:
        print("[ARIA checks] ruff not installed; skipping lint (CI runs it)")
    run_step("unit_tests", [python, "-m", "unittest", "discover", "-s", "tests"])

    npm = shutil.which("npm")
    if "--skip-web" in sys.argv:
        print("[ARIA checks] --skip-web: skipping landing checks")
    elif npm is None or not (LANDING / "node_modules").exists():
        print(
            "[ARIA checks] landing/: npm or node_modules missing; "
            "run `npm ci` in landing/ to include website checks"
        )
    else:
        for script in ("typecheck", "test", "build"):
            run_step(f"landing_{script}", [npm, "run", script], cwd=LANDING)
    print("[ARIA checks] all checks passed")


if __name__ == "__main__":
    main()
