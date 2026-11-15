"""Unit tests for ARIA."""

from __future__ import annotations

import sys
from pathlib import Path

# `python -m unittest discover -s tests` imports this package as top-level
# `unit`, so tests/__init__.py never runs; make `aria` importable here too.
SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
