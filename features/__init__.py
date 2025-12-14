"""Convenience shim so ``import features`` works from the repository root.

This module mirrors the public API of ``src.features`` while ensuring the
``src`` directory is present on ``sys.path``. It allows one-liner commands such
as ``python -c "from features import build_feature_table"`` to work without
manually setting ``PYTHONPATH``.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
_SRC_PATH = _PROJECT_ROOT / "src"
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

# Import the real package and re-export its public API.
_features_pkg = importlib.import_module("src.features")
from src.features import *  # noqa: F401,F403

__all__ = getattr(_features_pkg, "__all__", [])
