"""Shared project root, WSL environment validation and JSON persistence."""
from __future__ import annotations

import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[2]


def ensure_environment() -> None:
    if platform.system() != "Linux" or "microsoft" not in platform.release().lower():
        raise RuntimeError("Run this script inside WSL, using the vla-supervised Conda environment.")
    if Path(sys.prefix).name != "vla-supervised":
        raise RuntimeError(f"Expected vla-supervised environment; current interpreter: {sys.executable}")


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
