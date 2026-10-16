"""Compatibility import; implementation: tasks.nuscenes_sam3.python.sam3_nuscenes."""
import importlib
from pathlib import Path
import sys
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)
_implementation = importlib.import_module("tasks.nuscenes_sam3.python.sam3_nuscenes")
if __name__ == "__main__":
    if not hasattr(_implementation, "main"):
        raise SystemExit("This module is a library; use a task Bash entry point.")
    raise SystemExit(_implementation.main())
else:
    sys.modules[__name__] = _implementation
