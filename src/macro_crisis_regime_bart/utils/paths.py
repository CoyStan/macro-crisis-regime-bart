from __future__ import annotations

from pathlib import Path
from datetime import datetime


def make_run_dir(base_dir: str | Path = "outputs/runs", run_name: str | None = None) -> Path:
    """Create and return a run directory."""
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    name = run_name or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = base / name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir
