from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BenchmarkSpec:
    name: str
    model_config_path: str
