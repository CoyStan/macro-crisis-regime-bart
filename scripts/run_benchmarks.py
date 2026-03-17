#!/usr/bin/env python
from __future__ import annotations

import argparse
from macro_crisis_regime_bart.benchmarks.run_benchmarks import run_benchmark_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark experiment")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    args = parser.parse_args()
    run_dir = run_benchmark_experiment(args.config)
    print(f"Run completed: {run_dir}")


if __name__ == "__main__":
    main()
