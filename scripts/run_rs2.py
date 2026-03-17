#!/usr/bin/env python
from __future__ import annotations

import argparse
import time

from macro_crisis_regime_bart.benchmarks.run_benchmarks import run_benchmark_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick run for RS-2 model")
    parser.add_argument("--config", default="configs/experiments/quick_rs2.yaml")
    args = parser.parse_args()
    t0 = time.time()
    run_dir = run_benchmark_experiment(args.config)
    print(f"Run completed: {run_dir}")
    print(f"Runtime: {time.time()-t0:.2f}s")


if __name__ == "__main__":
    main()
