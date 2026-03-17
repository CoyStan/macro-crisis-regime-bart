#!/usr/bin/env python
from __future__ import annotations

import argparse
import pandas as pd

from macro_crisis_regime_bart.benchmarks.run_benchmarks import run_benchmark_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run final TVTP amplification model")
    parser.add_argument("--config", default="configs/experiments/quick_tvtp_amp.yaml")
    args = parser.parse_args()

    run_dir = run_benchmark_experiment(args.config)
    metrics = pd.read_csv(run_dir / "tables" / "metrics.csv")
    reg_path = run_dir / "diagnostics" / "tvtp_amp_regime_probs.csv"
    print(f"Run completed: {run_dir}")
    print(metrics.to_string(index=False))
    if reg_path.exists():
        reg = pd.read_csv(reg_path)
        print(f"Current stress probability: {reg['p_regime_2'].iloc[-1]:.3f}")
        print(f"Current enter-stress probability: {reg['p_enter_stress'].iloc[-1]:.3f}")
        print(f"Current stay-stress probability: {reg['p_stay_stress'].iloc[-1]:.3f}")


if __name__ == "__main__":
    main()
