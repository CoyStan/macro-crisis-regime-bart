# macro-crisis-regime-bart

Research-grade Python package for cross-country macro-financial crisis forecasting (country-month panel, 1990-2025).

## Implemented model stack

### Benchmarks
- pooled logistic / probit / elastic-net logistic
- random forest
- xgboost / monotone xgboost

### Custom Bayesian models
- **Static pmBART**: monotone probit BART baseline with native missing routing
- **RS-1**: latent global regimes + country effects + regime intercept + baseline pmBART
- **RS-2**: RS-1 + regime-specific nonlinear deviations
- **Final TVTP amplification model**: sticky TVTP regime switching with amplification

\[
z_{it}=\alpha_i + \delta_{s_t} + \lambda_{s_t} f_0(x_{it}) + \varepsilon_{it},\quad \varepsilon_{it}\sim N(0,1)
\]

with:
- \(\lambda_1=1\), \(\lambda_2=\exp(\eta_2)>1\)
- TVTP transitions using global covariates \(W_t\):
  \(P(s_t=2\mid s_{t-1}=j,w_t)=\text{logistic}(a_j+b_j'w_t)\)
- Pólya-Gamma updates for TVTP coefficients
- log-space FFBS for regime path sampling

## Data contract

Two input files (CSV/Parquet):
1. **Crisis dataset**: `country_id`, `year`, `month`, plus crisis columns (e.g., `crisis_any`)
2. **Feature dataset**: `country_id`, `year`, `month`, predictors

The feature registry separates:
- `X`: country-time predictors (for crisis latent index)
- `W_t`: global time-level predictors (`is_global: true`) used by TVTP transitions

## Staged burn-in for final TVTP model

The final model uses staged warm start:
1. **Phase A (N0)**: static warm-up (`s_t=1`, no amplification)
2. **Phase B (N1)**: activate regimes + deltas + TVTP, keep amplification fixed
3. **Phase C (burn-in)**: full sampler including `eta_2` update

Only draws after warm phases + burn-in are retained.

## Running experiments

### Benchmarks
```bash
python scripts/run_benchmarks.py --config configs/experiments/benchmark_forecast_12m.yaml
```

### Quick custom runs
```bash
python scripts/run_static_pmBART.py --config configs/experiments/quick_static_pmBART.yaml
python scripts/run_rs1.py --config configs/experiments/quick_rs1.yaml
python scripts/run_rs2.py --config configs/experiments/quick_rs2.yaml
python scripts/run_tvtp_amp.py --config configs/experiments/quick_tvtp_amp.yaml
```

### TVTP validation run template
```bash
python scripts/run_tvtp_amp.py --config configs/experiments/final_tvtp_amp_validation.yaml
```

## Outputs

Runs are saved under `outputs/runs/<run_name>/` with:
- `tables/metrics.csv`
- `predictions/test_predictions.csv`
- `models/*.joblib`
- `diagnostics/*_regime_probs.csv`
- config snapshot

## Deferred scope

- additional K>2 regimes
- final paper automation and Overleaf integration
- extended explainability exports beyond implemented component summaries
