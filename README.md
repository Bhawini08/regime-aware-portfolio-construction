# Building Portfolios for Regimes, Not Averages

A regime-aware multi-asset allocation research platform that asks whether portfolio construction improves when expected returns and risk are conditioned on observable macro and market environments instead of estimated from one long-run average.

## Research question

Traditional portfolio construction assumes one expected-return vector and one covariance matrix are representative through time. This project tests an alternative:

> Estimate the opportunity set using only information available at each rebalance date, condition those estimates on the prevailing growth, inflation, rate, and volatility environment, and compare the resulting portfolio with static allocation benchmarks.

## What the engine does

- Builds monthly multi-asset returns
- Constructs lagged macro and market regime features
- Classifies growth, inflation, rate, and volatility states
- Estimates expanding full-sample and regime-conditional moments
- Shrinks sparse regime estimates toward long-run expanding estimates
- Solves constrained long-only allocations
- Applies turnover penalties and transaction costs
- Runs a strictly chronological walk-forward backtest
- Compares adaptive allocation with equal weight, static minimum variance, and risk parity
- Reports regime occupancy, transition behavior, turnover, drawdown, Sharpe ratio, and downside statistics

## Live data

Asset proxies:
SPY, EFA, EEM, IEF, TLT, LQD, HYG, GLD, DBC, VNQ.

Macro / market features:
- Industrial Production (growth)
- CPI (inflation)
- 10-year Treasury yield (rates)
- VIX (volatility)

Macro series are shifted before portfolio formation so the strategy does not use unreleased contemporaneous observations.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src

pytest -q
python scripts/run_regime_analysis.py --mode synthetic
python scripts/run_regime_analysis.py --mode live

streamlit run dashboard/app.py
```

## Research discipline

The adaptive strategy is evaluated walk-forward only. Regime thresholds, conditional moments, portfolio weights, and risk estimates are formed using historical information available before the return being evaluated.

## What the repository is designed to establish

The research question is not whether one regime label can forecast markets. It is whether conditioning the opportunity set on information available at each rebalance changes portfolio decisions enough to improve the risk/return trade-off after implementation frictions. The comparison is therefore walk-forward and benchmark-relative rather than an in-sample optimization exercise.

## Validation

The implementation explicitly lags regime information before allocation, estimates moments using expanding historical information, and compares the adaptive portfolio with equal-weight, static minimum-variance, and risk-parity benchmarks. GitHub Actions runs the tests and synthetic analysis on every push and pull request.

## Limitations

Regime definitions are modeling choices, macro releases are revised, and conditional samples can be sparse. Shrinkage reduces but does not eliminate estimation error. Any apparent advantage is therefore interpreted together with turnover, transaction costs, drawdowns, regime occupancy, and benchmark behavior rather than as a standalone alpha claim.

