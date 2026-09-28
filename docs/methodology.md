# Methodology

## Why condition portfolios on regimes?

Expected returns, volatilities, and correlations are not stable. Estimating one set of moments over an entire sample can obscure economically meaningful differences across environments.

This project evaluates whether a portfolio can use observable state information without introducing look-ahead bias.

## Regime features

Four state variables are used:

- growth
- inflation
- rates
- volatility

Each feature is standardized using expanding historical mean and standard deviation, shifted by one period. The sign of the resulting z-score defines high/low states.

For live data, CPI and industrial production are additionally shifted one month to approximate publication delay.

## Conditional estimation

At every rebalance date:

1. Use returns strictly prior to the allocation date.
2. Estimate expanding full-history mean and covariance.
3. Identify prior observations matching the current four-dimensional regime.
4. Estimate regime-specific moments.
5. Shrink those moments toward expanding full-history estimates.

Sparse regimes therefore cannot dominate the portfolio simply because they contain a handful of extreme observations.

## Portfolio construction

The adaptive portfolio solves a long-only mean-variance problem with:

- maximum position weights
- turnover penalty
- covariance shrinkage
- full-investment constraint

Three benchmarks are evaluated with identical return timing:

- equal weight
- expanding minimum variance
- risk parity

## Walk-forward timing

Weights for month t are based only on observations before month t and state variables available at the rebalance date. The resulting weights earn month-t returns after transaction costs.

## Limitations

Regime classification is intentionally interpretable rather than optimized for prediction. Macro publication calendars are approximated rather than modeled at timestamp precision. The project evaluates whether conditional portfolio construction is useful, not whether macro regimes can be perfectly identified in real time.
