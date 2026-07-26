---
name: vectorbt-optimize
description: Optimize strategy parameters using VectorBT. Tests parameter combinations and generates heatmaps.
argument-hint: "[strategy] [symbol] [exchange] [interval]"
---

# VectorBT Parameter Optimization

Create a parameter optimization script for a VectorBT strategy.

## Arguments

- `$0` = strategy name (e.g., ema-crossover, rsi, donchian). Default: ema-crossover
- `$1` = symbol (e.g., SBIN, RELIANCE, NIFTY). Default: SBIN
- `$2` = exchange (e.g., NSE, NFO). Default: NSE
- `$3` = interval (e.g., D, 1h, 5m). Default: D

## Instructions

1. Create `backtesting/{strategy_name}/` directory if it doesn't exist
2. Create a `.py` file named `{symbol}_{strategy}_optimize.py`
3. The script must:
   - Use OpenAlgo ta for ALL indicators by default
   - Use `ta.exrem()` to clean signals
   - Define sensible parameter ranges for the chosen strategy
   - Use loop-based optimization with multiple metrics per combo
   - Track: total_return, sharpe_ratio, max_drawdown, trade_count
   - Use `tqdm` for progress bars
   - Generate Plotly heatmaps (`template="plotly_dark"`)
   - Compare best parameters vs benchmark
   - Print Strategy vs Benchmark comparison table
   - Explain results in plain language
   - Save results to CSV

## Default Parameter Ranges

| Strategy | Parameter 1 | Parameter 2 |
|----------|------------|-------------|
| ema-crossover | fast EMA: 5-50 | slow EMA: 10-60 |
| rsi | window: 5-30 | oversold: 20-40 |
| donchian | period: 5-50 | - |
| supertrend | period: 5-30 | multiplier: 1.0-5.0 |
