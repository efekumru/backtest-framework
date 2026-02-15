# 🧪 Backtest Framework

An event-driven backtesting engine built from scratch for evaluating algorithmic trading strategies. Designed to avoid common pitfalls like lookahead bias, unrealistic fills, and overfitting.

## Why Build Another Backtester?

Most backtesting libraries either oversimplify execution (vectorized backtests that ignore real-world constraints) or are too opaque to trust. This framework is:

- **Transparent** — every fill, fee, and slippage calculation is explicit
- **Rigorous** — walk-forward validation with purged gaps, automated lookahead bias checks
- **Practical** — generates HTML reports with the metrics that actually matter

## Features

- Event-driven architecture (market events → signal → order → fill → portfolio update)
- Walk-forward validation with expanding/sliding windows and purged cross-validation
- Realistic cost modeling: configurable commission, slippage, and market impact
- Automated lookahead bias detection test suite
- HTML report generation with equity curves, drawdown charts, and monthly returns heatmap
- Multi-asset support with portfolio-level risk metrics
- Strategy parameter optimization with cross-validated grid search

## Quick Start

```bash
git clone https://github.com/efekumru/backtest-framework.git
cd backtest-framework
pip install -r requirements.txt

# Run a simple moving average crossover strategy
python run.py --strategy sma_crossover --ticker AAPL --period 5y

# Run with walk-forward validation
python run.py --strategy momentum --ticker MSFT --walk-forward --train-months 24 --test-months 3

# Generate HTML report
python run.py --strategy mean_reversion --ticker GOOGL --report results/report.html

# Run lookahead bias tests
pytest tests/test_no_lookahead.py -v
```

## Example Output

```
══════════════════════════════════════════
  Backtest Results: SMA Crossover (AAPL)
  Period: 2019-01-01 → 2024-12-31
══════════════════════════════════════════
  Total Return:      +47.3%
  Annual Return:     +8.1%
  Sharpe Ratio:      1.12
  Sortino Ratio:     1.54
  Max Drawdown:      -14.8%
  Calmar Ratio:      0.55
  Win Rate:          54.2%
  Profit Factor:     1.38
  Total Trades:      127
  Avg Trade Duration: 12.3 days
══════════════════════════════════════════
```

## Project Structure

```
backtest-framework/
├── run.py                  # CLI entry point
├── engine/
│   ├── backtest.py         # Core event loop
│   ├── events.py           # Event types (Market, Signal, Order, Fill)
│   ├── portfolio.py        # Portfolio & position tracking
│   ├── execution.py        # Order execution with slippage model
│   └── data_handler.py     # OHLCV data feed
├── strategies/
│   ├── base.py             # Abstract strategy interface
│   ├── sma_crossover.py    # Simple moving average crossover
│   ├── momentum.py         # Momentum / trend-following
│   └── mean_reversion.py   # Mean reversion (z-score based)
├── validation/
│   ├── walk_forward.py     # Walk-forward splitter
│   └── optimizer.py        # Parameter grid search with CV
├── risk/
│   ├── metrics.py          # Sharpe, Sortino, Drawdown, Calmar, etc.
│   └── report.py           # HTML report generation
├── tests/
│   ├── test_engine.py      # Engine unit tests
│   ├── test_portfolio.py   # Portfolio math tests
│   └── test_no_lookahead.py # Automated lookahead bias detection
├── configs/                # Strategy YAML configs
└── results/                # Generated reports & charts
```

## Architecture

```
┌──────────┐   MarketEvent   ┌───────────┐  SignalEvent  ┌──────────┐
│   Data   │ ──────────────▶ │  Strategy  │ ───────────▶ │ Portfolio│
│  Handler │                 │            │              │  Manager │
└──────────┘                 └───────────┘              └────┬─────┘
                                                             │ OrderEvent
                                                      ┌─────▼──────┐
                                                      │  Execution  │
                                                      │   Handler   │
                                                      └─────┬──────┘
                                                             │ FillEvent
                                                      ┌─────▼──────┐
                                                      │  Portfolio  │
                                                      │   Update    │
                                                      └─────┬──────┘
                                                             │
                                                      ┌─────▼──────┐
                                                      │   Risk &   │
                                                      │  Reporting │
                                                      └────────────┘
```

## Tech Stack

Python · Pandas · NumPy · Matplotlib · Plotly · Jinja2 · yfinance · pytest

## License

MIT
