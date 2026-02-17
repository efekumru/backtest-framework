"""
Risk-adjusted performance metrics.
"""

import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    excess = returns - risk_free / 252
    if excess.std() == 0:
        return 0.0
    return float((excess.mean() / excess.std()) * np.sqrt(252))


def sortino_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    excess = returns - risk_free / 252
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return float((excess.mean() / downside.std()) * np.sqrt(252))


def max_drawdown(returns: pd.Series) -> float:
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return float(drawdown.min())


def calmar_ratio(returns: pd.Series) -> float:
    n_years = len(returns) / 252
    total = (1 + returns).prod() - 1
    annual = (1 + total) ** (1 / max(n_years, 0.01)) - 1
    mdd = abs(max_drawdown(returns))
    return float(annual / mdd) if mdd != 0 else 0.0


def win_rate(trades_pnl: list[float]) -> float:
    if not trades_pnl:
        return 0.0
    wins = sum(1 for p in trades_pnl if p > 0)
    return wins / len(trades_pnl) * 100


def profit_factor(trades_pnl: list[float]) -> float:
    gross_profit = sum(p for p in trades_pnl if p > 0)
    gross_loss = abs(sum(p for p in trades_pnl if p < 0))
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def monthly_returns(equity: pd.Series) -> pd.DataFrame:
    """Compute monthly return heatmap data."""
    monthly = equity.resample("ME").last().pct_change()
    df = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month,
        "return": monthly.values,
    })
    return df.pivot(index="year", columns="month", values="return")


def compute_all(returns: pd.Series, trades_pnl: list[float] = None) -> dict:
    metrics = {
        "sharpe_ratio": round(sharpe_ratio(returns), 3),
        "sortino_ratio": round(sortino_ratio(returns), 3),
        "max_drawdown_pct": round(max_drawdown(returns) * 100, 2),
        "calmar_ratio": round(calmar_ratio(returns), 3),
        "annual_volatility_pct": round(returns.std() * np.sqrt(252) * 100, 2),
    }
    if trades_pnl:
        metrics["win_rate_pct"] = round(win_rate(trades_pnl), 1)
        metrics["profit_factor"] = round(profit_factor(trades_pnl), 2)
    return metrics
