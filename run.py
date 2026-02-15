"""
Backtest Framework — Event-Driven Backtesting Engine
Core engine with event loop, portfolio management, and execution.
"""

import argparse
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from queue import Queue
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Events
# ──────────────────────────────────────────────


class EventType(Enum):
    MARKET = "MARKET"
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    FILL = "FILL"


class Direction(Enum):
    LONG = 1
    SHORT = -1
    EXIT = 0


@dataclass
class MarketEvent:
    type: EventType = field(default=EventType.MARKET, init=False)
    timestamp: datetime = datetime.now()


@dataclass
class SignalEvent:
    type: EventType = field(default=EventType.SIGNAL, init=False)
    ticker: str = ""
    direction: Direction = Direction.LONG
    strength: float = 1.0
    timestamp: datetime = datetime.now()


@dataclass
class OrderEvent:
    type: EventType = field(default=EventType.ORDER, init=False)
    ticker: str = ""
    direction: Direction = Direction.LONG
    quantity: int = 0
    timestamp: datetime = datetime.now()


@dataclass
class FillEvent:
    type: EventType = field(default=EventType.FILL, init=False)
    ticker: str = ""
    direction: Direction = Direction.LONG
    quantity: int = 0
    fill_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    timestamp: datetime = datetime.now()

    @property
    def total_cost(self) -> float:
        return self.commission + self.slippage


# ──────────────────────────────────────────────
# Data Handler
# ──────────────────────────────────────────────


class DataHandler:
    """OHLCV data feed that emits MarketEvents bar by bar."""

    def __init__(self, ticker: str, data: pd.DataFrame):
        self.ticker = ticker
        self.data = data.copy()
        self.index = 0
        self.total_bars = len(data)

    @classmethod
    def from_csv(cls, ticker: str, filepath: str) -> "DataHandler":
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        return cls(ticker, df)

    @classmethod
    def from_yfinance(cls, ticker: str, period: str = "5y") -> "DataHandler":
        import yfinance as yf

        df = yf.download(ticker, period=period, progress=False)
        return cls(ticker, df)

    def has_next(self) -> bool:
        return self.index < self.total_bars

    def get_next_bar(self) -> Optional[pd.Series]:
        if not self.has_next():
            return None
        bar = self.data.iloc[self.index]
        self.index += 1
        return bar

    def get_current_price(self) -> float:
        if self.index == 0:
            return self.data.iloc[0]["Close"]
        return self.data.iloc[self.index - 1]["Close"]

    def get_history(self, lookback: int) -> pd.DataFrame:
        """Return past N bars (no future data)."""
        end = self.index
        start = max(0, end - lookback)
        return self.data.iloc[start:end].copy()


# ──────────────────────────────────────────────
# Strategy Interface
# ──────────────────────────────────────────────


class BaseStrategy(ABC):
    """Abstract base class for trading strategies."""

    @abstractmethod
    def calculate_signals(
        self, data_handler: DataHandler
    ) -> Optional[SignalEvent]:
        ...


class SMACrossoverStrategy(BaseStrategy):
    """Simple Moving Average crossover strategy."""

    def __init__(self, fast_period: int = 20, slow_period: int = 50):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.prev_position: Direction = Direction.EXIT

    def calculate_signals(
        self, data_handler: DataHandler
    ) -> Optional[SignalEvent]:
        history = data_handler.get_history(self.slow_period + 1)
        if len(history) < self.slow_period:
            return None

        close = history["Close"]
        fast_ma = close.rolling(self.fast_period).mean().iloc[-1]
        slow_ma = close.rolling(self.slow_period).mean().iloc[-1]

        if fast_ma > slow_ma and self.prev_position != Direction.LONG:
            self.prev_position = Direction.LONG
            return SignalEvent(
                ticker=data_handler.ticker,
                direction=Direction.LONG,
                strength=1.0,
                timestamp=history.index[-1],
            )
        elif fast_ma < slow_ma and self.prev_position != Direction.SHORT:
            self.prev_position = Direction.EXIT
            return SignalEvent(
                ticker=data_handler.ticker,
                direction=Direction.EXIT,
                strength=1.0,
                timestamp=history.index[-1],
            )

        return None


class MomentumStrategy(BaseStrategy):
    """Momentum / trend-following strategy."""

    def __init__(self, lookback: int = 20, threshold: float = 0.02):
        self.lookback = lookback
        self.threshold = threshold
        self.prev_position: Direction = Direction.EXIT

    def calculate_signals(
        self, data_handler: DataHandler
    ) -> Optional[SignalEvent]:
        history = data_handler.get_history(self.lookback + 1)
        if len(history) < self.lookback:
            return None

        returns = history["Close"].pct_change(self.lookback).iloc[-1]

        if returns > self.threshold and self.prev_position != Direction.LONG:
            self.prev_position = Direction.LONG
            return SignalEvent(
                ticker=data_handler.ticker,
                direction=Direction.LONG,
                strength=min(returns / self.threshold, 2.0),
                timestamp=history.index[-1],
            )
        elif returns < -self.threshold and self.prev_position != Direction.EXIT:
            self.prev_position = Direction.EXIT
            return SignalEvent(
                ticker=data_handler.ticker,
                direction=Direction.EXIT,
                timestamp=history.index[-1],
            )

        return None


class MeanReversionStrategy(BaseStrategy):
    """Mean reversion using z-score of price."""

    def __init__(
        self,
        lookback: int = 20,
        entry_z: float = -1.5,
        exit_z: float = 0.0,
    ):
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.in_position = False

    def calculate_signals(
        self, data_handler: DataHandler
    ) -> Optional[SignalEvent]:
        history = data_handler.get_history(self.lookback + 1)
        if len(history) < self.lookback:
            return None

        close = history["Close"]
        mean = close.rolling(self.lookback).mean().iloc[-1]
        std = close.rolling(self.lookback).std().iloc[-1]

        if std == 0:
            return None

        z_score = (close.iloc[-1] - mean) / std

        if z_score <= self.entry_z and not self.in_position:
            self.in_position = True
            return SignalEvent(
                ticker=data_handler.ticker,
                direction=Direction.LONG,
                strength=abs(z_score),
                timestamp=history.index[-1],
            )
        elif z_score >= self.exit_z and self.in_position:
            self.in_position = False
            return SignalEvent(
                ticker=data_handler.ticker,
                direction=Direction.EXIT,
                timestamp=history.index[-1],
            )

        return None


# ──────────────────────────────────────────────
# Portfolio
# ──────────────────────────────────────────────


@dataclass
class Position:
    ticker: str
    quantity: int = 0
    avg_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


class Portfolio:
    """Tracks positions, cash, and portfolio value over time."""

    def __init__(self, initial_capital: float = 100_000.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: dict[str, Position] = {}
        self.equity_curve: list[dict] = []
        self.trades: list[dict] = []

    @property
    def total_value(self) -> float:
        positions_value = sum(
            p.quantity * p.avg_price for p in self.positions.values()
        )
        return self.cash + positions_value

    def update_market_value(self, ticker: str, price: float) -> None:
        if ticker in self.positions:
            pos = self.positions[ticker]
            pos.unrealized_pnl = (price - pos.avg_price) * pos.quantity

    def process_fill(self, fill: FillEvent) -> None:
        ticker = fill.ticker
        cost = fill.fill_price * fill.quantity + fill.total_cost

        if fill.direction == Direction.LONG:
            if ticker not in self.positions:
                self.positions[ticker] = Position(ticker=ticker)

            pos = self.positions[ticker]
            # Update average price
            total_qty = pos.quantity + fill.quantity
            if total_qty > 0:
                pos.avg_price = (
                    (pos.avg_price * pos.quantity)
                    + (fill.fill_price * fill.quantity)
                ) / total_qty
            pos.quantity = total_qty
            self.cash -= cost

        elif fill.direction == Direction.EXIT:
            if ticker in self.positions:
                pos = self.positions[ticker]
                pnl = (fill.fill_price - pos.avg_price) * pos.quantity
                pos.realized_pnl += pnl
                self.cash += fill.fill_price * pos.quantity - fill.total_cost

                self.trades.append(
                    {
                        "ticker": ticker,
                        "entry_price": pos.avg_price,
                        "exit_price": fill.fill_price,
                        "quantity": pos.quantity,
                        "pnl": round(pnl, 2),
                        "return_pct": round(
                            (fill.fill_price / pos.avg_price - 1) * 100, 2
                        ),
                        "timestamp": fill.timestamp,
                    }
                )
                del self.positions[ticker]

    def record_equity(self, timestamp: datetime, price: float) -> None:
        positions_value = sum(
            p.quantity * price
            for p in self.positions.values()
        )
        self.equity_curve.append(
            {
                "timestamp": timestamp,
                "equity": self.cash + positions_value,
                "cash": self.cash,
                "positions_value": positions_value,
            }
        )


# ──────────────────────────────────────────────
# Execution Handler
# ──────────────────────────────────────────────


class ExecutionHandler:
    """Simulates order execution with slippage and commission."""

    def __init__(
        self,
        commission_per_trade: float = 10.0,
        slippage_pct: float = 0.001,
    ):
        self.commission = commission_per_trade
        self.slippage_pct = slippage_pct

    def execute_order(
        self, order: OrderEvent, current_price: float
    ) -> FillEvent:
        # Apply slippage
        if order.direction == Direction.LONG:
            fill_price = current_price * (1 + self.slippage_pct)
        else:
            fill_price = current_price * (1 - self.slippage_pct)

        slippage_cost = abs(fill_price - current_price) * order.quantity

        return FillEvent(
            ticker=order.ticker,
            direction=order.direction,
            quantity=order.quantity,
            fill_price=round(fill_price, 4),
            commission=self.commission,
            slippage=round(slippage_cost, 4),
            timestamp=order.timestamp,
        )


# ──────────────────────────────────────────────
# Risk Metrics
# ──────────────────────────────────────────────


def compute_metrics(equity_curve: pd.DataFrame) -> dict:
    """Compute risk-adjusted performance metrics."""
    returns = equity_curve["equity"].pct_change().dropna()

    total_return = (
        equity_curve["equity"].iloc[-1] / equity_curve["equity"].iloc[0] - 1
    )
    n_years = len(returns) / 252
    annual_return = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1

    # Sharpe (assuming 0% risk-free rate)
    sharpe = (
        (returns.mean() / returns.std()) * np.sqrt(252)
        if returns.std() > 0
        else 0
    )

    # Sortino
    downside = returns[returns < 0]
    sortino = (
        (returns.mean() / downside.std()) * np.sqrt(252)
        if len(downside) > 0 and downside.std() > 0
        else 0
    )

    # Max drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    # Calmar
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

    return {
        "total_return": round(total_return * 100, 2),
        "annual_return": round(annual_return * 100, 2),
        "sharpe_ratio": round(sharpe, 2),
        "sortino_ratio": round(sortino, 2),
        "max_drawdown": round(max_drawdown * 100, 2),
        "calmar_ratio": round(calmar, 2),
        "volatility": round(returns.std() * np.sqrt(252) * 100, 2),
        "total_trades": 0,  # filled by caller
        "win_rate": 0.0,
    }


# ──────────────────────────────────────────────
# Backtest Engine
# ──────────────────────────────────────────────


class BacktestEngine:
    """Main event-driven backtest engine."""

    def __init__(
        self,
        data_handler: DataHandler,
        strategy: BaseStrategy,
        portfolio: Portfolio,
        execution: ExecutionHandler,
        position_size: int = 100,
    ):
        self.data = data_handler
        self.strategy = strategy
        self.portfolio = portfolio
        self.execution = execution
        self.position_size = position_size
        self.events: Queue = Queue()

    def run(self) -> dict:
        """Execute the backtest."""
        logger.info(
            f"Starting backtest: {self.data.ticker} "
            f"({self.data.total_bars} bars)"
        )

        while self.data.has_next():
            bar = self.data.get_next_bar()
            if bar is None:
                break

            timestamp = bar.name if hasattr(bar, "name") else datetime.now()
            price = bar["Close"]

            # Update portfolio market value
            self.portfolio.update_market_value(self.data.ticker, price)

            # Generate signals
            signal = self.strategy.calculate_signals(self.data)

            if signal is not None:
                # Convert signal to order
                order = OrderEvent(
                    ticker=signal.ticker,
                    direction=signal.direction,
                    quantity=self.position_size,
                    timestamp=timestamp,
                )

                # Execute order
                fill = self.execution.execute_order(order, price)

                # Update portfolio
                self.portfolio.process_fill(fill)

            # Record equity
            self.portfolio.record_equity(timestamp, price)

        # Compute results
        equity_df = pd.DataFrame(self.portfolio.equity_curve)
        metrics = compute_metrics(equity_df)
        metrics["total_trades"] = len(self.portfolio.trades)

        if self.portfolio.trades:
            wins = [t for t in self.portfolio.trades if t["pnl"] > 0]
            metrics["win_rate"] = round(
                len(wins) / len(self.portfolio.trades) * 100, 1
            )

        return metrics


# ──────────────────────────────────────────────
# Strategies Registry
# ──────────────────────────────────────────────

STRATEGIES: dict[str, type[BaseStrategy]] = {
    "sma_crossover": SMACrossoverStrategy,
    "momentum": MomentumStrategy,
    "mean_reversion": MeanReversionStrategy,
}


def main():
    parser = argparse.ArgumentParser(
        description="Backtest Framework — Run Strategy"
    )
    parser.add_argument("--strategy", type=str, default="sma_crossover",
                        choices=list(STRATEGIES.keys()))
    parser.add_argument("--ticker", type=str, default="AAPL")
    parser.add_argument("--period", type=str, default="5y")
    parser.add_argument("--capital", type=float, default=100_000)
    parser.add_argument("--commission", type=float, default=10.0)
    parser.add_argument("--slippage", type=float, default=0.001)
    parser.add_argument("--report", type=str, default=None)

    args = parser.parse_args()

    # Setup
    data = DataHandler.from_yfinance(args.ticker, args.period)
    strategy = STRATEGIES[args.strategy]()
    portfolio = Portfolio(initial_capital=args.capital)
    execution = ExecutionHandler(args.commission, args.slippage)

    # Run
    engine = BacktestEngine(data, strategy, portfolio, execution)
    metrics = engine.run()

    # Print results
    print(f"\n{'═'*50}")
    print(f"  Backtest Results: {args.strategy} ({args.ticker})")
    print(f"{'═'*50}")
    for key, value in metrics.items():
        label = key.replace("_", " ").title()
        if "return" in key or "drawdown" in key or "volatility" in key or "win_rate" in key:
            print(f"  {label:20s} {value:>+.1f}%")
        else:
            print(f"  {label:20s} {value}")
    print(f"{'═'*50}")


if __name__ == "__main__":
    main()
