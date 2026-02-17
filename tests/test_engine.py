"""
Tests for the backtest engine core components.
"""

import numpy as np
import pandas as pd
import pytest

from engine.events import Direction, FillEvent, OrderEvent, SignalEvent
from engine.portfolio import Portfolio
from engine.execution import ExecutionHandler, CostModel


def make_fill(ticker="AAPL", direction=Direction.LONG, qty=100, price=150.0) -> FillEvent:
    return FillEvent(
        ticker=ticker,
        direction=direction,
        quantity=qty,
        fill_price=price,
        commission=10.0,
        slippage=0.15,
    )


class TestPortfolio:
    def test_initial_state(self):
        p = Portfolio(initial_capital=100_000)
        assert p.cash == 100_000
        assert p.total_value == 100_000
        assert len(p.positions) == 0
        assert len(p.trades) == 0

    def test_open_position(self):
        p = Portfolio(initial_capital=100_000)
        fill = make_fill(price=150.0, qty=100)
        p.process_fill(fill)

        assert "AAPL" in p.positions
        assert p.positions["AAPL"].quantity == 100
        assert p.cash < 100_000

    def test_close_position_records_trade(self):
        p = Portfolio(initial_capital=100_000)

        # Open
        p.process_fill(make_fill(direction=Direction.LONG, price=150.0, qty=100))

        # Close
        p.process_fill(make_fill(direction=Direction.EXIT, price=160.0, qty=100))

        assert "AAPL" not in p.positions
        assert len(p.trades) == 1
        assert p.trades[0]["pnl"] > 0

    def test_losing_trade(self):
        p = Portfolio(initial_capital=100_000)
        p.process_fill(make_fill(direction=Direction.LONG, price=150.0, qty=100))
        p.process_fill(make_fill(direction=Direction.EXIT, price=140.0, qty=100))

        assert p.trades[0]["pnl"] < 0

    def test_equity_recording(self):
        p = Portfolio(initial_capital=100_000)
        p.record_equity("2024-01-01", {"AAPL": 150.0})
        p.record_equity("2024-01-02", {"AAPL": 151.0})

        assert len(p.equity_curve) == 2


class TestExecution:
    def test_long_fill_has_slippage(self):
        handler = ExecutionHandler(CostModel(slippage_pct=0.001))
        order = OrderEvent(ticker="AAPL", direction=Direction.LONG, quantity=100)
        fill = handler.execute(order, current_price=150.0)

        assert fill.fill_price > 150.0
        assert fill.commission > 0

    def test_exit_fill_has_slippage(self):
        handler = ExecutionHandler(CostModel(slippage_pct=0.001))
        order = OrderEvent(ticker="AAPL", direction=Direction.EXIT, quantity=100)
        fill = handler.execute(order, current_price=150.0)

        assert fill.fill_price < 150.0

    def test_fill_count_increments(self):
        handler = ExecutionHandler()
        order = OrderEvent(ticker="AAPL", direction=Direction.LONG, quantity=100)

        handler.execute(order, 150.0)
        handler.execute(order, 151.0)

        assert handler.fill_count == 2

    def test_reset(self):
        handler = ExecutionHandler()
        order = OrderEvent(ticker="AAPL", direction=Direction.LONG, quantity=100)
        handler.execute(order, 150.0)
        handler.reset()
        assert handler.fill_count == 0


class TestEvents:
    def test_fill_total_cost(self):
        fill = FillEvent(commission=10.0, slippage=5.0)
        assert fill.total_cost == 15.0

    def test_order_repr(self):
        order = OrderEvent(ticker="AAPL", direction=Direction.LONG, quantity=100)
        assert "LONG" in repr(order)
        assert "AAPL" in repr(order)
