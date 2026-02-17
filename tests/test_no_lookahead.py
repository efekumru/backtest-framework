"""
Automated lookahead bias detection.
Ensures data handler never exposes future data.
"""

import numpy as np
import pandas as pd
import pytest

from engine.data_handler import DataHandler


def make_sample_data(n: int = 100) -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.bdate_range("2023-01-01", periods=n)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame(
        {
            "Open": close + np.random.randn(n) * 0.1,
            "High": close + abs(np.random.randn(n) * 0.5),
            "Low": close - abs(np.random.randn(n) * 0.5),
            "Close": close,
            "Volume": np.random.randint(1_000_000, 10_000_000, n),
        },
        index=dates,
    )


class TestNoLookahead:
    def test_history_never_includes_future(self):
        df = make_sample_data(100)
        handler = DataHandler("TEST", df)

        for i in range(100):
            handler.get_next_bar()
            history = handler.get_history(50)

            # History should never include bars beyond current index
            assert len(history) <= handler.index
            if len(history) > 0:
                assert history.index[-1] <= df.index[handler.index - 1]

    def test_get_next_bar_advances_index(self):
        df = make_sample_data(10)
        handler = DataHandler("TEST", df)

        assert handler.index == 0
        handler.get_next_bar()
        assert handler.index == 1

    def test_history_lookback_limited(self):
        df = make_sample_data(100)
        handler = DataHandler("TEST", df)

        # Advance 10 bars
        for _ in range(10):
            handler.get_next_bar()

        history = handler.get_history(5)
        assert len(history) == 5

        history = handler.get_history(50)
        assert len(history) == 10  # Can't get more than we've seen

    def test_has_next_respects_bounds(self):
        df = make_sample_data(5)
        handler = DataHandler("TEST", df)

        for _ in range(5):
            assert handler.has_next()
            handler.get_next_bar()

        assert not handler.has_next()
        assert handler.get_next_bar() is None

    def test_reset_works(self):
        df = make_sample_data(10)
        handler = DataHandler("TEST", df)

        for _ in range(10):
            handler.get_next_bar()

        handler.reset()
        assert handler.index == 0
        assert handler.has_next()
