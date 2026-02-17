"""
Walk-forward validation with purged gap to prevent data leakage.
"""

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardSplit:
    """A single train/test split with metadata."""
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_size: int
    test_size: int


class WalkForwardValidator:
    """Generates time-respecting train/test splits with a purged gap."""

    def __init__(
        self,
        train_months: int = 24,
        test_months: int = 3,
        gap_days: int = 5,
        expanding: bool = False,
    ):
        self.train_months = train_months
        self.test_months = test_months
        self.gap_days = gap_days
        self.expanding = expanding

    def split(self, df: pd.DataFrame) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """Generate walk-forward splits.

        Args:
            df: DataFrame with DatetimeIndex.

        Returns:
            List of (train_df, test_df) tuples.
        """
        splits = []
        dates = df.index
        first_date = dates[0]
        train_start = first_date

        fold = 0
        while True:
            if self.expanding:
                t_start = first_date
            else:
                t_start = train_start

            train_end = train_start + pd.DateOffset(months=self.train_months)
            gap_end = train_end + pd.Timedelta(days=self.gap_days)
            test_end = gap_end + pd.DateOffset(months=self.test_months)

            if test_end > dates[-1]:
                break

            train_mask = (dates >= t_start) & (dates < train_end)
            test_mask = (dates >= gap_end) & (dates < test_end)

            if train_mask.sum() > 0 and test_mask.sum() > 0:
                fold += 1
                splits.append((df[train_mask], df[test_mask]))
                logger.info(
                    f"  Fold {fold}: train={train_mask.sum()} bars, "
                    f"test={test_mask.sum()} bars, "
                    f"gap={self.gap_days}d"
                )

            train_start += pd.DateOffset(months=self.test_months)

        logger.info(f"Generated {len(splits)} walk-forward splits")
        return splits

    def get_split_info(self, df: pd.DataFrame) -> list[WalkForwardSplit]:
        """Return metadata about splits without the actual data."""
        splits = self.split(df)
        info = []
        for i, (train, test) in enumerate(splits):
            info.append(WalkForwardSplit(
                fold=i + 1,
                train_start=str(train.index[0].date()),
                train_end=str(train.index[-1].date()),
                test_start=str(test.index[0].date()),
                test_end=str(test.index[-1].date()),
                train_size=len(train),
                test_size=len(test),
            ))
        return info
