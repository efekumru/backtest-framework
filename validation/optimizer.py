"""
Strategy parameter optimization with walk-forward cross-validation.
"""

import itertools
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result from a single parameter combination."""
    params: dict
    avg_sharpe: float
    avg_return: float
    avg_max_dd: float
    n_folds: int


class ParameterOptimizer:
    """Grid search over strategy parameters with walk-forward CV."""

    def __init__(self, param_grid: dict[str, list[Any]]):
        self.param_grid = param_grid
        self.results: list[OptimizationResult] = []

    def _generate_combinations(self) -> list[dict]:
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        return [dict(zip(keys, v)) for v in itertools.product(*values)]

    def optimize(self, backtest_fn, data, validator) -> OptimizationResult:
        """Run grid search.

        Args:
            backtest_fn: Function(params, train_data, test_data) -> dict with metrics
            data: Full dataset
            validator: WalkForwardValidator instance

        Returns:
            Best OptimizationResult by average Sharpe ratio.
        """
        combinations = self._generate_combinations()
        splits = validator.split(data)

        logger.info(
            f"Optimizing {len(combinations)} parameter combinations "
            f"across {len(splits)} folds"
        )

        self.results = []

        for params in combinations:
            fold_sharpes = []
            fold_returns = []
            fold_dds = []

            for train_df, test_df in splits:
                try:
                    metrics = backtest_fn(params, train_df, test_df)
                    fold_sharpes.append(metrics.get("sharpe_ratio", 0))
                    fold_returns.append(metrics.get("total_return_pct", 0))
                    fold_dds.append(metrics.get("max_drawdown_pct", 0))
                except Exception as e:
                    logger.warning(f"Fold failed for {params}: {e}")

            if fold_sharpes:
                result = OptimizationResult(
                    params=params,
                    avg_sharpe=round(np.mean(fold_sharpes), 3),
                    avg_return=round(np.mean(fold_returns), 2),
                    avg_max_dd=round(np.mean(fold_dds), 2),
                    n_folds=len(fold_sharpes),
                )
                self.results.append(result)

        self.results.sort(key=lambda r: r.avg_sharpe, reverse=True)

        if self.results:
            best = self.results[0]
            logger.info(
                f"Best params: {best.params} "
                f"(Sharpe={best.avg_sharpe}, Return={best.avg_return}%)"
            )
            return best

        raise ValueError("No valid results from optimization")

    def top_n(self, n: int = 5) -> list[OptimizationResult]:
        return self.results[:n]
