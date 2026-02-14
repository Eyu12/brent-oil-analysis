"""
Change Point Detection for Brent Oil Prices.
Supports multiple algorithms: PELT, Binary Segmentation, Sliding Window, Dynamic Programming.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
import numpy as np
import pandas as pd
import logging

# Optional: import change point detection packages
try:
    import ruptures as rpt
except ImportError:
    raise ImportError("Please install ruptures package: pip install ruptures")

logger = logging.getLogger(__name__)


@dataclass
class ChangePoint:
    """
    Dataclass representing a detected change point in price series.
    """
    date: pd.Timestamp
    price_before: float
    price_after: float
    change_pct: float
    direction: str  # 'Increase' or 'Decrease'
    significance: str  # 'High', 'Medium', 'Low', 'Minor'
    confidence: float  # 0-1
    volatility_before: float = 0.0
    volatility_after: float = 0.0


class ChangePointMethod(Enum):
    PELT = "pelt"
    BINARY_SEG = "binary_seg"
    WINDOW_SLIDING = "window"
    DYNAMIC_PROGRAMMING = "dynp"


class ChangePointDetector:
    """
    Detects change points in time series data using multiple methods.
    """

    def __init__(self, config):
        """
        Initialize detector with configuration.

        Args:
            config: Configuration object (can include penalty, min_size, jump, etc.)
        """
        self.config = config

    def detect(
        self,
        data: pd.DataFrame,
        method: ChangePointMethod = ChangePointMethod.PELT
    ) -> List[ChangePoint]:
        """
        Detect change points in a price series.

        Args:
            data: DataFrame with 'Price' column
            method: Detection method from ChangePointMethod enum

        Returns:
            List of ChangePoint objects
        """
        if "Price" not in data.columns:
            raise ValueError("Data must contain 'Price' column")

        prices = data["Price"].values
        dates = data.index

        # Choose detection method
        if method == ChangePointMethod.PELT:
            algo = rpt.Pelt(model="rbf", min_size=getattr(self.config, 'min_size', 5))
        elif method == ChangePointMethod.BINARY_SEG:
            algo = rpt.Binseg(model="rbf", min_size=getattr(self.config, 'min_size', 5))
        elif method == ChangePointMethod.WINDOW_SLIDING:
            algo = rpt.Window(width=getattr(self.config, 'window_size', 10), model="rbf")
        elif method == ChangePointMethod.DYNAMIC_PROGRAMMING:
            algo = rpt.Dynp(model="rbf", min_size=getattr(self.config, 'min_size', 5))
        else:
            raise ValueError(f"Unsupported method: {method}")

        # Fit and detect change points
        algo.fit(prices)
        pen = getattr(self.config, 'penalty', 3)
        try:
            if method == ChangePointMethod.DYNAMIC_PROGRAMMING:
                # For Dynp, specify number of change points if known
                n_bkps = getattr(self.config, 'n_bkps', 5)
                bkps = algo.predict(n_bkps=n_bkps)
            else:
                bkps = algo.predict(pen=pen)
        except Exception as e:
            logger.warning(f"Change point detection failed: {e}")
            bkps = []

        # Build ChangePoint objects
        change_points: List[ChangePoint] = []

        start_idx = 0
        rolling_vol = pd.Series(prices).rolling(window=getattr(self.config, 'vol_window', 20)).std()
        rolling_vol = rolling_vol.fillna(method='bfill')

        for idx in bkps:
            if idx >= len(prices):
                continue

            price_before = prices[start_idx] if start_idx >= 0 else prices[0]
            price_after = prices[idx - 1]

            change_pct = ((price_after - price_before) / price_before) * 100
            direction = "Increase" if change_pct > 0 else "Decrease"

            # Assign significance
            abs_change = abs(change_pct)
            if abs_change > getattr(self.config, 'high_threshold', 5):
                significance = "High"
            elif abs_change > getattr(self.config, 'medium_threshold', 2):
                significance = "Medium"
            elif abs_change > getattr(self.config, 'low_threshold', 0.5):
                significance = "Low"
            else:
                significance = "Minor"

            # Confidence as normalized change
            confidence = min(abs_change / getattr(self.config, 'max_expected_change', 10), 1.0)

            volatility_before = float(rolling_vol[start_idx])
            volatility_after = float(rolling_vol[idx - 1])

            cp = ChangePoint(
                date=dates[idx - 1],
                price_before=float(price_before),
                price_after=float(price_after),
                change_pct=float(change_pct),
                direction=direction,
                significance=significance,
                confidence=float(confidence),
                volatility_before=volatility_before,
                volatility_after=volatility_after
            )
            change_points.append(cp)
            start_idx = idx

        logger.info(f"Detected {len(change_points)} change points using {method.value.upper()}")
        return change_points
