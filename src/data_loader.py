"""
Data Loader module for Brent Oil Price Analysis.
Handles CSV loading, preprocessing, and summary statistics.
"""

import pandas as pd
from pathlib import Path
from typing import Dict
import logging

from config import DataConfig

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Loads and preprocesses financial time series data.
    """

    def __init__(self, config: DataConfig):
        """
        Initialize DataLoader with configuration.

        Args:
            config: DataConfig object containing file path and column names
        """
        self.config = config
        self.data: pd.DataFrame = pd.DataFrame()

    def load_data(self) -> pd.DataFrame:
        """
        Load CSV data and preprocess.

        Returns:
            pd.DataFrame with datetime index and price column
        """
        data_path = Path(self.config.data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        # Load CSV
        df = pd.read_csv(data_path)
        logger.info(f"CSV loaded with {len(df)} rows.")

        # Validate columns
        if self.config.date_column not in df.columns:
            raise ValueError(f"Date column '{self.config.date_column}' not found in data.")
        if self.config.price_column not in df.columns:
            raise ValueError(f"Price column '{self.config.price_column}' not found in data.")

        # Convert date column to datetime
        df[self.config.date_column] = pd.to_datetime(df[self.config.date_column])
        df = df.sort_values(by=self.config.date_column)

        # Set datetime index
        df.set_index(self.config.date_column, inplace=True)

        # Keep only price column
        df = df[[self.config.price_column]].copy()
        df.rename(columns={self.config.price_column: "Price"}, inplace=True)

        self.data = df
        return self.data

    def get_summary_statistics(self) -> Dict[str, float]:
        """
        Calculate basic summary statistics for the price series.

        Returns:
            Dictionary with summary statistics
        """
        if self.data.empty:
            raise ValueError("No data loaded. Please call load_data() first.")

        price_series = self.data["Price"]

        stats = {
            "count": int(price_series.count()),
            "min": float(price_series.min()),
            "max": float(price_series.max()),
            "mean": float(price_series.mean()),
            "median": float(price_series.median()),
            "std": float(price_series.std()),
            "start_date": str(price_series.index.min().date()),
            "end_date": str(price_series.index.max().date())
        }

        logger.info(f"Summary statistics calculated: {stats}")
        return stats
