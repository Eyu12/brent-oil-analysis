"""
Unit tests for data_loader module.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

from src.data_loader import DataLoader, DataValidationError
from src.config import DataConfig


class TestDataLoader:
    """Test suite for DataLoader class."""

    @pytest.fixture
    def sample_config(self):
        """Create sample configuration for testing."""
        return DataConfig(
            date_column="Date",
            price_column="Price",
            min_price=0.0,
            max_price=500.0,
            min_date="1980-01-01",
            max_date="2025-12-31"
        )

    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample dataframe for testing."""
        dates = pd.date_range(start="2020-01-01", periods=100, freq='B')
        prices = 50 + np.cumsum(np.random.randn(100) * 0.5)
        return pd.DataFrame({
            'Date': dates.strftime('%Y-%m-%d'),
            'Price': prices
        })

    @pytest.fixture
    def csv_file(self, sample_dataframe, tmp_path):
        """Write sample dataframe to temporary CSV file."""
        file_path = tmp_path / "sample.csv"
        sample_dataframe.to_csv(file_path, index=False)
        return file_path

    def test_load_valid_data(self, sample_config, csv_file):
        loader = DataLoader(sample_config)
        data = loader.load_data(csv_file)

        assert isinstance(data, pd.DataFrame)
        assert 'Price' in data.columns
        assert data.index.name == 'Date'
        assert len(data) == 100

    def test_missing_columns(self, sample_config, csv_file):
        df = pd.read_csv(csv_file)
        df.drop(columns=['Price'], inplace=True)
        df.to_csv(csv_file, index=False)

        loader = DataLoader(sample_config)
        with pytest.raises(DataValidationError) as exc:
            loader.load_data(csv_file)
        assert "Missing required columns" in str(exc.value)

    def test_date_processing(self, sample_config, csv_file):
        loader = DataLoader(sample_config)
        data = loader.load_data(csv_file)
        assert isinstance(data.index, pd.DatetimeIndex)
        assert data.index.is_monotonic_increasing

    def test_price_out_of_range(self, sample_config, csv_file):
        df = pd.read_csv(csv_file)
        df.loc[0, 'Price'] = 1000  # Above max_price
        df.to_csv(csv_file, index=False)

        loader = DataLoader(sample_config)
        data = loader.load_data(csv_file)
        assert data['Price'].max() <= sample_config.max_price

    def test_missing_value_handling(self, sample_config, csv_file):
        df = pd.read_csv(csv_file)
        df.loc[10:15, 'Price'] = np.nan
        df.to_csv(csv_file, index=False)

        loader = DataLoader(sample_config)
        data = loader.load_data(csv_file)
        assert not data['Price'].isna().any()

    def test_summary_statistics(self, sample_config, csv_file):
        loader = DataLoader(sample_config)
        loader.load_data(csv_file)
        stats = loader.get_summary_statistics()

        assert 'start_date' in stats
        assert 'end_date' in stats
        assert 'total_days' in stats
        assert stats['total_days'] == 100

    def test_nonexistent_file(self, sample_config):
        loader = DataLoader(sample_config)
        with pytest.raises(FileNotFoundError):
            loader.load_data(Path("nonexistent.csv"))

    def test_date_range_filtering(self, sample_config, csv_file):
        df = pd.read_csv(csv_file)
        old_date = (datetime.now() - timedelta(days=365*50)).strftime('%Y-%m-%d')
        df.loc[0, 'Date'] = old_date
        df.to_csv(csv_file, index=False)

        loader = DataLoader(sample_config)
        data = loader.load_data(csv_file)
        assert data.index.min() >= pd.to_datetime(sample_config.min_date)

    def test_data_consistency(self, sample_config, csv_file):
        loader = DataLoader(sample_config)
        data = loader.load_data(csv_file)

        assert (data['Price'] > 0).all()
        assert data.index.is_unique
        assert data.index.is_monotonic_increasing
