"""
Unit tests for change point detection module.
"""
import pytest
import pandas as pd
import numpy as np

from src.change_point import ChangePointDetector, ChangePointMethod, ChangePoint
from src.config import ModelConfig


class TestChangePointDetector:
    """Test suite for ChangePointDetector class."""

    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return ModelConfig(
            pelt_penalty=1.0,
            min_segment_length=5,
            confidence_level=0.95,
            n_bootstrap=100
        )

    @pytest.fixture
    def sample_data_with_change_points(self):
        """Time series with known change points."""
        np.random.seed(42)
        dates = pd.date_range(start="2020-01-01", periods=200, freq='B')
        prices = []
        prices.extend(50 + np.random.randn(50) * 1)   # Regime 1
        prices.extend(60 + np.random.randn(50) * 1.5) # Change 1
        prices.extend(45 + np.random.randn(50) * 2)   # Change 2
        prices.extend(55 + np.random.randn(50) * 1.2) # Regime 4
        return pd.DataFrame({'Price': prices}, index=dates)

    @pytest.fixture
    def sample_data_trend(self):
        """Time series with a linear trend and noise."""
        np.random.seed(42)
        dates = pd.date_range(start="2020-01-01", periods=200, freq='B')
        trend = np.linspace(50, 70, 200)
        noise = np.random.randn(200) * 0.5
        prices = trend + noise
        return pd.DataFrame({'Price': prices}, index=dates)

    def test_pelt_detection(self, sample_config, sample_data_with_change_points):
        detector = ChangePointDetector(sample_config)
        cps = detector.detect(sample_data_with_change_points, ChangePointMethod.PELT)
        assert isinstance(cps, list) and len(cps) > 0
        cp = cps[0]
        assert isinstance(cp, ChangePoint)
        assert hasattr(cp, 'date')
        assert hasattr(cp, 'change_pct')
        assert hasattr(cp, 'confidence')
        assert hasattr(cp, 'significance')

    def test_binary_segmentation(self, sample_config, sample_data_with_change_points):
        detector = ChangePointDetector(sample_config)
        cps = detector.detect(sample_data_with_change_points, ChangePointMethod.BINARY_SEG)
        assert isinstance(cps, list) and len(cps) > 0

    def test_window_sliding(self, sample_config, sample_data_with_change_points):
        detector = ChangePointDetector(sample_config)
        cps = detector.detect(sample_data_with_change_points, ChangePointMethod.WINDOW_SLIDING)
        assert isinstance(cps, list)

    def test_consensus_detection(self, sample_config, sample_data_with_change_points):
        detector = ChangePointDetector(sample_config)
        detector.detect(sample_data_with_change_points, ChangePointMethod.PELT)
        detector.data = sample_data_with_change_points
        consensus = detector.get_consensus_change_points(
            methods=[ChangePointMethod.PELT, ChangePointMethod.BINARY_SEG],
            min_confidence=0.5
        )
        assert isinstance(consensus, list)

    def test_confidence_calculation(self, sample_config):
        detector = ChangePointDetector(sample_config)
        before = np.array([50, 51, 49, 50, 52])
        after = np.array([60, 61, 59, 60, 62])
        confidence = detector._calculate_confidence(before, after, 50, 60)
        assert 0 <= confidence <= 1
        assert confidence > 0.5

    def test_no_change_point(self, sample_config, sample_data_trend):
        detector = ChangePointDetector(sample_config)
        cps = detector.detect(sample_data_trend)
        if cps:
            avg_conf = np.mean([cp.confidence for cp in cps])
            assert avg_conf < 0.8

    def test_change_point_properties(self, sample_config):
        cp = ChangePoint(
            index=50,
            date=pd.Timestamp("2020-03-01"),
            price_before=50.0,
            price_after=60.0,
            change_pct=20.0,
            method=ChangePointMethod.PELT,
            confidence=0.95,
            segment_mean_before=49.5,
            segment_mean_after=59.5,
            volatility_before=1.0,
            volatility_after=2.0
        )
        assert cp.significance == "High"
        assert cp.direction == "Increase"

        cp_low = ChangePoint(
            index=50,
            date=pd.Timestamp("2020-03-01"),
            price_before=50.0,
            price_after=51.0,
            change_pct=2.0,
            method=ChangePointMethod.PELT,
            confidence=0.95,
            segment_mean_before=49.5,
            segment_mean_after=50.5,
            volatility_before=1.0,
            volatility_after=1.1
        )
        assert cp_low.significance == "Minor"

    def test_impact_metrics(self, sample_config, sample_data_with_change_points):
        detector = ChangePointDetector(sample_config)
        detector.detect(sample_data_with_change_points)
        impacts = detector.calculate_impact_metrics()
        assert isinstance(impacts, dict)
        assert 'total_change_points' in impacts
        assert 'avg_change_pct' in impacts
        assert 'avg_confidence' in impacts

    def test_edge_cases(self, sample_config):
        short_data = pd.DataFrame({'Price': [50, 51, 52, 53, 54]},
                                  index=pd.date_range("2020-01-01", periods=5))
        detector = ChangePointDetector(sample_config)
        cps = detector.detect(short_data)
        assert isinstance(cps, list)

    def test_constant_series(self, sample_config):
        constant_data = pd.DataFrame({'Price': [50] * 100},
                                     index=pd.date_range("2020-01-01", periods=100))
        detector = ChangePointDetector(sample_config)
        cps = detector.detect(constant_data)
        if cps:
            for cp in cps:
                assert cp.confidence < 0.5
