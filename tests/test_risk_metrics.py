"""
Unit tests for risk metrics module.
"""
import pytest
import pandas as pd
import numpy as np

from src.risk_metrics import RiskMetricsCalculator


class TestRiskMetricsCalculator:
    """Test suite for RiskMetricsCalculator class."""

    @pytest.fixture
    def sample_returns(self):
        """Sample daily returns series."""
        np.random.seed(42)
        returns = np.random.randn(100) / 100  # small daily returns ~1%
        return pd.Series(returns, name="Returns")

    @pytest.fixture
    def constant_returns(self):
        """Series with constant returns."""
        return pd.Series([0.01] * 100, name="Returns")

    @pytest.fixture
    def negative_returns(self):
        """Series with negative returns."""
        return pd.Series([-0.01, -0.02, -0.005, -0.03, -0.01], name="Returns")

    def test_var_cvar_calculation(self, sample_returns):
        """Test Value at Risk (VaR) and Conditional VaR (CVaR) calculation."""
        calculator = RiskMetricsCalculator(sample_returns)
        var_95 = calculator.calculate_var(confidence_level=0.95)
        cvar_95 = calculator.calculate_cvar(confidence_level=0.95)
        
        assert isinstance(var_95, float)
        assert isinstance(cvar_95, float)
        assert var_95 <= 0  # VaR is negative for losses
        assert cvar_95 <= 0

    def test_volatility(self, sample_returns):
        """Test volatility calculation."""
        calculator = RiskMetricsCalculator(sample_returns)
        vol = calculator.calculate_volatility()
        assert vol > 0
        assert isinstance(vol, float)

    def test_max_drawdown(self, sample_returns):
        """Test maximum drawdown calculation."""
        calculator = RiskMetricsCalculator(sample_returns)
        mdd = calculator.calculate_max_drawdown()
        assert mdd <= 0
        assert isinstance(mdd, float)

    def test_sharpe_ratio(self, sample_returns):
        """Test Sharpe ratio calculation."""
        calculator = RiskMetricsCalculator(sample_returns)
        sharpe = calculator.calculate_sharpe_ratio(risk_free_rate=0.0)
        assert isinstance(sharpe, float)

    def test_empty_series(self):
        """Test handling of empty returns series."""
        empty_series = pd.Series([], dtype=float)
        calculator = RiskMetricsCalculator(empty_series)
        assert calculator.calculate_var() is None
        assert calculator.calculate_cvar() is None
        assert calculator.calculate_volatility() is None
        assert calculator.calculate_max_drawdown() is None
        assert calculator.calculate_sharpe_ratio() is None

    def test_constant_series(self, constant_returns):
        """Test metrics for constant returns (volatility=0)."""
        calculator = RiskMetricsCalculator(constant_returns)
        vol = calculator.calculate_volatility()
        sharpe = calculator.calculate_sharpe_ratio()
        assert vol == 0
        assert sharpe == float('inf') or sharpe > 1e6  # depending on implementation

    def test_negative_returns(self, negative_returns):
        """Test metrics for series with only negative returns."""
        calculator = RiskMetricsCalculator(negative_returns)
        var = calculator.calculate_var()
        cvar = calculator.calculate_cvar()
        mdd = calculator.calculate_max_drawdown()
        assert var <= 0
        assert cvar <= 0
        assert mdd <= 0
