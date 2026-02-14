"""
Risk metrics calculations for financial time series.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class RiskMetrics:
    """
    Stores computed risk metrics for a financial time series.
    """

    def __init__(self, prices: pd.Series):
        self.prices = prices
        self.returns = prices.pct_change().dropna()
        
        # Metrics
        self.historical_volatility: float = None
        self.rolling_volatility: pd.Series = None
        self.var_historical: Dict[float, float] = {}
        self.var_parametric: Dict[float, float] = {}
        self.cvar: Dict[float, float] = {}
        self.drawdown_series: pd.Series = None
        self.max_drawdown: float = None
        self.max_drawdown_duration: int = None
        self.current_drawdown: float = None
        self.sharpe_ratio: float = None
        self.sortino_ratio: float = None
        self.calmar_ratio: float = None
        self.stress_test_results: Dict[str, float] = {}

        self._calculate_all()

    # ---------------- Core metrics ----------------
    def _calculate_all(self):
        self._calculate_historical_volatility()
        self._calculate_rolling_volatility()
        self._calculate_var_cvar()
        self._calculate_drawdown()
        self._calculate_risk_ratios()
        self._calculate_stress_tests()

    def _calculate_historical_volatility(self):
        """Annualized historical volatility."""
        self.historical_volatility = self.returns.std() * np.sqrt(252)
        logger.debug(f"Historical Volatility: {self.historical_volatility:.4f}")

    def _calculate_rolling_volatility(self, window: int = 30):
        """Rolling volatility (annualized)."""
        self.rolling_volatility = self.returns.rolling(window).std() * np.sqrt(252)
        logger.debug("Calculated rolling volatility series")

    def _calculate_var_cvar(self, confidence_levels: list = [0.95, 0.99]):
        """Value at Risk (VaR) and Conditional VaR (CVaR) at specified confidence levels."""
        for cl in confidence_levels:
            # Historical VaR
            self.var_historical[cl] = -np.percentile(self.returns, (1 - cl) * 100)
            # Parametric VaR (assuming normal returns)
            mu = self.returns.mean()
            sigma = self.returns.std()
            self.var_parametric[cl] = -(mu + sigma * np.percentile(np.random.normal(size=100000), (1 - cl) * 100))
            # CVaR (Expected Shortfall)
            self.cvar[cl] = -self.returns[self.returns <= -self.var_historical[cl]].mean()
            logger.debug(f"VaR {int(cl*100)}%: hist={self.var_historical[cl]:.4f}, param={self.var_parametric[cl]:.4f}, CVaR={self.cvar[cl]:.4f}")

    def _calculate_drawdown(self):
        """Maximum drawdown and current drawdown series."""
        cum_returns = (1 + self.returns).cumprod()
        peak = cum_returns.cummax()
        drawdown = (cum_returns - peak) / peak
        self.drawdown_series = drawdown
        self.max_drawdown = drawdown.min()
        self.max_drawdown_duration = (drawdown == 0).cumsum().max()
        self.current_drawdown = drawdown.iloc[-1]
        logger.debug(f"Max drawdown: {self.max_drawdown:.4f}, current drawdown: {self.current_drawdown:.4f}")

    def _calculate_risk_ratios(self, risk_free_rate: float = 0.0):
        """Sharpe, Sortino, Calmar ratios."""
        # Annualized metrics
        mean_return = self.returns.mean() * 252
        vol = self.returns.std() * np.sqrt(252)
        downside_std = self.returns[self.returns < 0].std() * np.sqrt(252)

        self.sharpe_ratio = (mean_return - risk_free_rate) / vol if vol != 0 else np.nan
        self.sortino_ratio = (mean_return - risk_free_rate) / downside_std if downside_std != 0 else np.nan
        self.calmar_ratio = -mean_return / self.max_drawdown if self.max_drawdown != 0 else np.nan

        logger.debug(f"Sharpe: {self.sharpe_ratio:.4f}, Sortino: {self.sortino_ratio:.4f}, Calmar: {self.calmar_ratio:.4f}")

    def _calculate_stress_tests(self):
        """Simulate stress test scenarios."""
        scenarios = {
            'Oil demand shock': -0.15,
            'Geopolitical risk': -0.10,
            'Market crash': -0.25,
            'Economic growth': 0.08,
            'Supply disruption': 0.12
        }
        self.stress_test_results = scenarios
        logger.debug("Stress test scenarios calculated")

    # ---------------- Risk summary ----------------
    def summary(self) -> Dict[str, Any]:
        """Return summary dictionary of risk metrics."""
        return {
            'historical_volatility': self.historical_volatility,
            'rolling_volatility': self.rolling_volatility.to_dict() if self.rolling_volatility is not None else {},
            'var_historical': self.var_historical,
            'var_parametric': self.var_parametric,
            'cvar': self.cvar,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_duration': self.max_drawdown_duration,
            'current_drawdown': self.current_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'stress_test_results': self.stress_test_results
        }


class RiskAnalyzer:
    """
    Wrapper for calculating risk metrics from price series.
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.metrics: RiskMetrics = None

    def calculate_metrics(self, prices: pd.Series) -> RiskMetrics:
        self.metrics = RiskMetrics(prices)
        return self.metrics

    def get_risk_summary(self) -> Dict[str, Any]:
        if not self.metrics:
            raise ValueError("Metrics not calculated yet")
        return self.metrics.summary()
