"""
Configuration module for Brent Oil Change Point Detection and Risk Analysis.
Contains all default parameters and paths.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# =========================
# Data Configuration
# =========================

@dataclass
class DataConfig:
    data_path: Path = Path("data/brent.csv")
    date_column: str = "Date"
    price_column: str = "Price"


# =========================
# Model / Change Point Configuration
# =========================

@dataclass
class ModelConfig:
    method: str = "pelt"  # default method: pelt, can be overridden in CLI
    penalty: float = 10.0  # penalty for PELT method
    segment_window: int = 30  # window size for sliding window method
    n_bkps: int = 5  # max breakpoints for Binary Seg / Dynamic Programming
    signal_type: str = "returns"  # options: price, returns, volatility
    volatility_window: int = 30  # window for volatility calculation


# =========================
# Risk Analysis Configuration
# =========================

@dataclass
class RiskConfig:
    volatility_window: int = 30
    volatility_min_periods: int = 10
    var_confidence_levels: List[float] = field(default_factory=lambda: [0.95, 0.99])
    stress_test_shocks: List[float] = field(default_factory=lambda: [-0.10, -0.05, 0.05, 0.10])  # 10%,5% shocks


# =========================
# Application Configuration
# =========================

@dataclass
class AppConfig:
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    risk: RiskConfig = RiskConfig()

    def __post_init__(self):
        # Validate paths
        if not isinstance(self.data.data_path, Path):
            self.data.data_path = Path(self.data.data_path)
        if not 0 < self.model.penalty < 1e6:
            raise ValueError("Penalty must be positive and reasonable.")
        if not self.model.volatility_window > 0:
            raise ValueError("Volatility window must be > 0.")
