"""
Project-tailored utilities for Brent oil analysis, risk metrics, and change point detection.

"""
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from functools import wraps
import time
import hashlib
import pickle
from contextlib import contextmanager

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

def setup_logging(
    level: str = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    format_string: Optional[str] = None
) -> None:
    """Setup logging configuration."""
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=handlers
    )

class LoggerMixin:
    """Mixin for easy logger access."""
    @property
    def logger(self):
        return logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")

# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------
def timer(func):
    """Measure execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.debug(f"{func.__name__} executed in {elapsed:.4f}s")
        return result
    return wrapper

def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Retry decorator with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    logger.warning(
                        f"Attempt {attempt+1}/{max_attempts} failed for {func.__name__}: {e}"
                    )
                    if attempt < max_attempts - 1:
                        time.sleep(current_delay)
                        current_delay *= backoff
            raise last_exception
        return wrapper
    return decorator

def validate_input(*validators):
    """Decorator to validate function inputs."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            all_args = list(args) + list(kwargs.values())
            for i, validator in enumerate(validators):
                if i < len(all_args) and not validator(all_args[i]):
                    raise ValueError(f"Argument {i} failed validation in {func.__name__}")
            return func(*args, **kwargs)
        return wrapper
    return decorator

# ---------------------------------------------------------------------------
# Context Managers
# ---------------------------------------------------------------------------
@contextmanager
def temporary_change(obj, attr, new_value):
    old_value = getattr(obj, attr, None)
    setattr(obj, attr, new_value)
    try:
        yield
    finally:
        setattr(obj, attr, old_value)

@contextmanager
def timeit(name: str = "Operation"):
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        logger.info(f"{name} took {elapsed:.2f}s")

# ---------------------------------------------------------------------------
# Cache Manager
# ---------------------------------------------------------------------------
class CacheManager:
    """File-based cache manager with TTL."""
    def __init__(self, cache_dir: Union[str, Path] = ".cache", ttl: int = 3600):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl

    def _get_cache_path(self, key: str) -> Path:
        hashed = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{hashed}.pkl"

    def get(self, key: str) -> Optional[Any]:
        path = self._get_cache_path(key)
        if not path.exists():
            return None
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            if time.time() - data['timestamp'] > self.ttl:
                path.unlink()
                return None
            return data['value']
        except Exception as e:
            logger.warning(f"Error reading cache {key}: {e}")
            return None

    def set(self, key: str, value: Any) -> None:
        path = self._get_cache_path(key)
        try:
            data = {'timestamp': time.time(), 'key': key, 'value': value}
            with open(path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"Error writing cache {key}: {e}")

    def clear(self, key: Optional[str] = None) -> None:
        if key:
            path = self._get_cache_path(key)
            if path.exists():
                path.unlink()
        else:
            for f in self.cache_dir.glob("*.pkl"):
                f.unlink()

    def get_stats(self) -> Dict[str, Any]:
        files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in files)
        return {'total_files': len(files), 'total_size_bytes': total_size, 'total_size_mb': total_size/(1024*1024)}

# ---------------------------------------------------------------------------
# Date & Time Utilities
# ---------------------------------------------------------------------------
def parse_date(date_input: Union[str, datetime, pd.Timestamp]) -> pd.Timestamp:
    if isinstance(date_input, pd.Timestamp):
        return date_input
    if isinstance(date_input, datetime):
        return pd.Timestamp(date_input)
    if isinstance(date_input, str):
        try:
            return pd.to_datetime(date_input)
        except Exception as e:
            raise ValueError(f"Cannot parse date: {date_input}") from e
    raise ValueError(f"Unsupported date type: {type(date_input)}")

def get_business_day_range(start_date, end_date, include_start=True, include_end=True) -> List[pd.Timestamp]:
    start, end = parse_date(start_date), parse_date(end_date)
    if start > end:
        raise ValueError("Start date after end date")
    days = pd.date_range(start=start, end=end, freq='B').tolist()
    if not include_start and days and days[0] == start:
        days = days[1:]
    if not include_end and days and days[-1] == end:
        days = days[:-1]
    return days

def get_trading_years(days: int) -> float:
    return days / 252.0

# ---------------------------------------------------------------------------
# Financial Calculations
# ---------------------------------------------------------------------------
def calculate_daily_returns(prices: pd.Series, method: str = 'simple') -> pd.Series:
    if method == 'simple':
        returns = prices.pct_change()
    elif method == 'log':
        returns = np.log(prices / prices.shift(1))
    else:
        raise ValueError(f"Unknown return method: {method}")
    return returns.dropna()

def calculate_annualized_volatility(returns: pd.Series, trading_days: int = 252) -> float:
    if len(returns) < 2:
        return 0.0
    return returns.std() * np.sqrt(trading_days)

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02, trading_days: int = 252) -> float:
    if len(returns) < 2:
        return 0.0
    daily_rf = (1 + risk_free_rate) ** (1 / trading_days) - 1
    excess = returns - daily_rf
    annual_return = excess.mean() * trading_days
    annual_vol = returns.std() * np.sqrt(trading_days)
    return 0.0 if annual_vol == 0 else annual_return / annual_vol

def calculate_max_drawdown(prices: pd.Series) -> Dict[str, Any]:
    running_max = prices.expanding().max()
    drawdown = (prices - running_max) / running_max
    trough_idx = drawdown.idxmin()
    prior_prices = prices[:trough_idx]
    peak_idx = prior_prices.idxmax() if len(prior_prices) > 0 else trough_idx
    peak_price = prior_prices.max() if len(prior_prices) > 0 else prices[trough_idx]
    recovery_idx = next((i for i in prices[trough_idx:].index if prices[i] >= peak_price), None)
    recovery_days = (recovery_idx - trough_idx).days if recovery_idx else None
    return {
        'max_drawdown': float(drawdown.min()),
        'peak_date': peak_idx,
        'trough_date': trough_idx,
        'recovery_date': recovery_idx,
        'recovery_days': recovery_days,
        'drawdown_series': drawdown
    }

def calculate_var_cvar(returns: pd.Series, alpha: float = 0.05) -> dict:
    sorted_returns = returns.sort_values()
    var = sorted_returns.quantile(alpha)
    cvar = sorted_returns[sorted_returns <= var].mean()
    return {'VaR': var, 'CVaR': cvar}

def calculate_rolling_beta(returns: pd.Series, market_returns: pd.Series, window: int = 60) -> pd.Series:
    df = pd.concat([returns, market_returns], axis=1).dropna()
    if len(df) < window:
        return pd.Series(index=df.index, dtype=float)
    rolling_cov = df.iloc[:,0].rolling(window).cov(df.iloc[:,1])
    rolling_var = df.iloc[:,1].rolling(window).var()
    return rolling_cov / rolling_var

def label_trend(before: float, after: float, threshold: float = 0.01) -> str:
    """Label trend between two points."""
    change = (after - before)/before
    if change > threshold:
        return "Uptrend"
    elif change < -threshold:
        return "Downtrend"
    else:
        return "Stable"

# ---------------------------------------------------------------------------
# Time Series Utilities
# ---------------------------------------------------------------------------
def align_series(asset: pd.Series, market: pd.Series) -> pd.DataFrame:
    df = pd.concat([asset, market], axis=1).dropna()
    df.columns = ['asset', 'market']
    return df

def generate_test_data(
    n_points: int = 1000,
    change_points: Optional[List[int]] = None,
    noise_level: float = 1.0,
    seed: Optional[int] = None
) -> pd.DataFrame:
    if seed: np.random.seed(seed)
    if change_points is None: change_points = [n_points//3, 2*n_points//3]
    dates = pd.date_range("2020-01-01", periods=n_points, freq='B')
    signal = np.zeros(n_points)
    level = 50.0
    last_cp = 0
    for cp in sorted(change_points + [n_points]):
        signal[last_cp:cp] = level
        signal[last_cp:cp] += np.cumsum(np.random.randn(cp-last_cp)*0.1)
        level += np.random.choice([-10,10])*np.random.uniform(0.5,1.5)
        last_cp = cp
    prices = np.maximum(signal + np.random.randn(n_points)*noise_level, 1.0)
    return pd.DataFrame({'Price': prices}, index=dates)

# ---------------------------------------------------------------------------
# Data Validation
# ---------------------------------------------------------------------------
class DataValidator:
    @staticmethod
    def check_missing_values(df: pd.DataFrame, threshold: float = 0.05) -> bool:
        return (df.isna().mean() <= threshold).all()
    
    @staticmethod
    def check_date_range(dates: pd.DatetimeIndex, min_date: str, max_date: str) -> Tuple[bool, Optional[str]]:
        min_dt, max_dt = pd.to_datetime(min_date), pd.to_datetime(max_date)
        if dates.min() < min_dt: return False, f"Date {dates.min()} before {min_dt}"
        if dates.max() > max_dt: return False, f"Date {dates.max()} after {max_dt}"
        return True, None
    
    @staticmethod
    def check_price_range(prices: pd.Series, min_price: float = 0.0, max_price: float = 1000.0) -> Tuple[bool, Optional[str]]:
        if (prices < min_price).any(): return False, f"Prices below {min_price}"
        if (prices > max_price).any(): return False, f"Prices above {max_price}"
        return True, None
    
    @staticmethod
    def check_monotonic_increasing(index: pd.Index) -> Tuple[bool, Optional[str]]:
        return (index.is_monotonic_increasing, "Index not monotonic") if not index.is_monotonic_increasing else (True,None)
    
    @staticmethod
    def check_unique_index(index: pd.Index) -> Tuple[bool, Optional[str]]:
        if not index.is_unique:
            return False, f"Index has duplicates: {index[index.duplicated()].tolist()[:5]}"
        return True, None

# ---------------------------------------------------------------------------
# File & Config Utilities
# ---------------------------------------------------------------------------
def load_config_from_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    filepath = Path(filepath)
    if not filepath.exists(): raise FileNotFoundError(filepath)
    with open(filepath,'r') as f: return json.load(f)

def save_config_to_json(config: Dict[str, Any], filepath: Union[str, Path]) -> None:
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath,'w') as f: json.dump(config,f,indent=2,default=str)

def ensure_directory(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_file_hash(filepath: Union[str, Path], algorithm: str = 'md5') -> str:
    filepath = Path(filepath)
    if not filepath.exists(): raise FileNotFoundError(filepath)
    hash_func = hashlib.new(algorithm)
    with open(filepath,'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''): hash_func.update(chunk)
    return hash_func.hexdigest()

def safe_file_write(filepath: Union[str, Path], content: str, encoding: str = 'utf-8') -> None:
    filepath = Path(filepath)
    if filepath.exists():
        backup = filepath.with_suffix(filepath.suffix+'.backup')
        filepath.rename(backup)
        try: filepath.write_text(content,encoding=encoding); backup.unlink()
        except Exception: backup.rename(filepath); raise
    else:
        filepath.write_text(content,encoding=encoding)

# ---------------------------------------------------------------------------
# Formatting Utilities
# ---------------------------------------------------------------------------
def format_currency(value: float, include_symbol: bool = True) -> str:
    return f"${value:,.2f}" if include_symbol else f"{value:,.2f}"

def format_percent(value: float, decimals: int = 2) -> str:
    return f"{value*100:.{decimals}f}%"

def format_large_number(value: float) -> str:
    if abs(value) >= 1e9: return f"{value/1e9:.1f}B"
    if abs(value) >= 1e6: return f"{value/1e6:.1f}M"
    if abs(value) >= 1e3: return f"{value/1e3:.1f}K"
    return f"{value:.2f}"

def truncate_string(s: str, max_length: int = 50, suffix: str = "...") -> str:
    return s if len(s) <= max_length else s[:max_length-len(suffix)] + suffix

# ---------------------------------------------------------------------------
# Outlier & Statistical Utilities
# ---------------------------------------------------------------------------
def calculate_outlier_bounds(data: pd.Series, method: str = 'iqr', multiplier: float = 1.5) -> Tuple[float,float]:
    if method == 'iqr':
        Q1,Q3 = data.quantile(0.25), data.quantile(0.75)
        IQR = Q3-Q1
        return Q1-multiplier*IQR, Q3+multiplier*IQR
    elif method=='zscore':
        mean,std = data.mean(),data.std()
        return mean-multiplier*std, mean+multiplier*std
    else:
        raise ValueError(f"Unknown method {method}")

def winsorize_series(data: pd.Series, limits: Tuple[float,float]=(0.01,0.01)) -> pd.Series:
    lower, upper = limits
    return data.clip(data.quantile(lower), data.quantile(1-upper))

# ---------------------------------------------------------------------------
# Exported symbols
# ---------------------------------------------------------------------------
__all__ = [
    'timer','retry','validate_input','temporary_change','timeit','CacheManager',
    'parse_date','get_business_day_range','get_trading_years','calculate_daily_returns',
    'calculate_annualized_volatility','calculate_sharpe_ratio','calculate_max_drawdown',
    'calculate_var_cvar','calculate_rolling_beta','label_trend','align_series',
    'DataValidator','generate_test_data','load_config_from_json','save_config_to_json',
    'ensure_directory','get_file_hash','safe_file_write','format_currency','format_percent',
    'format_large_number','truncate_string','calculate_outlier_bounds','winsorize_series',
    'LoggerMixin','setup_logging'
]
