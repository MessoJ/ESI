import numpy as np
import pandas as pd


def winsorize_rolling(series: pd.Series, low_q: float = 0.01, high_q: float = 0.99, window: int = 1260) -> pd.Series:
    """Apply rolling winsorization to reduce the effect of outliers.

    Uses rolling quantiles to clip values within [q_low, q_high] for each window.
    Defaults target approx 5-year trading days window.
    """
    s = series.copy()
    q_low = s.rolling(window, min_periods=min(60, window//10)).quantile(low_q)
    q_high = s.rolling(window, min_periods=min(60, window//10)).quantile(high_q)
    return s.clip(lower=q_low, upper=q_high)


def robust_z(series: pd.Series, window_days: int, min_periods: int) -> pd.Series:
    """Robust z-score using rolling median and median absolute deviation (MAD).

    z_t = (x_t - med_t) / (1.4826 * MAD_t + eps)
    where 1.4826 scales MAD to approximate std under normality.
    """
    s = series.astype(float)
    med = s.rolling(window_days, min_periods=min_periods).median()
    mad = (s - med).abs().rolling(window_days, min_periods=min_periods).median()
    return (s - med) / (1.4826 * mad.replace(0, np.nan) + 1e-12)


def standard_z(series: pd.Series, window_days: int, min_periods: int) -> pd.Series:
    """Classical rolling z-score with mean and std."""
    s = series.astype(float)
    mu = s.rolling(window_days, min_periods=min_periods).mean()
    sd = s.rolling(window_days, min_periods=min_periods).std()
    return (s - mu) / (sd + 1e-12)


def apply_standardization(series: pd.Series, method: str = 'robust', winsorize: bool = True, window_days: int = 1260, min_periods: int = 252) -> pd.Series:
    """Apply winsorization then selected standardization.

    - method: 'robust' (median/MAD) or 'standard' (mean/std)
    - winsorize: apply rolling winsorization first to reduce outliers effect
    - window_days: rolling window length in days
    - min_periods: minimum points in window to compute stats
    """
    s = series.copy()
    if winsorize:
        s = winsorize_rolling(s, window=window_days)
    if method == 'robust':
        return robust_z(s, window_days, min_periods)
    return standard_z(s, window_days, min_periods)


def rolling_stats(series: pd.Series, window_days: int) -> pd.DataFrame:
    """Return rolling stats: mu, sd, med, mad, p1, p99."""
    s = series.astype(float)
    mu = s.rolling(window_days, min_periods=min(60, window_days//10)).mean()
    sd = s.rolling(window_days, min_periods=min(60, window_days//10)).std()
    med = s.rolling(window_days, min_periods=min(60, window_days//10)).median()
    mad = (s - med).abs().rolling(window_days, min_periods=min(60, window_days//10)).median()
    p1 = s.rolling(window_days, min_periods=min(60, window_days//10)).quantile(0.01)
    p99 = s.rolling(window_days, min_periods=min(60, window_days//10)).quantile(0.99)
    return pd.DataFrame({'mu': mu, 'sd': sd, 'med': med, 'mad': mad, 'p1': p1, 'p99': p99})



