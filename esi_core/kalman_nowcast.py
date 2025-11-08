import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.kalman_filter import KalmanFilter


def simple_nowcast_monthly_to_daily(monthly: pd.Series) -> pd.Series:
    """Spread monthly values evenly across days within the month (step function).
    This is a placeholder for a more sophisticated Kalman model.
    """
    m = monthly.copy()
    m.index = pd.to_datetime(m.index).to_period('M').to_timestamp()
    daily_index = pd.date_range(m.index.min(), m.index.max() + pd.offsets.MonthEnd(0), freq='D')
    s = pd.Series(index=daily_index, dtype=float)
    for d in s.index:
        mkey = pd.Timestamp(d.year, d.month, 1)
        if mkey in m.index:
            s.loc[d] = m.loc[mkey]
    return s



