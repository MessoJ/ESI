import csv
import datetime as dt
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import requests
from io import StringIO

RAW_PATH = Path("data/raw/ke_knbs.csv")


def _ensure_dirs():
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)


def fetch_knbs_open_api(cpi_url: str | None = None, ur_url: str | None = None) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    if cpi_url:
        r = requests.get(cpi_url, timeout=60)
        r.raise_for_status()
        df_cpi = pd.read_csv(StringIO(r.text)) if cpi_url.endswith('.csv') else pd.DataFrame(r.json())
        df_cpi.columns = [c.lower() for c in df_cpi.columns]
        # expect columns: date, cpi
        if 'date' in df_cpi.columns and 'cpi' in df_cpi.columns:
            df_cpi['date'] = pd.to_datetime(df_cpi['date']).dt.to_period('M').dt.to_timestamp()
            parts.append(df_cpi[['date', 'cpi']])
    if ur_url:
        r = requests.get(ur_url, timeout=60)
        r.raise_for_status()
        df_ur = pd.read_csv(StringIO(r.text)) if ur_url.endswith('.csv') else pd.DataFrame(r.json())
        df_ur.columns = [c.lower() for c in df_ur.columns]
        if 'date' in df_ur.columns and 'unemployment' in df_ur.columns:
            df_ur['date'] = pd.to_datetime(df_ur['date']).dt.to_period('M').dt.to_timestamp()
            parts.append(df_ur[['date', 'unemployment']])
    if not parts:
        return pd.DataFrame(columns=['date', 'cpi', 'unemployment'])
    df = parts[0]
    for p in parts[1:]:
        df = df.merge(p, on='date', how='outer')
    return df.sort_values('date')


def parse_monthly_release_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]
    # allow columns like year, month, cpi, unemployment
    if {'year', 'month'}.issubset(df.columns):
        df['date'] = pd.to_datetime(dict(year=df['year'], month=df['month'], day=1))
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date']).dt.to_period('M').dt.to_timestamp()
    else:
        raise ValueError('CSV must include year+month or date columns')
    cols = ['date']
    if 'cpi' in df.columns:
        cols.append('cpi')
    if 'unemployment' in df.columns:
        cols.append('unemployment')
    return df[cols].sort_values('date')


def write_csv(df: pd.DataFrame):
    _ensure_dirs()
    if RAW_PATH.exists():
        old = pd.read_csv(RAW_PATH)
        old['date'] = pd.to_datetime(old['date'])
        df = (
            pd.concat([old, df], ignore_index=True)
            .drop_duplicates(subset=['date'], keep='last')
            .sort_values('date')
        )
    df.to_csv(RAW_PATH, index=False)


if __name__ == '__main__':
    # Placeholder for manual invocation if open data endpoints known
    print('KNBS ingestion module. Use fetch_knbs_open_api or parse_monthly_release_csv.')


