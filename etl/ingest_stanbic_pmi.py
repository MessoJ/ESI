import os
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from io import StringIO

RAW_PATH = Path("data/raw/ke_pmi.csv")


def _ensure_dirs():
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)


def fetch_pmi_via_api(api_url: str, api_key: Optional[str] = None) -> pd.DataFrame:
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    r = requests.get(api_url, headers=headers, timeout=60)
    r.raise_for_status()
    # accept CSV or JSON
    if 'application/json' in r.headers.get('Content-Type',''):
        js = r.json()
        df = pd.DataFrame(js)
    else:
        df = pd.read_csv(StringIO(r.text))
    df.columns = [c.lower() for c in df.columns]
    if 'date' not in df.columns or 'pmi' not in df.columns:
        raise ValueError('PMI API must return columns: date, pmi')
    df['date'] = pd.to_datetime(df['date']).dt.to_period('M').dt.to_timestamp()
    return df[['date', 'pmi']].sort_values('date')


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


