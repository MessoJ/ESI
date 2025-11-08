import datetime as dt
from dataclasses import dataclass, asdict
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class QCRecord:
    run_id: str
    country: str
    series: str
    status: str  # ok|warn|fail
    detail: str
    timestamp: str


def missing_values(df: pd.DataFrame, series_cols: List[str]) -> List[QCRecord]:
    records: List[QCRecord] = []
    ts = dt.datetime.utcnow().isoformat()
    for s in series_cols:
        if s in df.columns:
            miss = int(df[s].isna().sum())
            status = 'ok' if miss == 0 else 'warn'
            records.append(QCRecord('-','- ', s, status, f'missing={miss}', ts))
    return records


def sudden_jump(zscores: pd.DataFrame, threshold: float = 6.0) -> List[QCRecord]:
    records: List[QCRecord] = []
    ts = dt.datetime.utcnow().isoformat()
    for c in zscores.columns:
        over = zscores[c].abs().max()
        status = 'ok' if over <= threshold else 'fail'
        records.append(QCRecord('-', '-', c, status, f'max_abs_z={over:.2f}', ts))
    return records


def staleness(df: pd.DataFrame, max_days: Dict[str, int]) -> List[QCRecord]:
    records: List[QCRecord] = []
    ts = dt.datetime.utcnow().isoformat()
    today = pd.Timestamp.today().normalize()
    for s, maxd in max_days.items():
        if s in df.columns:
            last_date = df[s].dropna().index.max()
            if pd.isna(last_date):
                records.append(QCRecord('-', '-', s, 'fail', 'no data', ts))
                continue
            days = (today - last_date.normalize()).days
            status = 'ok' if days <= maxd else 'fail'
            records.append(QCRecord('-', '-', s, status, f'stale_days={days}', ts))
    return records



