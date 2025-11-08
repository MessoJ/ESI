import os
from datetime import date
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd
import requests


RAW_DIR = Path("data/raw")
CBK_OUT = RAW_DIR / "ke_cbk.csv"
KNBS_OUT = RAW_DIR / "ke_knbs.csv"
PMI_OUT = RAW_DIR / "ke_pmi.csv"

TE_BASE = "https://api.tradingeconomics.com"
TE_CRED = os.getenv("TE_CREDENTIALS", "guest:guest")


def ensure_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def _te_get_csv(path: str, params: Optional[dict] = None) -> Optional[pd.DataFrame]:
    params = params or {}
    params.setdefault("c", TE_CRED)
    params.setdefault("format", "CSV")
    url = f"{TE_BASE}{path}"
    r = requests.get(url, params=params, timeout=60)
    if r.status_code != 200 or not r.text or r.text.strip() == "":
        return None
    try:
        from io import StringIO
        df = pd.read_csv(StringIO(r.text))
        return df
    except Exception:
        return None


def _clean_hist(df: pd.DataFrame, date_col_candidates: List[str], value_col_candidates: List[str]) -> Optional[pd.DataFrame]:
    cols = {c.lower(): c for c in df.columns}
    dcol = next((cols[c] for c in cols if c in [c2.lower() for c2 in date_col_candidates]), None)
    vcol = next((cols[c] for c in cols if c in [c2.lower() for c2 in value_col_candidates]), None)
    if not dcol or not vcol:
        # TE uses 'Date'/'Value' most of the time
        dcol = cols.get('date', None)
        vcol = cols.get('value', None)
    if not dcol or not vcol:
        return None
    out = df[[dcol, vcol]].copy()
    out.columns = ['date', 'value']
    out['date'] = pd.to_datetime(out['date'])
    out = out.dropna().sort_values('date')
    return out


def fetch_te_indicator_hist(country: str, indicator_candidates: Iterable[str]) -> Optional[pd.DataFrame]:
    # Prefer CSV historical indicator endpoint
    from urllib.parse import quote
    for ind in indicator_candidates:
        enc = quote(ind, safe="")
        path = f"/historical/indicator/{enc}/{country}"
        df = _te_get_csv(path, params={"d1": "2010-01-01", "d2": date.today().isoformat()})
        if df is not None and not df.empty:
            cleaned = _clean_hist(df, ["Date"], ["Value"]) or _clean_hist(df, ["date"], ["value"])
            if cleaned is not None and not cleaned.empty:
                return cleaned
    # Fallback to another variant
    for ind in indicator_candidates:
        enc = quote(ind, safe="")
        path = f"/historical/country/{country}/indicator/{enc}"
        df = _te_get_csv(path, params={"d1": "2010-01-01", "d2": date.today().isoformat()})
        if df is not None and not df.empty:
            cleaned = _clean_hist(df, ["Date"], ["Value"]) or _clean_hist(df, ["date"], ["value"])
            if cleaned is not None and not cleaned.empty:
                return cleaned
    return None


def upsert_csv(path: Path, df: pd.DataFrame, rename: Tuple[str, str]):
    # rename value column
    df = df.rename(columns={"value": rename[1]})
    if path.exists():
        old = pd.read_csv(path)
        old['date'] = pd.to_datetime(old['date'])
        df['date'] = pd.to_datetime(df['date'])
        merged = (
            pd.concat([old, df], ignore_index=True)
            .drop_duplicates(subset=['date'], keep='last')
            .sort_values('date')
        )
        merged.to_csv(path, index=False)
    else:
        df.to_csv(path, index=False)


def main():
    ensure_dirs()

    # 10Y Government Bond yield
    ten_y = fetch_te_indicator_hist(
        country="kenya",
        indicator_candidates=[
            "government bond 10y",
            "government bond 10 year",
            "10 year government bond",
            "kenya 10 year government bond yield",
        ],
    )
    if ten_y is not None:
        upsert_csv(CBK_OUT, ten_y, rename=("value", "ke_10y_bond_yield"))

    # 91-day Treasury Bill (3M)
    bill_3m = fetch_te_indicator_hist(
        country="kenya",
        indicator_candidates=[
            "treasury bill 3m",
            "treasury bills 3 months",
            "3 month t-bill",
            "treasury bill 91 day",
        ],
    )
    if bill_3m is not None:
        upsert_csv(CBK_OUT, bill_3m, rename=("value", "ke_91d_bill_yield"))

    # CPI (index) — try CPI or Consumer Price Index
    cpi = fetch_te_indicator_hist(
        country="kenya",
        indicator_candidates=[
            "consumer price index cpi",
            "cpi",
        ],
    )
    if cpi is not None:
        upsert_csv(KNBS_OUT, cpi, rename=("value", "cpi"))

    # Unemployment rate
    ur = fetch_te_indicator_hist(
        country="kenya",
        indicator_candidates=[
            "unemployment rate",
        ],
    )
    if ur is not None:
        upsert_csv(KNBS_OUT, ur, rename=("value", "unemployment"))

    # PMI
    pmi = fetch_te_indicator_hist(
        country="kenya",
        indicator_candidates=[
            "manufacturing pmi",
            "pmi",
        ],
    )
    if pmi is not None:
        upsert_csv(PMI_OUT, pmi, rename=("value", "pmi"))

    print("Done. Written:")
    for p in [CBK_OUT, KNBS_OUT, PMI_OUT]:
        print(" -", p, p.exists() and p.stat().st_size, "bytes")


if __name__ == "__main__":
    main()


