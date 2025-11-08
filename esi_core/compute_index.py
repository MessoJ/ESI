import os
import json
import datetime as dt
from functools import lru_cache
from typing import Dict, Optional

import numpy as np
import pandas as pd
import requests
import time
import logging
import os
import redis
import yaml


CONFIG_PATH = os.path.join("config", "config.yaml")
COUNTRY_DIR = os.path.join("config", "countries")
FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"


def _load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_config() -> Dict:
    return _load_yaml(CONFIG_PATH)


def _load_country(country: str) -> Dict:
    path = os.path.join(COUNTRY_DIR, f"{country}.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing country config: {path}")
    return _load_yaml(path)


def _fred_obs(series_id: str, start: str = "1990-01-01") -> pd.DataFrame:
    api_key = os.getenv("FRED_API_KEY", "")
    # If no key, skip remote call in dev; return empty to allow synthetic fallback
    if not api_key:
        return pd.DataFrame(columns=["date", series_id])
    params = dict(series_id=series_id, api_key=api_key, file_type="json", observation_start=start)
    # retry with backoff
    backoff = 1.0
    last_exc: Exception | None = None
    for _ in range(4):
        try:
            r = requests.get(FRED_BASE, params=params, timeout=30)
            r.raise_for_status()
            break
        except Exception as e:
            last_exc = e
            time.sleep(backoff)
            backoff *= 2
    else:
        if last_exc:
            raise last_exc
    js = r.json()
    obs = js.get("observations", [])
    if not obs:
        return pd.DataFrame(columns=["date", series_id])
    df = pd.DataFrame(obs)
    df["date"] = pd.to_datetime(df["date"])
    df[series_id] = pd.to_numeric(df["value"], errors="coerce")
    return df[["date", series_id]].dropna()


def _load_ke_local_series(series_id: str) -> pd.DataFrame:
    """Load Kenyan local series from data/raw CSVs.

    Returns DataFrame with columns [date, series_id] or empty DataFrame if not available.
    """
    raw_cbk = os.path.join("data", "raw", "ke_cbk.csv")
    raw_knbs = os.path.join("data", "raw", "ke_knbs.csv")
    raw_pmi = os.path.join("data", "raw", "ke_pmi.csv")

    try:
        if series_id == "KE_10Y_BOND" and os.path.exists(raw_cbk):
            df = pd.read_csv(raw_cbk)
            df["date"] = pd.to_datetime(df["date"])
            col = None
            for candidate in ["ke_10y_bond_yield", "10y", "bond10", "bond_10y"]:
                if candidate in df.columns:
                    col = candidate
                    break
            if col is None:
                return pd.DataFrame(columns=["date", series_id])
            return df[["date", col]].rename(columns={col: series_id}).dropna()
        if series_id == "KE_91D_BILL" and os.path.exists(raw_cbk):
            df = pd.read_csv(raw_cbk)
            df["date"] = pd.to_datetime(df["date"])
            col = None
            for candidate in ["ke_91d_bill_yield", "91d", "tbill_91d", "tbill91"]:
                if candidate in df.columns:
                    col = candidate
                    break
            if col is None:
                return pd.DataFrame(columns=["date", series_id])
            return df[["date", col]].rename(columns={col: series_id}).dropna()
        if series_id == "KE_CPI" and os.path.exists(raw_knbs):
            df = pd.read_csv(raw_knbs)
            df["date"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()
            if "cpi" not in df.columns:
                return pd.DataFrame(columns=["date", series_id])
            return df[["date", "cpi"]].rename(columns={"cpi": series_id}).dropna()
        if series_id == "KE_UNRATE" and os.path.exists(raw_knbs):
            df = pd.read_csv(raw_knbs)
            df["date"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()
            for candidate in ["unemployment", "unrate", "ur"]:
                if candidate in df.columns:
                    return df[["date", candidate]].rename(columns={candidate: series_id}).dropna()
            return pd.DataFrame(columns=["date", series_id])
        if series_id == "KE_STANBIC_PMI" and os.path.exists(raw_pmi):
            df = pd.read_csv(raw_pmi)
            df["date"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()
            if "pmi" not in df.columns:
                return pd.DataFrame(columns=["date", series_id])
            return df[["date", "pmi"]].rename(columns={"pmi": series_id}).dropna()
    except Exception:
        # Fail soft with empty
        return pd.DataFrame(columns=["date", series_id])
    return pd.DataFrame(columns=["date", series_id])


def _make_calendar(start: str = "1990-01-01") -> pd.DataFrame:
    end = dt.date.today()
    idx = pd.date_range(start=start, end=end, freq="D")
    return pd.DataFrame(index=idx)


def _zscore(s: pd.Series, win: int, eps: float = 1e-9) -> pd.Series:
    mu = s.rolling(win, min_periods=min(252, win // 5)).mean()
    sd = s.rolling(win, min_periods=min(252, win // 5)).std()
    return (s - mu) / (sd + eps)


def _synthetic_baseline(days: int = 180) -> pd.DataFrame:
    idx = pd.date_range(end=dt.date.today(), periods=days, freq="D")
    zeros = pd.Series(0.0, index=idx)
    out = pd.DataFrame(
        {
            "ESI": zeros,
            "ESI_smoothed": zeros,
            "A_financial": zeros,
            "B_macro": zeros,
            "C_sentiment": zeros,
        },
        index=idx,
    )
    return out


def compute_components(country: str = "US") -> pd.DataFrame:
    cfg = _load_config()
    cc = _load_country(country)

    win_days = int(cfg.get("default_window_days", 1260))
    span = int(cfg.get("smoothing_span", 7))
    max_ffill_m = int(cfg.get("max_ffill_monthly_days", 40))
    max_ffill_w = int(cfg.get("max_ffill_weekly_days", 10))

    series_map = cc.get("series", {})

    frames: Dict[str, pd.DataFrame] = {}
    fred_key_present = bool(os.getenv("FRED_API_KEY"))
    for k, sid in series_map.items():
        # For Kenya, load local CSV series for KE_* ids; otherwise fetch from FRED
        if country.upper() == "KE" and sid.startswith("KE_"):
            df = _load_ke_local_series(sid)
        else:
            # If no FRED key, skip external series to avoid timeouts
            if country.upper() == "KE" and not fred_key_present:
                df = pd.DataFrame(columns=["date", sid])
            else:
                df = _fred_obs(sid)
        frames[k] = df.set_index("date") if not df.empty else pd.DataFrame(columns=[sid])

    # Build calendar starting from earliest available data to speed up first compute
    min_date = None
    for df in frames.values():
        if not df.empty:
            d0 = df.index.min()
            if pd.notna(d0):
                min_date = d0 if min_date is None else min(min_date, d0)
    if min_date is None:
        min_date = pd.Timestamp("2005-01-01")
    cal = pd.DataFrame(index=pd.date_range(start=min_date.normalize(), end=dt.date.today(), freq="D"))

    # Attach raw series with canonical component keys (k)
    for k, df in frames.items():
        if not df.empty:
            first_col = df.columns[0]
            cal[k] = df[first_col]

    # Transforms and forward-fills (country-agnostic; country config decides sources)
    cal["slope"] = (
        cal.get("DGS10").ffill(limit=max_ffill_m) if "DGS10" in cal else np.nan
    ) - (
        cal.get("TB3M").ffill(limit=max_ffill_m) if "TB3M" in cal else np.nan
    )

    if "CPI" in cal:
        cal["cpi_yoy"] = (np.log(cal["CPI"]).diff(12) * 100).ffill(limit=max_ffill_m)
    if "UNRATE" in cal:
        cal["ur"] = cal["UNRATE"].ffill(limit=max_ffill_m)
    if "PMI" in cal:
        cal["pmi"] = cal["PMI"].ffill(limit=max_ffill_m)

    # Equity drawdown
    if "SP500" in cal:
        px = cal["SP500"].ffill()
        roll_max = px.rolling(30, min_periods=5).max()
        cal["dd30"] = (px / roll_max - 1.0) * 100.0

    # Credit spread proxy
    if "BAA" in cal and "AAA" in cal:
        cal["baa_aaa"] = cal["BAA"].ffill(limit=60) - cal["AAA"].ffill(limit=60)

    # Weekly NFCI forward?fill up to configured days
    if "NFCI" in cal:
        cal["nfci"] = cal["NFCI"].ffill(limit=max_ffill_w)

    # Component alignment: higher = more stress
    comp = pd.DataFrame(index=cal.index)
    if "VIX" in cal:
        comp["vix"] = cal["VIX"]
    if "slope" in cal:
        comp["slope_stress"] = -cal["slope"]
    if "nfci" in cal:
        comp["nfci"] = cal["nfci"]
    if "ur" in cal:
        comp["ur"] = cal["ur"]
    if "cpi_yoy" in cal:
        comp["cpi_yoy"] = cal["cpi_yoy"]
    if "pmi" in cal:
        comp["pmi_stress"] = -(cal["pmi"] - 50.0)
    if "dd30" in cal:
        comp["dd30_stress"] = -cal["dd30"]
    if "baa_aaa" in cal:
        comp["baa_aaa"] = cal["baa_aaa"]

    # Z-scores
    z = comp.apply(lambda s: _zscore(s, win_days)) if not comp.empty else pd.DataFrame(index=cal.index)

    buckets = _load_config().get("buckets", {})
    weights = _load_config().get("weights", {"financial": 0.4, "macro": 0.4, "sentiment": 0.2})

    def _bucket_mean(cols):
        if not cols:
            return pd.Series(index=z.index, dtype=float)
        avail = [c for c in cols if c in z.columns]
        if not avail:
            return pd.Series(index=z.index, dtype=float)
        return z[avail].mean(axis=1)

    A = _bucket_mean(buckets.get("financial", [])).fillna(0.0)
    B = _bucket_mean(buckets.get("macro", [])).fillna(0.0)
    C = _bucket_mean(buckets.get("sentiment", [])).fillna(0.0)

    esi = weights.get("financial", 0.4) * A + weights.get("macro", 0.4) * B + weights.get("sentiment", 0.2) * C
    esi_smooth = esi.ewm(span=span, adjust=False, min_periods=3).mean()

    out = pd.DataFrame(
        {
            "ESI": esi,
            "ESI_smoothed": esi_smooth,
            "A_financial": A,
            "B_macro": B,
            "C_sentiment": C,
        },
        index=cal.index,
    )

    # If we couldn't compute anything (e.g., no FRED key), optionally return a synthetic zero baseline
    if out["ESI"].dropna().empty and os.getenv("ESI_ALLOW_SYNTHETIC_BASELINE", "true").lower() == "true":
        return _synthetic_baseline(days=180)

    z.columns = [f"z_{c}" for c in z.columns]
    return out.join(z)


@lru_cache(maxsize=8)
def _compute_cache_key(country: str) -> str:
    # trivial cache key by country and today
    return f"{country}:{dt.date.today().isoformat()}"


_DF_CACHE: Dict[str, pd.DataFrame] = {}
_REDIS = None
try:
    _REDIS = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    _REDIS.ping()
except Exception:
    _REDIS = None


def compute_components_cached(country: str = "US") -> pd.DataFrame:
    key = _compute_cache_key(country)
    # memory cache
    if key in _DF_CACHE:
        return _DF_CACHE[key]
    # redis cache
    if _REDIS is not None:
        try:
            blob = _REDIS.get(f"esi:history:{country.upper()}:{dt.date.today().isoformat()}")
            if blob:
                import io
                df = pd.read_json(io.BytesIO(blob), orient="split")
                # ensure datetime index after deserialization
                if not isinstance(df.index, pd.DatetimeIndex):
                    try:
                        df.index = pd.to_datetime(df.index)
                    except Exception:
                        pass
                _DF_CACHE[key] = df
                return df
        except Exception:
            pass
    # compute
    try:
        df = compute_components(country)
    except Exception as e:
        # fallback to previous redis
        if _REDIS is not None:
            try:
                blob = _REDIS.get(f"esi:history:{country.upper()}:{dt.date.today().isoformat()}")
                if blob:
                    import io
                    df = pd.read_json(io.BytesIO(blob), orient="split")
                    if not isinstance(df.index, pd.DatetimeIndex):
                        try:
                            df.index = pd.to_datetime(df.index)
                        except Exception:
                            pass
                    return df
            except Exception:
                pass
        raise
    # sanitize
    if "ESI" in df.columns:
        df = df.dropna(subset=["ESI"])  
    else:
        df = df.dropna()
    _DF_CACHE[key] = df
    # store in redis
    if _REDIS is not None and not df.empty:
        try:
            import io
            buf = io.BytesIO()
            df.to_json(buf, orient="split")
            _REDIS.set(f"esi:history:{country.upper()}:{dt.date.today().isoformat()}", buf.getvalue(), ex=3600)
        except Exception:
            pass
    return df


def build_latest_payload(df: pd.DataFrame) -> Dict:
    last = df.iloc[-1]
    return {
        "as_of": str(df.index[-1].date()),
        "esi": float(last["ESI"]),
        "esi_smoothed": float(last["ESI_smoothed"]),
        "buckets": {
            "financial": float(last.get("A_financial", np.nan)),
            "macro": float(last.get("B_macro", np.nan)),
            "sentiment": float(last.get("C_sentiment", np.nan)),
        },
    }


def build_history_payload(df: pd.DataFrame, start: Optional[str] = None):
    if start:
        try:
            start_dt = pd.to_datetime(start)
            df = df[df.index >= start_dt]
        except Exception:
            pass
    data = df.reset_index().rename(columns={"index": "date"})
    data["date"] = data["date"].dt.strftime("%Y-%m-%d")
    return json.loads(data.to_json(orient="records"))


def build_components_payload(df: pd.DataFrame) -> Dict:
    last = df.iloc[-1]
    keys = [c for c in df.columns if c.startswith("z_")]
    return {k: float(last[k]) for k in keys}


