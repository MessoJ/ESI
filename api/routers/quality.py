import datetime as dt
from typing import Dict, List

import pandas as pd
from fastapi import APIRouter, HTTPException

from esi_core.compute_index import compute_components_cached
from etl.quality_checks import missing_values, sudden_jump, staleness

router = APIRouter()


@router.get("/api/index/quality")
def quality(country: str = "US"):
    df = compute_components_cached(country)
    if df.empty:
        raise HTTPException(status_code=404, detail="no data")
    # derive z columns
    zcols = [c for c in df.columns if c.startswith("z_")]
    series_cols = [c for c in df.columns if not c.startswith("z_")]
    qc = []
    qc += [r.__dict__ for r in missing_values(df, series_cols)]
    qc += [r.__dict__ for r in sudden_jump(df[zcols])]
    # staleness thresholds
    max_days = {
        "VIX": 5,
        "KE_10Y_BOND": 14,
        "KE_91D_BILL": 14,
        "CPI": 60,
        "KE_CPI": 60,
        "UNRATE": 60,
        "KE_UNRATE": 60,
        "KE_STANBIC_PMI": 60,
        "PMI": 60,
    }
    qc += [r.__dict__ for r in staleness(df, max_days)]
    return {"country": country, "ts": dt.datetime.utcnow().isoformat(), "checks": qc}



