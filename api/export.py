import io
import os
import datetime as dt
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

from esi_core.compute_index import compute_components_cached, build_history_payload

router = APIRouter()


def _df_for_range(country: str, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    df = compute_components_cached(country)
    if start:
        try:
            start_dt = pd.to_datetime(start)
            df = df[df.index >= start_dt]
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid start date: {start}")
    if end:
        try:
            end_dt = pd.to_datetime(end)
            df = df[df.index <= end_dt]
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid end date: {end}")
    return df


def _to_csv_stream(df: pd.DataFrame) -> StreamingResponse:
    out = io.StringIO()
    df_reset = df.reset_index().rename(columns={"index": "date"})
    df_reset["date"] = df_reset["date"].dt.strftime("%Y-%m-%d")
    # Ensure column order: date, ESI, ESI_smoothed, buckets, then z_*
    cols = list(df_reset.columns)
    fixed = ["date", "ESI", "ESI_smoothed", "A_financial", "B_macro", "C_sentiment"]
    zcols = [c for c in cols if c.startswith("z_")]
    other = [c for c in cols if c not in fixed + zcols]
    ordered = [c for c in fixed if c in cols] + other + zcols
    df_reset[ordered].to_csv(out, index=False)
    out.seek(0)
    return StreamingResponse(out, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=esi_export.csv"})


def _maybe_presign(df: pd.DataFrame) -> Optional[str]:
    # If too large, offload to S3/MinIO
    max_rows = int(os.getenv("EXPORT_STREAM_MAX_ROWS", "100000"))
    if len(df) <= max_rows:
        return None
    # Presign via MinIO if configured
    endpoint = os.getenv("S3_ENDPOINT")
    bucket = os.getenv("S3_BUCKET")
    access = os.getenv("S3_ACCESS_KEY")
    secret = os.getenv("S3_SECRET_KEY")
    if not all([endpoint, bucket, access, secret]):
        return None
    try:
        from minio import Minio
        from minio.error import S3Error
        client = Minio(endpoint.replace("https://", "").replace("http://", ""), access_key=access, secret_key=secret, secure=endpoint.startswith("https"))
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)
        # Write to a temp buffer and upload
        key = f"exports/esi_{dt.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.csv"
        csv_buf = io.StringIO()
        df_reset = df.reset_index().rename(columns={"index": "date"})
        df_reset["date"] = df_reset["date"].dt.strftime("%Y-%m-%d")
        df_reset.to_csv(csv_buf, index=False)
        data = csv_buf.getvalue().encode("utf-8")
        import io as _io
        client.put_object(bucket, key, _io.BytesIO(data), length=len(data), content_type="text/csv")
        # Presign
        url = client.presigned_get_object(bucket, key, expires=dt.timedelta(hours=1))
        return url
    except Exception as e:
        return None


@router.get("/api/index/export.csv")
def export_csv(country: str = "US", start: Optional[str] = None, end: Optional[str] = None, format: str = "csv"):
    df = _df_for_range(country, start, end)
    if df.empty:
        return JSONResponse({"message": "no data in range"})
    # Content negotiation
    if format.lower() == "json":
        payload = build_history_payload(df)
        return JSONResponse(payload)
    # Maybe presign
    presigned = _maybe_presign(df)
    if presigned:
        return JSONResponse({"presigned_url": presigned})
    # Stream CSV
    return _to_csv_stream(df)



