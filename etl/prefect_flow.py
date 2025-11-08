import os
import io
import datetime as dt
from pathlib import Path
from typing import List

import pandas as pd
from minio import Minio
from prefect import flow, task, get_run_logger
from sqlalchemy import create_engine, text


RAW_DIR = Path("data/raw")
S3_ENDPOINT = os.getenv("S3_ENDPOINT")
S3_BUCKET = os.getenv("S3_BUCKET", "esi-data")
S3_ACCESS = os.getenv("S3_ACCESS_KEY")
S3_SECRET = os.getenv("S3_SECRET_KEY")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./esi_series.db")


def _minio_client() -> Minio | None:
    if not all([S3_ENDPOINT, S3_BUCKET, S3_ACCESS, S3_SECRET]):
        return None
    return Minio(S3_ENDPOINT.replace("https://", "").replace("http://", ""), access_key=S3_ACCESS, secret_key=S3_SECRET, secure=S3_ENDPOINT.startswith("https"))


@task
def fetch() -> List[Path]:
    logger = get_run_logger()
    files = []
    for name in ["ke_cbk.csv", "ke_knbs.csv", "ke_pmi.csv"]:
        p = RAW_DIR / name
        if p.exists():
            files.append(p)
            logger.info(f"found {p}")
    if not files:
        logger.warning("no raw files found; ensure ingestion ran")
    return files


@task
def upload_raw(files: List[Path]) -> List[str]:
    logger = get_run_logger()
    cli = _minio_client()
    urls: List[str] = []
    if not cli:
        logger.warning("MinIO/S3 not configured; skipping upload")
        return urls
    for p in files:
        key = f"raw/{dt.datetime.utcnow().strftime('%Y%m%d')}/{p.name}"
        cli.fput_object(S3_BUCKET, key, str(p), content_type="text/csv")
        url = cli.presigned_get_object(S3_BUCKET, key, expires=dt.timedelta(days=7))
        urls.append(url)
        logger.info(f"uploaded {p} -> {key}")
    return urls


@task
def load_clean(files: List[Path]):
    logger = get_run_logger()
    engine = create_engine(DATABASE_URL)
    with engine.begin() as conn:
        # ensure table
        conn.execute(text(
            """
            CREATE TABLE IF NOT EXISTS series_points (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              date DATE,
              series VARCHAR(64),
              country VARCHAR(8),
              value FLOAT
            )
            """
        ))
        # upsert-like insert ignore duplicates
        for p in files:
            df = pd.read_csv(p)
            df['date'] = pd.to_datetime(df['date']).dt.date
            if p.name == 'ke_cbk.csv':
                mapping = {
                    'ke_91d_bill_yield': 'KE_91D_BILL',
                    'ke_10y_bond_yield': 'KE_10Y_BOND',
                }
                for col, series in mapping.items():
                    if col in df.columns:
                        for _, row in df[['date', col]].dropna().iterrows():
                            conn.execute(text("INSERT OR IGNORE INTO series_points(date, series, country, value) VALUES(:d,:s,'KE',:v)").bindparams(d=row['date'], s=series, v=float(row[col])))
            elif p.name == 'ke_knbs.csv':
                if 'cpi' in df.columns:
                    for _, row in df[['date','cpi']].dropna().iterrows():
                        conn.execute(text("INSERT OR IGNORE INTO series_points(date, series, country, value) VALUES(:d,'KE_CPI','KE',:v)").bindparams(d=row['date'], v=float(row['cpi'])))
                if 'unemployment' in df.columns:
                    for _, row in df[['date','unemployment']].dropna().iterrows():
                        conn.execute(text("INSERT OR IGNORE INTO series_points(date, series, country, value) VALUES(:d,'KE_UNRATE','KE',:v)").bindparams(d=row['date'], v=float(row['unemployment'])))
            elif p.name == 'ke_pmi.csv':
                if 'pmi' in df.columns:
                    for _, row in df[['date','pmi']].dropna().iterrows():
                        conn.execute(text("INSERT OR IGNORE INTO series_points(date, series, country, value) VALUES(:d,'KE_STANBIC_PMI','KE',:v)").bindparams(d=row['date'], v=float(row['pmi'])))
    logger.info("load complete")


@flow(name="esi-nightly-etl")
def nightly_flow():
    files = fetch()
    upload_raw(files)
    load_clean(files)


if __name__ == '__main__':
    nightly_flow()



