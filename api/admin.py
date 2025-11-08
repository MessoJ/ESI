from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from io import StringIO
import pandas as pd

from etl.ingest_stanbic_pmi import write_csv as write_pmi_csv
from .security import sanitize_cell, require_api_key

router = APIRouter()


@router.post("/api/admin/upload-pmi")
async def upload_pmi(file: UploadFile = File(...), _: bool = Depends(require_api_key)):
    try:
        if not file.filename.lower().endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        content = await file.read()
        text = content.decode('utf-8', errors='replace')
        # basic sanitization: strip null bytes and enforce csv
        text = text.replace('\x00', '')
        df = pd.read_csv(StringIO(text))
        df.columns = [c.lower() for c in df.columns]
        if 'date' not in df.columns or 'pmi' not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must include 'date' and 'pmi' columns")
        df['date'] = pd.to_datetime(df['date']).dt.to_period('M').dt.to_timestamp()
        # sanitize values
        df['pmi'] = df['pmi'].apply(lambda x: sanitize_cell(str(x)))
        # cast
        df['pmi'] = pd.to_numeric(df['pmi'], errors='coerce')
        write_pmi_csv(df[['date','pmi']].dropna())
        return {"ok": True, "rows": int(df.shape[0])}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


