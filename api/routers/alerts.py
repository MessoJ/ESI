import datetime as dt
import json
import os
from typing import List, Optional

import requests
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, EmailStr
from sqlalchemy.orm import Session

from api.db import SessionLocal
from api.security import require_api_key
from api.models import AlertRule
from esi_core.compute_index import compute_components_cached


router = APIRouter()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class AlertCreate(BaseModel):
    metric: str = Field(..., description="Metric to monitor, e.g., ESI or A_financial")
    op: str = Field(..., pattern=r"^(>=|<=|>|<)$")
    threshold: float
    country: str = "US"
    email: Optional[EmailStr] = None
    webhook_url: Optional[str] = None


class AlertOut(BaseModel):
    id: int
    metric: str
    op: str
    threshold: float
    country: str
    email: Optional[str]
    webhook_url: Optional[str]
    active: bool
    created_at: dt.datetime


@router.post("/api/alerts", response_model=AlertOut, dependencies=[Depends(require_api_key)])
def create_alert(rule: AlertCreate, db: Session = Depends(get_db)):
    if not rule.email and not rule.webhook_url:
        raise HTTPException(status_code=400, detail="Specify at least one of email or webhook_url")
    obj = AlertRule(
        metric=rule.metric,
        op=rule.op,
        threshold=rule.threshold,
        country=rule.country.upper(),
        email=rule.email,
        webhook_url=rule.webhook_url,
        active=True,
    )
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj


@router.get("/api/alerts", response_model=List[AlertOut])
def list_alerts(db: Session = Depends(get_db)):
    return db.query(AlertRule).order_by(AlertRule.created_at.desc()).all()


@router.delete("/api/alerts/{alert_id}")
def delete_alert(alert_id: int, db: Session = Depends(get_db)):
    obj = db.get(AlertRule, alert_id)
    if not obj:
        raise HTTPException(status_code=404, detail="Not found")
    db.delete(obj)
    db.commit()
    return {"ok": True}


def _compare(value: float, op: str, threshold: float) -> bool:
    return {
        ">": lambda v, t: v > t,
        "<": lambda v, t: v < t,
        ">=": lambda v, t: v >= t,
        "<=": lambda v, t: v <= t,
    }[op](value, threshold)


def _send_email(email: str, subject: str, text: str):
    postmark_key = os.getenv("POSTMARK_SERVER_TOKEN")
    sender = os.getenv("POSTMARK_SENDER", "alerts@example.com")
    if not postmark_key:
        return  # no-op
    try:
        r = requests.post(
            "https://api.postmarkapp.com/email",
            headers={"X-Postmark-Server-Token": postmark_key, "Accept": "application/json"},
            json={"From": sender, "To": email, "Subject": subject, "TextBody": text},
            timeout=10,
        )
        r.raise_for_status()
    except Exception:
        pass


def _post_webhook(url: str, payload: dict):
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception:
        pass


def evaluate_and_notify(db: Session):
    # Group rules by country
    rules: List[AlertRule] = db.query(AlertRule).filter(AlertRule.active == True).all()  # noqa: E712
    by_country = {}
    for r in rules:
        by_country.setdefault(r.country.upper(), []).append(r)
    for country, crules in by_country.items():
        df = compute_components_cached(country)
        if df.empty:
            continue
        last = df.iloc[-1]
        now_iso = dt.datetime.utcnow().isoformat()
        for rule in crules:
            val = float(last.get(rule.metric)) if rule.metric in last.index else None
            if val is None:
                continue
            if _compare(val, rule.op, rule.threshold):
                payload = {
                    "metric": rule.metric,
                    "value": val,
                    "op": rule.op,
                    "threshold": rule.threshold,
                    "country": country,
                    "timestamp": now_iso,
                }
                subj = f"ESI Alert: {rule.metric} {rule.op} {rule.threshold} in {country}"
                text = json.dumps(payload, indent=2)
                if rule.email:
                    _send_email(rule.email, subj, text)
                if rule.webhook_url:
                    _post_webhook(rule.webhook_url, payload)


