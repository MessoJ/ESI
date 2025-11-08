from fastapi import FastAPI, HTTPException, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from typing import Optional
import datetime as dt
from dotenv import load_dotenv
from esi_core.compute_index import compute_components_cached, build_latest_payload, build_history_payload, build_components_payload
from esi_core.pca_alt import compute_pca_index
from api.admin import router as admin_router
from api.export import router as export_router
from api.db import init_db
from api.routers.alerts import router as alerts_router
from api.routers.alerts import evaluate_and_notify, get_db
from api.routers.quality import router as quality_router
import asyncio
import json
import logging
import os

import redis
import sentry_sdk
from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware
from slowapi.errors import RateLimitExceeded
from fastapi.responses import JSONResponse
from pythonjsonlogger import jsonlogger

load_dotenv()

app = FastAPI(title="Economic Stress Index API", version="0.1.0")
limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"]) 
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

# Cache TTLs (seconds)
LATEST_TTL = int(os.getenv("CACHE_TTL_LATEST", "600"))
HISTORY_TTL = int(os.getenv("CACHE_TTL_HISTORY", "3600"))
COMPONENTS_TTL = int(os.getenv("CACHE_TTL_COMPONENTS", "600"))
STALE_TTL = int(os.getenv("CACHE_TTL_STALE", "60"))

origins = os.getenv("ALLOW_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
if os.getenv("FORCE_HTTPS", "false").lower() == "true":
    app.add_middleware(HTTPSRedirectMiddleware)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    # Basic hardening headers
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    # HSTS if HTTPS forced
    if os.getenv("FORCE_HTTPS", "false").lower() == "true":
        response.headers.setdefault("Strict-Transport-Security", "max-age=63072000; includeSubDomains; preload")
    return response


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"detail": "Too Many Requests"})


@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "ts": dt.datetime.utcnow().isoformat(),
        "redis": _state.get("redis_ok", False),
        "last_compute_ts": _state.get("last_compute_ts"),
        "stale_series": _state.get("stale_series", {}),
    }


@app.get("/api/index/latest")
def latest(country: str = "US", method: str = "weighted"):
    try:
        cache_key = f"esi:latest:{country.upper()}"
        cached = _redis_get(cache_key)
        if cached:
            return Response(content=cached, media_type="application/json", headers={"Cache-Control": f"max-age={LATEST_TTL}"})
        df = compute_components_cached(country)
        payload = build_latest_payload(df)
        if method == "pca":
            pca = compute_pca_index(df)
            pc1 = pca.pc1_series
            payload["esi_pca"] = float(pc1.dropna().iloc[-1]) if pc1 is not None and not pc1.dropna().empty else None
            # Optionally include explained variance of PC1
            if getattr(pca, "explained_variance_series", None) is not None:
                ev = pca.explained_variance_series
                payload["pca_pc1_variance_explained"] = float(ev.dropna().iloc[-1]) if ev is not None and not ev.dropna().empty else None
        body = json.dumps(payload)
        _redis_set(cache_key, body, ex=LATEST_TTL)
        _state["last_compute_ts"] = dt.datetime.utcnow().timestamp()
        return Response(content=body, media_type="application/json", headers={"Cache-Control": f"max-age={LATEST_TTL}"})
    except Exception as e:
        # fallback to cache
        cache_key = f"esi:latest:{country.upper()}"
        cached = _redis_get(cache_key)
        if cached:
            return Response(content=cached, media_type="application/json", headers={"X-Data-Stale": "true", "Cache-Control": f"max-age={STALE_TTL}"})
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/index/history")
def history(country: str = "US", start: Optional[str] = None):
    try:
        cache_key = f"esi:history:{country.upper()}:{start or ''}"
        cached = _redis_get(cache_key)
        if cached:
            return Response(content=cached, media_type="application/json", headers={"Cache-Control": f"max-age={HISTORY_TTL}"})
        df = compute_components_cached(country)
        payload = build_history_payload(df, start)
        body = json.dumps(payload)
        _redis_set(cache_key, body, ex=HISTORY_TTL)
        return Response(content=body, media_type="application/json", headers={"Cache-Control": f"max-age={HISTORY_TTL}"})
    except Exception as e:
        cache_key = f"esi:history:{country.upper()}:{start or ''}"
        cached = _redis_get(cache_key)
        if cached:
            return Response(content=cached, media_type="application/json", headers={"X-Data-Stale": "true", "Cache-Control": f"max-age={STALE_TTL}"})
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/index/components")
def components(country: str = "US"):
    try:
        cache_key = f"esi:components:{country.upper()}"
        cached = _redis_get(cache_key)
        if cached:
            return Response(content=cached, media_type="application/json", headers={"Cache-Control": f"max-age={COMPONENTS_TTL}"})
        df = compute_components_cached(country)
        payload = build_components_payload(df)
        body = json.dumps(payload)
        _redis_set(cache_key, body, ex=COMPONENTS_TTL)
        return Response(content=body, media_type="application/json", headers={"Cache-Control": f"max-age={COMPONENTS_TTL}"})
    except Exception as e:
        cache_key = f"esi:components:{country.upper()}"
        cached = _redis_get(cache_key)
        if cached:
            return Response(content=cached, media_type="application/json", headers={"X-Data-Stale": "true", "Cache-Control": f"max-age={STALE_TTL}"})
        raise HTTPException(status_code=500, detail=str(e))

# Routers
app.include_router(admin_router)
app.include_router(export_router)
app.include_router(alerts_router)
app.include_router(quality_router)


@app.on_event("startup")
async def on_startup():
    # Sentry
    if os.getenv("SENTRY_DSN"):
        sentry_sdk.init(dsn=os.getenv("SENTRY_DSN"))

    # Logging (JSON)
    logHandler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    logHandler.setFormatter(formatter)
    logging.getLogger().handlers = [logHandler]
    logging.getLogger().setLevel(logging.INFO)

    # Redis
    try:
        _state["redis"] = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
        _state["redis"].ping()
        _state["redis_ok"] = True
    except Exception:
        _state["redis_ok"] = False

    # Prometheus gauge
    _state["gauge_last_compute"] = Gauge("esi_last_compute_timestamp", "Unix ts of last ESI compute")
    init_db()

    async def _job():
        while True:
            try:
                # run daily
                db = next(get_db())
                evaluate_and_notify(db)
                # metrics
                if _state.get("last_compute_ts"):
                    _state["gauge_last_compute"].set(_state["last_compute_ts"])
            except Exception:
                pass
            await asyncio.sleep(24*3600)
    asyncio.create_task(_job())


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# helpers
_state: dict = {}


def _redis_get(key: str) -> str | None:
    r = _state.get("redis")
    if not r:
        return None
    try:
        val = r.get(key)
        return val.decode("utf-8") if val else None
    except Exception:
        return None


def _redis_set(key: str, value: str, ex: int):
    r = _state.get("redis")
    if not r:
        return
    try:
        r.set(key, value, ex=ex)
    except Exception:
        pass


