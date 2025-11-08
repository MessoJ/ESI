import os
import re
from typing import Optional

from fastapi import Header, HTTPException


def require_api_key(x_api_key: Optional[str] = Header(default=None, alias="X-API-KEY")):
    secret = os.getenv("API_KEY_SECRET")
    if not secret:
        # If not configured, allow for local dev
        return True
    if not x_api_key or x_api_key != secret:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True


_FORMULA_PREFIX = re.compile(r"^[=+\-@]")


def sanitize_cell(value: str) -> str:
    if isinstance(value, str) and _FORMULA_PREFIX.match(value):
        # Prevent CSV formula injection
        return "'" + value
    return value



