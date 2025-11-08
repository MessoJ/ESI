import csv
import datetime as dt
import os
import re
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup


RAW_PATH = Path("data/raw/ke_cbk.csv")
USER_AGENT = {
    "User-Agent": "ESI-Bot/1.0 (+https://example.com)"
}


def _ensure_dirs():
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)


def parse_cbk_auction_table(html: str) -> List[Tuple[dt.date, float, float]]:
    """Parse CBK auction page/table and extract (date, 91d_bill_yield, 10y_bond_yield).

    Returns a list of tuples; missing values are represented as None.
    """
    soup = BeautifulSoup(html, "lxml")
    rows = []

    # Try common table structures: look for rows with dates and yields
    for table in soup.find_all("table"):
        headers = [h.get_text(strip=True).lower() for h in table.find_all(["th", "td"])[:10]]
        if not any("91" in h and "bill" in h for h in headers) and not any("t-bill" in h for h in headers):
            continue
        for tr in table.find_all("tr"):
            tds = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
            if len(tds) < 3:
                continue
            # heuristic: first cell contains a date
            try:
                date = _parse_any_date(tds[0])
            except Exception:
                continue
            yields = " ".join(tds[1:]).lower()
            bill_y = _extract_percent(yields, patterns=[r"91[\-\s]?day.*?(\d+\.?\d*)%?", r"91.*?(\d+\.?\d*)%?"])
            bond10_y = _extract_percent(yields, patterns=[r"10\s?y.*?(\d+\.?\d*)%?", r"10\-?year.*?(\d+\.?\d*)%?"])
            rows.append((date, bill_y if bill_y is not None else float("nan"), bond10_y if bond10_y is not None else float("nan")))
    return rows


def _parse_any_date(text: str) -> dt.date:
    text = text.strip()
    # Try various formats
    fmts = ["%d %b %Y", "%d %B %Y", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"]
    for f in fmts:
        try:
            return dt.datetime.strptime(text, f).date()
        except Exception:
            pass
    # fallback: regex YYYY/MM/DD or DD-MM-YYYY
    m = re.search(r"(\d{1,2})[-/](\d{1,2})[-/](\d{2,4})", text)
    if m:
        d, mth, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if y < 100:
            y += 2000
        return dt.date(y, mth, d)
    raise ValueError(f"Unrecognized date: {text}")


def _extract_percent(blob: str, patterns: Iterable[str]) -> float | None:
    for pat in patterns:
        m = re.search(pat, blob)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                continue
    # loose fallback: first number with % sign
    m = re.search(r"(\d+\.?\d*)\s*%", blob)
    if m:
        return float(m.group(1))
    return None


def fetch_cbk_pages(urls: List[str]) -> List[Tuple[dt.date, float, float]]:
    out: List[Tuple[dt.date, float, float]] = []
    for url in urls:
        r = requests.get(url, headers=USER_AGENT, timeout=30)
        r.raise_for_status()
        out.extend(parse_cbk_auction_table(r.text))
    return out


def write_csv(rows: List[Tuple[dt.date, float, float]]):
    _ensure_dirs()
    df = pd.DataFrame(rows, columns=["date", "ke_91d_bill_yield", "ke_10y_bond_yield"])
    df = df.dropna(how="all")
    df = df.drop_duplicates(subset=["date"]).sort_values("date")
    if RAW_PATH.exists():
        # merge with existing
        old = pd.read_csv(RAW_PATH)
        old["date"] = pd.to_datetime(old["date"]).dt.date
        df["date"] = pd.to_datetime(df["date"]).dt.date
        merged = (
            pd.concat([old, df], ignore_index=True)
            .drop_duplicates(subset=["date"], keep="last")
            .sort_values("date")
        )
        merged.to_csv(RAW_PATH, index=False)
    else:
        df.to_csv(RAW_PATH, index=False)


def backfill_historical(url_list: List[str]):
    rows = fetch_cbk_pages(url_list)
    write_csv(rows)


if __name__ == "__main__":
    # Example historical pages (replace with real CBK archive URLs)
    example_urls = [
        "https://www.centralbank.go.ke/treasury-bills-auctions/",
    ]
    backfill_historical(example_urls)



