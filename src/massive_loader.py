# massive_loader.py
"""
Massive Loader (Single-Day, Single-Symbol)
-----------------------------------------

Purpose
    Load ONE trading day of 1-minute OHLCV + VWAP data for a single symbol
    using Polygon-compatible intraday aggregates.

Design goals
    - Minimal
    - Explicit
    - Forensic replay of ONE known Pine trade
"""

from __future__ import annotations

from datetime import date as date_type
from typing import Union
import os
import pandas as pd


class MassiveLoaderError(RuntimeError):
    pass


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass


def _parse_date(trade_date: Union[str, date_type]) -> str:
    if isinstance(trade_date, str):
        return trade_date
    return trade_date.isoformat()


def load_one_day(*, symbol: str, trade_date: Union[str, date_type]) -> pd.DataFrame:
    """
    Load one full trading day of 1-minute bars including extended hours.
    Requires POLYGON_API_KEY in .env
    """
    _load_env()

    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        raise MassiveLoaderError("POLYGON_API_KEY not found in environment")

    trade_date_str = _parse_date(trade_date)

    try:
        from polygon import RESTClient
    except Exception as e:
        raise MassiveLoaderError("polygon package not installed") from e

    client = RESTClient(api_key=api_key)

    rows = []
    for agg in client.list_aggs(
        symbol,
        1,
        "minute",
        trade_date_str,
        trade_date_str,
        limit=50000,
    ):
        rows.append(
            {
                "timestamp": pd.Timestamp(
                    agg.timestamp, unit="ms", tz="UTC"
                ).tz_convert("US/Eastern"),
                "open": float(agg.open),
                "high": float(agg.high),
                "low": float(agg.low),
                "close": float(agg.close),
                "volume": float(getattr(agg, "volume", float("nan"))),
                "vwap": float(getattr(agg, "vwap", float("nan"))),
            }
        )

    if not rows:
        raise MassiveLoaderError(
            f"No data returned for {symbol} on {trade_date_str}"
        )

    df = pd.DataFrame(rows).set_index("timestamp")
    df = df.sort_index()
    df = df.between_time("04:00", "16:00", inclusive="both")

    required = ["open", "high", "low", "close", "volume", "vwap"]
    for col in required:
        if col not in df.columns:
            raise MassiveLoaderError(f"Missing column: {col}")

    return df
