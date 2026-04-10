import os
from typing import Optional, Dict, Any
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

try:
    import finnhub
    _finnhub_key = os.getenv("FINNHUB_API_KEY")
    client = finnhub.Client(api_key=_finnhub_key) if _finnhub_key else None
except ImportError:
    client = None

def get_quote(symbol: str) -> Optional[Dict[str, Any]]:
    if client is None:
        return None
    try:
        return client.quote(symbol)
    except Exception:
        return None


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        level_0 = df.columns.get_level_values(0)
        level_1 = df.columns.get_level_values(1)
        if len(level_1) and str(level_1[0]).lower() in ("open", "high", "low", "close", "volume", "adj close"):
            df.columns = [str(c).lower() for c in level_1]
        else:
            df.columns = [str(c).lower() for c in level_0]
    else:
        df.columns = [str(c).lower() for c in df.columns]
    return df


def get_candles(symbol: str, period: str = "6mo") -> pd.DataFrame:
    df = yf.download(
        symbol,
        period=period,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )
    if df.empty:
        return df
    df = _normalize_columns(df)
    if "close" not in df.columns and "adj close" in df.columns:
        df["close"] = df["adj close"]
    return df


def get_candles_for_model(symbol: str, horizon_years: int = 10) -> pd.DataFrame:
    period = "10y" if horizon_years >= 5 else "2y"
    return get_candles(symbol, period=period)


if __name__ == "__main__":
    q = get_quote("AAPL")
    print("Quote (Finnhub):", q if q else "(no API key or error)")
    df = get_candles("AAPL", period="1y")
    print("Candles columns:", list(df.columns))
    print(get_candles("AAPL").tail())
