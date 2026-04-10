import pandas as pd
import numpy as np

RISK_FEATURES = [
    "volatility_20",
    "ma_5",
    "ma_20",
    "rsi",
    "volume",
    "daily_return",
]

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["close", "volume"]:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain '{col}'. Got: {list(df.columns)}")

    df["daily_return"] = df["close"].pct_change()
    df["volatility_20"] = df["daily_return"].rolling(20).std()

    df["ma_5"] = df["close"].rolling(5).mean()
    df["ma_20"] = df["close"].rolling(20).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss))

    df["bb_upper"] = df["ma_20"] + 2 * df["close"].rolling(20).std()
    df["bb_lower"] = df["ma_20"] - 2 * df["close"].rolling(20).std()

    df["next_return"] = df["daily_return"].shift(-1)

    df["high_risk"] = (df["next_return"] < -0.02).astype(int)
    df["risk_level"] = np.where(
        df["next_return"] < -0.02, 2, np.where(df["next_return"] < 0.005, 1, 0)
    )

    return df.dropna(subset=RISK_FEATURES)


def get_feature_summary(df: pd.DataFrame) -> dict:
    if df.empty or "daily_return" not in df.columns:
        return {}
    daily_returns = df["daily_return"].dropna()
    n = len(daily_returns)
    if n < 2:
        return {}
    mean_daily = daily_returns.mean()
    std_daily = daily_returns.std()
    if std_daily and not np.isnan(std_daily):
        annual_return = (1 + mean_daily) ** 252 - 1
        annual_vol = std_daily * np.sqrt(252)
    else:
        annual_return = 0.0
        annual_vol = 0.0
    latest_close = float(df["close"].iloc[-1]) if "close" in df.columns else None
    return {
        "annualized_return": annual_return,
        "annualized_volatility": annual_vol,
        "latest_close": latest_close,
        "trading_days": n,
    }
