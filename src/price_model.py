import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from processor import get_feature_summary, RISK_FEATURES


HORIZONS = (1, 5, 10)

PRICE_FEATURES = RISK_FEATURES


def predict_price_with_growth(
    latest_close: float,
    annualized_return: float,
    years: int,
) -> dict:
    r = np.clip(annualized_return, -0.5, 0.5)
    predicted = latest_close * ((1 + r) ** years)
    return {
        "predicted_price": round(float(predicted), 2),
        "annualized_return_used": round(float(r), 4),
        "years": years,
        "current_price": round(float(latest_close), 2),
    }


def _train_return_model(
    processed_df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
):
    if processed_df is None or processed_df.empty or "close" not in processed_df.columns:
        return None

    df = processed_df.copy()
    steps_ahead = 252
    df["future_1y_return"] = df["close"].shift(-steps_ahead) / df["close"] - 1

    cols_needed = list(PRICE_FEATURES) + ["future_1y_return"]
    train_df = df.dropna(subset=cols_needed)
    if len(train_df) < 80:
        return None

    X = train_df[PRICE_FEATURES]
    y = train_df["future_1y_return"]

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=False,
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def predict_future_price(
    processed_df: pd.DataFrame,
    years: int,
) -> dict:
    if years not in HORIZONS:
        years = min(HORIZONS, key=lambda h: abs(h - years))

    summary = get_feature_summary(processed_df)
    if not summary or summary.get("latest_close") is None:
        return {
            "predicted_price": None,
            "current_price": None,
            "years": years,
            "error": "Insufficient data",
        }

    latest_close = summary["latest_close"]

    model = _train_return_model(processed_df)

    if model is not None:
        latest_row = processed_df.dropna(subset=PRICE_FEATURES).iloc[[-1]]
        pred_ret_1y = float(model.predict(latest_row[PRICE_FEATURES])[0])
        ann_return = pred_ret_1y
        out = predict_price_with_growth(latest_close, ann_return, years)
        out["method"] = "learned_regression_1y"
        return out

    ann_return_hist = summary.get("annualized_return", 0.0) or 0.0
    out = predict_price_with_growth(latest_close, ann_return_hist, years)
    out["method"] = "historical_growth_fallback"
    return out
