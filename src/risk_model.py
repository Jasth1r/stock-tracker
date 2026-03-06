import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_absolute_error, r2_score
from src.fetcher import get_candles
from src.processor import add_features
SYMBOLS = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'META', 'NVDA',
           'NFLX', 'AMD', 'INTC', 'BABA', 'UBER', 'SHOP']

FEATURES = ['volatility_20', 'ma_5', 'ma_20', 'rsi', 'volume',
            'daily_return', 'volume_change', 'volume_spike',
            'return_3d', 'return_5d']

TARGET_CLASS = 'high_risk'
TARGET_REG   = 'next_return'

def load_data():
    frames = []
    for symbol in SYMBOLS:
        df = get_candles(symbol, period='5y')
        df = add_features(df)
        frames.append(df)
    return pd.concat(frames)

def train(df: pd.DataFrame):
    X = df[FEATURES]

    X_train, X_test, _, _ = train_test_split(
        X, X, test_size=0.2, random_state=42, shuffle=False
    )

    # --- Random Forest Classifier ---
    y_class = df[TARGET_CLASS]
    y_train_c, y_test_c = train_test_split(y_class, test_size=0.2, random_state=42, shuffle=False)

    classifier = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight='balanced'
    )
    classifier.fit(X_train, y_train_c)
    y_pred_c = classifier.predict(X_test)

    print("=== Random Forest Classifier ===")
    print(classification_report(y_test_c, y_pred_c))

    # --- Linear Regression ---
    y_reg = df[TARGET_REG]
    y_train_r, y_test_r = train_test_split(y_reg, test_size=0.2, random_state=42, shuffle=False)

    regressor = LinearRegression()
    regressor.fit(X_train, y_train_r)
    y_pred_r = regressor.predict(X_test)

    print("=== Linear Regression (next day return) ===")
    print(f"MAE:  {mean_absolute_error(y_test_r, y_pred_r):.4f}")
    print(f"R²:   {r2_score(y_test_r, y_pred_r):.4f}")

    return classifier, regressor

def predict_risk(classifier, regressor, latest_row: pd.DataFrame) -> str:
    prob         = classifier.predict_proba(latest_row[FEATURES])[0][1]
    label        = "🔴 HIGH RISK" if prob > 0.5 else "🟢 LOW RISK"
    predicted_return = regressor.predict(latest_row[FEATURES])[0]
    return (
        f"{label}  (confidence: {prob:.1%})\n"
        f"📈 Predicted next day return: {predicted_return:.2%}"
    )

if __name__ == "__main__":
    df = load_data()
    classifier, regressor = train(df)