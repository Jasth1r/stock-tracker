# Retail Radar

A stock analysis tool that evaluates **risk level** and **future price prediction** using machine learning models built on historical market data.

Author: **Jacob Zhang**

## Features

- **Price Prediction** -- Predicts stock price at 1, 5, or 10 year horizons using a Random Forest regression model (with compound-growth fallback)
- **Risk Assessment** -- Classifies stocks as Low / Medium / High risk using a Random Forest classifier trained on technical indicators
- **REST API** -- Flask-based API server with endpoints for analysis, watchlist, and available horizons
- **Frontend Mockup** -- Interactive HTML prototype with Chart.js visualizations and colorblind-friendly mode

## Project Structure

```
src/
  fetcher.py        # Data fetching: yfinance (historical OHLCV) + Finnhub (optional live quote)
  processor.py      # Feature engineering: returns, volatility, MA, RSI, Bollinger Bands, risk labels
  risk_model.py     # Random Forest classifier -> risk level (Low / Medium / High)
  price_model.py    # Random Forest regressor + compound growth -> predicted price at 1/5/10 years
  backend.py        # Pipeline entry point: run_pipeline(symbol, horizon_years)
  api_server.py     # Flask REST API server
notebooks/
  financial_mockup.html   # Frontend prototype
tests/
  test_stocks.py    # CLI test script
```

## Tech Stack

- **Data**: yfinance, Finnhub (optional)
- **ML**: scikit-learn (Random Forest)
- **Features**: pandas, numpy
- **API**: Flask

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# (Optional) Set Finnhub API key for live quotes
echo "FINNHUB_API_KEY=your_key_here" > .env
```

### Python

```python
from backend import run_pipeline

result = run_pipeline("AAPL", horizon_years=5)
print(result["predicted_price"])  # predicted price info
print(result["risk"])             # risk level, confidence, probabilities
```

### CLI

```bash
python tests/test_stocks.py --symbols AAPL NVDA TSLA
```

### API Server

```bash
python src/api_server.py
```

| Endpoint | Description |
|----------|-------------|
| `GET /api/analyze?symbol=AAPL&years=5` | Price prediction + risk assessment |
| `GET /api/horizons` | Available prediction horizons |
| `GET /api/watchlist` | Default stock watchlist |

## How It Works

1. **Data Fetching** -- Historical OHLCV data is pulled from yfinance. Optionally, a live quote is fetched from Finnhub.
2. **Feature Engineering** -- Technical indicators are computed: daily returns, 20-day volatility, 5/20-day moving averages, RSI, Bollinger Bands.
3. **Risk Classification** -- A Random Forest classifier is trained on the computed features with 3-class labels (Low/Medium/High) based on next-day return thresholds.
4. **Price Prediction** -- A Random Forest regressor predicts 1-year forward returns, which are then compounded to the selected horizon. Falls back to historical annualized return if insufficient training data.

## License

MIT
