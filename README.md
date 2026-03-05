# Stock Tracker / Risk Predictor

This is a student project that tracks stock prices and tries to predict whether a stock is at high or low risk of dropping the next day.

## What it does

It pulls historical stock data from Finnhub, calculates some common technical indicators (like RSI, moving averages, and Bollinger Bands), and uses a Random Forest model to classify whether a stock is likely to drop more than 2% the following day.

## Data

All stock data comes from the [Finnhub API](https://finnhub.io). You need a free API key from their website to run this project.

## How to run

1. Clone first
2. Install dependencies: `pip install -r requirements.txt`
3. Add your Finnhub API key to a `.env` file: `FINNHUB_API_KEY=your_key_here`
4. Run `python3 src/fetcher.py` to fetch data
5. Run `python3 src/risk_model.py` to train the model and get a prediction

## Tech

Python, Pandas, Scikit-learn, Finnhub API
