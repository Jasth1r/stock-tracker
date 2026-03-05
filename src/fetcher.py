import os
import finnhub
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

client = finnhub.Client(api_key=os.getenv("d6jk1d1r01qkvh5q8dsgd6jk1d1r01qkvh5q8dt0"))

def get_quote(symbol: str) -> dict:
    return client.quote(symbol)

def get_candles(symbol: str, resolution='D', count=100) -> pd.DataFrame:
    import time
    end = int(time.time())
    start = end - count * 24 * 3600

    res = client.stock_candles(symbol, resolution, start, end)
    if res['s'] != 'ok':
        return pd.DataFrame()

    df = pd.DataFrame({
        'timestamp': pd.to_datetime(res['t'], unit='s'),
        'open':  res['o'],
        'high':  res['h'],
        'low':   res['l'],
        'close': res['c'],
        'volume':res['v']
    })
    return df.set_index('timestamp')

if __name__ == "__main__":
    print(get_quote("AAPL"))
    print(get_candles("AAPL").tail())