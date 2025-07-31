import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def load_data(symbol, years=5):
    end = datetime.today()
    start = end - timedelta(days=365*years)
    df = yf.download(symbol, start=start, end=end)
    df = df[['Close']].rename(columns={'Close':'close'}).dropna()
    df.index = pd.to_datetime(df.index)
    return df
