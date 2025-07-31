import numpy as np

def moving_average_strategy(prices, short=5, long=20):
    short_ma = prices.rolling(short).mean()
    long_ma = prices.rolling(long).mean()
    signal = np.where(short_ma > long_ma, 1, 0)
    returns = prices.pct_change().fillna(0)
    strategy_returns = returns * np.roll(signal, 1)
    return (1+strategy_returns).cumprod()
