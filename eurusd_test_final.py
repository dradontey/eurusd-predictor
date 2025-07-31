import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

st.set_page_config(page_title="EURUSD 测试", page_icon="💱", layout="wide")
st.markdown("## 💱 EUR/USD 简单回测演示")

# 下载过去 5 年数据
end_date = datetime.today()
start_date = end_date - timedelta(days=365*5)
df = yf.download('EURUSD=X', start=start_date, end=end_date)
df = df[['Close']].rename(columns={'Close':'close'}).dropna()
df.index = pd.to_datetime(df.index)

# 固定只取最后 N 天
N = 258
test_dates = df.index[-N:]
close_prices = df['close'].values[-N:]

# ✅ 修复：转成 1 维
test_returns = pd.Series(close_prices.ravel()).pct_change().fillna(0).values

# 随机策略信号：长度也是 N
np.random.seed(42)
signal = np.random.choice([0, 1], size=N)

# 策略收益
strategy_returns = test_returns * np.roll(signal, 1)
cum_strategy = (1 + strategy_returns).cumprod()
cum_bh = (1 + test_returns).cumprod()

# 打印长度确认
print(f"test_dates:{len(test_dates)}, cum_strategy:{len(cum_strategy)}")

# 绘图
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(test_dates, cum_strategy, label='策略', color='green')
ax.plot(test_dates, cum_bh, label='买入持有', color='gray')
ax.legend()
st.pyplot(fig, use_container_width=True)

st.success("✅ 图表生成成功！")
