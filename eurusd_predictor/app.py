
import streamlit as st
import numpy as np
import pandas as pd
from data_loader import load_data
from moving_average import moving_average_strategy
from plot_utils import plot_equity_curve

st.set_page_config(page_title="EURUSD 策略回测", page_icon="💱", layout="wide")
st.title("💱 EUR/USD 策略回测 APP")

# 用户输入
symbol = st.text_input("输入交易对代码", value="EURUSD=X")
years = st.slider("回测年数", 1, 10, 5)

# 加载数据
df = load_data(symbol, years=years)
st.write("最新数据：", df.tail())

# 参数
short_window = st.slider("短期均线窗口", 5, 50, 20)
long_window = st.slider("长期均线窗口", 50, 200, 100)

# 策略信号和回测
df['signal'] = moving_average_strategy(df, short_window, long_window)
df['returns'] = df['close'].pct_change().fillna(0)
df['strategy_returns'] = df['returns'] * df['signal'].shift(1).fillna(0)

# 累计收益曲线
cum_bh = (1 + df['returns']).cumprod()
cum_strategy = (1 + df['strategy_returns']).cumprod()

# 显示图表
plot_equity_curve(df.index, [cum_bh, cum_strategy], ['买入持有', '均线策略'])

# 显示简单统计
st.markdown("### 📊 简单统计")
st.write(f"策略最终收益: {cum_strategy.iloc[-1]:.2f}")
st.write(f"买入持有最终收益: {cum_bh.iloc[-1]:.2f}")
