import streamlit as st
from data_loader import load_data
from moving_average import moving_average_signal
from plot_utils import plot_signals
import pandas as pd

st.set_page_config(page_title="EURUSD Predictor", page_icon="💱", layout="wide")
st.title("💱 EUR/USD 简单回测演示")

# 数据加载
df = load_data()

# 计算信号
df['signal'] = moving_average_signal(df, short_window=20, long_window=50)

# 绘图
plot_signals(df)

st.success("✅ 回测完成！")
