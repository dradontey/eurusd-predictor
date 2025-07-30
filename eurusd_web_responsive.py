import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import requests

plt.style.use('ggplot')

today_str = datetime.today().strftime('%Y-%m-%d')
st.markdown("## 📊 EUR/USD AI 预测工具")
st.caption(f"数据自动更新：{today_str}")

# 获取最新30天历史数据
end_date = datetime.today()
start_date = end_date - timedelta(days=30)
start_str = start_date.strftime('%Y-%m-%d')
end_str = end_date.strftime('%Y-%m-%d')

url = f"https://api.frankfurter.app/{start_str}..{end_str}?from=EUR&to=USD"
response = requests.get(url)
data = response.json()

rates = data['rates']
dates = sorted(rates.keys())
prices = [rates[date]['USD'] for date in dates]

df = pd.DataFrame({'date': pd.to_datetime(dates), 'close': prices})
df.set_index('date', inplace=True)

# 显示最新收盘价
st.metric("最新 EUR/USD 收盘价", f"{df['close'][-1]:.5f}")

# 模型预测
X = np.arange(len(df)).reshape(-1, 1)
y = df['close'].values
mlp = MLPRegressor(hidden_layer_sizes=(10,10), max_iter=1000, random_state=1)
mlp.fit(X, y)
lr = LinearRegression()
lr.fit(X, y)

future_days = 5
future_indexes = np.arange(len(df), len(df)+future_days).reshape(-1,1)
future_preds_mlp = mlp.predict(future_indexes)
future_preds_lr = lr.predict(future_indexes)

last_date = df.index[-1]
future_dates = [last_date + timedelta(days=i) for i in range(1, future_days+1)]

future_df = pd.DataFrame({
    'date': future_dates,
    '神经网络预测': future_preds_mlp,
    '线性回归预测': future_preds_lr
})
future_df.set_index('date', inplace=True)

# 技术指标
df['ma5'] = df['close'].rolling(window=5).mean()
df['ma10'] = df['close'].rolling(window=10).mean()
df['middle'] = df['close'].rolling(window=20).mean()
df['std'] = df['close'].rolling(window=20).std()
df['upper'] = df['middle'] + 2 * df['std']
df['lower'] = df['middle'] - 2 * df['std']

delta = df['close'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
df['rsi'] = 100 - (100 / (1 + rs))

ema12 = df['close'].ewm(span=12, adjust=False).mean()
ema26 = df['close'].ewm(span=26, adjust=False).mean()
df['macd'] = ema12 - ema26
df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()

# 显示预测表 + 下载
with st.container():
    st.markdown("### 📈 未来5天预测（多模型对比）")
    st.dataframe(future_df.style.format("{:.5f}"), use_container_width=True)
    csv = future_df.to_csv().encode('utf-8')
    st.download_button("📥 下载预测结果 CSV", data=csv, file_name='future_predictions.csv', mime='text/csv')

# 绘图
fig, axs = plt.subplots(3, 1, figsize=(8,10), sharex=True)

axs[0].plot(df.index, df['close'], label='实际收盘价', color='#1f77b4', linewidth=2)
axs[0].plot(future_df.index, future_df['神经网络预测'], label='神经网络预测', color='#ff7f0e', linestyle='--', linewidth=2)
axs[0].plot(future_df.index, future_df['线性回归预测'], label='线性回归预测', color='green', linestyle='--', linewidth=2)
axs[0].plot(df.index, df['ma5'], label='MA5', color='purple', alpha=0.8)
axs[0].plot(df.index, df['ma10'], label='MA10', color='red', alpha=0.8)
axs[0].plot(df.index, df['upper'], label='上轨', color='grey', linestyle='--', alpha=0.6)
axs[0].plot(df.index, df['middle'], label='中轨', color='black', linestyle='--', alpha=0.6)
axs[0].plot(df.index, df['lower'], label='下轨', color='grey', linestyle='--', alpha=0.6)
axs[0].legend(fontsize=8)
axs[0].grid(True, linestyle='--', alpha=0.3)
axs[0].set_title("收盘价 & AI预测 + 均线 + 布林带", fontsize=12)

axs[1].plot(df.index, df['rsi'], label='RSI(14)', color='brown', linewidth=1.5)
axs[1].axhline(70, color='red', linestyle='--', linewidth=1)
axs[1].axhline(30, color='green', linestyle='--', linewidth=1)
axs[1].legend(fontsize=8)
axs[1].grid(True, linestyle='--', alpha=0.3)
axs[1].set_title("RSI", fontsize=12)

axs[2].plot(df.index, df['macd'], label='MACD', color='cyan', linewidth=1.5)
axs[2].plot(df.index, df['signal'], label='Signal', color='magenta', linewidth=1.5)
axs[2].axhline(0, color='black', linestyle='--', linewidth=1)
axs[2].legend(fontsize=8)
axs[2].grid(True, linestyle='--', alpha=0.3)
axs[2].set_title("MACD", fontsize=12)

plt.tight_layout()
st.pyplot(fig, use_container_width=True)
