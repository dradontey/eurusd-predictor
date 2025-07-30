import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt
import requests

st.title("EUR/USD AI 多日预测（真实历史数据 + 神经网络）")

# 获取最近30天历史EUR/USD
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

# 神经网络预测
X = np.arange(len(df)).reshape(-1, 1)
y = df['close'].values
model = MLPRegressor(hidden_layer_sizes=(10,10), max_iter=1000, random_state=1)
model.fit(X, y)

# 预测未来 5 天
future_days = 5
future_indexes = np.arange(len(df), len(df)+future_days).reshape(-1,1)
future_preds = model.predict(future_indexes)

# 创建未来日期索引
last_date = df.index[-1]
future_dates = [last_date + timedelta(days=i) for i in range(1, future_days+1)]

# 拼接到 DataFrame
future_df = pd.DataFrame({'date': future_dates, 'close': future_preds})
future_df.set_index('date', inplace=True)

# 显示预测值
st.write("**未来 5 天预测收盘价：**")
st.dataframe(future_df)

# 绘图
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(df.index, df['close'], label='实际收盘价', color='blue')
ax.plot(future_df.index, future_df['close'], label='预测收盘价', color='red', linestyle='--')
ax.set_title("EUR/USD 实际收盘价 + 未来5天AI预测")
ax.legend()
ax.grid(True)

st.pyplot(fig)
