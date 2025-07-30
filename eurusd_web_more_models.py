import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import requests
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA

plt.style.use('ggplot')

today_str = datetime.today().strftime('%Y-%m-%d')
st.markdown("## 📊 EUR/USD AI 多模型预测工具")
st.caption(f"数据自动更新：{today_str}")

# 获取最近30天历史数据
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

# 模型准备
X = np.arange(len(df)).reshape(-1, 1)
y = df['close'].values

# 神经网络
mlp = MLPRegressor(hidden_layer_sizes=(10,10), max_iter=1000, random_state=1)
mlp.fit(X, y)

# 线性回归
lr = LinearRegression()
lr.fit(X, y)

# XGBoost
xgb = XGBRegressor(objective='reg:squarederror', n_estimators=100)
xgb.fit(X, y)

# ARIMA
model_arima = ARIMA(y, order=(2,1,2))
model_arima_fit = model_arima.fit()

# 未来5天预测
future_days = 5
future_indexes = np.arange(len(df), len(df)+future_days).reshape(-1,1)

future_preds_mlp = mlp.predict(future_indexes)
future_preds_lr = lr.predict(future_indexes)
future_preds_xgb = xgb.predict(future_indexes)
future_preds_arima = model_arima_fit.forecast(steps=future_days)

last_date = df.index[-1]
future_dates = [last_date + timedelta(days=i) for i in range(1, future_days+1)]

# 合成表格
future_df = pd.DataFrame({
    'date': future_dates,
    '神经网络': future_preds_mlp,
    '线性回归': future_preds_lr,
    'XGBoost': future_preds_xgb,
    'ARIMA': future_preds_arima
})
future_df.set_index('date', inplace=True)

# 显示预测表
st.markdown("### 📈 未来5天预测（四模型对比）")
st.dataframe(future_df.style.format("{:.5f}"), use_container_width=True)

# 下载按钮
csv = future_df.to_csv().encode('utf-8')
st.download_button("📥 下载预测结果 CSV", data=csv, file_name='future_predictions.csv', mime='text/csv')

# 绘图
fig, ax = plt.subplots(figsize=(10,6))

ax.plot(df.index, df['close'], label='实际收盘价', color='#1f77b4', linewidth=2)
ax.plot(future_df.index, future_df['神经网络'], label='神经网络', linestyle='--', color='#ff7f0e')
ax.plot(future_df.index, future_df['线性回归'], label='线性回归', linestyle='--', color='green')
ax.plot(future_df.index, future_df['XGBoost'], label='XGBoost', linestyle='--', color='purple')
ax.plot(future_df.index, future_df['ARIMA'], label='ARIMA', linestyle='--', color='brown')

ax.set_title("收盘价 & AI多模型预测", fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
st.pyplot(fig, use_container_width=True)
