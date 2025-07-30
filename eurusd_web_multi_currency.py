import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
import numpy as np
import matplotlib.pyplot as plt
import requests

plt.style.use('ggplot')
today_str = datetime.today().strftime('%Y-%m-%d')
st.markdown("## 📊 多币种 AI 预测工具")
st.caption(f"数据自动更新：{today_str}")

future_days = 5
currencies = ['USD', 'GBP', 'JPY']
models = {
    '神经网络': MLPRegressor(hidden_layer_sizes=(10,10), max_iter=1000, random_state=1),
    '线性回归': LinearRegression(),
    'XGBoost': XGBRegressor(objective='reg:squarederror', n_estimators=100)
}

all_future = {}

for target_currency in currencies:
    st.subheader(f"EUR/{target_currency}")
    # 获取最近30天数据
    end_date = datetime.today()
    start_date = end_date - timedelta(days=30)
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    url = f"https://api.frankfurter.app/{start_str}..{end_str}?from=EUR&to={target_currency}"
    response = requests.get(url)
    data = response.json()

    rates = data['rates']
    dates = sorted(rates.keys())
    prices = [rates[date][target_currency] for date in dates]

    df = pd.DataFrame({'date': pd.to_datetime(dates), 'close': prices})
    df.set_index('date', inplace=True)

    # 最新收盘价
    st.metric(f"最新 EUR/{target_currency} 收盘价", f"{df['close'][-1]:.5f}")

    X = np.arange(len(df)).reshape(-1, 1)
    y = df['close'].values

    future_indexes = np.arange(len(df), len(df)+future_days).reshape(-1,1)
    last_date = df.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, future_days+1)]

    # 保存每个模型预测结果
    predictions = {}

    for name, model in models.items():
        model.fit(X, y)
        preds = model.predict(future_indexes)
        predictions[name] = preds

    # 合成DataFrame
    future_df = pd.DataFrame({'date': future_dates})
    for name in models.keys():
        future_df[name] = predictions[name]
    future_df.set_index('date', inplace=True)

    all_future[target_currency] = future_df

    # 显示表格
    st.dataframe(future_df.style.format("{:.5f}"), use_container_width=True)

    # 绘图
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(df.index, df['close'], label='实际收盘价', color='#1f77b4', linewidth=2)
    colors = ['#ff7f0e', 'green', 'purple']
    for i, name in enumerate(models.keys()):
        ax.plot(future_df.index, future_df[name], label=name, linestyle='--', color=colors[i])
    ax.set_title(f"EUR/{target_currency} 多模型预测", fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

st.success("✅ 所有币种预测完成！")
