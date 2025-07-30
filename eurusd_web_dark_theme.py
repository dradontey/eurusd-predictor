import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
import numpy as np
import matplotlib.pyplot as plt
import requests

# 🎨 设置 matplotlib 深色主题
plt.style.use('dark_background')

# 🌙 设置页面配置
st.set_page_config(
    page_title="AI 外汇预测仪表盘",
    page_icon="🌙",
    layout="wide"
)

# 🖼 顶部 logo + 标题
logo_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/9/97/Circle-icons-currency.svg/1200px-Circle-icons-currency.svg.png"
st.markdown(f"""
<div style='text-align: center;'>
    <img src="{logo_url}" width="80">
    <h1 style='color: white;'>AI 多币种外汇预测</h1>
    <p style='color: gray;'>深色模式 • 自动更新 • 多模型预测 • EMA / ATR / KDJ</p>
</div>
""", unsafe_allow_html=True)

st.caption(f"数据自动更新：{datetime.today().strftime('%Y-%m-%d')}")

future_days = 5
currencies = ['USD', 'GBP', 'JPY']
models = {
    '神经网络': MLPRegressor(hidden_layer_sizes=(10,10), max_iter=1000, random_state=1),
    '线性回归': LinearRegression(),
    'XGBoost': XGBRegressor(objective='reg:squarederror', n_estimators=100)
}

for target_currency in currencies:
    st.markdown(f"## 🌍 EUR/{target_currency}")

    # 获取最近30天数据
    end_date = datetime.today()
    start_date = end_date - timedelta(days=30)
    url = f"https://api.frankfurter.app/{start_date.strftime('%Y-%m-%d')}..{end_date.strftime('%Y-%m-%d')}?from=EUR&to={target_currency}"
    data = requests.get(url).json()
    dates = sorted(data['rates'].keys())
    prices = [data['rates'][date][target_currency] for date in dates]

    df = pd.DataFrame({'date': pd.to_datetime(dates), 'close': prices})
    df.set_index('date', inplace=True)

    # 技术指标
    df['ema10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['high'] = df['close'].rolling(2).max()
    df['low'] = df['close'].rolling(2).min()
    df['tr'] = df['high'] - df['low']
    df['atr'] = df['tr'].rolling(window=14).mean()

    low_min = df['low'].rolling(window=9).min()
    high_max = df['high'].rolling(window=9).max()
    rsv = (df['close'] - low_min) / (high_max - low_min) * 100
    df['kdj_k'] = rsv.ewm(com=2).mean()
    df['kdj_d'] = df['kdj_k'].ewm(com=2).mean()
    df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']

    st.metric(f"最新收盘价 EUR/{target_currency}", f"{df['close'][-1]:.5f}")

    # 模型预测
    X = np.arange(len(df)).reshape(-1, 1)
    y = df['close'].values
    future_indexes = np.arange(len(df), len(df)+future_days).reshape(-1,1)
    last_date = df.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, future_days+1)]

    predictions = {}
    for name, model in models.items():
        model.fit(X, y)
        predictions[name] = model.predict(future_indexes)

    future_df = pd.DataFrame({'date': future_dates})
    for name in models.keys():
        future_df[name] = predictions[name]
    future_df.set_index('date', inplace=True)

    st.markdown("**未来5天预测（多模型对比）：**")
    st.dataframe(future_df.style.format("{:.5f}"), use_container_width=True)

    # 绘图
    fig, axs = plt.subplots(2, 1, figsize=(10,6), sharex=True)

    axs[0].plot(df.index, df['close'], label='收盘价', color='white')
    axs[0].plot(df.index, df['ema10'], label='EMA10', color='orange', alpha=0.8)
    colors = ['#ff7f0e', 'green', 'purple']
    for i, name in enumerate(models.keys()):
        axs[0].plot(future_df.index, future_df[name], label=name, linestyle='--', color=colors[i])
    axs[0].set_title("收盘价 + AI预测 + EMA", fontsize=12, color='white')
    axs[0].legend(fontsize=8)
    axs[0].grid(True, linestyle='--', alpha=0.3)

    axs[1].plot(df.index, df['atr'], label='ATR', color='red', linewidth=1.5)
    axs[1].plot(df.index, df['kdj_k'], label='K', color='cyan')
    axs[1].plot(df.index, df['kdj_d'], label='D', color='magenta')
    axs[1].plot(df.index, df['kdj_j'], label='J', color='yellow')
    axs[1].set_title("ATR & KDJ", fontsize=12, color='white')
    axs[1].legend(fontsize=8)
    axs[1].grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

st.success("✨ 深色模式 & logo & 多币种预测完成！")
