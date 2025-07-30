import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

plt.style.use('ggplot')
st.set_page_config(page_title="EUR/USD XGBoost 回测", page_icon="🧠", layout="wide")
st.markdown("## 🧠 EUR/USD XGBoost AI 策略回测（近5年）")

# 拉取数据
end_date = datetime.today()
start_date = end_date - timedelta(days=5*365)
df = yf.download('EURUSD=X', start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
df = df[['Close']].rename(columns={'Close':'close'}).dropna()
df.index = pd.to_datetime(df.index)

if df.empty:
    st.error("❌ 没有获取到数据")
else:
    st.caption(f"数据范围：{df.index[0].date()} ~ {df.index[-1].date()}，共 {len(df)} 天")

    # 特征工程
    df['return_1d'] = df['close'].pct_change()
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma10'] = df['close'].rolling(10).mean()
    df['ma5_ma10_diff'] = df['ma5'] - df['ma10']
    df = df.dropna()

    # 目标变量：明天涨跌（1/0）
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df = df.dropna()

    # 划分训练/测试
    split_idx = int(len(df)*0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = train_df[['return_1d', 'ma5_ma10_diff']]
    y_train = train_df['target']
    X_test = test_df[['return_1d', 'ma5_ma10_diff']]
    y_test = test_df['target']

    # XGBoost 模型
    model = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    # 预测
    preds = model.predict(X_test)

    # 策略：预测=1→做多，否则空仓
    test_df['signal'] = preds
    test_df['return'] = test_df['close'].pct_change()
    test_df['strategy_return'] = test_df['return'] * test_df['signal'].shift(1)

    # 累计收益
    test_df['cum_strategy'] = (1 + test_df['strategy_return']).cumprod()
    test_df['cum_buy_hold'] = (1 + test_df['return']).cumprod()

    # 最大回撤
    cummax = test_df['cum_strategy'].cummax()
    drawdown = (test_df['cum_strategy'] - cummax) / cummax
    max_drawdown = drawdown.min()

    final_return = test_df['cum_strategy'].iloc[-1] - 1
    final_bh = test_df['cum_buy_hold'].iloc[-1] - 1

    st.write(f"📊 **策略最终累计收益**：{final_return:.2%}")
    st.write(f"📊 **买入持有累计收益**：{final_bh:.2%}")
    st.write(f"📉 **策略最大回撤**：{max_drawdown:.2%}")

    # 图表
    fig, axs = plt.subplots(2, 1, figsize=(10,8), sharex=True)

    # 收盘价+预测信号
    axs[0].plot(test_df.index, test_df['close'], label='收盘价', color='white')
    axs[0].scatter(test_df.index[test_df['signal']==1], test_df['close'][test_df['signal']==1], label='预测做多', color='green', s=10)
    axs[0].set_title("收盘价 & AI预测信号", fontsize=12, color='white')
    axs[0].legend(fontsize=8)
    axs[0].grid(True, linestyle='--', alpha=0.3)

    # 收益曲线
    axs[1].plot(test_df.index, test_df['cum_strategy'], label='策略', color='green')
    axs[1].plot(test_df.index, test_df['cum_buy_hold'], label='买入持有', color='gray')
    axs[1].set_title("累计收益曲线", fontsize=12, color='white')
    axs[1].legend(fontsize=8)
    axs[1].grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    st.success("✅ XGBoost AI 策略回测完成！")
