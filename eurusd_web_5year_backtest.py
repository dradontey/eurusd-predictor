import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import yfinance as yf

plt.style.use('ggplot')

st.set_page_config(page_title="EUR/USD AI 5年回测", page_icon="📈", layout="wide")
st.markdown("## 📈 EUR/USD AI 预测 & 5年历史回测")

# 获取EUR/USD历史数据（用EURUSD=X，yfinance代码）
st.info("正在加载 EUR/USD 历史数据...")
end_date = datetime.today()
start_date = end_date - timedelta(days=5*365)
df = yf.download('EURUSD=X', start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

# 用收盘价
df = df[['Close']].rename(columns={'Close':'close'}).dropna()
df.index = pd.to_datetime(df.index)

st.caption(f"数据范围：{df.index[0].date()} ~ {df.index[-1].date()}，共 {len(df)} 天")

# 显示表格
st.dataframe(df.tail(10).style.format("{:.5f}"), use_container_width=True)

# 划分训练/测试
train_size = int(len(df) * 0.8)
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

# 模型训练
X_train = np.arange(len(train_df)).reshape(-1, 1)
y_train = train_df['close'].values

X_test = np.arange(len(train_df), len(df)).reshape(-1, 1)
y_test = test_df['close'].values

lr = LinearRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)

mlp = MLPRegressor(hidden_layer_sizes=(20,20), max_iter=1000, random_state=1)
mlp.fit(X_train, y_train)
pred_mlp = mlp.predict(X_test)

# 简单回测：画预测 vs 实际
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(df.index, df['close'], label='实际收盘价', color='white')
ax.plot(test_df.index, pred_lr, label='线性回归预测', color='cyan')
ax.plot(test_df.index, pred_mlp, label='神经网络预测', color='orange')
ax.set_title("EUR/USD 收盘价 vs AI预测", fontsize=14, color='white')
ax.legend(fontsize=9)
ax.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
st.pyplot(fig, use_container_width=True)

# 简单统计
mae_lr = np.mean(np.abs(y_test - pred_lr))
mae_mlp = np.mean(np.abs(y_test - pred_mlp))
st.write(f"📊 **回测结果 (最近约 {len(y_test)} 天)：**")
st.write(f"线性回归 MAE（平均绝对误差）: {mae_lr:.5f}")
st.write(f"神经网络 MAE（平均绝对误差）: {mae_mlp:.5f}")

st.success("✅ 简单5年回测完成！（更完整回测还可以加收益曲线、最大回撤等）")
