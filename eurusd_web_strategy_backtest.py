import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import yfinance as yf

plt.style.use('ggplot')
st.set_page_config(page_title="EUR/USD 策略回测", page_icon="📈", layout="wide")
st.markdown("## 📈 EUR/USD 双均线策略回测（近5年）")

# 拉取EUR/USD历史数据
end_date = datetime.today()
start_date = end_date - timedelta(days=5*365)
df = yf.download('EURUSD=X', start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
df = df[['Close']].rename(columns={'Close':'close'}).dropna()
df.index = pd.to_datetime(df.index)

st.caption(f"数据范围：{df.index[0].date()} ~ {df.index[-1].date()}，共 {len(df)} 天")

# 计算短期/长期均线
df['ma10'] = df['close'].rolling(window=10).mean()
df['ma30'] = df['close'].rolling(window=30).mean()

# 信号：短期均线 > 长期均线 → 持仓=1，否则=0
df['signal'] = np.where(df['ma10'] > df['ma30'], 1, 0)

# 计算每日收益率
df['return'] = df['close'].pct_change()
# 策略收益 = 收益率 * signal(昨天的信号)
df['strategy_return'] = df['return'] * df['signal'].shift(1)

# 累计收益曲线
df['cum_strategy'] = (1 + df['strategy_return']).cumprod()
df['cum_buy_hold'] = (1 + df['return']).cumprod()

# 最大回撤
cummax = df['cum_strategy'].cummax()
drawdown = (df['cum_strategy'] - cummax) / cummax
max_drawdown = drawdown.min()

# 显示最终累计收益
final_return = df['cum_strategy'].iloc[-1] - 1
final_bh = df['cum_buy_hold'].iloc[-1] - 1

st.write(f"📊 **策略最终累计收益**：{final_return:.2%}")
st.write(f"📊 **买入持有累计收益**：{final_bh:.2%}")
st.write(f"📉 **策略最大回撤**：{max_drawdown:.2%}")

# 图表
fig, axs = plt.subplots(2, 1, figsize=(10,8), sharex=True)

# 收盘价+均线
axs[0].plot(df.index, df['close'], label='收盘价', color='white')
axs[0].plot(df.index, df['ma10'], label='MA10', color='orange')
axs[0].plot(df.index, df['ma30'], label='MA30', color='cyan')
axs[0].set_title("EUR/USD 收盘价 & 双均线", fontsize=12, color='white')
axs[0].legend(fontsize=8)
axs[0].grid(True, linestyle='--', alpha=0.3)

# 收益曲线
axs[1].plot(df.index, df['cum_strategy'], label='策略', color='green')
axs[1].plot(df.index, df['cum_buy_hold'], label='买入持有', color='gray')
axs[1].set_title("累计收益曲线", fontsize=12, color='white')
axs[1].legend(fontsize=8)
axs[1].grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
st.pyplot(fig, use_container_width=True)

st.success("✅ 策略回测完成！（双均线策略）")
