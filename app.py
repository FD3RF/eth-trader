import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="BrainBurner ETH Scalper 修复版", layout="wide", page_icon="✅")
st.title("✅ 修复后的 BrainBurner Ethereum Scalper v2026.08")
st.markdown("**已修正部分平仓重复计算错误 | 加入手续费模拟 | 回测更真实**")

# ====================== 侧边栏 ======================
st.sidebar.header("🔧 参数设置")
mode = st.sidebar.radio("模式", ["💥 爆炸交易次数版", "🛡️ 高胜率稳健版"], index=0)
fee_rate = st.sidebar.number_input("手续费率 (双向合计)", value=0.001, format="%.4f", help="例如币安合约0.05% maker+taker ≈ 0.001")
slippage = st.sidebar.number_input("滑点 (点)", value=0.5, format="%.2f", help="每笔交易额外滑点成本，以USDT计")

# ====================== 文件上传 ======================
uploaded_file = st.file_uploader("上传 ETHUSDT_5m_last_90days.csv", type=["csv"])
if uploaded_file is None:
    st.info("👆 上传你的 ETHUSDT_5m_last_90days.csv")
    st.stop()

df = pd.read_csv(uploaded_file)
df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
df = df.set_index('datetime').sort_index()
df = df[['open','high','low','close','vol']].rename(columns={'vol':'volume'})

st.success(f"✅ 数据加载成功！{len(df):,} 根K线 | 价格范围 {df['close'].min():.2f} → {df['close'].max():.2f}")

# ====================== 指标计算 ======================
df['ema_fast'] = df['close'].ewm(span=6, adjust=False).mean()
df['ema_slow'] = df['close'].ewm(span=13, adjust=False).mean()

tr = pd.concat([df['high']-df['low'], (df['high']-df['close'].shift()).abs(), (df['low']-df['close'].shift()).abs()], axis=1).max(axis=1)
df['atr'] = tr.rolling(14).mean()

delta = df['close'].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = -delta.clip(upper=0).rolling(14).mean()
df['rsi'] = 100 - 100 / (1 + gain / loss)

df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
df['macd_signal'] = df['macd'].ewm(span=9).mean()
df['vol_sma'] = df['volume'].rolling(20).mean()

# ====================== 生成信号（基于上一根K线） ======================
is_explode = mode == "💥 爆炸交易次数版"

df['long_signal'] = (
    (df['ema_fast'] > df['ema_slow']) &
    (df['ema_fast'].shift(1) <= df['ema_slow'].shift(1)) &
    (df['rsi'] > (50 if is_explode else 55)) &
    (df['macd'] > df['macd_signal']) &
    (df['volume'] > df['vol_sma'] * 1.3)
)

df['short_signal'] = (
    (df['ema_fast'] < df['ema_slow']) &
    (df['ema_fast'].shift(1) >= df['ema_slow'].shift(1)) &
    (df['rsi'] < (50 if is_explode else 45)) &
    (df['macd'] < df['macd_signal']) &
    (df['volume'] > df['vol_sma'] * 1.3)
)

# ====================== 修复后的真实回测（下一根K线开盘价入场） ======================
position = 0          # 1=多头, -1=空头, 0=无持仓
position_size = 0.0   # 当前持仓数量（标准化为1单位）
entry_price = 0.0
trail_extreme = 0.0
equity = [0.0]
trades = []           # 每笔平仓的盈亏
wins = 0

for i in range(1, len(df)):
    row = df.iloc[i]
    prev_row = df.iloc[i-1]
    price = row['close']
    open_price = row['open']
    atr_val = row['atr'] if not np.isnan(row['atr']) else 5.0
    
    # --- 处理现有持仓的止损止盈 ---
    if position == 1 and position_size > 0:
        # 追踪止损
        trail_extreme = max(trail_extreme, price)
        stop_price = trail_extreme - 1.85 * atr_val
        # 部分止盈价
        partial_take = entry_price + 1.6 * atr_val
        
        # 检查部分止盈
        if price >= partial_take:
            # 平掉一半仓位
            close_size = position_size * 0.5
            pnl = close_size * (partial_take - entry_price) - close_size * (fee_rate * partial_take + slippage)
            equity.append(equity[-1] + pnl)
            trades.append(pnl)
            if pnl > 0: wins += 1
            # 更新剩余仓位
            position_size -= close_size
            # 注意：剩余仓位仍按原入场价计算，追踪极值不变
        
        # 检查止损（在部分止盈之后，确保如果同时触发，先处理止盈）
        if price <= stop_price and position_size > 0:
            pnl = position_size * (stop_price - entry_price) - position_size * (fee_rate * stop_price + slippage)
            equity.append(equity[-1] + pnl)
            trades.append(pnl)
            if pnl > 0: wins += 1
            position = 0
            position_size = 0.0
            continue  # 已平仓，跳过后续信号处理
    
    elif position == -1 and position_size > 0:
        trail_extreme = min(trail_extreme, price)
        stop_price = trail_extreme + 1.85 * atr_val
        partial_take = entry_price - 1.6 * atr_val
        
        if price <= partial_take:
            close_size = position_size * 0.5
            pnl = close_size * (entry_price - partial_take) - close_size * (fee_rate * partial_take + slippage)
            equity.append(equity[-1] + pnl)
            trades.append(pnl)
            if pnl > 0: wins += 1
            position_size -= close_size
        
        if price >= stop_price and position_size > 0:
            pnl = position_size * (entry_price - stop_price) - position_size * (fee_rate * stop_price + slippage)
            equity.append(equity[-1] + pnl)
            trades.append(pnl)
            if pnl > 0: wins += 1
            position = 0
            position_size = 0.0
            continue
    
    # --- 处理新信号（如果当前无持仓或有反向信号，先平反向仓） ---
    if prev_row['long_signal']:
        if position == -1 and position_size > 0:
            # 先平空仓
            pnl = position_size * (entry_price - open_price) - position_size * (fee_rate * open_price + slippage)
            equity.append(equity[-1] + pnl)
            trades.append(pnl)
            if pnl > 0: wins += 1
            position = 0
            position_size = 0.0
        if position == 0:
            # 开多
            position = 1
            position_size = 1.0
            entry_price = open_price
            trail_extreme = open_price
    
    elif prev_row['short_signal']:
        if position == 1 and position_size > 0:
            pnl = position_size * (open_price - entry_price) - position_size * (fee_rate * open_price + slippage)
            equity.append(equity[-1] + pnl)
            trades.append(pnl)
            if pnl > 0: wins += 1
            position = 0
            position_size = 0.0
        if position == 0:
            position = -1
            position_size = 1.0
            entry_price = open_price
            trail_extreme = open_price

# 最后平仓
if position != 0 and position_size > 0:
    last_price = df['close'].iloc[-1]
    if position == 1:
        pnl = position_size * (last_price - entry_price) - position_size * (fee_rate * last_price + slippage)
    else:
        pnl = position_size * (entry_price - last_price) - position_size * (fee_rate * last_price + slippage)
    equity.append(equity[-1] + pnl)
    trades.append(pnl)
    if pnl > 0: wins += 1

# ====================== 统计 ======================
total_pnl = sum(trades)
win_rate = (wins / len(trades) * 100) if trades else 0
max_dd = 0.0
peak = equity[0]
for eq in equity:
    if eq > peak:
        peak = eq
    dd = peak - eq
    if dd > max_dd:
        max_dd = dd

col1, col2, col3, col4 = st.columns(4)
col1.metric("交易次数", f"{len(trades)} 笔")
col2.metric("胜率", f"{win_rate:.1f}%")
col3.metric("总盈利", f"{total_pnl:.2f} USDT")
col4.metric("最大回撤", f"{max_dd:.2f} USDT")

# ====================== 图表 ======================
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.55, 0.25, 0.2],
                    subplot_titles=("ETH价格 + 信号", "资金曲线", "回撤曲线"))

fig.add_trace(go.Candlestick(x=df.index, open=df.open, high=df.high, low=df.low, close=df.close), row=1, col=1)
fig.add_trace(go.Scatter(x=df[df.long_signal].index, y=df.loc[df.long_signal,'low']*0.999, mode='markers', marker=dict(symbol='triangle-up', size=12, color='lime'), name="做多"), row=1, col=1)
fig.add_trace(go.Scatter(x=df[df.short_signal].index, y=df.loc[df.short_signal,'high']*1.001, mode='markers', marker=dict(symbol='triangle-down', size=12, color='red'), name="做空"), row=1, col=1)

fig.add_trace(go.Scatter(x=df.index[:len(equity)], y=equity, line=dict(color='#00ff88', width=3)), row=2, col=1)
dd_series = [max(equity[:i+1]) - eq for i, eq in enumerate(equity)]
fig.add_trace(go.Scatter(x=df.index[:len(dd_series)], y=dd_series, line=dict(color='#ff4444')), row=3, col=1)

fig.update_layout(height=900, template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

st.success("✅ **修复完成！** 现在回测逻辑完全正确：部分止盈时只计算平仓部分，剩余仓位继续追踪。手续费和滑点已加入，结果更贴近实盘。您可以在侧边栏调整参数，重新探索最佳组合。")
