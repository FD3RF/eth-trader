import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="5分钟纯K吞没突破 - 只做多头版", layout="wide")
st.title("🚀 5分钟纯K吞没突破 - 只做多头版（禁用空头）")
st.caption("纯K信号 | 无任何指标 | 只做多头")

# 参数
initial_capital = st.number_input("初始资金 (USDT)", value=10000.0)
fee_rate = st.number_input("手续费率 (双边)", value=0.0010, format="%.4f")
slippage = st.number_input("滑点", value=0.0005, format="%.4f")
rr = 2.5
sl_buffer = 5.0

st.button("🔄 刷新信号与统计")

# 数据加载 - 超鲁棒版
file = st.file_uploader("上传 5分钟 OHLCV CSV", type=["csv"])
if file is None:
    st.stop()

try:
    df = pd.read_csv(file)
    df.columns = [str(c).lower().strip() for c in df.columns]

    col_map = {
        'timestamp': 'ts', 'time': 'ts', 'date': 'ts', 'datetime': 'ts',
        'vol': 'volume', 'volume(usdt)': 'volume', 'qty': 'volume'
    }
    df = df.rename(columns=col_map)

    if 'ts' not in df.columns:
        st.error("未找到时间列（ts/timestamp/time），请检查CSV第一行列名")
        st.stop()

    df['ts'] = pd.to_datetime(df['ts'], unit='ms', errors='coerce')
    df = df.dropna(subset=['ts'])
    df = df[['ts', 'open', 'high', 'low', 'close']].sort_values('ts').reset_index(drop=True)

    st.success(f"数据加载成功！共 {len(df)} 根K线")
except Exception as e:
    st.error(f"文件读取失败：{str(e)}")
    st.stop()

# 纯K多头信号（空头禁用）
df['prev_open'] = df['open'].shift(1)
df['prev_high'] = df['high'].shift(1)
df['prev_low'] = df['low'].shift(1)
df['prev_close'] = df['close'].shift(1)

df['long_signal'] = (
    (df['prev_close'] < df['prev_open']) &
    (df['close'] > df['open']) &
    (df['open'] <= df['prev_close']) &
    (df['close'] >= df['prev_open']) &
    (df['close'] > df['prev_high'])
)
df['short_signal'] = False

# 回测
trades = []
marks = []
equity = initial_capital
equity_curve = [initial_capital]
position = 0
entry_price = 0

for i in range(1, len(df)):
    row = df.iloc[i]
    price = row['open']
    signal = 1 if row['long_signal'] else 0

    if position == 0 and signal == 1:
        entry_price = price + price * slippage
        sl_price = row['low'] - sl_buffer
        tp_price = entry_price + (entry_price - sl_price) * rr
        position = 1
        marks.append((i, entry_price, "buy"))

    elif position == 1:
        high, low = row['high'], row['low']
        exited = False
        exit_price = 0
        if low <= sl_price:
            exit_price = sl_price - row['open'] * slippage
            exited = True
        elif high >= tp_price:
            exit_price = tp_price + row['open'] * slippage
            exited = True

        if exited:
            pnl_points = exit_price - entry_price
            fee = abs(pnl_points) * fee_rate * 2
            net_pnl = pnl_points - fee
            equity += net_pnl
            trades.append({'idx': i, 'pnl': net_pnl, 'side': 'long',
                           'sl': sl_price, 'tp': tp_price, 'exit': exit_price})
            marks.append((i, exit_price, "sell"))
            position = 0
            equity_curve.append(equity)

# 统计
df_tr = pd.DataFrame(trades)
st.header("📊 只做多头回测结果")

if len(df_tr) > 0:
    last100 = df_tr.tail(100)
    win_all = (df_tr['pnl'] > 0).mean() * 100
    win_100 = (last100['pnl'] > 0).mean() * 100 if len(last100) > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("总交易数", len(df_tr))
    col2.metric("近100笔胜率", f"{win_100:.2f}%")
    col3.metric("总胜率", f"{win_all:.2f}%")
    col4.metric("最终资金", f"{equity:.2f} USDT")

    st.subheader("最近20笔信号")
    st.dataframe(df_tr.tail(20)[['idx', 'pnl', 'sl', 'tp', 'exit']], use_container_width=True)

# 资金曲线
fig_eq = go.Figure()
fig_eq.add_trace(go.Scatter(y=equity_curve, mode='lines', line=dict(color='cyan')))
fig_eq.update_layout(title="资金曲线（含手续费+滑点）", height=400)
st.plotly_chart(fig_eq, use_container_width=True)

# K线图 + 箭头
fig = go.Figure(data=[go.Candlestick(
    x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'],
    increasing_line_color="lime", decreasing_line_color="red"
)])

for idx, price, action in marks:
    fig.add_trace(go.Scatter(
        x=[df.index[idx]], y=[price],
        mode="markers+text",
        marker=dict(symbol="arrow-up" if action=="buy" else "arrow-down",
                    color="blue" if action=="buy" else "red", size=18),
        text="↑" if action=="buy" else "↓",
        textposition="top center"
    ))

fig.update_layout(title="📈 纯K只做多头（箭头标记）", xaxis_rangeslider_visible=False, height=800)
st.plotly_chart(fig, use_container_width=True)

st.success("✅ 纯K只做多头版运行完成！")
