import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="纯K吞没突破 - 只做多头版", layout="wide")
st.title("🚀 5分钟纯K吞没突破 - 只做多头版（禁用空头）")
st.caption("只做多头信号 | 纯K吞没突破 | 无任何指标")

# 参数
initial_capital = st.number_input("初始资金 (USDT)", value=10000.0)
fee_rate = st.number_input("手续费率 (双边)", value=0.0010, format="%.4f")
slippage = st.number_input("滑点", value=0.0005, format="%.4f")
rr = 2.5
sl_buffer = 5.0

st.button("🔄 刷新信号与统计")

# 数据加载
file = st.file_uploader("上传 5分钟 OHLCV CSV", type=["csv"])
if file is None: st.stop()

df = pd.read_csv(file)
df.columns = [c.lower() for c in df.columns]
df = df.rename(columns={"vol": "volume"})
df['ts'] = pd.to_datetime(df['ts'], unit='ms')
df = df[['ts', 'open', 'high', 'low', 'close']].sort_values('ts').reset_index(drop=True)

# 纯K多头信号（空头已禁用）
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

df['short_signal'] = False  # 强制禁用空头

# 回测逻辑（同之前）
trades = []
marks = []
equity = initial_capital
equity_curve = [initial_capital]
position = 0
entry_price = 0

for i in range(1, len(df)):
    row = df.iloc[i]
    price = row['open']
    signal = 1 if row['long_signal'] else (-1 if row['short_signal'] else 0)

    if position == 0 and signal != 0:
        entry_price = price + signal * price * slippage
        sl_price = row['low'] - sl_buffer if signal == 1 else row['high'] + sl_buffer
        tp_price = entry_price + signal * (entry_price - sl_price) * rr
        position = signal
        marks.append((i, entry_price, "buy" if signal==1 else "sell"))

    elif position != 0:
        high, low = row['high'], row['low']
        exited = False
        exit_price = 0
        if position == 1:
            if low <= sl_price:
                exit_price = sl_price - row['open'] * slippage
                exited = True
            elif high >= tp_price:
                exit_price = tp_price + row['open'] * slippage
                exited = True
        else:
            if high >= sl_price:
                exit_price = sl_price + row['open'] * slippage
                exited = True
            elif low <= tp_price:
                exit_price = tp_price - row['open'] * slippage
                exited = True

        if exited:
            pnl_points = (exit_price - entry_price) * position
            fee = abs(pnl_points) * fee_rate * 2
            net_pnl = pnl_points - fee
            equity += net_pnl
            trades.append({'idx': i, 'pnl': net_pnl, 'side': 'long' if position==1 else 'short',
                           'sl': sl_price, 'tp': tp_price, 'exit': exit_price})
            marks.append((i, exit_price, "sell" if position==1 else "buy"))
            position = 0
            equity_curve.append(equity)

# 统计显示（同之前）
df_tr = pd.DataFrame(trades)
if len(df_tr) > 0:
    last100 = df_tr.tail(100)
    win_all = (df_tr['pnl'] > 0).mean() * 100
    win_100 = (last100['pnl'] > 0).mean() * 100 if len(last100) > 0 else 0

    st.metric("总交易数", len(df_tr))
    st.metric("近100笔胜率", f"{win_100:.2f}%")
    st.metric("总胜率", f"{win_all:.2f}%")
    st.metric("最终资金", f"{equity:.2f} USDT")

    st.dataframe(df_tr.tail(20)[['idx','side','pnl','sl','tp','exit']])

# 资金曲线 + K线图（同之前代码，省略重复部分）
# ... 把你之前的资金曲线和K线图代码粘贴在这里即可 ...
