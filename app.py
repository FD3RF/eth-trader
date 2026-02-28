import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go

# ===========================
# 参数
# ===========================
SYMBOL = "ETH-USDT"
LIMIT = 300
CONF_THRESHOLD = 0.55
STRUCT_THRESHOLD = 0.003
LOOKAHEAD = 5

# ===========================
# 数据获取
# ===========================
def fetch_okx(bar="5m"):
    url = f"https://www.okx.com/api/v5/market/candles?instId={SYMBOL}&bar={bar}&limit={LIMIT}"
    try:
        r = requests.get(url, timeout=3).json()
        if r.get("code") == "0":
            df = pd.DataFrame(r["data"], columns=['ts','o','h','l','c','v','volC','volCQ','cf'])
            df = df[::-1].reset_index(drop=True)
            for col in ['o','h','l','c','v']:
                df[col] = df[col].astype(float)
            return df
    except:
        return None


# ===========================
# 特征
# ===========================
def build_features(df):
    df = df.copy()

    df['ema20'] = df['c'].ewm(span=20).mean()
    df['ema60'] = df['c'].ewm(span=60).mean()

    diff = df['c'].diff()
    gain = diff.clip(lower=0).rolling(14).mean()
    loss = -diff.clip(upper=0).rolling(14).mean().replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + (gain / loss)))

    tr = np.maximum(df['h'] - df['l'],
                   np.maximum(abs(df['h'] - df['c'].shift(1)),
                              abs(df['l'] - df['c'].shift(1))))
    df['atr'] = tr.rolling(14).mean()

    return df


# ===========================
# 等高等低
# ===========================
def detect_equal_levels(df):
    df = df.copy()
    df['equal_high_price'] = np.nan
    df['equal_low_price'] = np.nan

    for i in range(30, len(df)):
        window = df.iloc[i-30:i]
        tolerance = df['atr'].iloc[i] * 0.2

        highs = window['h'].values
        lows = window['l'].values

        for h in highs:
            if np.sum(np.abs(highs - h) < tolerance) >= 2:
                df.loc[df.index[i], 'equal_high_price'] = h
                break

        for l in lows:
            if np.sum(np.abs(lows - l) < tolerance) >= 2:
                df.loc[df.index[i], 'equal_low_price'] = l
                break

    return df


# ===========================
# 扫单
# ===========================
def detect_liquidity_sweep(df):
    df = df.copy()
    df['upper_sweep'] = False
    df['lower_sweep'] = False

    recent_high = df['h'].rolling(20).max().shift(1)
    recent_low = df['l'].rolling(20).min().shift(1)
    vol_mean = df['v'].rolling(20).mean()

    df['upper_sweep'] = (df['h'] > recent_high) & (df['c'] < recent_high) & (df['v'] > vol_mean)
    df['lower_sweep'] = (df['l'] < recent_low) & (df['c'] > recent_low) & (df['v'] > vol_mean)

    return df


# ===========================
# FVG
# ===========================
def detect_fvg(df):
    df = df.copy()

    df['bull_fvg_low'] = np.nan
    df['bull_fvg_high'] = np.nan
    df['bear_fvg_low'] = np.nan
    df['bear_fvg_high'] = np.nan

    for i in range(2, len(df)):
        # 多头 FVG
        if df['h'].iloc[i-2] < df['l'].iloc[i]:
            df.loc[df.index[i], 'bull_fvg_low'] = df['h'].iloc[i-2]
            df.loc[df.index[i], 'bull_fvg_high'] = df['l'].iloc[i]

        # 空头 FVG
        if df['l'].iloc[i-2] > df['h'].iloc[i]:
            df.loc[df.index[i], 'bear_fvg_high'] = df['l'].iloc[i-2]
            df.loc[df.index[i], 'bear_fvg_low'] = df['h'].iloc[i]

    return df


# ===========================
# 扫单反转 + FVG 入场
# ===========================
def detect_reversal_with_fvg(df):
    df = df.copy()
    df['ema20_slope'] = df['ema20'].diff()

    df['entry_long'] = False
    df['entry_short'] = False

    for i in range(2, len(df)):

        # 做多反转
        if df['lower_sweep'].iloc[i-1]:
            pool_low = df['equal_low_price'].iloc[i-1]

            if not np.isnan(pool_low):
                if df['c'].iloc[i] > df['h'].iloc[i-1] and df['ema20_slope'].iloc[i] > 0:
                    # 回补 FVG
                    fvg_high = df['bull_fvg_high'].iloc[i]
                    fvg_low = df['bull_fvg_low'].iloc[i]
                    if not np.isnan(fvg_low):
                        if df['l'].iloc[i] <= fvg_high and df['l'].iloc[i] >= fvg_low:
                            df.loc[df.index[i], 'entry_long'] = True

        # 做空反转
        if df['upper_sweep'].iloc[i-1]:
            pool_high = df['equal_high_price'].iloc[i-1]

            if not np.isnan(pool_high):
                if df['c'].iloc[i] < df['l'].iloc[i-1] and df['ema20_slope'].iloc[i] < 0:
                    fvg_high = df['bear_fvg_high'].iloc[i]
                    fvg_low = df['bear_fvg_low'].iloc[i]
                    if not np.isnan(fvg_low):
                        if df['h'].iloc[i] >= fvg_low and df['h'].iloc[i] <= fvg_high:
                            df.loc[df.index[i], 'entry_short'] = True

    return df


# ===========================
# 止损止盈
# ===========================
def calculate_trade_levels(df):
    df = df.copy()
    df['stop_loss'] = np.nan
    df['tp1'] = np.nan
    df['tp2'] = np.nan
    df['rr'] = np.nan

    for i in range(2, len(df)):

        # 多头
        if df['entry_long'].iloc[i]:
            entry = df['c'].iloc[i]
            sweep_low = df['l'].iloc[i-1]
            stop = sweep_low - df['atr'].iloc[i] * 0.2
            risk = entry - stop

            pool_high = df['equal_high_price'].iloc[i]
            if np.isnan(pool_high):
                pool_high = entry + 2 * risk

            tp1 = pool_high
            tp2 = entry + 2 * risk

            df.loc[df.index[i], 'stop_loss'] = stop
            df.loc[df.index[i], 'tp1'] = tp1
            df.loc[df.index[i], 'tp2'] = tp2
            df.loc[df.index[i], 'rr'] = (tp2 - entry) / risk

        # 空头
        if df['entry_short'].iloc[i]:
            entry = df['c'].iloc[i]
            sweep_high = df['h'].iloc[i-1]
            stop = sweep_high + df['atr'].iloc[i] * 0.2
            risk = stop - entry

            pool_low = df['equal_low_price'].iloc[i]
            if np.isnan(pool_low):
                pool_low = entry - 2 * risk

            tp1 = pool_low
            tp2 = entry - 2 * risk

            df.loc[df.index[i], 'stop_loss'] = stop
            df.loc[df.index[i], 'tp1'] = tp1
            df.loc[df.index[i], 'tp2'] = tp2
            df.loc[df.index[i], 'rr'] = (entry - tp2) / risk

    return df


# ===========================
# 仓位计算（1%风险）
# ===========================
def calc_position(account_balance, entry, stop):
    risk_amount = account_balance * 0.01
    per_unit = abs(entry - stop)
    if per_unit == 0:
        return 0
    return risk_amount / per_unit


# ===========================
# UI
# ===========================
st.set_page_config(page_title="V1100 结构分析", layout="wide")
st.title("🧠 V1100 专业结构分析终端")

df = fetch_okx()
if df is None:
    st.warning("数据获取失败")
    st.stop()

df = build_features(df)
df = detect_equal_levels(df)
df = detect_liquidity_sweep(df)
df = detect_fvg(df)
df = detect_reversal_with_fvg(df)
df = calculate_trade_levels(df)

trend = "UP" if df['ema20'].iloc[-1] > df['ema60'].iloc[-1] else "DOWN"

last = df.iloc[-1]

# 信号
signal = "观望"
if last['entry_long']:
    signal = "做多"
elif last['entry_short']:
    signal = "做空"

# 风险
account_balance = 10000
size = 0
if signal != "观望":
    size = calc_position(account_balance, last['c'], last['stop_loss'])

st.metric("趋势", trend)
st.metric("信号", signal)
st.metric("建议仓位", round(size, 4))

if signal != "观望":
    st.write("入场:", last['c'])
    st.write("止损:", last['stop_loss'])
    st.write("TP1:", last['tp1'])
    st.write("TP2:", last['tp2'])
    st.write("RR:", round(last['rr'], 2))

# 图表
fig = go.Figure(data=[go.Candlestick(
    x=df.index, open=df['o'], high=df['h'],
    low=df['l'], close=df['c']
)])
fig.add_trace(go.Scatter(x=df.index, y=df['ema20'], name="EMA20"))
fig.add_trace(go.Scatter(x=df.index, y=df['ema60'], name="EMA60"))
fig.update_layout(height=600, xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)
