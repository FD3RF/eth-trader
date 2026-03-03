"""
AI提示 + 小利润策略（优化版）
- 数据层：CSV缓存 + 增量更新
- 回测：RR、滑点、手续费
- 策略：双周期趋势 + 回踩
- 信号：进场提示（不自动交易）
"""

import streamlit as st
import pandas as pd
import ta
import plotly.graph_objects as go
import os
import numpy as np
from datetime import datetime, timedelta
import requests

st.set_page_config(layout="wide", page_title="AI提示+回测优化版")
st.title("📈 AI提示 + 小利润回测")

SYMBOL = "ETH-USDT-SWAP"
DATA_DIR = "market_data"
os.makedirs(DATA_DIR, exist_ok=True)

# ==========================
# 数据层（CSV缓存）
# ==========================
def get_filename(tf):
    return os.path.join(DATA_DIR, f"{SYMBOL}_{tf}.csv")

def load_local(tf):
    if os.path.exists(get_filename(tf)):
        return pd.read_csv(get_filename(tf), parse_dates=["ts"])
    return pd.DataFrame()

def save_local(df, tf):
    df = df.drop_duplicates(subset=["ts"]).sort_values("ts")
    df.to_csv(get_filename(tf), index=False)

def fetch_chunk(tf, before):
    url = "https://www.okx.com/api/v5/market/candles"
    try:
        r = requests.get(url, params={"instId": SYMBOL, "bar": tf, "limit": 1000, "before": before}, timeout=10)
        data = r.json().get("data", [])
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data, columns=["ts","open","high","low","close","vol","volCcy","volCcyQuote","confirm"])
        df["ts"] = pd.to_datetime(df["ts"].astype(int), unit="ms")
        for c in ["open","high","low","close","vol"]:
            df[c] = df[c].astype(float)
        return df
    except:
        return pd.DataFrame()

def update_local(tf, days=90):
    local = load_local(tf)
    if local.empty:
        return fetch_history(tf, days)
    last_ts = local["ts"].max()
    new = []
    current = datetime.now()
    while current > last_ts:
        chunk = fetch_chunk(tf, int(current.timestamp()*1000))
        if chunk.empty:
            break
        chunk = chunk[chunk["ts"] > last_ts]
        if not chunk.empty:
            new.append(chunk)
        current = chunk["ts"].min() if not chunk.empty else last_ts
    if new:
        combined = pd.concat([local, *new])
        save_local(combined, tf)
        return combined
    return local

def fetch_history(tf, days=90):
    end = datetime.now()
    start = end - timedelta(days=days)
    all_data = []
    current = end
    while current > start:
        chunk = fetch_chunk(tf, int(current.timestamp()*1000))
        if chunk.empty:
            break
        all_data.append(chunk)
        current = chunk["ts"].min()
    if all_data:
        df = pd.concat(all_data)
        df = df[df["ts"] >= start]
        save_local(df, tf)
        return df
    return pd.DataFrame()

# ==========================
# 策略参数
# ==========================
with st.sidebar:
    st.header("⚙ 参数")
    tf = st.selectbox("周期", ["5m","15m"], index=0)
    days = st.selectbox("回测天数", ["30","90","180"], index=1)
    lookback = int(days)

    st.divider()
    st.subheader("策略")
    use_trend = st.checkbox("趋势过滤", value=True)
    use_volume = st.checkbox("成交量", value=False)
    use_fake = st.checkbox("假突破", value=False)

    st.divider()
    st.subheader("回测")
    risk_pct = st.slider("单笔风险%", 0.2, 2.0, 1.0)
    slippage = st.number_input("滑点%", 0.05, 0.2, 0.05)
    run_btn = st.button("运行回测")

# ==========================
# 数据加载
# ==========================
if run_btn:
    with st.spinner("加载数据..."):
        df_raw = update_local(tf, days=lookback)
else:
    df_raw = load_local(tf)

if df_raw.empty:
    st.warning("无数据，请点击回测或更新数据")
    st.stop()

start = datetime.now() - timedelta(days=lookback)
df_raw = df_raw[df_raw["ts"] >= start].reset_index(drop=True)
st.success(f"数据：{len(df_raw)} 行 | {df_raw['ts'].min()} → {df_raw['ts'].max()}")

# ==========================
# 指标
# ==========================
df = df_raw.copy()
df["EMA9"] = ta.trend.ema_indicator(df["close"], 9)
df["EMA21"] = ta.trend.ema_indicator(df["close"], 21)
df["ADX"] = ta.trend.adx(df["high"], df["low"], df["close"], 14)
df["ATR"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], 10)
df["volume_ma"] = df["vol"].rolling(20).mean()
df = df.dropna().reset_index(drop=True)

# ==========================
# 结构点
# ==========================
def swing(df, w=3):
    highs, lows = [], []
    for i in range(w, len(df)-w):
        if df["high"].iloc[i] == max(df["high"].iloc[i-w:i+w+1]):
            highs.append((df["ts"].iloc[i], df["high"].iloc[i]))
        if df["low"].iloc[i] == min(df["low"].iloc[i-w:i+w+1]):
            lows.append((df["ts"].iloc[i], df["low"].iloc[i]))
    return highs, lows

s_high, s_low = swing(df)

# ==========================
# 回测引擎（小利润）
# ==========================
def backtest(df):
    capital = 1000
    trades = []
    equity = [capital]
    pos = None

    for i in range(1, len(df)-1):
        row = df.iloc[i]

        # 信号：趋势+回踩
        trend_up = row["EMA9"] > row["EMA21"]
        trend_down = row["EMA9"] < row["EMA21"]

        signal = None
        if use_trend:
            if trend_up and row["close"] <= row["EMA9"] and row["ADX"] > 15:
                signal = "多"
            elif trend_down and row["close"] >= row["EMA9"] and row["ADX"] > 15:
                signal = "空"
        else:
            if row["ADX"] > 15:
                signal = "多" if row["close"] > row["EMA9"] else "空"

        # 开仓
        if signal and pos is None:
            entry = df.iloc[i+1]["open"] * (1 + slippage/100 if signal=="多" else 1 - slippage/100)
            sl = entry - row["ATR"]*0.6 if signal=="多" else entry + row["ATR"]*0.6
            tp = entry + row["ATR"]*1.2 if signal=="多" else entry - row["ATR"]*1.2

            risk = capital * (risk_pct/100)
            dist = abs(entry - sl)
            qty = max(risk/dist, 0.01)
            fee = entry*qty*0.0005
            if capital > fee:
                capital -= fee
                pos = {"dir":signal,"entry":entry,"sl":sl,"tp":tp,"qty":qty,"idx":i+1}

        # 平仓
        if pos:
            cur = df.iloc[i]
            exit_price = None
            reason = None
            if pos["dir"]=="多":
                if cur["low"] <= pos["sl"]:
                    exit_price = pos["sl"]; reason="止损"
                elif cur["high"] >= pos["tp"]:
                    exit_price = pos["tp"]; reason="止盈"
            else:
                if cur["high"] >= pos["sl"]:
                    exit_price = pos["sl"]; reason="止损"
                elif cur["low"] <= pos["tp"]:
                    exit_price = pos["tp"]; reason="止盈"

            if exit_price:
                exit_price *= (1 - slippage/100 if pos["dir"]=="多" else 1 + slippage/100)
                pnl = (exit_price - pos["entry"]) * pos["qty"] if pos["dir"]=="多" else (pos["entry"] - exit_price)*pos["qty"]
                fee = exit_price*pos["qty"]*0.0005
                net = pnl - fee
                capital += net
                trades.append({"盈亏":net,"原因":reason})
                pos = None

        equity.append(capital)

    # 统计
    if trades:
        wins = sum(1 for t in trades if t["盈亏"]>0)
        total = len(trades)
        win_rate = wins/total*100
        net = capital-1000
        returns = np.diff(equity)/equity[:-1] if len(equity)>1 else []
        sharpe = np.mean(returns)/np.std(returns)*np.sqrt(365*24*12) if len(returns)>1 and np.std(returns)>0 else 0
        peak = np.maximum.accumulate(equity)
        dd = (peak-equity)/peak
        mdd = np.max(dd)*100
        return trades, win_rate, net, sharpe, mdd, equity
    return [],0,0,0,0,equity

# ==========================
# 回测运行
# ==========================
if run_btn:
    with st.spinner("回测中..."):
        trades, wr, net, sharpe, mdd, equity = backtest(df)

    st.subheader("回测结果")
    st.metric("交易", len(trades))
    st.metric("胜率", f"{wr:.1f}%")
    st.metric("净利润", f"{net:.2f}")
    st.metric("夏普", f"{sharpe:.2f}")
    st.metric("最大回撤", f"{mdd:.2f}%")

    # 资金曲线
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=equity, mode="lines"))
    fig.update_layout(title="资金曲线", height=400)
    st.plotly_chart(fig, use_container_width=True)

# ==========================
# AI进场提示（辅助）
# ==========================
latest = df.iloc[-1]
signal = None
if latest["ADX"] > 15:
    if latest["EMA9"] > latest["EMA21"] and latest["close"] <= latest["EMA9"]:
        signal = "多"
    elif latest["EMA9"] < latest["EMA21"] and latest["close"] >= latest["EMA9"]:
        signal = "空"

st.divider()
st.subheader("AI提示")
if signal:
    st.success(f"建议方向：{signal}")
    st.caption("理由：趋势+回踩+ADX")
else:
    st.info("无明确机会")
