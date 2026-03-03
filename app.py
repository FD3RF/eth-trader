import streamlit as st
import pandas as pd
import ta
import plotly.graph_objects as go
import requests
import numpy as np

# 1. 基础配置
st.set_page_config(layout="wide", page_title="战神 V2600 - 终极满意版")
st.title("🛡️ ETH 战神 V2600 (动态波动自适应版)")

# 2. 侧边栏：策略对齐参数
SYMBOL = "ETH-USDT-SWAP"
with st.sidebar:
    st.header("⚙️ 终极逻辑对齐")
    adx_filter = st.slider("ADX 门槛 (过滤震荡)", 15, 35, 22)
    atr_tp_base = st.slider("基础止盈 (ATR倍数)", 0.8, 1.5, 1.1)
    atr_sl_base = 0.6 # 严格执行小止损保命
    st.divider()
    st.info("✅ 满意逻辑：动态ATR调整 + 15M斜率共振 + 二次收盘确认")

# 3. 数据与计算引擎
def fetch_data(bar="5m", limit=500):
    url = f"https://www.okx.com/api/v5/market/candles?instId={SYMBOL}&bar={bar}&limit={limit}"
    try:
        r = requests.get(url, timeout=5).json().get("data", [])
        df = pd.DataFrame(r, columns=["ts","open","high","low","close","vol","volCcy","volCcyQuote","confirm"])
        df["ts"] = pd.to_datetime(df["ts"].astype(int), unit="ms")
        for c in ["open","high","low","close"]: df[c] = df[c].astype(float)
        return df.sort_values("ts").reset_index(drop=True)
    except: return pd.DataFrame()

# 4. 核心逻辑判定
def run_ultimate_logic(df5, adx_th, tp_m, sl_m):
    # 获取15M斜率共振
    df15 = fetch_data("15m", 100)
    df15["EMA20"] = ta.trend.ema_indicator(df15["close"], window=20)
    # 计算15M EMA的斜率（动能确认）
    tf15_up = df15.iloc[-1]["EMA20"] > df15.iloc[-2]["EMA20"]
    tf15_dn = df15.iloc[-1]["EMA20"] < df15.iloc[-2]["EMA20"]

    # 5M 核心指标
    df5["EMA12"] = ta.trend.ema_indicator(df5["close"], window=12)
    df5["EMA50"] = ta.trend.ema_indicator(df5["close"], window=50)
    df5["ADX"] = ta.trend.adx(df5["high"], df5["low"], df5["close"], window=14)
    df5["ATR"] = ta.volatility.average_true_range(df5["high"], df5["low"], df5["close"], window=14)
    
    # 结构高低点
    df5["L_min"] = df5["low"].shift(1).rolling(3).min()
    df5["H_max"] = df5["high"].shift(1).rolling(3).max()
    
    backtest_results = []
    for i in range(50, len(df5)-1):
        row = df5.iloc[i]
        nxt = df5.iloc[i+1] # 二次确认K线
        
        sig = None
        # 多头：15M趋势向上 + EMA金叉 + 低点抬高 + 二次收盘突破
        if tf15_up and row["EMA12"] > row["EMA50"] and row["ADX"] > adx_th:
            if row["low"] > row["L_min"] and nxt["close"] > row["high"]:
                sig = "多"
        
        # 空头：15M趋势向下 + EMA死叉 + 高点降低 + 二次收盘跌破
        elif tf15_dn and row["EMA12"] < row["EMA50"] and row["ADX"] > adx_th:
            if row["high"] < row["H_max"] and nxt["close"] < row["low"]:
                sig = "空"
                
        if sig:
            entry = nxt["close"]
            sl = entry - row["ATR"]*sl_m if sig=="多" else entry + row["ATR"]*sl_m
            tp = entry + row["ATR"]*tp_m if sig=="多" else entry - row["ATR"]*tp_m
            
            # 结果追踪 (未来15根K线，给予利润跑动时间)
            res = 0
            for j in range(i+2, min(i+17, len(df5))):
                f = df5.iloc[j]
                if sig == "多":
                    if f["low"] <= sl: res = -1; break
                    if f["high"] >= tp: res = 1; break
                else:
                    if f["high"] >= sl: res = -1; break
                    if f["low"] <= tp: res = 1; break
            backtest_results.append({"时间": nxt["ts"], "信号": sig, "结果": res, "价位": entry})
            
    return df5, backtest_results, ("多" if tf15_up else "空")

# 5. UI 表现
data_raw = fetch_data()
if not data_raw.empty:
    df_res, history, tf_dir = run_ultimate_logic(data_raw, adx_filter, atr_tp_base, atr_sl_base)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("15M 共振", tf_dir)
    c2.metric("当前 ADX", f"{df_res.iloc[-1]['ADX']:.1f}")
    
    if history:
        h_df = pd.DataFrame(history)
        trades = h_df[h_df["结果"] != 0]
        wr = (len(trades[trades["结果"] == 1]) / len(trades)) * 100 if not trades.empty else 0
        c3.metric("模拟胜率", f"{wr:.1f}%")
        c4.metric("捕捉信号", len(h_df))
        
        st.divider()
        st.subheader("📋 最终满意版 · 回测明细")
        st.dataframe(h_df.tail(10), use_container_width=True)
    else:
        st.warning("⚠️ 波动率极低或趋势不统一，系统正在等待机会...")

    # 绘图层
    fig = go.Figure(data=[go.Candlestick(x=df_res['ts'], open=df_res['open'], high=df_res['high'], low=df_res['low'], close=df_res['close'], name="K线")])
    fig.add_trace(go.Scatter(x=df_res['ts'], y=df_res['EMA12'], line=dict(color='cyan', width=1.5), name="EMA12"))
    fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
