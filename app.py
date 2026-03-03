import streamlit as st
import pandas as pd
import ta
import plotly.graph_objects as go
import requests
import numpy as np
from datetime import datetime

# 环境配置
st.set_page_config(layout="wide", page_title="战神 V2600 - 最终满意版")
st.title("🛡️ ETH 小利润·高胜率工程逻辑系统")

# --------------------------
# 侧边栏：核心风控参数
# --------------------------
SYMBOL = "ETH-USDT-SWAP"
with st.sidebar:
    st.header("⚙️ 核心逻辑对齐")
    adx_filter = st.slider("ADX 强度 (过滤横盘)", 15, 40, 26)
    atr_tp_val = st.slider("目标止盈 (ATR倍数)", 0.6, 2.0, 1.1) # 略微拉大以覆盖手续费
    atr_sl_val = 0.6 # 严格遵守小止损逻辑
    
    st.divider()
    st.info("💡 满意逻辑：15M共振 + 低点抬高 + 突破前高确认 + ATR波动适应")

# --------------------------
# 数据处理引擎
# --------------------------
@st.cache_data(ttl=5)
def get_market_data(bar="5m", limit=500):
    url = f"https://www.okx.com/api/v5/market/candles?instId={SYMBOL}&bar={bar}&limit={limit}"
    try:
        r = requests.get(url, timeout=5).json().get("data", [])
        df = pd.DataFrame(r, columns=["ts","open","high","low","close","vol","volCcy","volCcyQuote","confirm"])
        df["ts"] = pd.to_datetime(df["ts"].astype(int), unit="ms")
        for c in ["open","high","low","close","vol"]: df[c] = df[c].astype(float)
        return df.sort_values("ts").reset_index(drop=True)
    except: return pd.DataFrame()

# --------------------------
# 核心策略逻辑
# --------------------------
def execute_strategy(df5, adx_th, tp_m, sl_m):
    # 1. 15分钟多周期共振
    df15 = get_market_data("15m", 100)
    df15["EMA20"] = ta.trend.ema_indicator(df15["close"], window=20)
    tf15_trend = "多" if df15.iloc[-1]["EMA20"] > df15.iloc[-2]["EMA20"] else "空"

    # 2. 5分钟技术形态
    df5["EMA12"] = ta.trend.ema_indicator(df5["close"], window=12)
    df5["EMA50"] = ta.trend.ema_indicator(df5["close"], window=50)
    df5["ADX"] = ta.trend.adx(df5["high"], df5["low"], df5["close"], window=14)
    df5["ATR"] = ta.volatility.average_true_range(df5["high"], df5["low"], df5["close"], window=14)
    
    # 结构：局部高低点确认
    df5["L_min"] = df5["low"].shift(1).rolling(3).min()
    df5["H_max"] = df5["high"].shift(1).rolling(3).max()
    
    trade_history = []
    for i in range(50, len(df5)-1):
        curr = df5.iloc[i]
        nxt = df5.iloc[i+1] # 二次确认K线
        
        # 趋势强度过滤
        strong_trend = curr["ADX"] > adx_th
        ema_cross = curr["EMA12"] > curr["EMA50"] if tf15_trend == "多" else curr["EMA12"] < curr["EMA50"]
        
        signal = None
        # 多头逻辑：趋势对齐 + 低点抬高 + 突破上一根高点进场
        if strong_trend and ema_cross and tf15_trend == "多":
            if curr["low"] > curr["L_min"] and nxt["close"] > curr["high"]:
                signal = "多"
        
        # 空头逻辑：趋势对齐 + 高点降低 + 跌破上一根低点进场
        elif strong_trend and ema_cross and tf15_trend == "空":
            if curr["high"] < curr["H_max"] and nxt["close"] < curr["low"]:
                signal = "空"
                
        if signal:
            entry = nxt["close"]
            sl = entry - curr["ATR"]*sl_m if signal=="多" else entry + curr["ATR"]*sl_m
            tp = entry + curr["ATR"]*tp_m if signal=="多" else entry - curr["ATR"]*tp_m
            
            # 回测模拟 (检查未来10根K线)
            res = 0
            for j in range(i+2, min(i+12, len(df5))):
                future = df5.iloc[j]
                if signal == "多":
                    if future["low"] <= sl: res = -1; break
                    if future["high"] >= tp: res = 1; break
                else:
                    if future["high"] >= sl: res = -1; break
                    if future["low"] <= tp: res = 1; break
            trade_history.append({"时间": nxt["ts"], "方向": signal, "结果": res, "入场价": entry})
            
    return df5, trade_history, tf15_trend

# --------------------------
# UI 表现层
# --------------------------
df_raw = get_market_data()
if not df_raw.empty:
    df_plot, history, current_tf = execute_strategy(df_raw, adx_filter, atr_tp_val, atr_sl_val)
    
    # 顶部数据看板
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("15M 共振", current_tf)
    col2.metric("ADX 趋势强度", f"{df_plot.iloc[-1]['ADX']:.1f}")
    
    if history:
        h_df = pd.DataFrame(history)
        valid_trades = h_df[h_df["结果"] != 0]
        win_rate = (len(valid_trades[valid_trades["结果"] == 1]) / len(valid_trades)) * 100 if not valid_trades.empty else 0
        col3.metric("模拟胜率", f"{win_rate:.1f}%")
        col4.metric("捕捉信号", len(h_df))
        
        st.divider()
        st.subheader("📋 最终回测明细")
        st.dataframe(h_df.tail(10), use_container_width=True)
    else:
        st.warning("⚠️ 当前市场波动率或强度不足，系统严格保持空仓观望。")

    # 动态图表 
    fig = go.Figure(data=[go.Candlestick(x=df_plot['ts'], open=df_plot['open'], high=df_plot['high'], low=df_plot['low'], close=df_plot['close'], name="ETH-SWAP")])
    fig.add_trace(go.Scatter(x=df_plot['ts'], y=df_plot['EMA12'], line=dict(color='yellow', width=1.2), name="EMA12"))
    fig.add_trace(go.Scatter(x=df_plot['ts'], y=df_plot['EMA50'], line=dict(color='orange', width=1.2, dash='dot'), name="EMA50"))
    fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
