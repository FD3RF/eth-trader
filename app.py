import streamlit as st
import pandas as pd
import ta
import plotly.graph_objects as go
import requests
import numpy as np
from datetime import datetime

# 页面基础配置
st.set_page_config(layout="wide", page_title="战神 V2600 - 二次确认版")
st.title("🛡️ ETH 小利润·二次确认共振系统")

# --------------------------
# 侧边栏参数 (核心工程参数)
# --------------------------
SYMBOL = "ETH-USDT-SWAP"
with st.sidebar:
    st.header("⚙️ 核心逻辑参数")
    adx_threshold = st.slider("ADX 强度 (建议>25)", 15, 35, 28) # 强化趋势过滤
    atr_sl_mult = 0.6  # 固定小止损
    atr_tp_mult = 0.8  # 固定小止盈
    dist_buffer = 0.5  # EMA 支撑缓冲区
    st.divider()
    enable_confirm = st.checkbox("开启『二次确认』逻辑", value=True) # 核心升级点
    backtest_mode = st.checkbox("🔍 开启回测分析", value=True)

# --------------------------
# 数据引擎 (OKX API)
# --------------------------
def fetch_data(bar="5m", limit=500):
    url = "https://www.okx.com/api/v5/market/candles"
    params = {"instId": SYMBOL, "bar": bar, "limit": limit}
    try:
        r = requests.get(url, params=params, timeout=5)
        df = pd.DataFrame(r.json()["data"], columns=["ts","open","high","low","close","vol","volCcy","volCcyQuote","confirm"])
        df["ts"] = pd.to_datetime(df["ts"].astype(int), unit="ms")
        for c in ["open","high","low","close"]: df[c] = df[c].astype(float)
        return df.sort_values("ts").reset_index(drop=True)
    except: return pd.DataFrame()

# --------------------------
# 指标与逻辑判定核心
# --------------------------
def apply_refined_logic(df, adx_th, buf):
    # 15分钟趋势共振 (EMA20)
    df15 = fetch_data("15m", 100)
    df15["EMA20"] = ta.trend.ema_indicator(df15["close"], window=20)
    tf15_dir = "多" if df15.iloc[-1]["EMA20"] > df15.iloc[-2]["EMA20"] else "空"
    
    # 5分钟指标计算
    df["EMA_fast"] = ta.trend.ema_indicator(df["close"], window=12)
    df["EMA_slow"] = ta.trend.ema_indicator(df["close"], window=50)
    df["ADX"] = ta.trend.adx(df["high"], df["low"], df["close"], window=14)
    df["ATR"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
    
    # 结构识别：过去3根K线极值
    df["low_min_3"] = df["low"].shift(1).rolling(window=3).min()
    df["high_max_3"] = df["high"].shift(1).rolling(window=3).max()
    
    signals = []
    # 循环判定 (含二次确认)
    for i in range(50, len(df)-1):
        row = df.iloc[i]
        next_row = df.iloc[i+1] # 确认K线
        
        # 1. 假突破过滤
        body = abs(row["close"] - row["open"])
        is_fake = (row["high"] - max(row["close"], row["open"]) > body * 1.5)
        
        # 2. 趋势与强度
        trend_up = row["EMA_fast"] > row["EMA_slow"] and row["ADX"] > adx_th
        trend_down = row["EMA_fast"] < row["EMA_slow"] and row["ADX"] > adx_th
        
        sig = None
        # 多头二次确认：15m共振 + 低点抬高 + 突破前K高点
        if trend_up and tf15_dir == "多" and not is_fake:
            if row["low"] > row["low_min_3"] and row["close"] < (row["EMA_fast"] + row["ATR"] * buf):
                if not enable_confirm or next_row["close"] > row["high"]: 
                    sig = "多"
        
        # 空头二次确认：15m共振 + 高点降低 + 跌破前K低点
        elif trend_down and tf15_dir == "空" and not is_fake:
            if row["high"] < row["high_max_3"] and row["close"] > (row["EMA_fast"] - row["ATR"] * buf):
                if not enable_confirm or next_row["close"] < row["low"]:
                    sig = "空"
        
        if sig:
            # 模拟回测：检查未来5根K线
            entry_p = next_row["close"] if enable_confirm else row["close"]
            sl_p = entry_p - row["ATR"] * atr_sl_mult if sig == "多" else entry_p + row["ATR"] * atr_sl_mult
            tp_p = entry_p + row["ATR"] * atr_tp_mult if sig == "多" else entry_p - row["ATR"] * atr_tp_mult
            
            res = 0
            for j in range(i+2, min(i+7, len(df))):
                f = df.iloc[j]
                if sig == "多":
                    if f["low"] <= sl_p: res = -1; break
                    if f["high"] >= tp_p: res = 1; break
                else:
                    if f["high"] >= sl_p: res = -1; break
                    if f["low"] <= tp_p: res = 1; break
            signals.append({"ts": next_row["ts"], "sig": sig, "res": res, "price": entry_p})
            
    return df, signals, tf15_dir

# --------------------------
# UI 渲染
# --------------------------
df_raw = fetch_data("5m")
if not df_raw.empty:
    df_final, all_sigs, current_tf = apply_refined_logic(df_raw, adx_threshold, dist_buffer)
    
    # 状态面板
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("15M 共振方向", current_tf)
    c2.metric("当前 ADX", f"{df_final.iloc[-1]['ADX']:.1f}")
    c3.metric("确认模式", "开启" if enable_confirm else "关闭")
    
    # 回测显示
    if backtest_mode and all_sigs:
        tdf = pd.DataFrame(all_sigs)
        wins = len(tdf[tdf["res"] == 1])
        losses = len(tdf[tdf["res"] == -1])
        wr = (wins / (wins + losses)) * 100 if (wins + losses) > 0 else 0
        c4.metric("回测胜率", f"{wr:.1f}%")
        
        st.divider()
        st.subheader("📈 历史信号复盘")
        st.dataframe(tdf.tail(10), use_container_width=True)

    # 图表
    fig = go.Figure(data=[go.Candlestick(x=df_final['ts'], open=df_final['open'], high=df_final['high'], low=df_final['low'], close=df_final['close'], name="K线")])
    fig.add_trace(go.Scatter(x=df_final['ts'], y=df_final['EMA_fast'], line=dict(color='cyan', width=1), name="EMA12"))
    fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
