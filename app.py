import streamlit as st
import pandas as pd
import ta
import plotly.graph_objects as go
import requests
import numpy as np

# 1. 基础配置
st.set_page_config(layout="wide", page_title="战神 V2600 - 极致优化版")
st.title("🛡️ ETH 战神 V2600 (高胜率·盈亏优化版)")

# 2. 侧边栏：核心逻辑对齐 (这些参数经过回测调优)
SYMBOL = "ETH-USDT-SWAP"
with st.sidebar:
    st.header("⚙️ 盈利引擎参数")
    adx_threshold = st.slider("ADX 强度 (建议>25过滤震荡)", 15, 40, 26)
    atr_tp_mult = st.slider("止盈 ATR 倍数 (拉高盈亏比)", 0.8, 2.5, 1.3)
    atr_sl_mult = 0.6  # 严格止损
    
    st.divider()
    st.info("💡 盈利核心：15M趋势锁死 + 5M放量确认 + 动态止盈")
    if st.button("🔄 手动同步行情"):
        st.rerun()

# 3. 高效数据获取
@st.cache_data(ttl=5)
def get_clean_data(bar="5m", limit=500):
    url = f"https://www.okx.com/api/v5/market/candles?instId={SYMBOL}&bar={bar}&limit={limit}"
    try:
        r = requests.get(url, timeout=5).json().get("data", [])
        df = pd.DataFrame(r, columns=["ts","open","high","low","close","vol","volCcy","volCcyQuote","confirm"])
        df["ts"] = pd.to_datetime(df["ts"].astype(int), unit="ms")
        for c in ["open","high","low","close","vol"]: df[c] = df[c].astype(float)
        return df.sort_values("ts").reset_index(drop=True)
    except:
        return pd.DataFrame()

# 4. 极致回测逻辑引擎
def run_profit_engine(df5, adx_th, tp_m, sl_m):
    # 15M 大趋势动能确认
    df15 = get_clean_data("15m", 100)
    df15["EMA20"] = ta.trend.ema_indicator(df15["close"], window=20)
    tf15_direction = "多" if df15.iloc[-1]["EMA20"] > df15.iloc[-2]["EMA20"] else "空"

    # 5M 指标体系
    df5["EMA12"] = ta.trend.ema_indicator(df5["close"], window=12)
    df5["EMA50"] = ta.trend.ema_indicator(df5["close"], window=50)
    df5["ADX"] = ta.trend.adx(df5["high"], df5["low"], df5["close"], window=14)
    df5["ATR"] = ta.volatility.average_true_range(df5["high"], df5["low"], df5["close"], window=14)
    
    # 局部形态：高低点结构
    df5["L_min_3"] = df5["low"].shift(1).rolling(3).min()
    df5["H_max_3"] = df5["high"].shift(1).rolling(3).max()
    
    results = []
    # 核心循环：增加“二次确认”与“入场间隔”
    for i in range(50, len(df5)-1):
        curr = df5.iloc[i]
        nxt = df5.iloc[i+1]
        
        # 趋势共振判定
        trend_match = (tf15_direction == "多" and curr["EMA12"] > curr["EMA50"]) or \
                      (tf15_direction == "空" and curr["EMA12"] < curr["EMA50"])
        
        signal = None
        # 多头：结构抬高 + 收盘价突破确认
        if trend_match and tf15_direction == "多" and curr["ADX"] > adx_th:
            if curr["low"] > curr["L_min_3"] and nxt["close"] > curr["high"]:
                signal = "多"
        
        # 空头：结构降低 + 收盘价跌破确认
        elif trend_match and tf15_direction == "空" and curr["ADX"] > adx_th:
            if curr["high"] < curr["H_max_3"] and nxt["close"] < curr["low"]:
                signal = "空"
                
        if signal:
            entry_p = nxt["close"]
            sl_p = entry_p - curr["ATR"]*sl_m if signal=="多" else entry_p + curr["ATR"]*sl_m
            tp_p = entry_p + curr["ATR"]*tp_m if signal=="多" else entry_p - curr["ATR"]*tp_m
            
            # 回测模拟：追踪未来15根K线 (寻找盈利最大化)
            trade_res = 0
            for j in range(i+2, min(i+17, len(df5))):
                future = df5.iloc[j]
                if signal == "多":
                    if future["low"] <= sl_p: trade_res = -1; break
                    if future["high"] >= tp_p: trade_res = 1; break
                else:
                    if future["high"] >= sl_p: trade_res = -1; break
                    if future["low"] <= tp_p: trade_res = 1; break
            results.append({"ts": nxt["ts"], "sig": signal, "res": trade_res, "entry": entry_p})
            
    return df5, results, tf15_direction

# 5. 表现层绘制
df_5m = get_clean_data()
if not df_5m.empty:
    df_plot, history, tf_dir = run_profit_engine(df_5m, adx_threshold, atr_tp_mult, atr_sl_mult)
    
    # 顶部面板：实时状态
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("15M 共振方向", tf_dir)
    c2.metric("当前趋势强度 (ADX)", f"{df_plot.iloc[-1]['ADX']:.1f}")
    
    if history:
        h_df = pd.DataFrame(history)
        valid = h_df[h_df["res"] != 0]
        wr = (len(valid[valid["res"] == 1]) / len(valid)) * 100 if not valid.empty else 0
        c3.metric("回测胜率", f"{wr:.1f}%")
        c4.metric("捕捉优质信号", len(h_df))
        
        st.divider()
        st.subheader("📋 盈利信号复盘 (高价值入场)")
        st.dataframe(h_df.tail(10), use_container_width=True)
    else:
        st.warning("⏳ 市场正处于磨损震荡期，系统自动锁定入场权限。")

    # 动态K线图 
    fig = go.Figure(data=[go.Candlestick(x=df_plot['ts'], open=df_plot['open'], high=df_plot['high'], low=df_plot['low'], close=df_plot['close'], name="ETH主图")])
    fig.add_trace(go.Scatter(x=df_plot['ts'], y=df_plot['EMA12'], line=dict(color='yellow', width=1.5), name="EMA12"))
    fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
