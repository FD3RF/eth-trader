import streamlit as st
import pandas as pd
import ta
import plotly.graph_objects as go
import requests
import numpy as np

# 页面基础配置
st.set_page_config(layout="wide", page_title="战神 V2600 - 强化版")
st.title("🛡️ ETH 小利润·趋势共振工程版")

# --------------------------
# 参数与数据获取
# --------------------------
SYMBOL = "ETH-USDT-SWAP"
with st.sidebar:
    st.header("⚙️ 核心逻辑参数")
    adx_threshold = st.slider("ADX 强度阈值 (趋势过滤)", 15, 35, 25)
    atr_sl_mult = 0.6  # 止损系数
    atr_tp_mult = 0.8  # 止盈系数
    dist_buffer = 0.5   # EMA 支撑缓冲区 (ATR倍数)
    backtest_mode = st.checkbox("🔍 开启深度逻辑回测", value=True)

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
# 指标计算引擎
# --------------------------
def apply_indicators(df, df15):
    # 15分钟共振：EMA20方向
    df15["EMA20"] = ta.trend.ema_indicator(df15["close"], window=20)
    latest_tf15_dir = "多" if df15.iloc[-1]["EMA20"] > df15.iloc[-2]["EMA20"] else "空"
    
    # 5分钟指标
    df["EMA_fast"] = ta.trend.ema_indicator(df["close"], window=12)
    df["EMA_slow"] = ta.trend.ema_indicator(df["close"], window=50)
    df["ADX"] = ta.trend.adx(df["high"], df["low"], df["close"], window=14)
    df["ATR"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
    
    # 结构识别：过去3根K线的局部极值（不含当前K线）
    df["low_min_3"] = df["low"].shift(1).rolling(window=3).min()
    df["high_max_3"] = df["high"].shift(1).rolling(window=3).max()
    
    return df, latest_tf15_dir

# --------------------------
# 逻辑判定与回测核心
# --------------------------
def run_strategy_logic(df, tf_dir, adx_th, buf, sl_m, tp_m):
    results = []
    for i in range(50, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        
        # 假突破过滤
        body = abs(row["close"] - row["open"])
        is_fake = (row["high"] - max(row["close"], row["open"]) > body * 1.5) or \
                  (min(row["close"], row["open"]) - row["low"] > body * 1.5)
        
        # 趋势与共振
        trend_up = row["EMA_fast"] > row["EMA_slow"] and row["ADX"] > adx_th
        trend_down = row["EMA_fast"] < row["EMA_slow"] and row["ADX"] > adx_th
        
        signal = None
        if trend_up and tf_dir == "多" and not is_fake:
            if row["low"] > row["low_min_3"] and row["EMA_fast"] < row["close"] < (row["EMA_fast"] + row["ATR"] * buf):
                signal = "多"
        elif trend_down and tf_dir == "空" and not is_fake:
            if row["high"] < row["high_max_3"] and (row["EMA_fast"] - row["ATR"] * buf) < row["close"] < row["EMA_fast"]:
                signal = "空"
        
        if signal:
            # 记录信号用于回测模拟结果
            entry_p = row["close"]
            sl_p = entry_p - row["ATR"] * sl_m if signal == "多" else entry_p + row["ATR"] * sl_m
            tp_p = entry_p + row["ATR"] * tp_m if signal == "多" else entry_p - row["ATR"] * tp_m
            
            # 简单模拟：检查未来5根K线是否触及
            trade_res = 0 # 0: 未果, 1: 胜, -1: 损
            for j in range(i+1, min(i+6, len(df))):
                f = df.iloc[j]
                if signal == "多":
                    if f["low"] <= sl_p: {trade_res := -1}; break
                    if f["high"] >= tp_p: {trade_res := 1}; break
                else:
                    if f["high"] >= sl_p: {trade_res := -1}; break
                    if f["low"] <= tp_p: {trade_res := 1}; break
            
            results.append({"ts": row["ts"], "signal": signal, "res": trade_res, "price": entry_p})
            
    return results

# --------------------------
# 执行与显示
# --------------------------
df_5m = fetch_data("5m")
df_15m = fetch_data("15m")

if not df_5m.empty and not df_15m.empty:
    df_final, tf_dir = apply_indicators(df_5m, df_15m)
    all_signals = run_strategy_logic(df_final, tf_dir, adx_threshold, dist_buffer, atr_sl_mult, atr_tp_mult)
    
    # 当前实时状态
    last_sig = all_signals[-1] if all_signals and all_signals[-1]["ts"] == df_final.iloc[-1]["ts"] else None
    
    st.subheader("📊 实时状态面板")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("15M 共振方向", tf_dir)
    c2.metric("ADX 强度", f"{df_final.iloc[-1]['ADX']:.1f}")
    c3.metric("价格", f"{df_final.iloc[-1]['close']:.2f}")
    c4.metric("检测到信号", "有" if last_sig else "无")

    if backtest_mode:
        st.divider()
        st.subheader("📈 深度回测报告 (基于强化逻辑)")
        if all_signals:
            trades_df = pd.DataFrame(all_signals)
            total = len(trades_df)
            wins = len(trades_df[trades_df["res"] == 1])
            losses = len(trades_df[trades_df["res"] == -1])
            win_rate = (wins / (wins + losses)) * 100 if (wins + losses) > 0 else 0
            
            r1, r2, r3 = st.columns(3)
            r1.metric("总交易次数", total)
            r2.metric("模拟胜率", f"{win_rate:.1f}%")
            r3.metric("净胜场次", f"{wins - losses}")
            
            st.caption("注：回测模拟未来5根K线内的固定止盈止损达成情况。")
        else:
            st.warning("当前参数设置下，回测区间内未触发任何严苛信号。")

    # 图表绘制
    fig = go.Figure(data=[go.Candlestick(x=df_final['ts'], open=df_final['open'], high=df_final['high'], low=df_final['low'], close=df_final['close'], name="K线")])
    fig.add_trace(go.Scatter(x=df_final['ts'], y=df_final['EMA_fast'], line=dict(color='yellow', width=1), name="EMA12"))
    fig.update_layout(height=500, template="plotly_dark", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.error("数据加载失败，请检查网络连接。")
