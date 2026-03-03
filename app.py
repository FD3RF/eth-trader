import streamlit as st
import pandas as pd
import ta
import plotly.graph_objects as go
import requests
import numpy as np
from datetime import datetime

# 页面配置
st.set_page_config(layout="wide", page_title="小利润趋势系统 V2.0")
st.title("⚡ ETH 小利润高周转系统 (最终工程优化版)")

# --------------------------
# 配置参数 (侧边栏)
# --------------------------
SYMBOL = "ETH-USDT-SWAP"
with st.sidebar:
    st.header("⚙️ 策略调优")
    adx_min = st.slider("ADX 强度阈值", 15, 30, 20)
    atr_sl_mult = st.slider("止损倍数 (ATR)", 0.4, 1.0, 0.6)
    atr_tp_mult = st.slider("止盈倍数 (ATR)", 0.5, 1.5, 0.8)
    dist_mult = st.slider("回调允许跨度 (ATR)", 0.5, 1.0, 0.7)
    fee_rate = 0.0005  # 预设 0.05% 手续费+滑点

# --------------------------
# 核心引擎：数据与指标
# --------------------------
@st.cache_data(ttl=10) # 缓存10秒，避免频繁请求被封
def fetch_klines(bar="5m", limit=300):
    url = "https://www.okx.com/api/v5/market/candles"
    params = {"instId": SYMBOL, "bar": bar, "limit": limit}
    try:
        r = requests.get(url, params=params, timeout=5)
        data = r.json().get("data", [])
        df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume","volCcy","volCcyQuote","confirm"])
        df["ts"] = pd.to_datetime(df["ts"].astype(int), unit="ms")
        for c in ["open","high","low","close","volume"]:
            df[c] = df[c].astype(float)
        return df.sort_values("ts").reset_index(drop=True)
    except Exception as e:
        st.error(f"数据获取失败: {e}")
        return pd.DataFrame()

def apply_indicators(df):
    # 基础指标
    df["EMA_fast"] = ta.trend.ema_indicator(df["close"], window=12)
    df["EMA_slow"] = ta.trend.ema_indicator(df["close"], window=50)
    df["ADX"] = ta.trend.adx(df["high"], df["low"], df["close"], window=14)
    df["ATR"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
    
    # 结构逻辑优化：判断过去3根K线的局部高低点
    df["lowest_3"] = df["low"].rolling(window=3).min()
    df["highest_3"] = df["high"].rolling(window=3).max()
    
    # 15min共振缓存逻辑（简化版）
    return df.dropna()

# --------------------------
# 核心逻辑：信号识别
# --------------------------
def get_signal(df, adx_th, dist_m):
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    # 1. 趋势与强度
    trend_up = latest["EMA_fast"] > latest["EMA_slow"] and latest["ADX"] > adx_th
    trend_down = latest["EMA_fast"] < latest["EMA_slow"] and latest["ADX"] > adx_th
    
    # 2. 假突破过滤逻辑 (影线/实体比)
    body = abs(latest["close"] - latest["open"])
    upper_sh = latest["high"] - max(latest["close"], latest["open"])
    lower_sh = min(latest["close"], latest["open"]) - latest["low"]
    is_fake = (upper_sh > body * 1.5) or (lower_sh > body * 1.5)
    
    signal = None
    if trend_up and not is_fake:
        # 低点抬高 (当前低点 > 前一根低点) + 距离EMA够近
        if latest["low"] > prev["low"] and abs(latest["close"] - latest["EMA_fast"]) < latest["ATR"] * dist_m:
            signal = "多"
    elif trend_down and not is_fake:
        # 高点降低 + 距离EMA够近
        if latest["high"] < prev["high"] and abs(latest["close"] - latest["EMA_fast"]) < latest["ATR"] * dist_m:
            signal = "空"
    
    return signal, latest

# --------------------------
# 主程序逻辑
# --------------------------
df_raw = fetch_klines()
if not df_raw.empty:
    df = apply_indicators(df_raw)
    signal, last_row = get_signal(df, adx_min, dist_mult)

    # 计算止损止盈
    sl, tp = 0.0, 0.0
    if signal == "多":
        sl = last_row["close"] - last_row["ATR"] * atr_sl_mult
        tp = last_row["close"] + last_row["ATR"] * atr_tp_mult
    elif signal == "空":
        sl = last_row["close"] + last_row["ATR"] * atr_sl_mult
        tp = last_row["close"] - last_row["ATR"] * atr_tp_mult

    # --------------------------
    # UI 显示
    # --------------------------
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        st.metric("实时价格", f"{last_row['close']:.2f}")
    with col_s2:
        status = "🔥 信号出现" if signal else "⏳ 等待共振"
        st.subheader(f"{status} [{signal if signal else ''}]")
    with col_s3:
        st.metric("ATR 波动", f"{last_row['ATR']:.4f}")

    if signal:
        st.success(f"建议入场: {last_row['close']:.2f} | 止损: {sl:.2f} | 止盈: {tp:.2f} | 盈亏比: {(abs(tp-last_row['close'])/abs(sl-last_row['close'])):.2f}")

    # --------------------------
    # 图表绘制
    # --------------------------
    fig = go.Figure()
    # K线图
    fig.add_trace(go.Candlestick(x=df["ts"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="行情"))
    # 均线
    fig.add_trace(go.Scatter(x=df["ts"], y=df["EMA_fast"], line=dict(color='cyan', width=1.5), name="EMA12"))
    fig.add_trace(go.Scatter(x=df["ts"], y=df["EMA_slow"], line=dict(color='orange', width=1.5), name="EMA50"))
    
    # 如果有信号，画出TP/SL参考线
    if signal:
        fig.add_hline(y=tp, line_dash="dash", line_color="green", annotation_text="止盈目标")
        fig.add_hline(y=sl, line_dash="dash", line_color="red", annotation_text="止损防线")
        # 标记入场点
        fig.add_trace(go.Scatter(x=[last_row["ts"]], y=[last_row["close"]], mode="markers", marker=dict(symbol="star", size=15, color="yellow"), name="入场点"))

    fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # --------------------------
    # 优化版回测逻辑
    # --------------------------
    if st.checkbox("运行逻辑回测 (基于当前数据)"):
        balance = 1000.0
        trade_count = 0
        win_count = 0
        
        for i in range(50, len(df)-1):
            row = df.iloc[i]
            p_row = df.iloc[i-1]
            
            # 复用信号逻辑
            body = abs(row['close'] - row['open'])
            is_fake = (row['high'] - max(row['close'], row['open']) > body * 1.5)
            
            sig = None
            if row["EMA_fast"] > row["EMA_slow"] and row["ADX"] > adx_min:
                if row["low"] > p_row["low"] and not is_fake: sig = "多"
            elif row["EMA_fast"] < row["EMA_slow"] and row["ADX"] > adx_min:
                if row["high"] < p_row["high"] and not is_fake: sig = "空"
            
            if sig:
                trade_count += 1
                entry = row["close"]
                # 模拟小利润止盈止盈
                curr_sl = entry - row["ATR"] * atr_sl_mult if sig == "多" else entry + row["ATR"] * atr_sl_mult
                curr_tp = entry + row["ATR"] * atr_tp_mult if sig == "多" else entry - row["ATR"] * atr_tp_mult
                
                # 检查后续1-3根K线是否触及
                for j in range(i+1, min(i+5, len(df))):
                    future = df.iloc[j]
                    if sig == "多":
                        if future["low"] <= curr_sl: # 损
                            balance -= (entry * 0.01) # 假设1%风险
                            break
                        if future["high"] >= curr_tp: # 盈
                            balance += (entry * 0.01 * (atr_tp_mult/atr_sl_mult))
                            win_count += 1
                            break
                    else:
                        if future["high"] >= curr_sl:
                            balance -= (entry * 0.01)
                            break
                        if future["low"] <= curr_tp:
                            balance += (entry * 0.01 * (atr_tp_mult/atr_sl_mult))
                            win_count += 1
                            break
        
        # 扣除手续费模拟
        balance -= (trade_count * balance * fee_rate)
        
        st.write(f"### 模拟报告")
        c1, c2, c3 = st.columns(3)
        c1.metric("最终盈亏", f"{balance:.2f} USDT")
        c2.metric("交易次数", trade_count)
        c3.metric("模拟胜率", f"{(win_count/trade_count*100 if trade_count else 0):.1f}%")

else:
    st.warning("等待 API 响应中...")
