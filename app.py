import streamlit as st
import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
import time
from datetime import datetime

# =============================
# 1. 核心配置与初始化
# =============================
SYMBOL = "ETH/USDT"
REFRESH_INTERVAL = 3 
LEVERAGE = 100
CIRCUIT_BREAKER_PCT = 0.005 # 0.5% 闪崩熔断

st.set_page_config(layout="wide", page_title="ETH 100x Pro 10-Level System")
exchange = ccxt.binance({"enableRateLimit": True})

if 'last_price' not in st.session_state: st.session_state.last_price = 0
if 'system_halted' not in st.session_state: st.session_state.system_halted = False

# =============================
# 2. 升级版核心函数
# =============================

def detect_regime(df):
    """使用 15m 判断大趋势结构，过滤 5m 噪音"""
    df["adx"] = ta.adx(df["h"], df["l"], df["c"], 14)["ADX_14"]
    df["atr"] = ta.atr(df["h"], df["l"], df["c"], 14)
    df["ema21"] = ta.ema(df["c"], 21)
    
    adx_mean = df["adx"].tail(20).mean()
    atr_val = df["atr"].iloc[-1]
    slope = df["ema21"].iloc[-1] - df["ema21"].iloc[-5]
    
    if adx_mean > 25 and abs(slope) > 0.1: return "TREND", atr_val
    elif atr_val > (df["c"].iloc[-1] * 0.003): return "VOLATILE", atr_val
    else: return "RANGE", atr_val

def tf_score(df):
    """Pro 级评分：结合 EMA, MACD, RSI, ADX 和量能确认"""
    df["ema9"] = ta.ema(df["c"], 9)
    df["ema21"] = ta.ema(df["c"], 21)
    df["rsi"] = ta.rsi(df["c"], 14)
    df["adx"] = ta.adx(df["h"], df["l"], df["c"], 14)["ADX_14"]
    macd = ta.macd(df["c"])
    df["hist"] = macd["MACDh_12_26_9"]

    last = df.iloc[-1]
    score = 0

    # 基础动能评分
    score += 20 if last["ema9"] > last["ema21"] else -20
    score += 20 if last["hist"] > 0 else -20
    score += 25 if last["adx"] > 25 else 0

    # RSI 强弱过滤
    if last["rsi"] > 60: score += 15
    elif last["rsi"] < 40: score -= 15

    # 量能确认 (Volume Confirmation)
    vol_mean = df["v"].rolling(20).mean().iloc[-1]
    if last["v"] > vol_mean * 1.2:
        score += 20 if score > 0 else -20

    return score

def exhaustion_prob(df):
    """核心：计算趋势衰竭概率"""
    # 纠错：确保指标存在
    if "adx" not in df or "hist" not in df: return 0
    adx_drop = df["adx"].iloc[-1] < df["adx"].iloc[-3]
    hist_shrink = abs(df["hist"].iloc[-1]) < abs(df["hist"].iloc[-2])
    vol_drop = df["v"].iloc[-1] < df["v"].rolling(20).mean().iloc[-1]
    return sum([adx_drop, hist_shrink, vol_drop]) / 3

# =============================
# 3. 主作战循环
# =============================
st.title("🛡️ ETH 100x AI 专家自适应系统 (Pro)")

if st.sidebar.button("🔌 重置系统熔断"):
    st.session_state.system_halted = False
    st.session_state.last_price = 0

placeholder = st.empty()

while True:
    try:
        # 1. 毫秒级熔断检测 (修复 Bug 版)
        ticker = exchange.fetch_ticker(SYMBOL)
        current_price = ticker['last']
        change = 0
        
        if st.session_state.last_price != 0:
            change = abs(current_price - st.session_state.last_price) / st.session_state.last_price
            if change > CIRCUIT_BREAKER_PCT:
                st.session_state.system_halted = True
        
        st.session_state.last_price = current_price
        
        if st.session_state.system_halted:
            st.error(f"🚨 系统熔断！价格瞬间异常波动 {change:.2%}。请手动复位。")
            time.sleep(5); continue

        # 2. 获取多周期数据 (5m 执行, 15m 结构)
        b5 = exchange.fetch_ohlcv(SYMBOL, "5m", 150)
        b15 = exchange.fetch_ohlcv(SYMBOL, "15m", 150)
        df5 = pd.DataFrame(b5, columns=["t","o","h","l","c","v"])
        df15 = pd.DataFrame(b15, columns=["t","o","h","l","c","v"])

        # 3. 核心计算
        regime, atr = detect_regime(df15) # 15m 判定结构
        score_5 = tf_score(df5)          # 5m 判定入场动能
        exhaust = exhaustion_prob(df5)   # 5m 判定衰竭

        # 4. 10级 Pro 动态 TP 逻辑
        strength_factor = abs(score_5) / 100
        tp_multiplier = 1.2 + (strength_factor * 2.5) # 强趋势下 TP 扩张
        if exhaust > 0.66: tp_multiplier *= 0.7       # 衰竭时强制收缩 TP

        with placeholder.container():
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("ETH 实时价", f"${current_price}")
            c2.metric("15m 结构", regime)
            c3.metric("5m 强度评分", f"{score_5} pt")
            c4.metric("衰竭概率", f"{round(exhaust*100, 1)}%")

            # 5. 执行计划决策
            if abs(score_5) >= 50:
                side = "LONG" if score_5 > 0 else "SHORT"
                # 100x 风控：ATR 止损与 0.3% 硬止损取最小值
                sl_dist = min(atr * 1.2, current_price * 0.003)
                sl = current_price - sl_dist if side == "LONG" else current_price + sl_dist
                tp = current_price + (current_price - sl) * tp_multiplier if side == "LONG" else current_price - (sl - current_price) * tp_multiplier
                
                # 衰竭高风险警示
                if exhaust > 0.66:
                    st.error(f"⚠️ 动能衰竭高风险：当前倾向于反转或横盘。TP 已下调。")
                
                st.write(f"### 🎯 Pro 作战计划 ({side})")
                sc1, sc2, sc3, sc4 = st.columns(4)
                sc1.success(f"**入场位:** {current_price}")
                sc2.error(f"**止损位:** {round(sl, 2)}")
                sc3.info(f"**动态止盈:** {round(tp, 2)}")
                sc4.metric("盈亏比", f"1:{round(tp_multiplier, 2)}")
            else:
                st.info("💎 扫描中... 5m 动能评分未达阈值，15m 结构信号不明确。")

            # 可视化
            fig = go.Figure(data=[go.Candlestick(x=pd.to_datetime(df5['t'], unit='ms'),
                            open=df5['o'], high=df5['h'], low=df5['l'], close=df5['c'])])
            fig.update_layout(height=400, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.sidebar.error(f"异常: {e}")
    
    time.sleep(REFRESH_INTERVAL)
