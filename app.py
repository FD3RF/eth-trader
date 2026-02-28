import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime

# ==================== 1. 核心配置 ====================
st.set_page_config(page_title="ETH V75 高频捕捉版", layout="wide")

def fetch_okx_data(endpoint, params=""):
    url = f"https://www.okx.com/api/v5/{endpoint}?instId=ETH-USDT{params}"
    try:
        r = requests.get(url, timeout=5).json()
        return r if r.get('code') == '0' else None
    except: return None

# ==================== 2. V75 高频双周期引擎 ====================
def engine_v75(df_1m, df_5m, net_flow, buy_ratio):
    # --- 1min 灵敏指标 ---
    df_1m['ema20'] = df_1m['c'].ewm(span=20, adjust=False).mean()
    df_1m['std'] = df_1m['c'].rolling(20).std()
    df_1m['upper'] = df_1m['ema20'] + (df_1m['std'] * 1.8) # 缩窄通道，提高频率
    df_1m['lower'] = df_1m['ema20'] - (df_1m['std'] * 1.8)
    
    # --- 5min 趋势背景 ---
    ema5_20 = df_5m['c'].ewm(span=20, adjust=False).mean().iloc[-1]
    ema5_60 = df_5m['c'].ewm(span=60, adjust=False).mean().iloc[-1]
    main_trend = "BULL" if ema5_20 > ema5_60 else "BEAR"

    # --- 资金评分 (V75 加强版) ---
    f_score = 0
    if net_flow > 2: f_score += 1      # 降低金额门槛
    if net_flow > 8: f_score += 2
    if buy_ratio > 52: f_score += 1
    if net_flow < -2: f_score -= 1
    if net_flow < -8: f_score -= 2
    if buy_ratio < 48: f_score -= 1

    # --- 信号决策 (1min 级别突破) ---
    curr = df_1m.iloc[-1]
    vol_mean = df_1m['v'].rolling(10).mean().iloc[-1]
    signal = {"type": None, "dir": None, "score": 0, "desc": ""}

    # 逻辑 A：高频趋势追随 (顺 5m 势，1m 回踩/放量)
    if main_trend == "BULL" and f_score >= 1:
        if curr['c'] > df_1m['ema20'].iloc[-1] and curr['v'] > vol_mean:
            signal = {"type": "1M-趋势点火", "dir": "LONG", "score": 5 + f_score, "desc": "5M看多，1M动能放量"}
    elif main_trend == "BEAR" and f_score <= -1:
        if curr['c'] < df_1m['ema20'].iloc[-1] and curr['v'] > vol_mean:
            signal = {"type": "1M-趋势点火", "dir": "SHORT", "score": 5 + abs(f_score), "desc": "5M看空，1M动能下破"}

    # 逻辑 B：极短线乖离 (1m 布林突破)
    if not signal['type'] and abs(f_score) >= 2:
        if curr['c'] <= df_1m['lower'].iloc[-1]:
            signal = {"type": "1M-极短抄底", "dir": "LONG", "score": 7, "desc": "1M超跌且主力护盘"}
        elif curr['c'] >= df_1m['upper'].iloc[-1]:
            signal = {"type": "1M-极短摸顶", "dir": "SHORT", "score": 7, "desc": "1M超买且主力撤离"}

    return signal, main_trend, f_score

# ==================== 3. 渲染层 ====================
# 获取 1m 和 5m 数据
k1_raw = fetch_okx_data("market/candles", "&bar=1m&limit=100")
k5_raw = fetch_okx_data("market/candles", "&bar=5m&limit=100")
t_raw = fetch_okx_data("market/trades", "&limit=50") # 提高成交获取频率

if k1_raw and k5_raw and t_raw:
    # 处理 1m
    df1 = pd.DataFrame(k1_raw['data'], columns=['ts','o','h','l','c','v','volCcy','volCcyQ','confirm'])
    df1[['o','h','l','c','v']] = df1[['o','h','l','c','v']].astype(float)
    df1 = df1[::-1].reset_index(drop=True)
    # 处理 5m
    df5 = pd.DataFrame(k5_raw['data'], columns=['ts','o','h','l','c','v','volCcy','volCcyQ','confirm'])
    df5[['c']] = df5[['c']].astype(float)
    
    # 资金流 (1min 内的极速变动)
    tdf = pd.DataFrame(t_raw['data'])
    buy_v = tdf[tdf['side']=='buy']['sz'].astype(float).sum()
    sell_v = tdf[tdf['side']=='sell']['sz'].astype(float).sum()
    net_flow = buy_v - sell_v
    buy_ratio = (buy_v / (buy_v + sell_v)) * 100 if (buy_v + sell_v) > 0 else 50
    
    sig, m_trend, f_score = engine_v75(df1, df5, net_flow, buy_ratio)
    curr_p = df1.iloc[-1]['c']
    
    with st.sidebar:
        st.header("⚡ V75 高频终端")
        st.subheader(f"5M 主趋势: {'📈 多头' if m_trend == 'BULL' else '📉 空头'}")
        st.write(f"1M 资金评分: {f_score}")
        
        if sig['type']:
            color = "#00FFCC" if sig['dir'] == "LONG" else "#FF4B4B"
            st.markdown(f"""
                <div style="border:3px solid {color}; padding:15px; border-radius:12px; background:rgba(0,0,0,0.5)">
                    <h2 style="color:{color}; margin:0">{sig['type']}</h2>
                    <p style="font-size:12px">{sig['desc']}</p>
                    <hr>
                    <p>📍 进场: ${curr_p:.2f}</p>
                    <p style="color:#FF4B4B">❌ 止损: ${curr_p - 5 if sig['dir']=='LONG' else curr_p + 5:.2f}</p>
                    <p style="color:#00FFCC">💰 止盈: ${curr_p + 10 if sig['dir']=='LONG' else curr_p - 10:.2f}</p>
                </div>
            """, unsafe_allow_html=True)
            st.button("🔊 播放信号预警音")
        else:
            st.write("🔎 1min 扫描中，等待高频共振...")

    st.title("🛡️ ETH 高频捕捉终端 V75")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("1min 净流", f"{net_flow:+.2f} ETH")
    c2.metric("实时买压", f"{buy_ratio:.1f}%")
    c3.metric("5M 状态", m_trend)
    c4.metric("资金动能", f_score)

    # 画图：显示 1min 精确走势
    fig = go.Figure(data=[go.Candlestick(x=df1.index, open=df1['o'], high=df1['h'], low=df1['l'], close=df1['c'], name="1m K线")])
    fig.add_trace(go.Scatter(x=df1.index, y=df1['ema20'], line=dict(color='yellow', width=1), name="1m EMA20"))
    fig.add_trace(go.Scatter(x=df1.index, y=df1['upper'], line=dict(color='rgba(255,255,255,0.1)', dash='dot'), name="1m 上轨"))
    fig.add_trace(go.Scatter(x=df1.index, y=df1['lower'], line=dict(color='rgba(255,255,255,0.1)', dash='dot'), name="1m 下轨"))
    
    fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.error("正在同步 1min 高频数据流...")
