import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go

# ==================== 1. 系统核心配置 ====================
st.set_page_config(page_title="ETH V80 火眼金睛版", layout="wide")

def fetch_okx_data(endpoint, params=""):
    url = f"https://www.okx.com/api/v5/{endpoint}?instId=ETH-USDT{params}"
    try:
        r = requests.get(url, timeout=5).json()
        return r if r.get('code') == '0' else None
    except: return None

# ==================== 2. V80 核心辨伪引擎 ====================
def vision_engine_v80(df_1m, df_5m, net_flow, buy_ratio):
    # --- 1min 基础指标 ---
    df_1m['ema20'] = df_1m['c'].ewm(span=20, adjust=False).mean()
    df_1m['atr'] = (df_1m['h'] - df_1m['l']).rolling(14).mean()
    
    # --- 核心：动能质量验证 (辨别假动作) ---
    curr, prev = df_1m.iloc[-1], df_1m.iloc[-2]
    vol_mean = df_1m['v'].rolling(20).mean().iloc[-1]
    
    # 1. 价量共振检查
    # 如果价格波动大但成交量低于均值，判定为“空涨/空跌”，即假动作
    price_move_ratio = abs(curr['c'] - curr['o']) / curr['atr']
    vol_surge_ratio = curr['v'] / vol_mean
    is_real_move = vol_surge_ratio > 1.2 or (price_move_ratio < 1.5)
    
    # 2. 压力/支撑 识别
    df_5m['ema20'] = df_5m['c'].ewm(span=20, adjust=False).mean()
    ema5_20 = df_5m['ema20'].iloc[-1]
    
    # --- 资金雷达 (对敲识别逻辑) ---
    # 如果买压 100% 但净流入极小，判定为对敲诱多
    is_spoofing = buy_ratio > 90 and net_flow < 2.0
    
    f_score = 0
    if net_flow > 5 and not is_spoofing: f_score += 3
    if net_flow < -5: f_score -= 3
    if buy_ratio > 60: f_score += 1
    if buy_ratio < 40: f_score -= 1

    # --- 决策逻辑 ---
    signal = {"type": None, "dir": None, "score": 0, "warning": ""}
    
    # 判定 1: 真实趋势火爆点
    if is_real_move and abs(f_score) >= 3:
        if curr['c'] > curr['ema20'] and f_score > 0:
            signal = {"type": "真·趋势点火", "dir": "LONG", "score": 7 + f_score}
        elif curr['c'] < curr['ema20'] and f_score < 0:
            signal = {"type": "真·驱势下破", "dir": "SHORT", "score": 7 + abs(f_score)}
            
    # 判定 2: 假动作预警
    if not is_real_move and price_move_ratio > 2.0:
        signal["warning"] = "⚠️ 检测到无量空拉/空砸，疑似假突破！"
    if is_spoofing:
        signal["warning"] = "🚨 监测到高频对敲，庄家可能在诱多！"

    return signal, curr['atr'], f_score, is_real_move

# ==================== 3. 终端 UI 渲染 ====================
k1_raw = fetch_okx_data("market/candles", "&bar=1m&limit=100")
k5_raw = fetch_okx_data("market/candles", "&bar=5m&limit=100")
t_raw = fetch_okx_data("market/trades", "&limit=50")

if k1_raw and k5_raw and t_raw:
    df1 = pd.DataFrame(k1_raw['data'], columns=['ts','o','h','l','c','v','volCcy','volCcyQ','confirm'])
    df1[['o','h','l','c','v']] = df1[['o','h','l','c','v']].astype(float)
    df1 = df1[::-1].reset_index(drop=True)
    df5 = pd.DataFrame(k5_raw['data'], columns=['ts','o','h','l','c','v','volCcy','volCcyQ','confirm'])
    df5['c'] = df5['c'].astype(float)
    
    tdf = pd.DataFrame(t_raw['data'])
    buy_v = tdf[tdf['side']=='buy']['sz'].astype(float).sum()
    sell_v = tdf[tdf['side']=='sell']['sz'].astype(float).sum()
    net_flow = buy_v - sell_v
    buy_ratio = (buy_v / (buy_v + sell_v)) * 100 if (buy_v + sell_v) > 0 else 50
    
    sig, atr, f_score, is_real = vision_engine_v80(df1, df5, net_flow, buy_ratio)
    curr_p = df1.iloc[-1]['c']
    
    with st.sidebar:
        st.header("👁️ V80 火眼金睛")
        st.write(f"动能真实性: {'✅ 真实' if is_real else '❌ 虚假'}")
        
        if sig['warning']:
            st.error(sig['warning'])
        
        if sig['type']:
            color = "#00FFCC" if sig['dir'] == "LONG" else "#FF4B4B"
            st.markdown(f"""
                <div style="border:4px solid {color}; padding:15px; border-radius:12px; background:rgba(0,0,0,0.4)">
                    <h2 style="color:{color}; margin:0">{sig['type']}</h2>
                    <p><b>辨伪分: {sig['score']}</b></p>
                    <hr>
                    <p>进场: ${curr_p:.2f}</p>
                    <p>止损: ${curr_p - 1.5*atr if sig['dir']=='LONG' else curr_p + 1.5*atr:.2f}</p>
                    <p>止盈: ${curr_p + 3*atr if sig['dir']=='LONG' else curr_p - 3*atr:.2f}</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.info("正在过滤市场噪音...")

    st.title("🛡️ ETH 辨伪决策终端 V80")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("1min 净流", f"{net_flow:+.2f} ETH")
    c2.metric("实时买压", f"{buy_ratio:.1f}%")
    c3.metric("动能真实性", "PASS" if is_real else "FAIL")
    c4.metric("资金因子", f_score)

    fig = go.Figure(data=[go.Candlestick(x=df1.index, open=df1['o'], high=df1['h'], low=df1['l'], close=df1['c'], name="1m K线")])
    fig.add_trace(go.Scatter(x=df1.index, y=df1['ema20'], line=dict(color='yellow', width=1), name="1m EMA20"))
    fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
