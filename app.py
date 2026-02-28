import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go

# ==================== 1. 系统核心配置 ====================
st.set_page_config(page_title="ETH V71.0 职业机构版", layout="wide")

def fetch_okx_data(endpoint, params=""):
    url = f"https://www.okx.com/api/v5/{endpoint}?instId=ETH-USDT{params}"
    try:
        with requests.Session() as s:
            s.trust_env = True  
            r = s.get(url, timeout=5).json()
            return r if r.get('code') == '0' else None
    except: return None

# ==================== 2. 核心智能引擎 ====================
def pro_commander_engine(df):
    # --- 基础指标体系 ---
    df['ema20'] = df['c'].ewm(span=20, adjust=False).mean()
    df['ema60'] = df['c'].ewm(span=60, adjust=False).mean()
    
    # 波动率与布林带 (用于捕捉横盘变盘)
    df['std'] = df['c'].rolling(20).std()
    df['upper'] = df['ema20'] + (df['std'] * 2)
    df['lower'] = df['ema20'] - (df['std'] * 2)
    
    # ATR 风控
    df['tr'] = np.maximum(df['h'] - df['l'], np.maximum(abs(df['h'] - df['c'].shift(1)), abs(df['l'] - df['c'].shift(1))))
    df['atr'] = df['tr'].rolling(14).mean()
    atr = df['atr'].iloc[-1]
    
    # 偏离度与动能
    df['dev'] = (df['c'] - df['ema20']).abs() / atr
    delta = df['c'].diff()
    df['rsi'] = 100 - (100 / (1 + (delta.where(delta > 0, 0)).rolling(14).mean() / (-delta.where(delta < 0, 0)).rolling(14).mean()))
    
    # --- 市场环境诊断 ---
    curr, prev = df.iloc[-1], df.iloc[-2]
    vol_mean = df['v'].rolling(20).mean().iloc[-1]
    
    # 判定横盘后的变盘 (Squeeze Breakout)
    is_squeezed = (df['upper'] - df['lower']).iloc[-5:].mean() < (atr * 4) # 布林带收窄
    break_up = curr['c'] > df['upper'].iloc[-2] and curr['v'] > vol_mean * 1.3
    break_down = curr['c'] < df['lower'].iloc[-2] and curr['v'] > vol_mean * 1.3

    signal = {"type": None, "dir": None, "score": 0}
    mode = "区间缠绕"
    
    # 1. 变盘突袭引擎 (最高优先级)
    if is_squeezed:
        mode = "缩量变盘中"
        if break_up: signal = {"type": "动能突破", "dir": "LONG", "score": 9}
        elif break_down: signal = {"type": "动能突破", "dir": "SHORT", "score": 9}
    
    # 2. 趋势回踩引擎 (次高优先级)
    if not signal['type']:
        if curr['ema20'] > curr['ema60'] and curr['l'] <= curr['ema20']:
            mode = "多头回补"
            signal = {"type": "趋势回踩", "dir": "LONG", "score": 6}
        elif curr['ema20'] < curr['ema60'] and curr['h'] >= curr['ema20']:
            mode = "空头回补"
            signal = {"type": "趋势回踩", "dir": "SHORT", "score": 6}

    # 3. 乖离修正引擎 (最低优先级)
    if not signal['type'] and curr['dev'] > 1.8:
        mode = "极端超买/卖"
        if curr['rsi'] < 30: signal = {"type": "乖离修复", "dir": "LONG", "score": 7}
        elif curr['rsi'] > 70: signal = {"type": "乖离修复", "dir": "SHORT", "score": 7}

    return signal, atr, mode, curr['dev']

# ==================== 3. 执行渲染层 ====================
k_raw = fetch_okx_data("market/candles", "&bar=5m&limit=300")
if k_raw:
    df = pd.DataFrame(k_raw['data'], columns=['ts','o','h','l','c','v','volCcy','volCcyQ','confirm'])
    df[['o','h','l','c','v']] = df[['o','h','l','c','v']].astype(float)
    df = df[::-1].reset_index(drop=True)
    
    sig, atr, mode, dev = pro_commander_engine(df)
    curr_p = df.iloc[-1]['c']
    
    with st.sidebar:
        st.header("💂 V71.0 职业指挥部")
        st.metric("变盘诊断", mode)
        
        # 动态风控计划
        if sig['type']:
            sl_c, tp_c = (1.1, 2.8) if sig['type'] == "动能突破" else (1.5, 2.0)
            sl = curr_p - (sl_c * atr) if sig['dir'] == "LONG" else curr_p + (sl_c * atr)
            tp = curr_p + (tp_c * atr) if sig['dir'] == "LONG" else curr_p - (tp_c * atr)
            color = "#00f0ff" if sig['type'] == "动能突破" else "#00ffcc"
            
            st.markdown(f"""
                <div style="border:3px solid {color}; padding:15px; border-radius:10px; background:rgba(0,0,0,0.3)">
                    <h2 style="color:{color}; margin:0">{sig['type']} 🔥</h2>
                    <p style="color:{color}"><b>评分: {sig['score']} | 方向: {sig['dir']}</b></p>
                    <p style="font-size:13px">捕获到{mode}后的放量信号，执行自适应风控。</p>
                    <hr>
                    <p>📍 进场: ${curr_p:.2f}</p>
                    <p style="color:#ff4b4b">❌ 止损: ${sl:.2f}</p>
                    <p style="color:#00ffcc">💰 止盈: ${tp:.2f}</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ 行情极度缩量中，耐心等待变盘。")

    st.title(f"🛡️ ETH 职业机构终端 V71.0")
    col1, col2, col3 = st.columns(3)
    col1.metric("现价", f"${curr_p}")
    col2.metric("ATR 波动", f"{atr:.2f}")
    col3.metric("EMA 偏离", f"{dev:.2f}")

    # 绘图：增加布林带显示
    fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['o'], high=df['h'], low=df['l'], close=df['c'], name="K线")])
    fig.add_trace(go.Scatter(x=df.index, y=df['upper'], line=dict(color='rgba(255,255,255,0.2)', dash='dot'), name="上轨"))
    fig.add_trace(go.Scatter(x=df.index, y=df['lower'], line=dict(color='rgba(255,255,255,0.2)', dash='dot'), name="下轨"))
    fig.add_trace(go.Scatter(x=df.index, y=df['ema20'], line=dict(color='yellow', width=1.5), name="EMA20"))
    
    fig.update_layout(template="plotly_dark", height=700, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
