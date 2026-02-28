import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go

# ==================== 1. 系统核心配置 ====================
st.set_page_config(page_title="ETH V90 全能战神整合版", layout="wide")

def fetch_okx(endpoint, params=""):
    url = f"https://www.okx.com/api/v5/{endpoint}?instId=ETH-USDT{params}"
    try:
        r = requests.get(url, timeout=5).json()
        return r if r.get('code') == '0' else None
    except: return None

# ==================== 2. 终极算法引擎 (修复 V88 报错并整合全功能) ====================
def ultimate_engine_v90(df1, df5, net_flow, buy_ratio):
    # --- 指标预处理 (统一类型，修复 V88 报错) ---
    for d in [df1, df5]:
        for col in ['o', 'h', 'l', 'c', 'v']:
            d[col] = d[col].astype(float)
        d['ema20'] = d['c'].ewm(span=20, adjust=False).mean()
        d['ema60'] = d['c'].ewm(span=60, adjust=False).mean()
        d['std'] = d['c'].rolling(20).std()
        d['upper'] = d['ema20'] + (d['std'] * 2)
        d['lower'] = d['ema20'] - (d['std'] * 2)
        # 修复 ATR 计算逻辑
        d['atr'] = (d['h'] - d['l']).rolling(14).mean()

    curr1, curr5 = df1.iloc[-1], df5.iloc[-1]
    
    # 【功能整合 1：V74 模式识别】
    slope5 = (df5['ema20'].iloc[-1] - df5['ema20'].iloc[-5]) / 5
    is_trend = abs(slope5) > (df5['atr'].iloc[-1] * 0.15)
    market_mode = "🌊 趋势模式" if is_trend else "⚖️ 震荡模式"

    # 【功能整合 2：V80 动能辨伪】
    vol_mean = df1['v'].rolling(20).mean().iloc[-1]
    # 动能品质验证
    is_real = (curr1['v'] / vol_mean > 1.2) or (abs(curr1['c'] - curr1['o']) / curr1['atr'] < 1.5)
    # 诱多诱空识别 (Spoofing)
    is_spoofing = buy_ratio > 92 and net_flow < 1.5

    # 【功能整合 3：V72 资金评分】
    f_score = 0
    if net_flow > 4 and not is_spoofing: f_score += 3
    if net_flow < -4: f_score -= 3
    if buy_ratio > 60: f_score += 1
    if buy_ratio < 40: f_score -= 1

    # 【功能整合 4：V65 AI 战术计划】
    sig = {"type": None, "dir": None, "reason": ""}
    
    # 策略 A: 趋势共振 (V81 增强)
    if is_trend and is_real:
        if curr1['c'] > curr1['ema20'] and f_score >= 3:
            sig = {"type": "趋势点火", "dir": "LONG", "reason": "5M趋势向上且1M动能真实放量"}
        elif curr1['c'] < curr1['ema20'] and f_score <= -3:
            sig = {"type": "趋势崩塌", "dir": "SHORT", "reason": "5M趋势向下且资金大幅流出"}
    
    # 策略 B: 震荡逆势 (V74 增强)
    elif not is_trend:
        if curr1['c'] <= curr1['lower'] and f_score >= 1:
            sig = {"type": "低位吸筹", "dir": "LONG", "reason": "1M触碰下轨且资金暗中护盘"}
        elif curr1['c'] >= curr1['upper'] and f_score <= -1:
            sig = {"type": "高位派发", "dir": "SHORT", "reason": "1M触碰上轨且主力高位出货"}

    return sig, market_mode, f_score, is_real, is_spoofing, curr1['atr']

# ==================== 3. UI 界面布局 (整合 V14-V85 所有视觉元素) ====================
k1_raw = fetch_okx("market/candles", "&bar=1m&limit=100")
k5_raw = fetch_okx("market/candles", "&bar=5m&limit=100")
t_raw = fetch_okx("market/trades", "&limit=50")

if k1_raw and k5_raw and t_raw:
    df1 = pd.DataFrame(k1_raw['data'], columns=['ts','o','h','l','c','v','volCcy','volCcyQ','confirm'])
    df5 = pd.DataFrame(k5_raw['data'], columns=['ts','o','h','l','c','v','volCcy','volCcyQ','confirm'])
    df1 = df1[::-1].reset_index(drop=True)
    df5 = df5[::-1].reset_index(drop=True)
    
    tdf = pd.DataFrame(t_raw['data'])
    buy_v = tdf[tdf['side']=='buy']['sz'].astype(float).sum()
    sell_v = tdf[tdf['side']=='sell']['sz'].astype(float).sum()
    net_f = buy_v - sell_v
    buy_r = (buy_v / (buy_v + sell_v)) * 100 if (buy_v + sell_v) > 0 else 50
    
    # 运行 V90 引擎
    sig, mode, f_score, is_real, is_spoofing, atr = ultimate_engine_v90(df1, df5, net_f, buy_r)
    curr_p = df1.iloc[-1]['c']

    # --- 侧边栏：AI 战略指挥中心 (V81+V85 整合) ---
    with st.sidebar:
        st.header(f"🧠 V90 指挥部")
        st.info(f"市场环境: {mode}")
        st.write(f"品质校验: {'✅ 真实' if is_real else '⚠️ 噪音'}")
        
        if sig['type']:
            color = "#00FFCC" if sig['dir'] == "LONG" else "#FF4B4B"
            tp1 = curr_p + 1.8*atr if sig['dir']=="LONG" else curr_p - 1.8*atr
            tp2 = curr_p + 3.5*atr if sig['dir']=="LONG" else curr_p - 3.5*atr
            sl = curr_p - 1.5*atr if sig['dir']=="LONG" else curr_p + 1.5*atr
            
            st.markdown(f"""
                <div style="border:3px solid {color}; padding:15px; border-radius:12px; background:rgba(0,0,0,0.4)">
                    <h2 style="color:{color}; margin:0">{sig['type']}</h2>
                    <p style="font-size:13px; margin:5px 0"><b>理由:</b> {sig['reason']}</p>
                    <hr>
                    <p>📍 进场: ${curr_p:.2f}</p>
                    <p style="color:#FF4B4B">❌ 止损: ${sl:.2f}</p>
                    <p style="color:#00FFCC">💰 止盈1: ${tp1:.2f}</p>
                    <p style="color:#00FFCC">💎 止盈2: ${tp2:.2f}</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("🔭 正在过滤市场虚假波动...")
            if is_spoofing: st.error("🚨 警告：检测到虚假对敲诱多！")

    # --- 主页面：数据大屏 (整合 V14/V72/V80) ---
    st.title("🛡️ ETH 全能战神整合终端 V90")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("1min 净流", f"{net_f:+.2f} ETH")
    col2.metric("买压占比", f"{buy_r:.1f}%")
    col3.metric("资金动能", f_score)
    col4.metric("波动率 ATR", f"{atr:.2f}")

    # 图表层
    fig = go.Figure(data=[go.Candlestick(x=df1.index, open=df1['o'], high=df1['h'], low=df1['l'], close=df1['c'], name="1M K线")])
    fig.add_trace(go.Scatter(x=df1.index, y=df1['ema20'], line=dict(color='yellow', width=1), name="EMA20"))
    fig.add_trace(go.Scatter(x=df1.index, y=df1['upper'], line=dict(color='rgba(255,255,255,0.1)', dash='dot'), name="上轨"))
    fig.add_trace(go.Scatter(x=df1.index, y=df1['lower'], line=dict(color='rgba(255,255,255,0.1)', dash='dot'), name="下轨"))
    
    fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False, margin=dict(l=10,r=10,t=30,b=10))
    st.plotly_chart(fig, use_container_width=True)

else:
    st.error("数据连接异常，正在重新同步...")
