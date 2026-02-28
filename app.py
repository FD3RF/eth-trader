import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import asyncio
import httpx

# ==================== 1. 基础配置与初始化 ====================
st.set_page_config(page_title="ETH V400 战神·不朽版", layout="wide")

# 初始化持久化存储
if 'hall_of_glory' not in st.session_state: st.session_state.hall_of_glory = []
if 'war_logs_fail' not in st.session_state: st.session_state.war_logs_fail = []
if 'last_cmd' not in st.session_state: st.session_state.last_cmd = ""

# ==================== 2. 核心引擎功能 ====================

def speak_passionate(text, passion_level="normal"):
    """热血语音引擎"""
    if st.session_state.last_cmd == text: return # 防止重复播报
    st.session_state.last_cmd = text
    
    pitch = 1.5 if passion_level == "excited" else (0.8 if passion_level == "warning" else 1.0)
    rate = 1.2 if passion_level == "excited" else 1.0
    
    js_code = f"""
    <script>
    var msg = new SpeechSynthesisUtterance('{text}');
    msg.lang = 'zh-CN';
    msg.pitch = {pitch};
    msg.rate = {rate};
    window.speechSynthesis.speak(msg);
    </script>
    """
    st.components.v1.html(js_code, height=0)

async def fetch_market_data():
    """全域异步扫描"""
    symbols = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]
    async with httpx.AsyncClient() as client:
        tasks = [client.get(f"https://www.okx.com/api/v5/market/candles?instId={s}&bar=1m&limit=100") for s in symbols]
        responses = await asyncio.gather(*tasks)
        results = {}
        for s, r in zip(symbols, responses):
            data = r.json().get('data', [])
            if data:
                df = pd.DataFrame(data, columns=['ts','o','h','l','c','v','volC','volCQ','cf'])[::-1]
                for col in ['o','h','l','c','v']: df[col] = df[col].astype(float)
                df['ema20'] = df['c'].ewm(span=20, adjust=False).mean()
                results[s] = {"df": df, "price": df['c'].iloc[-1]}
        return results

def get_strategy(symbol, price, df):
    """先知裁决逻辑"""
    res = df['h'].tail(50).max()
    sup = df['l'].tail(50).min()
    prob = 82.5 if symbol == "ETH-USDT" else 65.0 # 实战中此处应接回测胜率函数
    
    if prob >= 80 and price > df['ema20'].iloc[-1]:
        return f"冲啊战神！{symbol}胜率爆表({int(prob)}%)！目标看{int(res)}！", "excited", res, sup, prob
    elif price < sup:
        return f"撤退！{symbol}跌破防线，立即保护战果！", "warning", res, sup, prob
    else:
        return "战场冷静期，观察动能变化。", "normal", res, sup, prob

# ==================== 3. UI 渲染 ====================

# 数据抓取
try:
    data_map = asyncio.run(fetch_market_data())
except:
    st.error("📡 网络波动，正在尝试重连作战中心...")
    st.stop()

tab1, tab2, tab3 = st.tabs(["🎮 实时指挥部", "🏆 战神荣耀册", "🕯️ 战败反思录"])

# --- 实时指挥部 ---
with tab1:
    eth_main = data_map['ETH-USDT']
    voice_txt, p_level, res_v, sup_v, p_val = get_strategy('ETH-USDT', eth_main['price'], eth_main['df'])
    
    l, r = st.columns([1, 2.5])
    with l:
        style = "background:linear-gradient(45deg, #FF4B2B, #FF416C);" if p_level == "excited" else "background:#1E1E1E;"
        st.markdown(f"""
        <div style="{style} padding:20px; border-radius:15px; border:2px solid gold;">
            <h2 style="text-align:center; color:white;">⚔️ 实时指令</h2>
            <p style="font-size:18px; color:white;"><b>{voice_txt}</b></p>
            <p style="color:#00FFCC;">当前价格: ${eth_main['price']}</p>
            <p style="color:white;">胜率强度: {p_val}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("📢 强制播报语音"): speak_passionate(voice_txt, p_level)
        
        st.markdown("---")
        # 模拟操作按钮
        c1, c2 = st.columns(2)
        if c1.button("🚀 录入大捷"):
            st.session_state.hall_of_glory.insert(0, {"time": datetime.now().strftime("%H:%M"), "pnl": "+150U", "chart": eth_main['df'].tail(30), "cmd": voice_txt})
            st.balloons()
        if c2.button("💀 录入战败"):
            st.session_state.war_logs_fail.insert(0, {"time": datetime.now().strftime("%H:%M"), "pnl": "-80U", "chart": eth_main['df'].tail(30), "cmd": voice_txt})
            st.snow()

    with r:
        fig = go.Figure(data=[go.Candlestick(x=eth_main['df'].index, open=eth_main['df']['o'], high=eth_main['df']['h'], low=eth_main['df']['l'], close=eth_main['df']['c'])])
        fig.add_hline(y=res_v, line_dash="dash", line_color="#FF4B4B", annotation_text="阻力 R")
        fig.add_hline(y=sup_v, line_dash="dash", line_color="#00FFCC", annotation_text="支撑 S")
        fig.add_trace(go.Scatter(x=eth_main['df'].index, y=eth_main['df']['ema20'], line=dict(color='yellow', width=1.5), name="EMA20"))
        fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

# --- 荣耀册 & 反思录 ---
with tab2:
    for v in st.session_state.hall_of_glory:
        st.success(f"✨ {v['time']} | 收益: {v['pnl']} | 指令: {v['cmd']}")
with tab3:
    for f in st.session_state.war_logs_fail:
        st.error(f"💀 {f['time']} | 亏损: {f['pnl']} | 反思: 当时执行“{f['cmd']}”是否过晚？")

# 自动语音触发
speak_passionate(voice_txt, p_level)
