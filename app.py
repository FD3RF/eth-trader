import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from collections import deque

# ==========================================
# 1. UI 配置 (合约深度定制版)
# ==========================================
st.set_page_config(layout="wide", page_title="ETH CONTRACT V15", page_icon="🕵️")

st.markdown("""
<style>
    .stApp { background-color: #0E1117; }
    .status-card { background: #1c2128; border: 1px solid #30363d; padding: 15px; border-radius: 10px; margin-bottom: 20px; }
    .pnl-long { color: #00FFC2; font-family: 'monospace'; font-weight: bold; font-size: 1.5rem; }
    .pnl-short { color: #FF4B4B; font-family: 'monospace'; font-weight: bold; font-size: 1.5rem; }
    .div-alert { color: #FFA500; font-weight: bold; animation: blinker 1.5s linear infinite; }
    @keyframes blinker { 50% { opacity: 0; } }
</style>
""", unsafe_allow_html=True)

# 初始化合约状态
if 'contract' not in st.session_state:
    st.session_state.contract = {"side": "空仓", "entry": 0.0, "lev": 20, "liq": 0.0}
if 'alert' not in st.session_state: st.session_state.alert = ""

with st.sidebar:
    st.header("⚡ 合约核心配置")
    lev = st.select_slider("杠杆倍数", options=[1, 10, 20, 50, 100], value=20)
    stop_loss = st.slider("硬性止损 (%)", 1, 50, 15)
    
    st.divider()
    c1, c2 = st.columns(2)
    if c1.button("🟢 开多 (LONG)", use_container_width=True):
        st.session_state.contract = {"side": "多单", "entry": st.session_state.last_p, "lev": lev, "liq": st.session_state.last_p * (1 - 0.9/lev)}
    if c2.button("🔴 开空 (SHORT)", use_container_width=True):
        st.session_state.contract = {"side": "空单", "entry": st.session_state.last_p, "lev": lev, "liq": st.session_state.last_p * (1 + 0.9/lev)}
    
    if st.button("⏹️ 立即全平", use_container_width=True):
        st.session_state.contract = {"side": "空仓", "entry": 0.0, "lev": lev, "liq": 0.0}

st.title("🕵️ ETH 合约背离扫描终端 (V15)")

# 核心数据区
m1, m2, m3 = st.columns([1, 2, 1])
p_ph = m1.empty()
status_ph = m2.empty()
div_ph = m3.empty()

chart_ph = st.empty()

# ==========================================
# 2. 引擎逻辑 (增加背离算法)
# ==========================================
if 'v15_data' not in st.session_state:
    st.session_state.v15_data = {
        't': deque([time.strftime("%M:%S", time.localtime(time.time()-i)) for i in range(120, 0, -1)], maxlen=120),
        'c': deque([2800.0 + np.sin(i/10)*10 for i in range(120)], maxlen=120)
    }

while True:
    # A. 模拟行情
    prev_p = st.session_state.v15_data['c'][-1]
    new_p = prev_p + np.random.normal(0, 3.2)
    st.session_state.last_p = new_p
    
    st.session_state.v15_data['t'].append(time.strftime("%M:%S"))
    st.session_state.v15_data['c'].append(new_p)
    
    df = pd.DataFrame(st.session_state.v15_data)
    df['ma'] = df['c'].rolling(20).mean().ffill().bfill()
    df['macd'] = df['c'].ewm(span=12).mean() - df['c'].ewm(span=26).mean()
    df['hist'] = df['macd'] - df['macd'].ewm(span=9).mean()

    # B. 背离扫描算法
    st.session_state.alert = "扫描中..."
    if len(df) > 30:
        # 简单背离逻辑：价格创新低但MACD回升
        if df['c'].iloc[-1] < df['c'].iloc[-20:-1].min() and df['hist'].iloc[-1] > df['hist'].iloc[-20:-1].min():
            st.session_state.alert = "⚠️ 底背离 (BULLISH)"
        elif df['c'].iloc[-1] > df['c'].iloc[-20:-1].max() and df['hist'].iloc[-1] < df['hist'].iloc[-20:-1].max():
            st.session_state.alert = "⚠️ 顶背离 (BEARISH)"

    # C. 持仓盈亏
    con = st.session_state.contract
    pnl_text, pnl_class, liq_info = "等待机会", "pnl-long", ""
    if con["side"] != "空仓":
        raw_pnl = (new_p - con["entry"]) if con["side"] == "多单" else (con["entry"] - new_p)
        pct = (raw_pnl / con["entry"]) * 100 * con["lev"]
        pnl_class = "pnl-long" if pct >= 0 else "pnl-short"
        pnl_text = f"{con['side']} {pct:+.2f}%"
        liq_info = f"强平价: ${con['liq']:,.2f} | 杠杆: {con['lev']}x"
