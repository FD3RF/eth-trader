import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from collections import deque

# ==========================================
# 1. 极致 UI 架构 (V13 实战版)
# ==========================================
st.set_page_config(layout="wide", page_title="ETH QUANTUM V13", page_icon="⚔️")

st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: white; }
    [data-testid="stMetricValue"] { color: #A491FF !important; font-family: 'monospace'; }
    .stMetric { background-color: #161B22; border-radius: 8px; border: 1px solid #30363d; padding: 10px; }
    .pos-card { padding: 15px; border-radius: 10px; border: 1px solid #30363d; background: #1c2128; margin-bottom: 10px; }
    .pnl-plus { color: #00FFC2; font-weight: bold; font-size: 1.2rem; }
    .pnl-minus { color: #FF4B4B; font-weight: bold; font-size: 1.2rem; }
</style>
""", unsafe_allow_html=True)

# 初始化持仓状态
if 'pos' not in st.session_state:
    st.session_state.pos = {"type": "空仓", "entry": 0.0, "pnl": 0.0}

with st.sidebar:
    st.header("⚔️ 实战指挥部")
    if st.button("🚀 现价开多", use_container_width=True):
        st.session_state.pos = {"type": "多单", "entry": st.session_state.last_price, "pnl": 0.0}
    if st.button("🔥 现价开空", use_container_width=True):
        st.session_state.pos = {"type": "空单", "entry": st.session_state.last_price, "pnl": 0.0}
    if st.button("⏹️ 一键平仓", use_container_width=True):
        st.session_state.pos = {"type": "空仓", "entry": 0.0, "pnl": 0.0}
    st.divider()
    st.info("模式：实时模拟盘 | V13 引擎")

st.title("⚔️ ETH 量子实战决策终端")

# 顶层状态
c1, c2, c3, c4 = st.columns(4)
p_ph, s_ph, r_ph, m_ph = c1.empty(), c2.empty(), c3.empty(), c4.empty()

# 中部持仓面板
pos_ph = st.empty()

# 主图
chart_ph = st.empty()

# ==========================================
# 2. 稳健数据与持仓引擎
# ==========================================
if 'v13_cache' not in st.session_state:
    st.session_state.v13_cache = {
        't': deque([time.strftime("%M:%S", time.localtime(time.time()-i)) for i in range(80, 0, -1)], maxlen=80),
        'o': deque([2800.0] * 80, maxlen=80),
        'h': deque([2805.0] * 80, maxlen=80),
        'l': deque([2795.0] * 80, maxlen=80),
        'c': deque([2800.0] * 80, maxlen=80)
    }

while True:
    # A. 价格模拟
    prev_c = st.session_state.v13_cache['c'][-1]
    new_c = prev_c + np.random.normal(0, 3.8)
    st.session_state.last_price = new_c # 存入 session 供侧边栏读取
    
    st.session_state.v13_cache['t'].append(time.strftime("%M:%S"))
    st.session_state.v13_cache['o'].append(prev_c)
    st.session_state.v13_cache['h'].append(max(prev_c, new_c) + 1)
    st.session_state.v13_cache['l'].append(min(prev_c, new_c) - 1)
    st.session_state.v13_cache['c'].append(new_c)
    
    df = pd.DataFrame(st.session_state.v13_cache)
    
    # B. 指标计算 (冷启动保护)
    df['ma'] = df['c'].rolling(20).mean().ffill().bfill()
    df['up'] = df['ma'] + (1.6 * df['c'].rolling(20).std().ffill().bfill())
    df['dn'] = df['ma'] - (1.6 * df['c'].rolling(20).std().ffill().bfill())
    
    # RSI & MACD
    delta = df['c'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = (100 - (100 / (1 + (gain / (loss + 1e-9))))).ffill().bfill()
    df['macd'] = df['c'].ewm(span=12).mean() - df['c'].ewm(span=26).mean()
    df['hist'] = df['macd'] - df['macd'].ewm(span=9).mean()

    # C. 持仓盈亏计算
    pnl_val, pnl_pct, pnl_class = 0.0, 0.0, "pnl-plus"
    if st.session_state.pos["type"] != "空仓":
        entry = st.session_state.pos["entry"]
        pnl_val = (new_c - entry) if st.session_state.pos["type"] == "多单" else (entry - new_c)
        pnl_pct = (pnl_val / entry) * 100
        pnl_class = "pnl-plus" if pnl_val >= 0 else "pnl-minus"

    # D. UI 渲染
    p_ph.metric("ETH 现价", f"${new_c:,.2f}", f"{new_c-prev_c:.2f}")
    
    # 渲染持仓卡片
    pos_type = st.session_state.pos["type"]
    entry_price = f"${st.session_state.pos['entry']:.2f}" if st.session_state.pos["entry"] > 0 else "--"
    pos_ph.markdown(f"""
    <div class='pos-card'>
        <div style='display:flex; justify-content:space-between;'>
            <span>当前持仓: <b>{pos_type}</b></span>
            <span>入场均价: {entry_price}</span>
            <span>实时盈亏: <span class='{pnl_class}'>{pnl_pct:+.2f}% (${pnl_val:+.2f})</span></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # E. 主图渲染
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.5, 0.25, 0.25])
    fig.add_trace(go.Candlestick(x=df['t'], open=df['o'], high=df['h'], low=df['l'], close=df['c']), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['t'], y=df['up'], line=dict(color='rgba(164,145,255,0.2)', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['t'], y=df['dn'], line=dict(color='rgba(164,145,255,0.2)', width=1), fill='tonexty'), row=1, col=1)
    
    # 持仓入场线
    if st.session_state.pos["entry"] > 0:
        fig.add_hline(y=st.session_state.pos["entry"], line_dash="dash", line_color="yellow", row=1, col=1)

    fig.add_trace(go.Bar(x=df['t'], y=df['hist'], marker_color=['#00FFC2' if x>0 else '#FF4B4B' for x in df['hist']]), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['t'], y=df['rsi'], line=dict(color='#A491FF', width=2)), row=3, col=1)
    
    fig.update_layout(template="plotly_dark", height=650, margin=dict(l=0,r=0,t=0,b=0), xaxis_rangeslider_visible=False, showlegend=False)
    chart_ph.plotly_chart(fig, key=f"v13_{time.time_ns()}")

    time.sleep(1)
