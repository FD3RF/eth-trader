import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from collections import deque

# ==========================================
# 1. UI 配置 (合约实战风格)
# ==========================================
st.set_page_config(layout="wide", page_title="ETH CONTRACT V14", page_icon="⚡")

st.markdown("""
<style>
    .stApp { background-color: #0E1117; }
    .metric-box { background: #161B22; border: 1px solid #30363d; padding: 12px; border-radius: 8px; text-align: center; }
    .pnl-long { color: #00FFC2; font-family: 'monospace'; font-weight: bold; font-size: 1.4rem; }
    .pnl-short { color: #FF4B4B; font-family: 'monospace'; font-weight: bold; font-size: 1.4rem; }
    .liq-price { color: #FFA500; font-size: 0.9rem; margin-top: 5px; }
</style>
""", unsafe_allow_html=True)

# 持仓状态机
if 'trade' not in st.session_state:
    st.session_state.trade = {"side": "空仓", "entry": 0.0, "lev": 20, "liq": 0.0}

with st.sidebar:
    st.header("⚡ 合约参数")
    lev = st.select_slider("杠杆倍数 (Leverage)", options=[1, 5, 10, 20, 50, 100], value=20)
    st.session_state.trade["lev"] = lev
    
    col_a, col_b = st.columns(2)
    if col_a.button("📈 开多 (LONG)", use_container_width=True):
        entry = st.session_state.last_c
        st.session_state.trade = {"side": "多单", "entry": entry, "lev": lev, "liq": entry * (1 - 1/lev)}
    if col_b.button("📉 开空 (SHORT)", use_container_width=True):
        entry = st.session_state.last_c
        st.session_state.trade = {"side": "空单", "entry": entry, "lev": lev, "liq": entry * (1 + 1/lev)}
    
    if st.button("⏹️ 立即平仓 (CLOSE)", use_container_width=True):
        st.session_state.trade = {"side": "空仓", "entry": 0.0, "lev": lev, "liq": 0.0}
    
    st.divider()
    st.warning("合约风险提示：杠杆倍数越高，强平距离越近。")

st.title("⚡ ETH 永续合约上帝视角终端")

# 顶层指标
m1, m2, m3, m4 = st.columns(4)
price_ph, pnl_ph, rsi_ph, macd_ph = m1.empty(), m2.empty(), m3.empty(), m4.empty()

chart_ph = st.empty()

# ==========================================
# 2. 核心引擎
# ==========================================
if 'history_v14' not in st.session_state:
    st.session_state.history_v14 = {
        't': deque([time.strftime("%M:%S", time.localtime(time.time()-i)) for i in range(100, 0, -1)], maxlen=100),
        'o': deque([2800.0]*100, maxlen=100), 'h': deque([2805.0]*100, maxlen=100),
        'l': deque([2795.0]*100, maxlen=100), 'c': deque([2800.0]*100, maxlen=100)
    }

while True:
    # A. 市场模拟器
    pc = st.session_state.history_v14['c'][-1]
    nc = pc + np.random.normal(0, 3.5)
    st.session_state.last_c = nc
    
    st.session_state.history_v14['t'].append(time.strftime("%M:%S"))
    st.session_state.history_v14['o'].append(pc); st.session_state.history_v14['h'].append(max(pc, nc)+1)
    st.session_state.history_v14['l'].append(min(pc, nc)-1); st.session_state.history_v14['c'].append(nc)
    
    df = pd.DataFrame(st.session_state.history_v14)
    
    # B. 技术指标 (布林带 + RSI + MACD)
    df['ma'] = df['c'].rolling(20).mean().ffill().bfill()
    df['std'] = df['c'].rolling(20).std().ffill().bfill()
    df['up'], df['dn'] = df['ma'] + 2*df['std'], df['ma'] - 2*df['std']
    
    delta = df['c'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = (100 - (100 / (1 + (gain / (loss + 1e-9))))).ffill().bfill()
    df['macd'] = df['c'].ewm(span=12).mean() - df['c'].ewm(span=26).mean()
    df['hist'] = df['macd'] - df['macd'].ewm(span=9).mean()

    # C. 合约盈亏计算
    t = st.session_state.trade
    pnl_str, pnl_class, liq_str = "无持仓", "pnl-long", ""
    if t["side"] != "空仓":
        pnl_raw = (nc - t["entry"]) if t["side"] == "多单" else (t["entry"] - nc)
        pnl_pct = (pnl_raw / t["entry"]) * 100 * t["lev"]
        pnl_class = "pnl-long" if pnl_pct >= 0 else "pnl-short"
        pnl_str = f"{t['side']} {pnl_pct:+.2f}%"
        liq_str = f"估算强平价: ${t['liq']:,.2f}"

    # D. 界面渲染
    price_ph.metric("ETH 现价", f"${nc:,.2f}", f"{nc-pc:.2f}")
    pnl_ph.markdown(f"<div class='metric-box'><span class='{pnl_class}'>{pnl_str}</span><div class='liq-price'>{liq_str}</div></div>", unsafe_allow_html=True)
    rsi_ph.metric("RSI (14)", f"{df['rsi'].iloc[-1]:.1f}", "超买" if df['rsi'].iloc[-1]>70 else "超卖" if df['rsi'].iloc[-1]<30 else "震荡")
    macd_ph.metric("MACD 柱", f"{df['hist'].iloc[-1]:.2f}", "多头自强" if df['hist'].iloc[-1]>0 else "空头占优")

    # E. 三轴联动 K 线图
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.5, 0.25, 0.25])
    fig.add_trace(go.Candlestick(x=df['t'], open=df['o'], high=df['h'], low=df['l'], close=df['c'], name="K线"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['t'], y=df['up'], line=dict(color='rgba(255,255,255,0.1)', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['t'], y=df['dn'], line=dict(color='rgba(255,255,255,0.1)', width=1), fill='tonexty'), row=1, col=1)
    
    # 标注持仓线与强平线
    if t["entry"] > 0:
        fig.add_hline(y=t["entry"], line_dash="dash", line_color="cyan", annotation_text="入场价", row=1, col=1)
        fig.add_hline(y=t["liq"], line_dash="dot", line_color="red", annotation_text="强平线", row=1, col=1)

    fig.add_trace(go.Bar(x=df['t'], y=df['hist'], marker_color=['#00FFC2' if x>0 else '#FF4B4B' for x in df['hist']]), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['t'], y=df['rsi'], line=dict(color='#A491FF', width=2)), row=3, col=1)
    
    fig.update_layout(template="plotly_dark", height=700, margin=dict(l=0,r=0,t=10,b=0), xaxis_rangeslider_visible=False, showlegend=False)
    chart_ph.plotly_chart(fig, key=f"v14_{time.time_ns()}")

    time.sleep(1)
