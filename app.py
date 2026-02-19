import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from collections import deque

# ==========================================
# 1. 极致 UI 架构 (2026 稳定标准)
# ==========================================
st.set_page_config(layout="wide", page_title="ETH QUANTUM V12", page_icon="📈")

st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: white; }
    [data-testid="stMetricValue"] { color: #A491FF !important; font-size: 1.6rem !important; }
    .stMetric { background-color: #161B22; border-radius: 8px; border: 1px solid #30363d; padding: 10px; }
    .signal-hero { padding: 15px; border-radius: 10px; text-align: center; font-size: 1.4rem; font-weight: bold; margin-bottom: 20px; }
    .advice-card { background: #161B22; padding: 10px; border-radius: 5px; border-left: 4px solid #A491FF; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ 引擎控制台")
    is_live = st.toggle("同步实时行情流", value=True)
    refresh_rate = st.select_slider("心跳频率 (秒)", options=[0.5, 1, 2], value=1)
    st.divider()
    st.info("架构：V12 零报错版 | 已激活冷启动保护")

st.title("🛡️ ETH 实时上帝视角决策终端")

# 顶层状态栏
c1, c2, c3, c4 = st.columns(4)
price_ph, sig_ph, rsi_ph, macd_ph = c1.empty(), c2.empty(), c3.empty(), c4.empty()

# 核心图表区
chart_ph = st.empty()

# 底部决策区
st.subheader("🛠️ 智能交易建议")
d1, d2, d3 = st.columns(3)
adv1, adv2, adv3 = d1.empty(), d2.empty(), d3.empty()

# ==========================================
# 2. 稳健数据引擎 (带对齐保护)
# ==========================================
if 'v12_cache' not in st.session_state:
    # 初始预填充 80 个点，确保指标计算有足够样本
    st.session_state.v12_cache = {
        't': deque([time.strftime("%H:%M:%S", time.localtime(time.time()-i)) for i in range(80, 0, -1)], maxlen=80),
        'o': deque([2800.0] * 80, maxlen=80),
        'h': deque([2805.0] * 80, maxlen=80),
        'l': deque([2795.0] * 80, maxlen=80),
        'c': deque([2800.0] * 80, maxlen=80)
    }

while is_live:
    # A. 模拟以太坊实时波动
    prev_c = st.session_state.v12_cache['c'][-1]
    new_o = prev_c
    new_c = new_o + np.random.normal(0, 4.2)
    new_h = max(new_o, new_c) + np.random.uniform(0, 2)
    new_l = min(new_o, new_c) - np.random.uniform(0, 2)
    
    st.session_state.v12_cache['t'].append(time.strftime("%M:%S"))
    st.session_state.v12_cache['o'].append(new_o); st.session_state.v12_cache['h'].append(new_h)
    st.session_state.v12_cache['l'].append(new_l); st.session_state.v12_cache['c'].append(new_c)
    
    df = pd.DataFrame(st.session_state.v12_cache)
    
    # B. 安全指标计算 (强制填充 NaN 以防报错)
    df['ma'] = df['c'].rolling(20).mean().ffill().bfill()
    df['std'] = df['c'].rolling(20).std().ffill().bfill()
    df['up'] = df['ma'] + (1.6 * df['std'])
    df['dn'] = df['ma'] - (1.6 * df['std'])
    
    # RSI & MACD
    delta = df['c'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = (100 - (100 / (1 + (gain / (loss + 1e-9))))).ffill().bfill()
    df['macd'] = df['c'].ewm(span=12).mean() - df['c'].ewm(span=26).mean()
    df['hist'] = df['macd'] - df['macd'].ewm(span=9).mean()

    # C. 决策逻辑
    cur_c, cur_rsi, cur_h = df['c'].iloc[-1], df['rsi'].iloc[-1], df['hist'].iloc[-1]
    decision, color = "⌛ 扫描信号中", "#808080"
    if cur_c < df['dn'].iloc[-1] and cur_rsi < 35:
        decision, color = "🟢 强力做多 (LONG)", "#00FFC2"
    elif cur_c > df['up'].iloc[-1] and cur_rsi > 65:
        decision, color = "🔴 强力做空 (SHORT)", "#FF4B4B"

    # D. 渲染指标卡
    price_ph.metric("ETH 现价", f"${cur_c:,.2f}", f"{cur_c-df['c'].iloc[-2]:.2f}")
    sig_ph.markdown(f"<div class='signal-hero' style='background:{color}22; border: 1px solid {color}; color:{color}'>{decision}</div>", unsafe_allow_html=True)
    rsi_ph.metric("RSI 指数", f"{cur_rsi:.1f}", "超买区" if cur_rsi > 70 else "超卖区" if cur_rsi < 30 else "常态")
    macd_ph.metric("MACD 柱", f"{cur_h:.2f}", "多头增强" if cur_h > 0 else "空头增强")

    # E. 渲染主图 (3子图联动)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.5, 0.25, 0.25])
    fig.add_trace(go.Candlestick(x=df['t'], open=df['o'], high=df['h'], low=df['l'], close=df['c'], name="K线"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['t'], y=df['up'], line=dict(color='rgba(164,145,255,0.2)', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['t'], y=df['dn'], line=dict(color='rgba(164,145,255,0.2)', width=1), fill='tonexty'), row=1, col=1)
    fig.add_trace(go.Bar(x=df['t'], y=df['hist'], marker_color=['#00FFC2' if x>0 else '#FF4B4B' for x in df['hist']]), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['t'], y=df['rsi'], line=dict(color='#A491FF', width=2)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#FF4B4B", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#00FFC2", row=3, col=1)
    
    fig.update_layout(template="plotly_dark", height=600, margin=dict(l=0,r=0,t=0,b=0), xaxis_rangeslider_visible=False, showlegend=False)
    chart_ph.plotly_chart(fig, key=f"v12_{time.time_ns()}")

    # F. 渲染决策卡 (替代不稳定的 DataFrame 渲染)
    adv1.markdown(f"<div class='advice-card'><b>轨道位置:</b><br>{'偏离底轨' if cur_c < df['ma'].iloc[-1] else '偏离顶轨'}</div>", unsafe_allow_html=True)
    adv2.markdown(f"<div class='advice-card'><b>RSI 建议:</b><br>{'等待反转' if cur_rsi < 40 else '等待回调'}</div>", unsafe_allow_html=True)
    adv3.markdown(f"<div class='advice-card'><b>操作逻辑:</b><br>{decision}</div>", unsafe_allow_html=True)

    time.sleep(refresh_rate)
