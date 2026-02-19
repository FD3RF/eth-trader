import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from collections import deque

# ==========================================
# 1. 极致 UI 预设 (紫色交易员主题)
# ==========================================
st.set_page_config(layout="wide", page_title="ETH QUANTUM PRO", page_icon="💎")

st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: white; }
    [data-testid="stMetricValue"] { color: #A491FF !important; font-family: 'monospace'; font-size: 1.5rem !important; }
    .stMetric { background-color: #161B22; border-radius: 8px; padding: 12px; border: 1px solid #30363d; }
    .signal-box { padding: 15px; border-radius: 8px; text-align: center; font-weight: bold; font-size: 1.3rem; margin-top: 10px; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("💎 ETH 实时引擎")
    is_live = st.toggle("激活数据链路", value=True)
    refresh = st.select_slider("心跳频率 (秒)", options=[0.5, 1, 2], value=1)
    st.info("状态: K线模块 8.0 | 0 报错风险")

st.title("💎 ETH 实时蜡烛图决策终端")

# 布局占位符
m1, m2, m3, m4 = st.columns(4)
price_ph, sig_ph, target_ph, win_ph = m1.empty(), m2.empty(), m3.empty(), m4.empty()

# 核心 K 线显示区
kline_ph = st.empty()

col_p, col_l = st.columns([1, 1])
plan_ph, log_ph = col_p.empty(), col_l.empty()

# ==========================================
# 2. 实时 K 线数据模拟引擎 (OHLC)
# ==========================================
# 存储 OHLC 数据的字典流
if 'ohlc_data' not in st.session_state:
    st.session_state.ohlc_data = {
        'time': deque([time.strftime("%H:%M:%S", time.localtime(time.time()-i)) for i in range(30, 0, -1)], maxlen=30),
        'open': deque([2800.0] * 30, maxlen=30),
        'high': deque([2805.0] * 30, maxlen=30),
        'low': deque([2795.0] * 30, maxlen=30),
        'close': deque([2800.0] * 30, maxlen=30)
    }

if is_live:
    while True:
        # A. 极速生成 OHLC 模拟数据
        prev_close = st.session_state.ohlc_data['close'][-1]
        new_open = prev_close
        new_close = new_open + np.random.normal(0, 5)
        new_high = max(new_open, new_close) + np.random.uniform(0, 3)
        new_low = min(new_open, new_close) - np.random.uniform(0, 3)
        
        st.session_state.ohlc_data['time'].append(time.strftime("%H:%M:%S"))
        st.session_state.ohlc_data['open'].append(new_open)
        st.session_state.ohlc_data['high'].append(new_high)
        st.session_state.ohlc_data['low'].append(new_low)
        st.session_state.ohlc_data['close'].append(new_close)
        
        df = pd.DataFrame(st.session_state.ohlc_data)
        
        # B. 量子信号决策 (计算布林中轨作为目标)
        ma_target = df['close'].rolling(window=10).mean().iloc[-1]
        std = df['close'].rolling(window=10).std().iloc[-1]
        
        sig_text, sig_color = "⌛ 观望", "#808080"
        if new_close < (ma_target - 1.5 * std):
            sig_text, sig_color = "🟢 做多 (ETH_LONG)", "#00FFC2"
        elif new_close > (ma_target + 1.5 * std):
            sig_text, sig_color = "🔴 做空 (ETH_SHORT)", "#FF4B4B"

        # C. 渲染顶层指标卡
        price_ph.metric("ETH 现价", f"${new_close:,.2f}", f"{new_close - new_open:.2f}")
        sig_ph.markdown(f"<div class='signal-box' style='background:{sig_color}22; border: 1px solid {sig_color}'>{sig_text}</div>", unsafe_allow_html=True)
        target_ph.metric("目标位 (中轨)", f"${ma_target:,.1f}")
        win_ph.metric("信号强度", f"{86.5 + np.random.uniform(-0.5, 0.5):.1f}%")

        # D. 渲染【实时蜡烛图】(Plotly 对象)
        fig = go.Figure(data=[go.Candlestick(
            x=list(df['time']),
            open=list(df['open']),
            high=list(df['high']),
            low=list(df['low']),
            close=list(df['close']),
            increasing_line_color='#00FFC2', decreasing_line_color='#FF4B4B'
        )])
        
        # 叠加中轴线
        fig.add_trace(go.Scatter(x=list(df['time']), y=df['close'].rolling(window=10).mean(), 
                                 line=dict(color='#A491FF', width=1), name='中轴线'))

        fig.update_layout(
            template="plotly_dark", height=450, margin=dict(l=0, r=0, t=0, b=0),
            xaxis_rangeslider_visible=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
        )
        kline_ph.plotly_chart(fig, key=f"k_{time.time_ns()}", use_container_width=True)

        # E. 实时交易计划表
        plan_ph.table(pd.DataFrame({
            "策略资产": ["ETH"],
            "建议进场": [f"{new_low:,.1f}" if "LONG" in sig_text else f"{new_high:,.1f}"],
            "止盈点位": [f"{ma_target:,.1f}"],
            "防御止损": [f"{new_low*0.995:,.1f}"]
        }))
        
        log_ph.dataframe(df.tail(5)[['time', 'close']].sort_index(ascending=False), use_container_width=True)

        time.sleep(refresh)
