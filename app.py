import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from collections import deque

# ==========================================
# 1. 极致 UI 初始化
# ==========================================
st.set_page_config(layout="wide", page_title="ETH QUANTUM V11", page_icon="🏦")

st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: white; }
    [data-testid="stMetricValue"] { color: #A491FF !important; font-size: 1.8rem !important; font-weight: bold; }
    .stMetric { background-color: #161B22; border-radius: 10px; border: 1px solid #30363d; padding: 15px; }
    .sig-card { padding: 20px; border-radius: 10px; text-align: center; font-weight: bold; font-size: 1.4rem; border: 2px solid; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("🏦 ETH 决策核心")
    is_live = st.toggle("同步以太坊数据链", value=True)
    sens = st.slider("信号灵敏度", 1.0, 2.5, 1.6)
    st.divider()
    st.success("V11 引擎：冷启动保护已开启")

st.title("🏦 ETH 实时上帝视角决策终端")

# 占位符定义
m1, m2, m3, m4 = st.columns(4)
p_ph, s_ph, r_ph, d_ph = m1.empty(), m2.empty(), m3.empty(), m4.empty()

chart_ph = st.empty()

col_plan, col_log = st.columns([1, 1])
plan_ph, log_ph = col_plan.empty(), col_log.empty()

# ==========================================
# 2. 稳健数据引擎
# ==========================================
if 'history' not in st.session_state:
    # 预填充 60 个点，防止冷启动报错
    st.session_state.history = {
        't': deque([time.strftime("%H:%M:%S", time.localtime(time.time()-i)) for i in range(60, 0, -1)], maxlen=60),
        'o': deque([2800.0 + np.random.randn() for _ in range(60)], maxlen=60),
        'h': deque([2805.0] * 60, maxlen=60),
        'l': deque([2795.0] * 60, maxlen=60),
        'c': deque([2800.0] * 60, maxlen=60)
    }

if is_live:
    while True:
        # A. 实时价格合成
        last_c = st.session_state.history['c'][-1]
        n_o = last_c
        n_c = n_o + np.random.normal(0, 4)
        n_h = max(n_o, n_c) + np.random.uniform(0, 3)
        n_l = min(n_o, n_c) - np.random.uniform(0, 3)
        
        st.session_state.history['t'].append(time.strftime("%M:%S"))
        st.session_state.history['o'].append(n_o); st.session_state.history['h'].append(n_h)
        st.session_state.history['l'].append(n_l); st.session_state.history['c'].append(n_c)
        
        df = pd.DataFrame(st.session_state.history)
        
        # B. 技术指标计算 (带对齐保护)
        df['ma'] = df['c'].rolling(20).mean()
        df['std'] = df['c'].rolling(20).std()
        df['up'] = df['ma'] + (sens * df['std'])
        df['dn'] = df['ma'] - (sens * df['std'])
        
        # RSI 计算
        delta = df['c'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
        
        # MACD 计算
        df['ema12'] = df['c'].ewm(span=12).mean()
        df['ema26'] = df['c'].ewm(span=26).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['sig'] = df['macd'].ewm(span=9).mean()
        df['hist'] = df['macd'] - df['sig']

        # C. 共振决策引擎
        curr_c = df['c'].iloc[-1]
        curr_rsi = df['rsi'].iloc[-1]
        curr_hist = df['hist'].iloc[-1]
        
        decision, color = "⌛ 扫描信号中", "#808080"
        if curr_c < df['dn'].iloc[-1] and curr_rsi < 40:
            decision, color = "🚀 做多 (STRONG LONG)", "#00FFC2"
        elif curr_c > df['up'].iloc[-1] and curr_rsi > 60:
            decision, color = "🔥 做空 (STRONG SHORT)", "#FF4B4B"

        # D. UI 渲染 - 指标卡
        p_ph.metric("ETH 实时价", f"${curr_c:,.2f}", f"{curr_c - df['c'].iloc[-2]:.2f}")
        s_ph.markdown(f"<div class='sig-card' style='color:{color}; border-color:{color}; background:{color}11'>{decision}</div>", unsafe_allow_html=True)
        r_r = f"{curr_rsi:.1f}" if not np.isnan(curr_rsi) else "计算中"
        r_ph.metric("RSI 强度", r_r, "超卖" if curr_rsi < 30 else "超买" if curr_rsi > 70 else "中性")
        d_ph.metric("MACD 动能", f"{curr_hist:.2f}", "多头胜" if curr_hist > 0 else "空头胜")

        # E. 专业 K 线图渲染 (三子图)
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2])
        # 蜡烛图与布林带
        fig.add_trace(go.Candlestick(x=df['t'], open=df['o'], high=df['h'], low=df['l'], close=df['c'], name="Price"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['t'], y=df['up'], line=dict(color='rgba(164,145,255,0.2)', width=1), name="Upper"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['t'], y=df['dn'], line=dict(color='rgba(164,145,255,0.2)', width=1), fill='tonexty', name="Lower"), row=1, col=1)
        # MACD
        fig.add_trace(go.Bar(x=df['t'], y=df['hist'], marker_color=['#00FFC2' if x>0 else '#FF4B4B' for x in df['hist']]), row=2, col=1)
        # RSI
        fig.add_trace(go.Scatter(x=df['t'], y=df['rsi'], line=dict(color='#A491FF')), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#FF4B4B", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#00FFC2", row=3, col=1)

        fig.update_layout(template="plotly_dark", height=650, margin=dict(l=0, r=0, t=0, b=0), showlegend=False, xaxis_rangeslider_visible=False)
        chart_ph.plotly_chart(fig, key=f"v11_{time.time_ns()}")

        # F. 实时计划计划表 (修复长度对齐)
        plan_data = {
            "维度": ["BOLL轨道", "RSI状态", "MACD趋势"],
            "数据": [f"{'触底' if curr_c < df['ma'].iloc[-1] else '冲顶'}", f"{r_r}", f"{'金叉' if curr_hist > 0 else '死叉'}"],
            "建议行动": [decision, decision, decision]
        }
        plan_ph.dataframe(pd.DataFrame(plan_data), hide_index=True)
        log_ph.dataframe(df.tail(5)[['t', 'c']].sort_index(ascending=False), hide_index=True)

        time.sleep(1)
