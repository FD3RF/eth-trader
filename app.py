import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time

# ==========================================
# 🎨 1. 基础配置与视觉样式
# ==========================================
st.set_page_config(layout="wide", page_title="QUANTUM PRO", page_icon="👁️")

# 注入 CSS 确保暗黑模式下的文字清晰度
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: white; }
    [data-testid="stMetricValue"] { color: #00FFC2 !important; font-family: 'monospace'; }
    .stMetric { background-color: #161B22; border-radius: 8px; padding: 12px; border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 🖥️ 2. 静态布局占位符 (防止渲染顺序混乱)
# ==========================================
with st.sidebar:
    st.markdown("### 🤖 自动化交易引擎")
    run_live = st.toggle("启动实盘监控", value=True)
    st.divider()
    trigger_val = st.slider("触发价差 (%)", 0.1, 1.0, 0.35)

st.title("👁️ QUANTUM PRO: 实时终端")

# 第一行：指标卡
m1, m2, m3, m4 = st.columns(4)
eq_ph = m1.empty()
rs_ph = m2.empty()
lt_ph = m3.empty()
st_ph = m4.empty()

# 第二行：主要内容
col_left, col_right = st.columns([2, 1])
with col_left:
    st.markdown("#### 🌐 风险矩阵")
    matrix_ph = st.empty()  # 矩阵占位符

with col_right:
    st.markdown("#### 📜 审计流水")
    log_ph = st.empty()     # 日志占位符

# ==========================================
# 🔄 3. 同步刷新引擎 (方案 A 增强版)
# ==========================================
# 核心修复：将绘图逻辑彻底从深层嵌套中移出，防止 IndentationError
def render_frame():
    symbols = ["BTC", "ETH", "SOL", "BNB", "ARB"]
    
    # A. 模拟计算
    sim_data = np.random.randn(25, len(symbols))
    df_corr = pd.DataFrame(sim_data, columns=symbols).corr()
    
    # B. 更新指标卡
    eq_ph.metric("账户权益", "$10,000.00")
    rs_ph.metric("安全系数", f"{np.random.uniform(70, 95):.1f}%")
    lt_ph.metric("系统延迟", f"{np.random.randint(5, 15)}ms")
    st_ph.metric("状态", "LIVE" if run_live else "IDLE")

    # C. 渲染热力图 (关键修复：物理对齐，使用唯一 Key)
    
    fig = px.imshow(
        df_corr, text_auto=".2f",
        color_continuous_scale='RdBu_r', range_color=[-1, 1],
        template="plotly_dark", aspect="auto"
    )
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=400)
    
    # 动态 Key 防止截图中的 Duplicate Element ID 报错
    matrix_ph.plotly_chart(fig, key=f"mat_{int(time.time())}", use_container_width=True)

    # D. 更新流水
    log_data = pd.DataFrame({
        "time": [time.strftime("%H:%M:%S")] * 2,
        "symbol": ["BTC/USDT", "ETH/USDT"],
        "side": ["BUY", "SELL"]
    })
    log_ph.dataframe(log_data, use_container_width=True)

# --- 启动循环 ---
if st.button("🚀 激活量子监控链路", use_container_width=True):
    while True:
        render_frame()
        time.sleep(2)  # 每2秒刷新一次
