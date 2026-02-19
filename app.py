import streamlit as st
import asyncio
import pandas as pd
import numpy as np
import plotly.express as px
import time
import sqlite3

# --- 1. 基础配置与视觉样式 ---
st.set_page_config(layout="wide", page_title="QUANTUM PRO", page_icon="👁️")

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: white; }
    [data-testid="stMetricValue"] { color: #00FFC2 !important; font-family: 'monospace'; }
    .stMetric { background-color: #161B22; border-radius: 8px; padding: 15px; border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 页面布局占位符（扁平化结构，防止缩进报错） ---
with st.sidebar:
    st.title("🤖 交易引擎")
    run_live = st.toggle("启动实盘监控", value=True)
    st.divider()
    st.slider("触发价差 (%)", 0.1, 1.0, 0.35)

st.title("👁️ QUANTUM PRO: 实时上帝视角终端")

# 预设四大指标容器
c1, c2, c3, c4 = st.columns(4)
eq_ph = c1.empty()
rs_ph = c2.empty()
lt_ph = c3.empty()
st_ph = c4.empty()

# 预设主图表与日志容器
col_left, col_right = st.columns([2, 1])
with col_left:
    st.markdown("#### 🌐 全球流动性风险矩阵")
    matrix_ph = st.empty()

with col_right:
    st.markdown("#### 📜 实时审计流水")
    log_ph = st.empty()

# --- 3. 核心刷新引擎（无嵌套逻辑，绝对对齐） ---
async def update_terminal():
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "ARB/USDT"]
    
    while True:
        try:
            # A. 生成模拟数据
            sim_data = np.random.randn(25, len(symbols))
            df_corr = pd.DataFrame(sim_data, columns=symbols).corr()
            
            # B. 刷新指标卡
            eq_ph.metric("账户权益", "$10,000.00")
            rs_ph.metric("安全系数", f"{85.0 + np.random.uniform(-5, 5):.1f}%")
            lt_ph.metric("系统延迟", f"{np.random.randint(5, 15)}ms")
            st_ph.metric("运行状态", "LIVE 现场演出" if run_live else "IDLE")

            # C. 渲染热力图（关键：直接作用于占位符，不使用嵌套 with）
            fig = px.imshow(
                df_corr, text_auto=".2f",
                color_continuous_scale='RdBu_r', range_color=[-1, 1],
                template="plotly_dark", aspect="auto"
            )
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=400)
            
            # 使用动态 key 彻底解决截图中的 Duplicate Element ID 潜在问题
            matrix_ph.plotly_chart(fig, key=f"hm_{int(time.time())}", width="stretch")

            # D. 刷新简易日志
            log_data = pd.DataFrame({
                "symbol": ["BTC/USDT", "ETH/USDT"],
                "side": ["BUY", "SELL"],
                "exec": [65432.1, 3456.7]
            })
            log_ph.dataframe(log_data, width="stretch", height=350)

        except Exception as e:
            st.error(f"刷新异常: {e}")
            
        await asyncio.sleep(2)

# --- 4. 安全启动入口 ---
if st.button("🚀 重新激活监控链路", width="stretch"):
    asyncio.run(update_terminal())
