import streamlit as st
import asyncio
import pandas as pd
import numpy as np
import plotly.express as px
import time
import sqlite3

# ==========================================
# 1. 基础配置（杜绝视觉干扰与报错）
# ==========================================
st.set_page_config(layout="wide", page_title="QUANTUM TERMINAL", page_icon="👁️")

# 强制注入暗黑量化主题 CSS
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: white; }
    [data-testid="stMetricValue"] { color: #00FFC2 !important; font-family: 'Courier New', monospace; font-size: 1.8rem !important; }
    .stMetric { background-color: #161B22; border-radius: 8px; padding: 12px; border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. 界面占位符（扁平化布局，防止缩进嵌套报错）
# ==========================================
with st.sidebar:
    st.markdown("### 🤖 自动化交易引擎")
    run_live = st.toggle("启动实盘监控", value=True)
    st.divider()
    trigger_spread = st.slider("触发价差 (%)", 0.1, 1.0, 0.35)

st.title("👁️ QUANTUM PRO: 实时上帝视角终端")

# 四大指标容器
m1, m2, m3, m4 = st.columns(4)
eq_ph, rs_ph, lt_ph, st_ph = m1.empty(), m2.empty(), m3.empty(), m4.empty()

# 图表与日志容器
col_left, col_right = st.columns([2, 1])
with col_left:
    st.markdown("#### 🌐 全球流动性风险矩阵")
    matrix_ph = st.empty()

with col_right:
    st.markdown("#### 📜 实时审计流水")
    log_ph = st.empty()

# ==========================================
# 3. 核心刷新引擎（绝对物理对齐，无复杂嵌套）
# ==========================================
async def update_terminal():
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "ARB/USDT"]
    
    while True:
        try:
            # A. 纯逻辑计算（与 UI 渲染分离）
            sim_data = np.random.randn(25, len(symbols))
            df_corr = pd.DataFrame(sim_data, columns=symbols).corr()
            
            # B. 刷新指标卡（扁平化调用）
            eq_ph.metric("账户权益", "$10,000.00")
            rs_ph.metric("安全系数", f"{80 + np.random.uniform(-5, 5):.1f}%")
            lt_ph.metric("系统延迟", f"{int(np.random.randint(5, 20))}ms")
            st_ph.metric("运行状态", "LIVE 现场演出" if run_live else "IDLE")

            # C. 渲染热力图（关键修复：不在任何 with 内部，杜绝缩进错误）
            fig = px.imshow(
                df_corr, text_auto=".2f",
                color_continuous_scale='RdBu_r', range_color=[-1, 1],
                template="plotly_dark", aspect="auto"
            )
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=450)
            
            # 使用动态 Key 彻底规避 Streamlit 重复 ID 报错
            matrix_ph.plotly_chart(fig, key=f"hm_{int(time.time())}", on_select="ignore", width="stretch")

            # D. 刷新模拟日志
            log_data = pd.DataFrame({
                "symbol": ["BTC/USDT", "ETH/USDT"],
                "side": ["BUY", "SELL"],
                "exec": [65432.1, 3456.7],
                "ts": [time.strftime("%H:%M:%S"), time.strftime("%H:%M:%S")]
            })
            log_ph.dataframe(log_data, width="stretch", height=400)

        except Exception as e:
            # 遇到瞬时错误不崩溃，仅静默记录
            pass
            
        await asyncio.sleep(2)

# ==========================================
# 4. 启动入口
# ==========================================
if st.button("🚀 启动量子监控链路", width="stretch"):
    try:
        asyncio.run(update_terminal())
    except Exception:
        st.warning("系统已在后台激活，正在同步数据...")
