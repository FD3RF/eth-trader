import streamlit as st
import asyncio
import pandas as pd
import numpy as np
import plotly.express as px
import time
import sqlite3
from datetime import datetime
import ccxt.async_support as ccxt

# ==========================================
# 🛡️ 系统配置与核心引擎
# ==========================================
CONFIG = {
    "symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "ARB/USDT"],
    "timeframe": "1h",
    "initial_equity": 10000.0
}

class QuantumCore:
    def __init__(self, api="", sec=""):
        # 即使没有密钥，模拟模式也能运行
        self.ex = ccxt.binance({
            "apiKey": api, "secret": sec,
            "options": {"defaultType": "future", "adjustForTimeDifference": True},
            "enableRateLimit": True
        })
        self.db_path = "quantum_audit.db"
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ledger (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                symbol TEXT, side TEXT, exec REAL, slip REAL
            )
        """)
        conn.close()

# ==========================================
# 🎨 UI 界面与 CSS 样式 (已修正报错)
# ==========================================
st.set_page_config(layout="wide", page_title="QUANTUM PRO TERMINAL", page_icon="👁️")

# 修正核心错误：使用正确的 unsafe_allow_html=True
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: white; }
    [data-testid="stMetricValue"] { color: #00FFC2 !important; font-family: 'Courier New', monospace; font-size: 1.8rem !important; }
    .stMetric { background-color: #161B22; border-radius: 10px; padding: 15px; border: 1px solid #30363d; }
    .stDataFrame { border: 1px solid #30363d; border-radius: 10px; }
    /* 隐藏多余边距 */
    .block-container { padding-top: 2rem; }
    </style>
    """, unsafe_allow_html=True)

# 初始化引擎
if 'core' not in st.session_state:
    st.session_state.core = QuantumCore()

core = st.session_state.core

# ==========================================
# 🖥️ 侧边栏布局 (对应您的截图)
# ==========================================
with st.sidebar:
    st.markdown("### 🤖 自动化交易计划")
    run_live = st.toggle("启动实盘执行计划", value=False)
    
    st.markdown("---")
    trigger_spread = st.slider("触发价差 (%)", 0.1, 1.0, 0.30)
    safe_factor = st.slider("最小安全系数 (%)", 50.0, 100.0, 95.00)
    
    st.divider()
    with st.expander("🔑 API 密钥配置"):
        api_key = st.text_input("API Key", type="password")
        api_sec = st.text_input("Secret Key", type="password")
        if st.button("更新密钥"):
            st.session_state.core = QuantumCore(api_key, api_sec)
            st.success("API 核心已就绪")

# ==========================================
# 📊 主界面指标与图表 (对应您的截图)
# ==========================================
st.title("👁️ QUANTUM PRO: 实时上帝视角终端")

# 四个核心指标占位符
m1, m2, m3, m4 = st.columns(4)
equity_ph = m1.empty()
risk_ph = m2.empty()
latency_ph = m3.empty()
status_ph = m4.empty()

# 中央显示区：左侧热力图，右侧审计流水
col_left, col_right = st.columns([2, 1])

with col_left:
    st.markdown("#### 🌐 全球流动性风险矩阵")
    matrix_ph = st.empty()

with col_right:
    st.markdown("#### 📜 实时审计流水")
    log_ph = st.empty()

# ==========================================
# 🔄 实时异步循环
# ==========================================
async def update_terminal():
    while True:
        start_ts = time.time()
        
        # 1. 模拟行情数据与相关性计算
        # 实盘环境下此处将调用 core.ex.fetch_ohlcv
        sim_data = np.random.randn(100, len(CONFIG["symbols"]))
        df_sim = pd.DataFrame(sim_data, columns=CONFIG["symbols"])
        corr_matrix = df_sim.corr()
        
        # 2. 渲染热力图
        with matrix_ph.container():
            fig = px.imshow(
                corr_matrix,
                text_auto=".2f",
                color_continuous_scale='RdBu_r', # 红蓝对比，对应您截图的风格
                range_color=[-1, 1],
                template="plotly_dark",
                aspect="auto"
            )
            fig.update_layout(
                margin=dict(l=0, r=0, t=10, b=0),
                height=450,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True, key="risk_matrix")

        # 3. 计算指标值
        latency = (time.time() - start_ts) * 1000
        current_safe_score = (1 - corr_matrix.mean().mean()) * 100
        
        # 4. 更新前端指标卡
        equity_ph.metric("账户权益 (Equity)", f"${CONFIG['initial_equity']:,.0f},...")
        risk_ph.metric(
            "安全系数 (Safety)", 
            f"{current_safe_score:.1f}%",
            delta=f"{current_safe_score - safe_factor:.1f}%",
            delta_color="normal" if current_safe_score >= safe_factor else "inverse"
        )
        latency_ph.metric("系统延迟 (Late...)", f"{int(latency)}ms")
        status_ph.metric("运行状态", "LIVE..." if run_live else "IDLE...")

        # 5. 读取审计流水
        with log_ph.container():
            conn = sqlite3.connect(core.db_path)
            try:
                df_audit = pd.read_sql("SELECT symbol, side, exec, ts FROM ledger ORDER BY ts DESC LIMIT 15", conn)
                if df_audit.empty:
                    st.write("等待首笔信号执行...")
                else:
                    st.dataframe(df_audit, use_container_width=True, height=400)
            except:
                st.write("审计数据库同步中...")
            finally:
                conn.close()

        # 6. 安全报警 (UI 实时反馈)
        if run_live and current_safe_score < safe_factor:
            st.toast(f"风险预警：安全系数 {current_safe_score:.1f}% 低于阈值", icon="⚠️")

        await asyncio.sleep(2) # 刷新间隔

# 启动引擎按钮
if st.button("🚀 启动量子监控链路", use_container_width=True):
    asyncio.run(update_terminal())
