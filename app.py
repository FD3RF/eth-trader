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
# 🛡️ 系统配置
# ==========================================
CONFIG = {
    "symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "ARB/USDT"],
    "timeframe": "1h",
    "initial_equity": 10000.0
}

class QuantumCore:
    def __init__(self, api="", sec=""):
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
# 🎨 UI 界面样式 (修复参数报错)
# ==========================================
st.set_page_config(layout="wide", page_title="QUANTUM PRO", page_icon="👁️")

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: white; }
    [data-testid="stMetricValue"] { color: #00FFC2 !important; font-family: 'Courier New', monospace; font-size: 1.8rem !important; }
    .stMetric { background-color: #161B22; border-radius: 10px; padding: 15px; border: 1px solid #30363d; }
    .stDataFrame { border: 1px solid #30363d; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

if 'core' not in st.session_state:
    st.session_state.core = QuantumCore()

# ==========================================
# 🖥️ 侧边栏
# ==========================================
with st.sidebar:
    st.markdown("### 🤖 自动化交易计划")
    run_live = st.toggle("启动实盘执行计划", value=False)
    st.divider()
    trigger_spread = st.slider("触发价差 (%)", 0.1, 1.0, 0.30)
    safe_factor = st.slider("最小安全系数 (%)", 50.0, 100.0, 95.00)
    st.divider()
    with st.expander("🔑 API 密钥配置"):
        api_key = st.text_input("API Key", type="password")
        api_sec = st.text_input("Secret Key", type="password")
        if st.button("更新密钥"):
            st.session_state.core = QuantumCore(api_key, api_sec)
            st.toast("核心已重载")

# ==========================================
# 📊 主界面布局
# ==========================================
st.title("👁️ QUANTUM PRO: 实时上帝视角终端")

m1, m2, m3, m4 = st.columns(4)
eq_ph = m1.empty()
rs_ph = m2.empty()
lt_ph = m3.empty()
st_ph = m4.empty()

col_left, col_right = st.columns([2, 1])
with col_left:
    st.markdown("#### 🌐 全球流动性风险矩阵")
    matrix_container = st.empty()

with col_right:
    st.markdown("#### 📜 实时审计流水")
    log_container = st.empty()

# ==========================================
# 🔄 修复 DuplicateKey 问题的核心循环
# ==========================================
async def update_terminal():
    # 使用 session_state 跟踪运行状态，防止重复触发
    while True:
        start_ts = time.time()
        
        # 1. 模拟行情与矩阵计算
        sim_data = np.random.randn(50, len(CONFIG["symbols"]))
        df_corr = pd.DataFrame(sim_data, columns=CONFIG["symbols"]).corr()
        
        # 2. 渲染热力图 (移除静态 key，防止 DuplicateElementKey 错误)
        with matrix_container.container():
            fig = px.imshow(
                df_corr, text_auto=".2f",
                color_continuous_scale='RdBu_r', range_color=[-1, 1],
                template="plotly_dark", aspect="auto"
            )
            fig.update_layout(
                margin=dict(l=10, r=10, t=10, b=10), height=450,
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
            )
            # 关键修复：不再手动指定 key，让 Streamlit 自动处理容器内元素
            st.plotly_chart(fig, use_container_width=True)

        # 3. 更新指标卡
        latency = (time.time() - start_ts) * 1000
        safe_score = (1 - df_corr.mean().mean()) * 100
        
        eq_ph.metric("账户权益", f"${CONFIG['initial_equity']:,.0f}")
        rs_ph.metric("安全系数", f"{safe_score:.1f}%", delta=f"{safe_score - safe_factor:.1f}%")
        lt_ph.metric("系统延迟", f"{int(latency)}ms")
        st_ph.metric("运行状态", "LIVE" if run_live else "IDLE")

        # 4. 更新审计日志
        with log_container.container():
            conn = sqlite3.connect(st.session_state.core.db_path)
            try:
                df_log = pd.read_sql("SELECT symbol, side, exec, ts FROM ledger ORDER BY ts DESC LIMIT 10", conn)
                st.dataframe(df_log, use_container_width=True, height=400)
            except:
                st.info("等待信号...")
            finally:
                conn.close()

        await asyncio.sleep(2)

# 启动逻辑
if st.button("🚀 启动量子监控链路", use_container_width=True):
    # 确保在运行环境中只启动一个异步循环
    asyncio.run(update_terminal())
