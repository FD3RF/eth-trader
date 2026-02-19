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
# 🛡️ 1. 系统核心配置
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
# 🎨 2. UI 样式修复 (彻底解决截图中的 TypeError)
# ==========================================
st.set_page_config(layout="wide", page_title="QUANTUM PRO TERMINAL", page_icon="👁️")

# 修正：删除非法参数 unsafe_allow_password，确保 CSS 正常加载
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: white; }
    [data-testid="stMetricValue"] { color: #00FFC2 !important; font-family: 'Courier New', monospace; font-size: 1.8rem !important; }
    .stMetric { background-color: #161B22; border-radius: 10px; padding: 15px; border: 1px solid #30363d; }
    .stDataFrame { border: 1px solid #30363d; border-radius: 10px; }
    .block-container { padding-top: 1.5rem; }
    </style>
    """, unsafe_allow_html=True)

if 'core' not in st.session_state:
    st.session_state.core = QuantumCore()

# ==========================================
# 🖥️ 3. 侧边栏布局 (对应截图)
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
        if st.button("更新连接"):
            st.session_state.core = QuantumCore(api_key, api_sec)
            st.toast("核心已重新挂载")

# ==========================================
# 📊 4. 主界面：实时指标与矩阵 (对应截图 UI)
# ==========================================
st.title("👁️ QUANTUM PRO: 实时上帝视角终端")

# 四大指标卡占位符
m1, m2, m3, m4 = st.columns(4)
eq_ph = m1.empty()
rs_ph = m2.empty()
lt_ph = m3.empty()
st_ph = m4.empty()

col_left, col_right = st.columns([2, 1])
with col_left:
    st.markdown("#### 🌐 全球流动性风险矩阵")
    matrix_ph = st.empty()

with col_right:
    st.markdown("#### 📜 实时审计流水")
    log_ph = st.empty()

# ==========================================
# 🔄 5. 核心刷新循环 (解决 DuplicateKey 及 Deprecation 问题)
# ==========================================
async def update_terminal():
    while True:
        start_ts = time.time()
        
        # A. 数据模拟（实盘可替换为异步 API 请求）
        sim_data = np.random.randn(50, len(CONFIG["symbols"]))
        df_corr = pd.DataFrame(sim_data, columns=CONFIG["symbols"]).corr()
        
        # B. 刷新指标卡
        latency = (time.time() - start_ts) * 1000
        safe_score = (1 - df_corr.mean().mean()) * 100
        
        eq_ph.metric("账户权益 (Equity)", f"${CONFIG['initial_equity']:,.0f}")
        rs_ph.metric("安全系数 (Safety)", f"{safe_score:.1f}%", delta=f"{safe_score - safe_factor:.1f}%")
        lt_ph.metric("系统延迟 (Latency)", f"{int(latency)}ms")
        st_ph.metric("运行状态", "LIVE" if run_live else "IDLE")

        # C. 渲染热力图 (关键修复：使用 container 动态刷新，不设固定 key)
        with matrix_ph.container():
            fig = px.imshow(
                df_corr, text_auto=".2f",
                color_continuous_scale='RdBu_r', range_color=[-1, 1],
                template="plotly_dark", aspect="auto"
            )
            fig.update_layout(
                margin=dict(l=10, r=10, t=10, b=10), height=450,
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
            )
            # 使用 width="stretch" 适配最新版本警告
            st.plotly_chart(fig, on_select="ignore", key=f"corr_{int(time.time())}", width="stretch")

        # D. 刷新审计流水
        with log_ph.container():
            conn = sqlite3.connect(st.session_state.core.db_path)
            try:
                df_log = pd.read_sql("SELECT symbol, side, exec, ts FROM ledger ORDER BY ts DESC LIMIT 15", conn)
                st.dataframe(df_log, width="stretch", height=400)
            except:
                st.info("等待执行信号...")
            finally:
                conn.close()

        await asyncio.sleep(2) # 设置平稳的刷新频率

# ==========================================
# 🏁 6. 运行入口
# ==========================================
if st.button("🚀 启动量子监控链路", width="stretch"):
    try:
        asyncio.run(update_terminal())
    except Exception as e:
        # 捕获 asyncio.run 常见的嵌套运行错误
        st.warning("监控链路正在运行中或已手动停止。")
