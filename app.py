import streamlit as st
import asyncio
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import sqlite3
from datetime import datetime
import ccxt.async_support as ccxt

# ==========================================
# 🛡️ 核心配置与后端架构
# ==========================================
CONFIG = {
    "symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "ARB/USDT"],
    "timeframe": "1h",
    "leverage": 3,
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
                symbol TEXT, side TEXT, size REAL, 
                entry REAL, exec REAL, slip REAL, var REAL
            )
        """)
        conn.close()

# ==========================================
# 🎨 UI 布局与样式修正 (彻底修复报错)
# ==========================================
st.set_page_config(layout="wide", page_title="QUANTUM PRO TERMINAL", page_icon="👁️")

# 修正：使用正确的 unsafe_allow_html 参数
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: white; }
    [data-testid="stMetricValue"] { color: #00FFC2 !important; font-family: 'Courier New', monospace; }
    .stMetric { background-color: #161B22; border-radius: 10px; padding: 15px; border: 1px solid #30363d; }
    .stDataFrame { border: 1px solid #30363d; border-radius: 10px; }
    /* 隐藏 Streamlit 默认页眉 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# 初始化核心引擎
if 'core' not in st.session_state:
    st.session_state.core = QuantumCore()
    st.session_state.loop_active = False

core = st.session_state.core

# ==========================================
# 🖥️ 侧边栏控件 (对应截图)
# ==========================================
with st.sidebar:
    st.markdown("### 🤖 自动化交易计划")
    run_live = st.toggle("启动实盘执行计划", value=False)
    
    st.markdown("---")
    trigger_spread = st.slider("触发价差 (%)", 0.1, 1.0, 0.3)
    safe_factor = st.slider("最小安全系数 (%)", 50.0, 100.0, 95.0)
    
    st.divider()
    with st.expander("🔑 API 密钥配置"):
        api_key = st.text_input("API Key", type="password")
        api_sec = st.text_input("Secret Key", type="password")
        if st.button("更新连接"):
            st.session_state.core = QuantumCore(api_key, api_sec)
            st.toast("API 核心已重载", icon="✅")

# ==========================================
# 📊 主界面：实时上帝视角
# ==========================================
st.title("👁️ QUANTUM PRO: 实时上帝视角终端")

# 指标行
m1, m2, m3, m4 = st.columns(4)
equity_val = m1.empty()
risk_val = m2.empty()
latency_val = m3.empty()
status_val = m4.empty()

# 中央显示区
chart_col, log_col = st.columns([2, 1])

with chart_col:
    st.markdown("#### 🌐 全球流动性风险矩阵")
    matrix_placeholder = st.empty()

with log_col:
    st.markdown("#### 📜 实时审计流水")
    log_placeholder = st.empty()

# ==========================================
# 🔄 核心刷新逻辑 (异步)
# ==========================================
async def update_terminal():
    while True:
        start_time = time.time()
        
        # 1. 模拟行情与相关性生成 (实盘时此处替换为核心 API 调用)
        # 生成 50 个时间点的随机收益率以计算相关性
        sim_returns = pd.DataFrame(
            np.random.randn(50, len(CONFIG["symbols"])), 
            columns=CONFIG["symbols"]
        )
        corr_matrix = sim_returns.corr()
        
        # 2. 渲染 Plotly 热力图
        with matrix_placeholder.container():
            fig = px.imshow(
                corr_matrix,
                text_auto=".2f",
                color_continuous_scale='RdBu_r',
                range_color=[-1, 1],
                template="plotly_dark",
                aspect="auto"
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=20, b=20),
                height=450
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # 3. 计算并更新指标
        latency = (time.time() - start_time) * 1000
        current_safe_score = (1 - corr_matrix.mean().mean()) * 100
        
        equity_val.metric("账户权益 (Equity)", f"${CONFIG['initial_equity']:,.2f}")
        risk_val.metric(
            "安全系数 (Safety)", 
            f"{current_safe_score:.1f}%",
            delta=f"{current_safe_score - safe_factor:.1f}%"
        )
        latency_place = f"{latency:.0f}ms"
        latency_val.metric("系统延迟 (Latency)", latency_place)
        status_val.metric("运行状态", "LIVE" if run_live else "IDLE")

        # 4. 读取审计数据库
        with log_placeholder.container():
            try:
                conn = sqlite3.connect(core.db_path)
                # 尝试读取数据，若为空则显示空表
                df_log = pd.read_sql("SELECT symbol, side, exec, ts FROM ledger ORDER BY ts DESC LIMIT 12", conn)
                st.dataframe(df_log, use_container_width=True, height=400)
                conn.close()
            except:
                st.info("等待首笔交易审计落盘...")

        # 5. 风险报警
        if run_live and current_safe_score < safe_factor:
            st.warning(f"🚨 风险预警：当前安全系数 {current_safe_score:.1f}% 低于阈值！")

        await asyncio.sleep(2) # 刷新频率

# 启动按钮
if not st.session_state.get('started', False):
    if st.button("🚀 启动量子监控链路", use_container_width=True):
        st.session_state.started = True
        asyncio.run(update_terminal())
else:
    # 自动重连逻辑
    asyncio.run(update_terminal())
