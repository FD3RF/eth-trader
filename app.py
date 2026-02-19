import streamlit as st
import asyncio
import pandas as pd
import numpy as np
import plotly.express as px
import time
import sqlite3
from datetime import datetime
import ccxt.async_support as ccxt
from decimal import Decimal, ROUND_DOWN

# ==========================================
# 🛡️ V14 系统配置
# ==========================================
CONFIG = {
    "symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "ARB/USDT"],
    "timeframe": "1h",
    "leverage": 3,
    "risk_per_trade": 0.01,
    "live": False  # 初始保持 False，UI 切换启动
}

# ==========================================
# 📦 工业级后端核心
# ==========================================

class V14Core:
    def __init__(self, api="", sec=""):
        self.ex = ccxt.binance({
            "apiKey": api, "secret": sec,
            "options": {"defaultType": "future", "adjustForTimeDifference": True},
            "enableRateLimit": True
        })
        self.equity = 10000.0
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
                entry REAL, exec REAL, slip REAL, 
                var REAL, latency REAL
            )
        """)
        conn.close()

    async def fetch_all_data(self):
        tasks = [self.ex.fetch_ohlcv(s, CONFIG['timeframe'], limit=50) for s in CONFIG['symbols']]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return dict(zip(CONFIG['symbols'], results))

    def compute_matrix(self, market_results):
        returns_data = {}
        for s, data in market_results.items():
            if isinstance(data, list):
                df = pd.DataFrame(data, columns=['t','o','h','l','c','v'])
                returns_data[s] = df['c'].pct_change().dropna()
        return pd.DataFrame(returns_data)

# ==========================================
# 🎨 UI & 实时上帝视角
# ==========================================

st.set_page_config(layout="wide", page_title="QUANTUM PRO", page_icon="👁️")

# 自定义 CSS 适配暗色主题
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: white; }
    .stMetric { background-color: #161B22; border-radius: 10px; padding: 15px; border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_password=True)

# 初始化 Session State
if 'core' not in st.session_state:
    st.session_state.core = V14Core()
    st.session_state.initialized = False

core = st.session_state.core

# 侧边栏：同步你截图的 UI 控件
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2091/2091665.png", width=50)
    st.title("自动化交易计划")
    run_live = st.toggle("启动实盘执行计划", value=False)
    trigger_spread = st.slider("触发价差 (%)", 0.1, 1.0, 0.3)
    safe_factor = st.slider("最小安全系数 (%)", 50.0, 100.0, 95.0)
    
    st.divider()
    api_key = st.text_input("API Key", type="password")
    api_sec = st.text_input("Secret Key", type="password")
    if st.button("更新密钥"):
        st.session_state.core = V14Core(api_key, api_sec)
        st.success("密钥已更新")

# 主界面布局
st.title("👁️ QUANTUM PRO: 实时上帝视角终端")

m1, m2, m3, m4 = st.columns(4)
equity_metric = m1.empty()
risk_metric = m2.empty()
latency_metric = m3.empty()
status_metric = m4.empty()

# 中央黑色显示区域
chart_col, log_col = st.columns([2, 1])

with chart_col:
    st.subheader("🌐 全球流动性风险矩阵")
    matrix_container = st.empty()

with log_col:
    st.subheader("📜 实时审计流水")
    log_container = st.empty()

# ==========================================
# 🔄 实时高频循环
# ==========================================

async def main_loop():
    await core.ex.load_markets()
    
    while True:
        start_time = time.time()
        
        # 1. 获取数据 (异步扇出)
        market_results = await core.fetch_all_data()
        returns_matrix = core.compute_matrix(market_results)
        
        # 2. 渲染相关性热力图
        if not returns_matrix.empty:
            corr = returns_matrix.corr()
            fig = px.imshow(
                corr, text_auto=".2f", aspect="auto",
                color_continuous_scale='RdBu_r', range_color=[-1, 1],
                template="plotly_dark"
            )
            fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=400, paper_bgcolor='rgba(0,0,0,0)')
            matrix_container.plotly_chart(fig, use_container_width=True)
            
            # 计算风险
            avg_corr = corr.mean().mean()
            current_safe_score = (1 - avg_corr) * 100
        else:
            current_safe_score = 100.0
            avg_corr = 0.0

        # 3. 更新指标卡
        latency = (time.time() - start_time) * 1000
        equity_metric.metric("账户权益", f"${core.equity:,.2f}")
        risk_metric.metric("安全系数", f"{current_safe_score:.1f}%", 
                           delta=f"{current_safe_score - safe_factor:.1f}%",
                           delta_color="normal" if current_safe_score >= safe_factor else "inverse")
        latency_metric.metric("核心延迟", f"{latency:.0f}ms")
        status_metric.metric("系统状态", "LIVE" if run_live else "IDLE")

        # 4. 执行逻辑判断 (如果安全系数达标)
        if run_live and current_safe_score < safe_factor:
            st.toast(f"风险过载: 安全系数 {current_safe_score:.1f}% 低于设定值", icon="⚠️")

        # 5. 读取审计日志
        conn = sqlite3.connect(core.db_path)
        df_log = pd.read_sql("SELECT symbol, side, exec, slip, ts FROM ledger ORDER BY ts DESC LIMIT 8", conn)
        conn.close()
        log_container.dataframe(df_log, use_container_width=True, height=350)

        await asyncio.sleep(2) # 刷新频率

# 启动引擎
if st.button("🚀 链接上帝视角", use_container_width=True):
    asyncio.run(main_loop())
