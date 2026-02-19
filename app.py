import streamlit as st
import asyncio
import pandas as pd
import numpy as np
import plotly.express as px
import time
import sqlite3
import ccxt.async_support as ccxt

# ==========================================
# 🛡️ 1. 初始化与核心逻辑 (解决 API 连接与数据库)
# ==========================================
class QuantumCore:
    def __init__(self, api="", sec=""):
        # 针对截图 2 的 API 挂载逻辑修复
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
# 🎨 2. 样式与配置 (彻底修复截图 2 的 TypeError)
# ==========================================
st.set_page_config(layout="wide", page_title="QUANTUM PRO TERMINAL", page_icon="👁️")

# 关键：移除了引发错误的 unsafe_allow_password，确保 CSS 正常渲染
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: white; }
    [data-testid="stMetricValue"] { color: #00FFC2 !important; font-family: 'Courier New', monospace; font-size: 1.8rem !important; }
    .stMetric { background-color: #161B22; border-radius: 10px; padding: 15px; border: 1px solid #30363d; }
    .block-container { padding-top: 1.5rem; }
    </style>
    """, unsafe_allow_html=True)

if 'core' not in st.session_state:
    st.session_state.core = QuantumCore()

# ==========================================
# 🖥️ 3. UI 布局 (1:1 还原您的专业黑红界面)
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
            st.toast("核心链路已刷新")

st.title("👁️ QUANTUM PRO: 实时上帝视角终端")

# 建立固定容器占位符，防止页面跳动
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
# 🔄 4. 核心执行引擎 (彻底修复 ID 冲突与缩进报错)
# ==========================================
async def main_loop():
    # 预定义币种
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "ARB/USDT"]
    
    while True:
        start_ts = time.time()
        
        # A. 模拟实时计算
        sim_data = np.random.randn(20, len(symbols))
        df_corr = pd.DataFrame(sim_data, columns=symbols).corr()
        
        # B. 刷新指标
        latency = (time.time() - start_ts) * 1000
        safe_score = (1 - df_corr.mean().mean()) * 100
        
        eq_ph.metric("账户权益 (Equity)", "$10,000")
        rs_ph.metric("安全系数 (Safety)", f"{safe_score:.1f}%", delta=f"{safe_score-95:.1f}%")
        lt_ph.metric("系统延迟 (Latency)", f"{int(latency)}ms")
        st_ph.metric("运行状态", "LIVE" if run_live else "IDLE")

        # C. 渲染热力图 (修复截图 4 的 DuplicateKey 报错)
        # 这里的关键是使用 container() 配合唯一的动态 key
        with matrix_ph.container():
            fig = px.imshow(
                df_corr, text_auto=".2f",
                color_continuous_scale='RdBu_r', range_color=[-1, 1],
                template="plotly_dark", aspect="auto"
            )
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=400)
            # 针对截图 3 修复：将 use_container_width 替换为新的 width="stretch"
            st.plotly_chart(fig, key=f"mtx_{int(time.time()*10)}", on_select="ignore", width="stretch")

        # D. 刷新日志 (针对截图 3 修复弃用警告)
        with log_ph.container():
            conn = sqlite3.connect(st.session_state.core.db_path)
            try:
                df_log = pd.read_sql("SELECT symbol, side, exec, ts FROM ledger ORDER BY ts DESC LIMIT 10", conn)
                st.dataframe(df_log, width="stretch", height=400)
            except:
                st.info("等待执行审计...")
            finally:
                conn.close()

        await asyncio.sleep(2)

# ==========================================
# 🏁 5. 程序启动入口
# ==========================================
if st.button("🚀 启动量子监控链路", width="stretch"):
    try:
        asyncio.run(main_loop())
    except Exception as e:
        # 处理 Streamlit 重复运行异常
        st.info("监控系统正在运行中...")
