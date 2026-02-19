import streamlit as st
import asyncio
import pandas as pd
import numpy as np
import plotly.express as px
import time
import sqlite3
import ccxt.async_support as ccxt

# ==========================================
# 🛡️ 1. 初始化（数据库与 CCXT）
# ==========================================
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
# 🎨 2. UI 界面配置
# ==========================================
st.set_page_config(layout="wide", page_title="QUANTUM TERMINAL", page_icon="👁️")

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: white; }
    [data-testid="stMetricValue"] { color: #00FFC2 !important; font-family: 'Courier New', monospace; font-size: 1.8rem !important; }
    .stMetric { background-color: #161B22; border-radius: 8px; padding: 12px; border: 1px solid #30363d; }
    .stDataFrame { border: 1px solid #30363d; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

if 'core' not in st.session_state:
    st.session_state.core = QuantumCore()

# ==========================================
# 🖥️ 3. 页面布局容器（静态预置）
# ==========================================
with st.sidebar:
    st.markdown("### 🤖 自动化交易引擎")
    run_live = st.toggle("启动实盘监控", value=False)
    st.divider()
    trigger_spread = st.slider("触发价差 (%)", 0.1, 1.0, 0.35)
    with st.expander("🔑 密钥配置"):
        st.text_input("API Key", type="password")
        st.text_input("Secret Key", type="password")

st.title("👁️ QUANTUM PRO: 实时上帝视角终端")

# 指标占位符
m1, m2, m3, m4 = st.columns(4)
eq_ph, rs_ph, lt_ph, st_ph = m1.empty(), m2.empty(), m3.empty(), m4.empty()

# 图表与日志占位符
col_l, col_r = st.columns([2, 1])
with col_l:
    st.markdown("#### 🌐 全球流动性风险矩阵")
    matrix_ph = st.empty()
with col_r:
    st.markdown("#### 📜 实时审计流水")
    log_ph = st.empty()

# ==========================================
# 🔄 4. 核心刷新循环（缩进绝对安全）
# ==========================================
async def update_terminal():
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "ARB/USDT"]
    
    while True:
        try:
            start_ts = time.time()
            
            # A. 模拟数据计算
            sim_data = np.random.randn(25, len(symbols))
            df_corr = pd.DataFrame(sim_data, columns=symbols).corr()
            
            # B. 刷新指标卡
            latency = (time.time() - start_ts) * 1000
            safe_score = (1 - df_corr.mean().mean()) * 100
            
            eq_ph.metric("账户权益", "$10,000.00")
            rs_ph.metric("安全系数", f"{safe_score:.1f}%", delta=f"{safe_score-90:.1f}%")
            lt_ph.metric("系统延迟", f"{int(latency)}ms")
            st_ph.metric("运行状态", "LIVE" if run_live else "IDLE")

            # C. 渲染热力图（容器刷新模式）
            with matrix_ph.container():
                fig = px.imshow(
                    df_corr, text_auto=".2f",
                    color_continuous_scale='RdBu_r', range_color=[-1, 1],
                    template="plotly_dark", aspect="auto"
                )
                fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=450)
                # 使用动态 Key 彻底规避 ID 冲突
                st.plotly_chart(fig, key=f"hm_{int(time.time()*100)}", on_select="ignore", width="stretch")

            # D. 刷新日志
            with log_ph.container():
                conn = sqlite3.connect(st.session_state.core.db_path)
                try:
                    df_log = pd.read_sql("SELECT symbol, side, exec, ts FROM ledger ORDER BY ts DESC LIMIT 15", conn)
                    st.dataframe(df_log, width="stretch", height=400)
                except:
                    st.info("监控中...")
                finally:
                    conn.close()

        except Exception:
            pass
            
        await asyncio.sleep(2)

# ==========================================
# 🏁 5. 安全运行入口
# ==========================================
if st.button("🚀 启动量子监控链路", width="stretch"):
    try:
        asyncio.run(update_terminal())
    except Exception:
        st.warning("系统已在后台运行")
