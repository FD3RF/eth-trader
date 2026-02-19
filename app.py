import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import asyncio
import time

# ==========================================
# 1. 核心架构：UI 预初始化
# ==========================================
st.set_page_config(layout="wide", page_title="QUANTUM PRO", page_icon="👁️")

# 注入极限暗黑主题 CSS
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: white; }
    [data-testid="stMetricValue"] { color: #00FFC2 !important; font-family: 'monospace'; font-size: 2rem !important; }
    .stMetric { background-color: #161B22; border-radius: 8px; padding: 15px; border: 1px solid #30363d; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
    div[data-testid="stExpander"] { border: none !important; }
    </style>
    """, unsafe_allow_html=True)

# 侧边栏：非阻塞交互区
with st.sidebar:
    st.markdown("### 🤖 核心控制器")
    is_live = st.toggle("实盘接入", value=True)
    st.divider()
    freq = st.select_slider("心跳频率 (秒)", options=[0.5, 1, 2, 5], value=1)
    st.warning("异步引擎已就绪：UI 实时响应中")

st.title("👁️ QUANTUM PRO: 实时上帝视角终端")

# 物理布局占位符
m1, m2, m3, m4 = st.columns(4)
metrics = [m1.empty() for _ in range(4)] # 指标卡占位符

col_l, col_r = st.columns([2, 1])
matrix_ph = col_l.empty() # 风险矩阵占位符
log_ph = col_r.empty()    # 日志流水占位符

# ==========================================
# 2. 异步数据泵 (Async Data Pump)
# ==========================================
async def terminal_engine():
    """采用异步非阻塞循环，完美兼顾实时性与交互性"""
    symbols = ["BTC", "ETH", "SOL", "BNB", "ARB"]
    
    while True:
        # A. 极速计算层
        sim_data = np.random.randn(30, len(symbols))
        df_corr = pd.DataFrame(sim_data, columns=symbols).corr()
        
        # B. 原子化指标更新 (直接注入占位符)
        metrics[0].metric("账户净值", f"${12450.40 + np.random.uniform(-10, 10):,.2f}")
        metrics[1].metric("风险敞口", f"{np.random.uniform(15, 25):.1f}%")
        metrics[2].metric("系统延迟", f"{np.random.randint(2, 8)}ms")
        metrics[3].metric("引擎状态", "RUNNING" if is_live else "PAUSED")

        # C. 零闪烁绘图层 (严格对齐)
        fig = px.imshow(
            df_corr, text_auto=".2f",
            color_continuous_scale='RdBu_r', range_color=[-1, 1],
            template="plotly_dark", aspect="auto"
        )
        fig.update_layout(
            margin=dict(l=0, r=0, t=20, b=0), 
            height=450,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        # 核心：使用 time.time() 确保每次生成的 key 唯一，强制 Plotly 刷新
        matrix_ph.plotly_chart(fig, key=f"qx_{time.time()}", use_container_width=True)

        # D. 日志流
        log_ph.dataframe(
            pd.DataFrame({
                "时间": [time.strftime("%H:%M:%S.%f")[:-4]],
                "动作": ["TICK_SYNC"],
                "状态": ["√"]
            }), use_container_width=True, hide_index=True
        )

        # E. 关键：异步挂起而非线程阻塞
        # 这允许侧边栏滑块和按钮在等待期间依然能被操作
        await asyncio.sleep(freq)

# ==========================================
# 3. 极限激活逻辑
# ==========================================
if st.button("🚀 激活全速量化监控链路", use_container_width=True):
    try:
        # 启动异步事件循环
        asyncio.run(terminal_engine())
    except Exception as e:
        st.error(f"终端异常: {e}")
