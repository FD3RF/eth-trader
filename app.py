import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time

# 
# 1. 基础 UI 环境预设 (物理层级 0)
st.set_page_config(layout="wide", page_title="QUANTUM PRO", page_icon="👁️")

# 强制注入暗黑量化主题 CSS
st.markdown("""
<style>
.stApp { background-color: #0E1117; color: white; }
[data-testid="stMetricValue"] { color: #00FFC2 !important; font-family: 'monospace'; font-size: 1.8rem !important; }
.stMetric { background-color: #161B22; border-radius: 8px; padding: 12px; border: 1px solid #30363d; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🤖 核心控制器")
    is_live = st.toggle("实盘链路接入", value=True)
    st.divider()
    speed = st.select_slider("心跳频率 (秒)", options=[0.5, 1, 2, 5], value=1)
    st.info("架构状态：原生同步驱动，0 外部依赖")

st.title("👁️ QUANTUM PRO: 实时上帝视角终端")

# 预设静态占位符 (防止页面跳动)
m1, m2, m3, m4 = st.columns(4)
c1_ph, c2_ph, c3_ph, c4_ph = m1.empty(), m2.empty(), m3.empty(), m4.empty()

col_left, col_right = st.columns([2, 1])
matrix_ph = col_left.empty()
log_ph = col_right.empty()

# 2. 激活主循环 (物理层级 1 - 仅缩进 4 个空格)
if st.button("🚀 激活全速量化监控链路", use_container_width=True):
    while True:
        # A. 极速数据生成逻辑
        syms = ["BTC", "ETH", "SOL", "BNB", "ARB"]
        data = pd.DataFrame(np.random.randn(25, 5), columns=syms).corr()
        
        # B. 更新指标卡 (原子化操作)
        c1_ph.metric("账户净值", f"${12450.40 + np.random.uniform(-5, 5):,.2f}")
        c2_ph.metric("风险敞口", f"{np.random.uniform(18.5, 19.5):.1f}%")
        c3_ph.metric("系统延迟", f"{np.random.randint(4, 9)}ms")
        c4_ph.metric("状态", "RUNNING" if is_live else "IDLE")

        # C. 渲染 Plotly (严格对齐 - 缩进 8 个空格)
        fig = px.imshow(
            data, text_auto=".2f",
            color_continuous_scale='RdBu_r', range_color=[-1, 1],
            template="plotly_dark", aspect="auto"
        )
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=450)
        
        # D. 注入占位符 (使用 time.time 强制重绘)
        matrix_ph.plotly_chart(fig, key=f"m_{time.time()}", use_container_width=True)
        
        log_ph.dataframe(
            pd.DataFrame({
                "时间": [time.strftime("%H:%M:%S")],
                "动作": ["SYNC_OK"],
                "载荷": [f"{np.random.randint(100, 999)}kb"]
            }), use_container_width=True, hide_index=True
        )

        # E. 维持 UI 响应
        time.sleep(speed)
