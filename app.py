import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time

# # ==========================================
# 1. 极限 UI 预初始化 (顶层无缩进)
# ==========================================
st.set_page_config(layout="wide", page_title="QUANTUM PRO", page_icon="👁️")

# 注入极致暗黑量化主题
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
    refresh_speed = st.select_slider("心跳频率 (秒)", options=[0.5, 1, 2, 5], value=1)
    st.info("架构状态：原生组件驱动，0 外部依赖")

st.title("👁️ QUANTUM PRO: 实时上帝视角终端")

# 预设静态容器占位符
m1, m2, m3, m4 = st.columns(4)
metrics_placeholders = [m.empty() for m in [m1, m2, m3, m4]]

col_left, col_right = st.columns([2, 1])
matrix_ph = col_left.empty()
log_ph = col_right.empty()

# ==========================================
# 2. 极限渲染引擎 (顶层 if 结构)
# ==========================================
if st.button("🚀 激活全速量化监控链路", use_container_width=True):
    # 使用 while 循环配合 empty 占位符实现非重载刷新
    while True:
        # A. 极速数据生成
        symbols = ["BTC", "ETH", "SOL", "BNB", "ARB"]
        df_corr = pd.DataFrame(np.random.randn(25, 5), columns=symbols).corr()
        
        # B. 原子化更新指标 (绝对对齐)
        metrics_placeholders[0].metric("账户净值", f"${12450.40 + np.random.uniform(-5, 5):,.2f}")
        metrics_placeholders[1].metric("风险敞口", f"{np.random.uniform(18.5, 19.5):.1f}%")
        metrics_placeholders[2].metric("系统延迟", f"{np.random.randint(4, 9)}ms")
        metrics_placeholders[3].metric("引擎状态", "RUNNING" if is_live else "IDLE")

        # C. 渲染风险矩阵 (注意：此段代码严格缩进 4 个空格)
        fig = px.imshow(
            df_corr, text_auto=".2f",
            color_continuous_scale='RdBu_r', range_color=[-1, 1],
            template="plotly_dark", aspect="auto"
        )
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=450)
        
        # 使用动态 Key 强制局部重绘，避免 Duplicate ID 错误
        matrix_ph.plotly_chart(fig, key=f"mtx_{time.time()}", use_container_width=True)

        # D. 日志流刷新
        log_ph.dataframe(
            pd.DataFrame({
                "时间": [time.strftime("%H:%M:%S")],
                "动作": ["SYNC_OK"],
                "载荷": [f"{np.random.randint(100, 999)}kb"]
            }), use_container_width=True, hide_index=True
        )

        # E. 释放线程，维持 UI 响应
        time.sleep(refresh_speed)
