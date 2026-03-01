import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import numpy as np
import plotly.io as pio
import threading
import traceback
import random
from streamlit_autorefresh import st_autorefresh
import feedparser

# ====================== 页面配置 & 全局极致紧凑CSS ======================
st.set_page_config(layout="wide", page_title="专业量化决策引擎·至尊版", page_icon="📊")

st.markdown("""
<style>
    /* 全局极致压缩 */
    .main > div { padding: 0 4px !important; }
    .block-container { max-width: 99% !important; padding-top: 4px !important; padding-bottom: 4px !important; }
    section[data-testid="stSidebar"] > div { padding: 8px 4px !important; }
    div[data-testid="column"] { padding: 0 3px !important; gap: 4px !important; }

    /* 小卡精致基类 */
    .compact-card {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
    }
    .compact-card > div {
        width: 100px;
        height: 55px;
        padding: 6px;
        border-radius: 8px;
        background: rgba(255,255,255,0.04);
        border: 0.8px solid rgba(255,255,255,0.12);
        text-align: center;
        transition: all 0.15s;
    }
    .compact-card > div:hover { border-color: rgba(0,204,119,0.4); }

    .label { font-size: 0.65rem; color: #aaa; margin-bottom: 1px; }
    .value { font-size: 1.95rem; font-weight: 900; line-height: 1.05; }
    .status { font-size: 0.68rem; margin-top: 1px; }

    .green { color: #00cc77; }
    .red { color: #ff6b6b; }
    .yellow { color: #ffcc00; }

    /* 响应：手机折列 */
    @media (max-width: 768px) {
        .compact-card > div { width: calc(50% - 4px); height: 50px; font-size: 0.9em; }
    }
    @media (max-width: 600px) {
        .compact-card > div { width: calc(33.3% - 4px); }
    }

    /* 子图标签极淡 */
    .subplot-label { font-size: 0.62rem !important; fill: #888 !important; opacity: 0.75; }

    /* 新闻极小 */
    .news-expander { font-size: 0.85rem !important; }
    .news-item { font-size: 0.78rem; line-height: 1.3; margin: 4px 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
</style>
""", unsafe_allow_html=True)

pio.templates['custom_dark'] = pio.templates['plotly_dark']

# ====================== 状态 & 工具函数（保持原样，略） ======================
# ...（这里省略你的 init_state / light_cleanup / send_telegram / safe_request / get_ls_ratio 等函数，保持不变）

# ====================== get_candles 函数（已包含新指标） ======================
# ...（保持你最新的版本，包含 Ichimoku / Stochastic / ADX 计算）

# ====================== generate_signal（保持你最新逻辑） ======================
# ...（保持不变，返回 prob, direction, entry_zone, sl, tp, reason, details 等）

# ====================== 主界面 main() - 完整完美布局 ======================
def main():
    # 数据获取部分（保持不变）
    ls_ratio = get_ls_ratio()
    ls_history = get_ls_history(24)
    if not ls_history.empty:
        st.session_state.ls_history = ls_history

    df = get_candles(bar="15m", limit=200)  # 示例用15m，你可动态
    if df is None:
        st.error("数据加载失败")
        return

    current_price = df['c'].iloc[-1]
    atr = df['atr'].iloc[-1]
    # 假设你已有 prob, direction 等计算
    prob = 50.0  # 示例，替换成你的 generate_signal 调用
    direction = 0
    entry_zone = "观望"
    sl = tp = None
    reason = "示例理由"

    # ====================== 层1: 顶部标题 - 极薄 ======================
    st.markdown("""
    <div style="height:32px; padding:4px 0; text-align:center; border-bottom:0.4px solid #00cc77;">
        <h1 style="font-size:2.0rem; margin:0; color:#00cc77;">专业量化决策引擎·至尊版</h1>
        <p style="font-size:0.75rem; color:#888; margin:0;">ETH-USDT | 15m | {}</p>
    </div>
    """.format(datetime.now().strftime('%Y-%m-%d %H:%M')), unsafe_allow_html=True)

    # ====================== 层2: 核心指标 - 横排紧凑 ======================
    with st.container():
        st.markdown("""
        <div class="compact-card">
            <div><div class="label">💲 价格</div><div class="value green">${:.2f}</div></div>
            <div><div class="label">📈 胜率</div><div class="value green">{:.1f}%</div></div>
            <div><div class="label">⚖️ 多空比</div><div class="value yellow">{:.2f}</div></div>
            <div><div class="label">💰 净流</div><div class="value green">0 ETH</div></div>
            <div><div class="label">🌊 ATR</div><div class="value yellow">{:.2f}</div></div>
        </div>
        """.format(current_price, prob, ls_ratio, atr), unsafe_allow_html=True)

    # ====================== 层3: 超级英雄信号 - 压缩 ======================
    dir_label = "做多" if direction == 1 else "做空" if direction == -1 else "观望"
    dir_color = "#00cc77" if direction == 1 else "#ff6b6b" if direction == -1 else "#ffcc00"
    st.markdown(f"""
    <div style="height:90px; margin:4px 0; padding:6px; border:0.7px solid {dir_color}; border-radius:10px; background:rgba(0,0,0,0.025); display:flex; align-items:center;">
        <div style="flex:0.6;">
            <h2 style="font-size:1.7rem; color:{dir_color}; margin:0;">{dir_label}</h2>
            <p style="font-size:0.85rem; color:#bbb; margin:2px 0;">{reason[:22]}...</p>
        </div>
        <div style="flex:0.4; text-align:center;">
            <svg viewBox="0 0 36 36" style="width:78px;height:78px;">
                <path d="M18 2.0845 a15.9155 15.9155 0 0 1 0 31.831" fill="none" stroke="#333" stroke-width="1.8"/>
                <path d="M18 2.0845 a15.9155 15.9155 0 0 1 0 31.831" fill="none" stroke="{dir_color}" stroke-width="1.8" stroke-dasharray="{prob*3.14:.0f},100"/>
                <text x="18" y="21" text-anchor="middle" fill="{dir_color}" font-size="5">{int(prob)}%</text>
            </svg>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ====================== 层4: 进场4卡 - 极扁 ======================
    st.markdown("""
    <div style="display:flex; gap:5px; margin:4px 0;">
        <div style="width:102px;height:40px;padding:4px;border:0.8px solid #00cc77;border-radius:8px;text-align:center;">
            <p style="font-size:0.62rem;color:#bbb;margin:0;">入场区</p>
            <h5 style="font-size:1.45rem;color:#00cc77;margin:0;">1980-1990</h5>
        </div>
        <!-- 其余3卡类似，替换对应值和颜色 -->
    </div>
    """, unsafe_allow_html=True)

    # ====================== 层5: 仓位 + 雷达 - 45°标签 ======================
    col_left, col_right = st.columns([26, 74])
    with col_left:
        st.markdown("""
        <div style="height:110px;padding:6px;border-radius:10px;background:rgba(0,0,0,0.03);">
            <h3 style="font-size:1.4rem;color:#00cc77;margin:0;">实时仓位</h3>
            <p style="font-size:0.68rem;color:#bbb;margin:2px 0;">1%风险 10000 USDT</p>
            <h1 style="font-size:2.15rem;color:#00cc77;margin:0;">10.40 ETH</h1>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        # 雷达图 - 45°标签
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=[8, 6, 5, 4, 7, 9, 3, 10, 6, 8, 4, 5],  # 示例贡献值
            theta=["EMA", "RSI", "MACD", "布林", "量", "流", "多空", "一目", "Stoch", "ADX", "Fib", "MVRV"],
            fill='toself',
            line_color='#00cc77'
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(range=[0,11], showticklabels=False),
                angularaxis=dict(rotation=45, direction='clockwise')  # 45°旋转起点
            ),
            showlegend=False,
            height=150,
            margin=dict(l=10,r=10,t=10,b=10),
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # ====================== 层6: 情绪温度计 - 超扁 ======================
    st.markdown("""
    <div style="height:45px;margin:4px 0;padding:4px;border-radius:8px;background:rgba(0,0,0,0.02);">
        <div style="display:flex;align-items:center;gap:8px;">
            <span style="font-size:0.95rem;color:#ffcc00;">中性</span>
            <span style="font-size:0.8rem;color:#aaa;">多空比 1.00</span>
        </div>
        <div style="height:8px;background:#444;border-radius:3px;margin:2px 0;">
            <div style="height:8px;width:50%;background:#ffcc00;border-radius:3px;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ====================== 层7: 技术4小卡 ======================
    st.markdown("""
    <div class="compact-card" style="margin:4px 0;">
        <div><div class="label">RSI</div><div class="value green">33.0</div><div class="status">中🟡</div></div>
        <div><div class="label">MACD</div><div class="value red">-1.77</div><div class="status">负🔴</div></div>
        <div><div class="label">布林</div><div class="value yellow">中轨</div><div class="status">区间🟡</div></div>
        <div><div class="label">成交</div><div class="value gray">缩量</div><div class="status">↓🟡</div></div>
    </div>
    """, unsafe_allow_html=True)

    # ====================== 层8: 信号历史 - 矮table ======================
    st.markdown("<div style='height:120px;overflow:hidden;margin:4px 0;'>", unsafe_allow_html=True)
    # 示例数据
    hist_data = pd.DataFrame({
        "时间": ["11:03", "11:02", "11:01"],
        "方向": ["空头", "空头", "空头"],
        "胜率": ["5.0%", "25.0%", "31.0%"],
        "价格": ["$1985", "$1986", "$1986"]
    })
    st.table(hist_data.style.set_table_styles([{'selector': 'th', 'props': [('font-size', '0.62rem'), ('color', '#888')]}]))
    st.markdown("</div>", unsafe_allow_html=True)

    # ====================== 层9: K线图 - 大霸屏 ======================
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.004,
                        row_heights=[0.78, 0.075, 0.075, 0.07])
    # 添加你的K线、EMA、云层等（保持你原逻辑）
    fig.update_layout(height=520, margin=dict(l=10,r=10,t=10,b=10), hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

    # ====================== 层10: 新闻 - 彻底最小化 ======================
    with st.expander("📰 最新加密新闻", expanded=False):
        st.markdown("""
        <div style="max-height:130px;overflow-y:auto;">
            <div class="news-item">- Ether, Solana, XRP surge up to 10%... (Sun, Mar 01 2026)</div>
            <div class="news-item">- Polymarket attracts record volumes... (Sun, Mar 01 2026)</div>
            <div class="news-item">- Bitcoin tops $68k after... (Sun, Mar 01 2026)</div>
            <!-- 更多条目 -->
        </div>
        """, unsafe_allow_html=True)

    # 自动刷新（保持你原逻辑）
    st_autorefresh(interval=15000)

if __name__ == "__main__":
    main()
