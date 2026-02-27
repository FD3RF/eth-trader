import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime

# ==================== 1. 配置与深度引擎 ====================
OKX_CONFIG = {
    "api_key": "a2a2a452-49e6-4e76-95f3-fb54eb982e7b",
    "secret_key": "330FABB2CAD3585677716686C2BF3872",
    "passphrase": "123321aA@", 
}

def fetch_okx(endpoint, params=""):
    url = f"https://www.okx.com/api/v5/{endpoint}?instId=ETH-USDT{params}"
    try:
        with requests.Session() as s:
            s.trust_env = True
            r = s.get(url, timeout=5)
            return r.json()
    except: return None

# ==================== 2. 核心计算：盘口平衡比 ====================
def analyze_order_book(depth_data):
    if not depth_data or 'data' not in depth_data:
        return 0.5, 0.5, 0, 0
    
    # 提取前 20 档数据
    asks = pd.DataFrame(depth_data['data'][0]['asks']).iloc[:, :2].astype(float)
    bids = pd.DataFrame(depth_data['data'][0]['bids']).iloc[:, :2].astype(float)
    
    # 计算总量 (买卖单深度)
    total_ask_vol = asks[1].sum()
    total_bid_vol = bids[1].sum()
    
    total = total_ask_vol + total_bid_vol
    ask_pct = (total_ask_vol / total) * 100
    bid_pct = (total_bid_vol / total) * 100
    
    return ask_pct, bid_pct, total_ask_vol, total_bid_vol

# ==================== 3. 界面渲染 ====================
st.set_page_config(page_title="ETH V43.0 盘口平衡站", layout="wide")

# 同步数据
depth_raw = fetch_okx("market/books", "&sz=20")
k_data = fetch_okx("market/candles", "&bar=5m&limit=100")

if depth_raw and k_data:
    ask_p, bid_p, ask_v, bid_v = analyze_order_book(depth_raw)
    curr_p = float(k_data['data'][0][4])
    
    st.title("🛡️ ETH V43.0 决策平衡指挥部")

    # --- 第一行：核心饼图与多空胜算 ---
    c1, c2, c3 = st.columns([1.5, 1, 1])
    
    with c1:
        # 创建多空对比饼图
        fig_pie = go.Figure(data=[go.Pie(
            labels=['卖压 (Asks)', '买压 (Bids)'],
            values=[ask_v, bid_v],
            hole=.4,
            marker_colors=['#ff4b4b', '#00ffcc'],
            textinfo='percent+label'
        )])
        fig_pie.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),
            height=300,
            showlegend=False,
            template="plotly_dark",
            title="20档实时买卖挂单占比"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        st.write("🚦 **胜算评估 (Win Rate)**")
        # 简单逻辑：买压 > 60% 且价格在 EMA 上方 = 胜算极高
        win_rate = 50 + (bid_p - 50) * 1.5
        win_rate = max(min(win_rate, 95), 5) # 限制在 5%-95%
        st.metric("多头胜算预测", f"{win_rate:.1f}%", f"{bid_p - ask_p:.1f}% 净买压")
        st.progress(win_rate / 100)
        st.caption("基于盘口深度与量能偏差计算")

    with c3:
        st.metric("ETH 实时价", f"${curr_p:.2f}")
        status = "🟢 护盘中" if bid_p > 55 else ("🔴 压盘中" if ask_p > 55 else "⚪ 均衡")
        st.subheader(f"盘口态势: {status}")

    # --- 第二行：主图渲染 (延续 V42 择时功能) ---
    st.divider()
    st.write("📈 **实时择时主图 (K线 + 深度拦截)**")
    raw_df = pd.DataFrame(k_data['data'], columns=['ts','o','h','l','c','v','volCcy','volCcyQ','confirm'])
    raw_df[['o','h','l','c','v']] = raw_df[['o','h','l','c','v']].astype(float)
    df = raw_df[::-1].reset_index(drop=True)
    
    fig_main = go.Figure(data=[go.Candlestick(x=df.index, open=df['o'], high=df['h'], low=df['l'], close=df['c'])])
    fig_main.update_layout(template="plotly_dark", height=450, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig_main, use_container_width=True)

    # --- 第三行：大单细节 ---
    st.subheader("🕵️ 庄家挂单墙 (Order Book Wall)")
    col_ask, col_bid = st.columns(2)
    with col_ask:
        st.write("🔴 上方压单总量:", f"{ask_v:.2f} ETH")
    with col_bid:
        st.write("🟢 下方托单总量:", f"{bid_v:.2f} ETH")

    st.caption(f"🚀 V43.0 | 深度 20 档实时计算 | 更新: {datetime.now().strftime('%H:%M:%S')}")
else:
    st.warning("🔄 盘口数据对接中...")
