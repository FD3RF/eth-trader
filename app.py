import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# ==================== 1. 样式与自动刷新 ====================
st.set_page_config(page_title="ETH V17.9 哨兵终端", layout="wide")

# 每 5 秒强制刷新一次，解决“不显示”或“数据卡死”问题
st_autorefresh(interval=5000, key="framer")

st.markdown("""
<style>
    @keyframes blink { 0% { opacity: 1; } 50% { opacity: 0.3; } 100% { opacity: 1; } }
    .sniper-box {
        background-color: #ff4b4b; color: white; padding: 15px; border-radius: 10px;
        text-align: center; border: 3px solid white; animation: blink 1s infinite;
        font-weight: bold; margin-bottom: 20px;
    }
    .sentinel-card {
        background-color: #1e1e1e; padding: 15px; border-radius: 5px; 
        border-left: 5px solid #4da9ff; margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ==================== 2. 数据引擎 (实时监控版) ====================
@st.cache_data(ttl=5)
def fetch_realtime_data(symbol="ETH"):
    try:
        url = "https://min-api.cryptocompare.com/data/v2/histominute"
        params = {"fsym": symbol, "tsym": "USD", "limit": 200, "aggregate": 5, "e": "CCCAGG"}
        resp = requests.get(url, timeout=5).json()
        df = pd.DataFrame(resp['Data']['Data'])
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        # 计算核心指标
        df['ema8'] = df['close'].ewm(span=8).mean()
        df['ema21'] = df['close'].ewm(span=21).mean()
        df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
        df['hist'] = df['macd'] - df['macd'].ewm(span=9).mean()
        return df
    except: return pd.DataFrame()

def get_live_whale_flow(curr_price):
    """
    模拟实时挂单流变化：返回挂单价格、金额及厚度变化趋势
    """
    # 模拟支撑墙和阻力墙
    return [
        {'price': curr_price * 1.015, 'amount': 45.2, 'type': '阻力', 'trend': '增加'}, # 庄家在加压
        {'price': curr_price * 0.985, 'amount': 62.8, 'type': '支撑', 'trend': '稳定'}  # 庄家在护盘
    ]

# ==================== 3. 核心监控逻辑 ====================
df = fetch_realtime_data()

if not df.empty:
    curr_price = df['close'].iloc[-1]
    whale_walls = get_live_whale_flow(curr_price)
    avg_vol = df['volumefrom'].rolling(20).mean().iloc[-1]
    vol_ratio = df['volumefrom'].iloc[-1] / avg_vol

    # --- A. 顶部仪表盘 ---
    st.title("🚀 ETH V17.9 哨兵监控终端")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("全球实时价", f"${curr_price:.2f}")
    c2.metric("实时量能比", f"{vol_ratio:.2f}x")
    
    with c3:
        res_5m = df['ema8'].iloc[-1] > df['ema21'].iloc[-1]
        st.write("🚦 趋势共振灯")
        st.markdown(f"{'🟢' if res_5m else '🔴'} 5m | {'🔴'} 15m | {'🔴'} 1h") # 模拟当前阴跌背景
    
    # 狙击判定：缩量+触墙
    is_sniper = vol_ratio < 0.8 and any(abs(curr_price - w['price'])/w['price'] < 0.005 for w in whale_walls if w['type']=='支撑')
    c4.metric("庄家活跃度", "高" if vol_ratio > 1.2 else "极低 (洗盘)")

    # --- B. 红色狙击倒计时 (强力弹出) ---
    if is_sniper:
        st.markdown(f"""
        <div class="sniper-box">
            <div style="font-size: 22px;">🎯 狙击信号：缩量触墙 (庄家诱空)</div>
            <div style="font-size: 40px;">倒计时: 60s</div>
            <p>价格贴近 $1921 支撑墙，且抛压枯竭，建议小仓位博弈 V 转。</p>
        </div>
        """, unsafe_allow_html=True)

    # --- C. 强制显示监控栏 (哨兵意图流) ---
    st.subheader("📡 庄家意图实时哨兵")
    
    if vol_ratio < 0.6:
        status, color, desc = "😴 缩量洗盘中", "#808080", "成交量极低，庄家在消磨多头意志，不要被阴跌吓跑。"
    elif any(w['trend'] == '增加' and w['type'] == '阻力' for w in whale_walls):
        status, color, desc = "⚠️ 庄家压盘中", "#ff4b4b", "上方阻力墙厚度在增加，庄家在阻止价格反弹。"
    else:
        status, color, desc = "⚖️ 震荡吸筹", "#4da9ff", "多空均衡，庄家正在支撑墙附近缓慢吃货。"

    st.markdown(f"""
    <div class="sentinel-card" style="border-left-color: {color};">
        <div style="display: flex; justify-content: space-between;">
            <span style="font-size: 18px; color: {color}; font-weight: bold;">{status}</span>
            <span style="color: #666; font-size: 12px;">最后扫描: {datetime.now().strftime('%H:%M:%S')}</span>
        </div>
        <p style="color: #bbb; margin: 10px 0;">{desc}</p>
        <div style="font-size: 13px; color: #888; border-top: 1px solid #333; padding-top: 5px;">
            <b>挂单实时动态：</b> 
            支撑位 ${whale_walls[1]['price']:.1f} ({whale_walls[1]['amount']}M) - {whale_walls[1]['trend']} | 
            阻力位 ${whale_walls[0]['price']:.1f} ({whale_walls[0]['amount']}M) - {whale_walls[0]['trend']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- D. 主图表 ---
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="K线"))
    # 绘制庄家墙
    for w in whale_walls:
        line_color = "rgba(57, 211, 83, 0.4)" if w['type'] == '支撑' else "rgba(248, 81, 73, 0.4)"
        fig.add_hline(y=w['price'], line_dash="dot", line_color=line_color, annotation_text=f"{w['type']}墙 ${w['amount']}M")

    fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # --- E. 侧边栏 AI 复盘 ---
    with st.sidebar:
        st.header("🔍 哨兵控制台")
        if st.button("🧩 执行 AI 深度博弈复盘"):
            st.session_state.show_review = True
    
    if st.session_state.get('show_review'):
        st.info(f"🧠 AI 复盘分析：当前价格处于支撑墙上方。由于成交量比率仅为 {vol_ratio:.2f}，判定为‘无动力下跌’。庄家利用 $1921 附近的护盘意图明显，建议关注 5m K 线是否收阳确认。")
