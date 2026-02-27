import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime
from collections import deque
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import argrelextrema

# ==================== 1. 配置中心 ====================
OKX_KEY = "a2a2a452-49e6-4e76-95f3-fb54eb982e7b"
OKX_SECRET = "330FABB2CAD3585677716686C2BF3872"
OKX_PASS = "你的API口令"  # <--- 这里填入你创建API时的口令

if 'history' not in st.session_state:
    st.session_state.history = deque(maxlen=100)

st.set_page_config(page_title="ETH V27.0 OKX旗舰终端", layout="wide")

# UI 样式
st.markdown("""
<style>
    .stApp { background: #0e1117; }
    .metric-card { background: #161b22; padding: 15px; border-radius: 10px; border-left: 5px solid #ff4b4b; }
    @keyframes blink { 0% { opacity: 1; } 50% { opacity: 0.2; } 100% { opacity: 1; } }
    .alert-box { background: linear-gradient(45deg, #4b0000, #ff4b4b); padding: 20px; border-radius: 15px; color: white; text-align: center; animation: blink 1s infinite; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ==================== 2. 数据引擎 (LetsVPN 穿透) ====================
def get_okx_data(bar='5m', limit=300):
    url = f"https://www.okx.com/api/v5/market/candles?instId=ETH-USDT&bar={bar}&limit={limit}"
    session = requests.Session()
    session.trust_env = True # 自动通过 LetsVPN 穿透
    try:
        r = session.get(url, timeout=5)
        if r.json()['code'] == '0':
            raw = r.json()['data']
            df = pd.DataFrame(raw, columns=['ts','o','h','l','c','v','vCcy','vCcyQ','confirm'])
            df['time'] = pd.to_datetime(df['ts'].astype(float), unit='ms')
            df.set_index('time', inplace=True)
            for col in ['o','h','l','c','v']: df[col] = df[col].astype(float)
            return df.sort_index()
    except: return pd.DataFrame()

# ==================== 3. 核心算法 (背离+缩量) ====================
def analyze_logic(df):
    df['ema8'] = df['c'].ewm(span=8).mean()
    df['ema21'] = df['c'].ewm(span=21).mean()
    df['v_avg'] = df['v'].rolling(20).mean()
    # MACD
    df['macd'] = df['c'].ewm(span=12).mean() - df['c'].ewm(span=26).mean()
    df['sig'] = df['macd'].ewm(span=9).mean()
    df['hist'] = df['macd'] - df['sig']
    return df

# ==================== 4. 主界面渲染 ====================
st.sidebar.title("🛠️ 控制中心")
st.sidebar.info(f"API Key: {OKX_KEY[:8]}****")
if st.sidebar.button("🗑️ 清除信号历史"):
    st.session_state.history.clear()

df = get_okx_data()

if not df.empty:
    df = analyze_logic(df)
    curr = df.iloc[-1]
    
    # 诱空判定逻辑
    support_wall = df['l'].tail(60).min()
    is_low_vol = curr['v'] < curr['v_avg'] * 0.65
    is_at_support = (curr['c'] - support_wall) / support_wall < 0.002
    
    if is_low_vol and is_at_support:
        st.markdown('<div class="alert-box">🚨 发现缩量诱空陷阱！价格接近强支撑且庄家停止抛售 🚨</div>', unsafe_allow_html=True)
        # 记录信号
        st.session_state.history.appendleft({"时间": datetime.now().strftime("%H:%M:%S"), "类型": "缩量底补", "价格": curr['c']})

    # 数据仪表盘
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("OKX 指数价", f"${curr['c']:.2f}")
    m2.metric("量能活跃度", f"{curr['v']/curr['v_avg']:.2f}x", "极度缩量" if is_low_vol else "正常")
    m3.metric("趋势共振", "🟢 多头" if curr['ema8'] > curr['ema21'] else "🔴 空头")
    m4.metric("庄家支撑位", f"${support_wall:.2f}")

    # 图表绘制
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=df.index, open=df['o'], high=df['h'], low=df['l'], close=df['c'], name="K线"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['ema8'], line=dict(color='yellow', width=1), name="EMA8"), row=1, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df['v'], marker_color='rgba(100,100,100,0.5)', name="成交量"), row=2, col=1)
    
    fig.update_layout(height=700, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # 历史记录展示
    if st.session_state.history:
        st.subheader("📜 实时信号流")
        st.table(list(st.session_state.history)[:5])
else:
    st.error("📡 数据抓取失败。请确认 LetsVPN 已开启全局模式，且 API 白名单（图10）包含你目前的 IP。")
