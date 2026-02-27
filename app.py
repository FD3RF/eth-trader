import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# ==================== 1. 账号密钥区 (图10) ====================
OKX_KEY = "a2a2a452-49e6-4e76-95f3-fb54eb982e7b"
OKX_SECRET = "330FABB2CAD3585677716686C2BF3872"
OKX_PASS = "这里填入你的API口令" # <--- 唯一需要你手动改的地方

# ==================== 2. UI 渲染引擎 (图8风格) ====================
st.set_page_config(page_title="ETH V28.0 OKX实战终端", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #e6edf3; }
    .status-bar { padding: 10px; border-radius: 5px; margin-bottom: 20px; border: 1px solid #30363d; }
    .alert-active { background: linear-gradient(90deg, #8a0606, #ff4b4b); color: white; padding: 15px; border-radius: 10px; text-align: center; font-weight: bold; animation: pulse 1.5s infinite; }
    @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
</style>
""", unsafe_allow_html=True)

# ==================== 3. LetsVPN 穿透抓取 ====================
@st.cache_data(ttl=10) # 每10秒自动更新，不卡顿
def get_live_data():
    # 自动适配 Python 3.13 的系统代理环境
    url = "https://www.okx.com/api/v5/market/candles?instId=ETH-USDT&bar=5m&limit=150"
    try:
        # 强制信任 LetsVPN 提供的代理隧道
        with requests.Session() as s:
            s.trust_env = True 
            r = s.get(url, timeout=5)
            if r.json()['code'] == '0':
                df = pd.DataFrame(r.json()['data'], columns=['ts','o','h','l','c','v','volCcy','volCcyQ','confirm'])
                df['time'] = pd.to_datetime(df['ts'].astype(float), unit='ms')
                df.set_index('time', inplace=True)
                for col in ['o','h','l','c','v']: df[col] = df[col].astype(float)
                return df.sort_index()
    except Exception as e:
        return None

# ==================== 4. 逻辑计算与绘图 ====================
df = get_live_data()

if df is not None:
    # 指标计算 (V15.3 核心算法)
    df['ema8'] = df['c'].ewm(span=8).mean()
    df['ema21'] = df['c'].ewm(span=21).mean()
    df['v_avg'] = df['v'].rolling(20).mean()
    
    curr = df.iloc[-1]
    support_wall = df['l'].tail(40).min()
    
    # 侧边栏：保留图8的参数控制感
    st.sidebar.title("🎮 策略控制台")
    sensitivity = st.sidebar.slider("缩量敏感度", 0.5, 0.9, 0.7)
    st.sidebar.divider()
    st.sidebar.write(f"📡 链路: LetsVPN (已穿透)")
    st.sidebar.write(f"🔑 API: OKX (只读模式)")

    # 顶部状态栏
    if curr['v'] < df['v_avg'].iloc[-1] * sensitivity and curr['c'] <= support_wall * 1.002:
        st.markdown('<div class="alert-active">🎯 监测到庄家锁仓：极度缩量 + 支撑位震荡</div>', unsafe_allow_html=True)

    # 仪表盘
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ETH 当前价", f"${curr['c']:.2f}", f"{curr['c']-df['c'].iloc[-2]:.2f}")
    c2.metric("当前量比", f"{curr['v']/df['v_avg'].iloc[-1]:.2f}x")
    c3.metric("EMA趋势", "🟢 多头" if curr['ema8'] > curr['ema21'] else "🔴 空头")
    c4.metric("支撑防线", f"${support_wall:.2f}")

    # 主图渲染
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=df.index, open=df['o'], high=df['h'], low=df['l'], close=df['c'], name="K线"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['ema8'], line=dict(color='#FFD700', width=1), name="EMA8"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['ema21'], line=dict(color='#00FFFF', width=1), name="EMA21"), row=1, col=1)
    
    # 量能柱 (带颜色识别)
    colors = ['#ff4b4b' if c < o else '#00ff41' for c, o in zip(df['c'], df['o'])]
    fig.add_trace(go.Bar(x=df.index, y=df['v'], marker_color=colors, name="成交量"), row=2, col=1)

    fig.update_layout(height=650, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=5, r=5, t=5, b=5))
    st.plotly_chart(fig, use_container_width=True)
    
    st.caption(f"数据实时同步中 (OKX V5 API) | 刷新时间: {datetime.now().strftime('%H:%M:%S')}")

else:
    st.warning("🔄 正在等待 LetsVPN 握手信号... 如果长时间无反应，请尝试切换 VPN 节点为‘香港’。")
