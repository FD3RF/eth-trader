import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from datetime import datetime

# ==================== 1. 核心实战配置 (已根据你的信息填入) ====================
OKX_CONFIG = {
    "api_key": "a2a2a452-49e6-4e76-95f3-fb54eb982e7b",
    "secret_key": "330FABB2CAD3585677716686C2BF3872",
    "passphrase": "123321aA@",  # <--- 已填入你刚才提供的口令
}

st.set_page_config(page_title="ETH V29.0 终极实战终端", layout="wide")

# ==================== 2. 数据引擎 (穿透 LetsVPN) ====================
@st.cache_data(ttl=5)
def get_okx_candles():
    # 自动使用你 LetsVPN 建立的香港链路 (图4)
    url = "https://www.okx.com/api/v5/market/candles?instId=ETH-USDT&bar=5m&limit=100"
    try:
        # 强制信任系统代理环境
        with requests.Session() as s:
            s.trust_env = True 
            r = s.get(url, timeout=8)
            res = r.json()
            if res.get('code') == '0':
                df = pd.DataFrame(res['data'], columns=['ts','o','h','l','c','v','volCcy','volCcyQ','confirm'])
                df['time'] = pd.to_datetime(df['ts'].astype(float), unit='ms')
                df.set_index('time', inplace=True)
                for col in ['o','h','l','c','v']: df[col] = df[col].astype(float)
                return df.sort_index()
            else:
                st.sidebar.error(f"OKX 报错: {res.get('msg')}")
    except Exception as e:
        st.sidebar.warning("📡 正在等待 LetsVPN 响应...")
    return pd.DataFrame()

# ==================== 3. 界面与逻辑渲染 ====================
st.title("🛡️ ETH V29.0 实战防御系统")

df = get_okx_candles()

if not df.empty:
    # 核心算法逻辑
    df['ema8'] = df['c'].ewm(span=8).mean()
    df['ema21'] = df['c'].ewm(span=21).mean()
    curr = df.iloc[-1]
    
    # 阴跌/缩量判定 (基于图8核心算法)
    vol_avg = df['v'].rolling(15).mean().iloc[-1]
    is_low_vol = curr['v'] < vol_avg * 0.7
    is_down_trend = curr['c'] < curr['ema21']
    risk_score = 98.9 if is_low_vol and is_down_trend else 15.0

    # 顶部仪表盘
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ETH 现价", f"${curr['c']:.2f}", f"{curr['c']-df.iloc[-2]['c']:.2f}")
    c2.metric("阴跌风险", f"{risk_score}", "高危" if risk_score > 80 else "安全", delta_color="inverse")
    c3.metric("量能状态", "极度缩量" if is_low_vol else "活跃")
    c4.metric("EMA 趋势", "🟢 多头" if curr['ema8'] > curr['ema21'] else "🔴 空头")

    # 警报显示
    if risk_score > 80:
        st.error("⚠️ 监测到高风险阴跌模式：价格受压于 EMA21 且成交量急剧萎缩！")

    # K线图表
    fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['o'], high=df['h'], low=df['l'], close=df['c'], name="OKX 实时")])
    fig.add_trace(go.Scatter(x=df.index, y=df['ema8'], line=dict(color='yellow', width=1), name="EMA8"))
    fig.add_trace(go.Scatter(x=df.index, y=df['ema21'], line=dict(color='cyan', width=1), name="EMA21"))
    fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.success(f"✅ OKX 数据流已连接 | 更新时间: {datetime.now().strftime('%H:%M:%S')}")
else:
    st.info("🔄 正在通过 LetsVPN 握手 OKX 节点，请稍候...")
