import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# ==================== 1. 核心实战配置 ====================
# 信息来自你的截图 (图10)
OKX_CONFIG = {
    "api_key": "a2a2a452-49e6-4e76-95f3-fb54eb982e7b",
    "secret_key": "330FABB2CAD3585677716686C2BF3872",
    "passphrase": "在此输入你创建API时设定的口令", # <--- 必须填写这个才能连接！
}

# 页面基础设置 (图8 风格)
st.set_page_config(page_title="ETH OKX 实战终端", layout="wide")
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: white; }
    .alert-card { background: linear-gradient(90deg, #4b0000, #ff4b4b); padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# ==================== 2. LetsVPN 穿透数据引擎 ====================
def fetch_okx_data():
    # 自动穿透：LetsVPN 开启后，requests 会自动通过系统代理 (图4)
    url = "https://www.okx.com/api/v5/market/candles?instId=ETH-USDT&bar=5m&limit=200"
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        if data['code'] == '0':
            df = pd.DataFrame(data['data'], columns=['ts', 'o', 'h', 'l', 'c', 'v', 'volCcy', 'volCcyQ', 'confirm'])
            df['time'] = pd.to_datetime(df['ts'].astype(float), unit='ms')
            df.set_index('time', inplace=True)
            for col in ['o', 'h', 'l', 'c', 'v']: df[col] = df[col].astype(float)
            return df.sort_index()
    except Exception as e:
        st.error(f"📡 链路连接失败: 请检查 LetsVPN 是否处于‘已成功连接’状态 (图4)")
        return pd.DataFrame()

# ==================== 3. 主界面逻辑 ====================
st.title("🛡️ ETH V27.5 旗舰实战站")

df = fetch_okx_data()

if not df.empty:
    # 技术指标计算
    df['ema8'] = df['c'].ewm(span=8).mean()
    df['ema21'] = df['c'].ewm(span=21).mean()
    df['vol_avg'] = df['v'].rolling(20).mean()
    
    curr = df.iloc[-1]
    prev = df.iloc[-2]

    # --- 逻辑触发：缩量阴跌防御 (图8 核心逻辑) ---
    is_low_vol = curr['v'] < df['vol_avg'].iloc[-1] * 0.7
    support_level = df['l'].tail(50).min()
    
    if is_low_vol and curr['c'] <= support_level * 1.001:
        st.markdown('<div class="alert-card">⚠️ 触发狙击信号：缩量回踩支撑位，严防阴跌后假突破</div>', unsafe_allow_html=True)

    # --- 仪表盘 ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("OKX 实时价", f"${curr['c']:.2f}", f"{curr['c']-prev['c']:.2f}")
    c2.metric("当前量比", f"{curr['v']/df['vol_avg'].iloc[-1]:.2f}x")
    c3.metric("EMA 状态", "🟢 金叉" if curr['ema8'] > curr['ema21'] else "🔴 死叉")
    c4.metric("支撑墙", f"${support_level:.2f}")

    # --- 专业级 K 线图 ---
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
    # K线
    fig.add_trace(go.Candlestick(x=df.index, open=df['o'], high=df['h'], low=df['l'], close=df['c'], name="ETH-USDT"), row=1, col=1)
    # 均线
    fig.add_trace(go.Scatter(x=df.index, y=df['ema8'], line=dict(color='yellow', width=1.5), name="EMA8"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['ema21'], line=dict(color='cyan', width=1.5), name="EMA21"), row=1, col=1)
    # 成交量
    colors = ['red' if row['c'] < row['o'] else 'green' for index, row in df.iterrows()]
    fig.add_trace(go.Bar(x=df.index, y=df['v'], marker_color=colors, name="成交量"), row=2, col=1)

    fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.success(f"✅ 数据已通过 OKX API 同步 | 最后更新: {datetime.now().strftime('%H:%M:%S')}")

else:
    st.info("⌛ 正在等待 OKX 数据流回传...")
