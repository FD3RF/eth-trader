import streamlit as st
import pandas as pd
import ccxt
import plotly.graph_objects as go
from datetime import datetime
import time

# 1. 基础配置
st.set_page_config(page_title="ETH V2600 战神系统", layout="wide")

# 2. 核心：语音播报与金色闪烁逻辑
def trigger_action(text, is_alert=False):
    # 语音脚本
    js_voice = f"var msg = new SpeechSynthesisUtterance('{text}'); msg.lang = 'zh-CN'; window.speechSynthesis.speak(msg);"
    
    # 金色闪烁视觉效果脚本 (仅在放量突破/跌破时触发)
    js_flash = ""
    if is_alert:
        js_flash = "document.body.style.backgroundColor = '#FFD700'; setTimeout(() => { document.body.style.backgroundColor = '#0e1117'; }, 500);"
    
    st.components.v1.html(f"<script>{js_voice} {js_flash}</script>", height=0)

# 3. 初始化 OKX (使用你已配置好的 Key)
@st.cache_resource
def init_okx():
    return ccxt.okx({
        'apiKey': 'a2a2a452-49e6-4e76-95f3-fb54eb982e7b',
        'secret': '330FABB2CAD3585677716686C2BF3872',
        'password': '123321aA@',
        'enableRateLimit': True,
    })

okx = init_okx()

# 4. 获取数据逻辑
def fetch_data():
    try:
        # 获取 ETH 永续合约 5分钟 K线
        bars = okx.fetch_ohlcv('ETH/USDT:USDT', timeframe='5m', limit=100)
        df = pd.DataFrame(bars, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        return df
    except Exception as e:
        st.error(f"数据获取异常: {e}")
        return pd.DataFrame()

# 5. 核心口诀判定引擎
def analyze_v2600(df):
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    # 计算 20 周期均量
    avg_vol = df['vol'].iloc[-21:-1].mean()
    vol_ratio = curr['vol'] / avg_vol
    
    # 锁定前 30 根 K 线的高低点
    recent_low = df['low'].iloc[-30:].min()
    recent_high = df['high'].iloc[-30:].max()
    
    signal = {"motto": "量能不明", "action": "观望", "color": "#2b2b2b", "voice": "", "alert": False}

    # --- 做多口诀判定 ---
    if vol_ratio < 0.6 and abs(curr['low'] - recent_low)/recent_low < 0.002:
        signal.update({"motto": "缩量回踩，低点不破", "action": "准备动手", "color": "#1E90FF"})
    elif vol_ratio > 1.8 and curr['close'] > recent_high:
        signal.update({"motto": "放量起涨，突破前高", "action": "直接开多", "color": "#00C853", "voice": "放量起涨突破前高直接开多", "alert": True})
    
    # --- 做空口诀判定 ---
    elif vol_ratio < 0.6 and abs(curr['high'] - recent_high)/recent_high < 0.002:
        signal.update({"motto": "缩量反弹，高点不破", "action": "准备动手", "color": "#FF9100"})
    elif vol_ratio > 1.8 and curr['close'] < recent_low:
        signal.update({"motto": "放量下跌，跌破前低", "action": "直接开空", "color": "#FF3D00", "voice": "放量下跌跌破前低直接开空", "alert": True})

    return signal, vol_ratio, recent_low, recent_high

# --- 界面展示 ---
st.title("🛡️ ETH V2600 战神·不朽大衍系统")

df = fetch_data()
if not df.empty:
    sig, vr, sup, res = analyze_v2600(df)
    
    # 执行语音与警报
    if sig['voice']:
        trigger_action(sig['voice'], sig['alert'])

    # 顶层看板
    st.markdown(f"""
        <div style="background-color:{sig['color']}; padding:35px; border-radius:20px; text-align:center; border: 3px solid gold;">
            <h1 style="color:white; font-size:50px; margin:0;">{sig['action']}</h1>
            <h2 style="color:gold;">“{sig['motto']}”</h2>
            <p style="color:white; font-size:18px;">当前量比: {vr:.2f}x | 支撑: {sup} | 压力: {res}</p>
        </div>
    """, unsafe_allow_html=True)

    # 绘制 K 线图
    fig = go.Figure(data=[go.Candlestick(x=df['ts'], open=df['open'], high=df['high'], low=df['low'], close=df['close'])])
    fig.add_hline(y=sup, line_dash="dash", line_color="cyan", annotation_text="支撑线")
    fig.add_hline(y=res, line_dash="dash", line_color="magenta", annotation_text="压力线")
    fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

st.caption(f"数据实时刷新中... 当前时间: {datetime.now().strftime('%H:%M:%S')}")
