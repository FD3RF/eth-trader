import streamlit as st
import pandas as pd
import ccxt
import plotly.graph_objects as go
from datetime import datetime

# 1. 页面与安全配置
st.set_page_config(page_title="OKX 5m口诀机器人", layout="wide")

# 2. 浏览器语音引擎
def trigger_speech(text):
    js = f"""<script>
    var msg = new SpeechSynthesisUtterance("{text}");
    msg.lang = 'zh-CN';
    msg.rate = 1.1;
    window.speechSynthesis.speak(msg);
    </script>"""
    st.components.v1.html(js, height=0)

# 3. 初始化 OKX 连接 (使用你提供的凭证)
@st.cache_resource
def init_okx():
    return ccxt.okx({
        'apiKey': 'a2a2a452-49e6-4e76-95f3-fb54eb982e7b',
        'secret': '330FABB2CAD3585677716686C2BF3872',
        'password': '123321aA@',
        'enableRateLimit': True, # 自动避开频率限制
    })

okx = init_okx()

def fetch_data():
    try:
        # 获取 ETH-USDT-SWAP (永续合约) 5分钟数据
        bars = okx.fetch_ohlcv('ETH/USDT:USDT', timeframe='5m', limit=100)
        df = pd.DataFrame(bars, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        return df
    except Exception as e:
        st.error(f"OKX 数据接入异常: {e}")
        return pd.DataFrame()

# 4. 核心判定逻辑
def get_signal(df):
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    # 均量计算与量比
    avg_vol = df['vol'].iloc[-21:-1].mean()
    vol_ratio = curr['vol'] / avg_vol
    
    # 锁定前30根K线的高低点
    recent_low = df['low'].iloc[-30:].min()
    recent_high = df['high'].iloc[-30:].max()
    
    res = {"motto": "量能不明", "action": "保持观望", "color": "#2b2b2b", "voice": ""}

    # 做多：缩量不破底，放量突破买
    if vol_ratio < 0.6 and abs(curr['low'] - recent_low)/recent_low < 0.003:
        res.update({"motto": "缩量回踩，低点不破", "action": "准备动手", "color": "#1E90FF"})
    elif vol_ratio > 1.8 and curr['close'] > recent_high:
        res.update({"motto": "放量起涨，突破前高", "action": "直接开多", "color": "#00C853", "voice": "放量起涨突破前高直接开多"})
        
    # 做空：缩量不过顶，放量跌破空
    elif vol_ratio < 0.6 and abs(curr['high'] - recent_high)/recent_high < 0.003:
        res.update({"motto": "缩量反弹，高点不破", "action": "准备动手", "color": "#FF9100"})
    elif vol_ratio > 1.8 and curr['close'] < recent_low:
        res.update({"motto": "放量下跌，跌破前低", "action": "直接开空", "color": "#FF3D00", "voice": "放量下跌跌破前低直接开空"})

    return res, vol_ratio, recent_low, recent_high

# 5. 执行与渲染
st.title("🤖 ETH V2600 实时口诀播报系统")
df = fetch_data()

if not df.empty:
    sig, vr, low_p, high_p = get_signal(df)
    
    if sig['voice']:
        trigger_speech(sig['voice'])

    # 看板布局
    st.markdown(f"""
        <div style="background-color:{sig['color']}; padding:30px; border-radius:15px; text-align:center; border: 2px solid white;">
            <h1 style="color:white; font-size:42px;">{sig['action']}</h1>
            <h2 style="color:white; opacity:0.8;">“{sig['motto']}”</h2>
            <p style="color:white;">实时量比: {vr:.2f}x | 支撑: {low_p} | 压力: {high_p}</p>
        </div>
    """, unsafe_allow_html=True)

    # 绘制K线
    fig = go.Figure(data=[go.Candlestick(x=df['ts'], open=df['open'], high=df['high'], low=df['low'], close=df['close'])])
    fig.add_hline(y=low_p, line_dash="dash", line_color="cyan")
    fig.add_hline(y=high_p, line_dash="dash", line_color="magenta")
    fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

st.write(f"最后更新: {datetime.now().strftime('%H:%M:%S')}")
