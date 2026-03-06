import streamlit as st
import pandas as pd
import ccxt
import plotly.graph_objects as go
from datetime import datetime

# 页面基础配置
st.set_page_config(page_title="ETH 5m精准口诀机器人", layout="wide")

# 1. 语音播报函数 (利用浏览器 Web Speech API)
def trigger_voice(text):
    js_script = f"""
    <script>
    var msg = new SpeechSynthesisUtterance('{text}');
    msg.lang = 'zh-CN';
    msg.rate = 1.1; 
    window.speechSynthesis.speak(msg);
    </script>
    """
    st.components.v1.html(js_script, height=0)

# 2. 获取币安实时数据
exchange = ccxt.binance()

def fetch_eth_data():
    try:
        # 获取最近50根5分钟K线
        ohlcv = exchange.fetch_ohlcv('ETH/USDT', timeframe='5m', limit=50)
        df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        return df
    except Exception as e:
        st.error(f"获取数据失败: {e}")
        return pd.DataFrame()

# 3. 核心决策逻辑引擎
def analyze_market(df):
    curr = df.iloc[-1]   # 当前未收盘K线
    prev = df.iloc[-2]   # 上一根已收盘K线
    
    # 计算均量 (参考最近5根)
    avg_vol = df['vol'].iloc[-6:-1].mean()
    vol_ratio = curr['vol'] / avg_vol
    
    # 锁定波段高低点
    recent_low = df['low'].iloc[-30:].min()
    recent_high = df['high'].iloc[-30:].max()
    
    motto, action, color, sound = "量能不明", "保持观望", "#2b2b2b", ""

    # --- 做多判定 ---
    if curr['low'] >= recent_low and curr['close'] <= recent_low * 1.002 and vol_ratio < 0.7:
        motto, action, color = "缩量回踩，低点不破", "准备动手", "#1E90FF"
    elif vol_ratio > 1.5 and curr['close'] > prev['high'] and prev['close'] < prev['open']:
        motto, action, color, sound = "放量起涨，突破阴线", "马上跟多", "#00C853", "马上跟多"
    elif vol_ratio > 2.0 and curr['low'] < recent_low and curr['close'] > recent_low:
        motto, action, color, sound = "放量暴跌，低点不破", "假跌真买", "#00E676", "这是机会"

    # --- 做空判定 ---
    elif curr['high'] <= recent_high and curr['close'] >= recent_high * 0.998 and vol_ratio < 0.7:
        motto, action, color = "缩量反弹，高点不破", "准备动手", "#FF9100"
    elif vol_ratio > 1.5 and curr['close'] < prev['low'] and prev['close'] > prev['open']:
        motto, action, color, sound = "放量下跌，跌破阳线", "马上跟空", "#FF3D00", "马上跟空"
    elif vol_ratio > 2.0 and curr['high'] > recent_high and curr['close'] < recent_high:
        motto, action, color, sound = "放量急涨，顶部不破", "假涨真空", "#D50000", "这是机会"

    return motto, action, color, sound, vol_ratio, recent_low, recent_high

# 4. 界面渲染
st.title("⚡ ETH 5分钟量价口诀实时机器人")
df = fetch_eth_data()

if not df.empty:
    motto, action, color, sound, vr, sup, res = analyze_market(df)
    
    # 触发语音
    if sound:
        trigger_voice(sound)
    
    # 看板显示
    st.markdown(f"""
        <div style="background-color:{color}; padding:25px; border-radius:12px; text-align:center;">
            <h1 style="color:white; margin:0;">{action}</h1>
            <h3 style="color:white; opacity:0.8;">“{motto}”</h3>
            <p style="color:white;">量比: {vr:.2f}x | 支撑: {sup} | 压力: {res}</p>
        </div>
    """, unsafe_allow_html=True)

    # 绘制K线图
    fig = go.Figure(data=[go.Candlestick(x=df['ts'], open=df['open'], high=df['high'], low=df['low'], close=df['close'])])
    fig.add_hline(y=sup, line_dash="dash", line_color="cyan")
    fig.add_hline(y=res, line_dash="dash", line_color="magenta")
    fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

st.caption(f"刷新频率：10秒 | 当前时间：{datetime.now().strftime('%H:%M:%S')}")
