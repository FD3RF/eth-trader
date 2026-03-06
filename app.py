import streamlit as st
import pandas as pd
import ccxt
import plotly.graph_objects as go
from datetime import datetime
import components.html as html # Streamlit 原生支持嵌入 HTML

# 页面基础配置
st.set_page_config(page_title="ETH精准口诀量价机器人-语音版", layout="wide")

# 1. 接入真实数据
exchange = ccxt.binance()

def get_realtime_data():
    try:
        bars = exchange.fetch_ohlcv('ETH/USDT', timeframe='5m', limit=100)
        df = pd.DataFrame(bars, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        return df
    except:
        return pd.DataFrame()

# 2. 语音播报组件 (JavaScript)
def text_to_speech(text):
    """通过 JavaScript 调用浏览器语音引擎"""
    js_code = f"""
    <script>
    var msg = new SpeechSynthesisUtterance('{text}');
    msg.lang = 'zh-CN';
    msg.rate = 1.2; // 语速稍快一点，适合合约节奏
    window.speechSynthesis.speak(msg);
    </script>
    """
    st.components.v1.html(js_code, height=0)

# 3. 核心逻辑引擎
def analyze_logic(df):
    if df.empty: return "等待", "数据加载中", "#2b2b2b", 1, 0, 0
    
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    avg_vol = df['vol'].iloc[-6:-1].mean()
    vol_ratio = curr['vol'] / avg_vol
    
    recent_window = df.iloc[-30:]
    support_level = recent_window['low'].min()
    resistance_level = recent_window['high'].max()
    
    motto = "量能不明"
    action = "保持观望"
    color = "#2b2b2b"
    should_speak = False

    # 做多口诀
    if curr['low'] >= support_level and curr['close'] <= support_level * 1.003 and vol_ratio < 0.8:
        motto = "缩量回踩，低点不破"
        action = "准备动手"
        color = "#1E90FF"
    elif vol_ratio > 1.5 and curr['close'] > prev['high'] and prev['close'] < prev['open']:
        motto = "放量起涨，突破阴线"
        action = "马上跟多"
        color = "#00C853"
        should_speak = True
    elif vol_ratio > 2.0 and curr['low'] < support_level and curr['close'] > support_level:
        motto = "放量暴跌，低点不破"
        action = "假跌真买，这是机会"
        color = "#00E676"
        should_speak = True

    # 做空口诀
    elif curr['high'] <= resistance_level and curr['close'] >= resistance_level * 0.997 and vol_ratio < 0.8:
        motto = "缩量反弹，高点不破"
        action = "准备动手"
        color = "#FF9100"
    elif vol_ratio > 1.5 and curr['close'] < prev['low'] and prev['close'] > prev['open']:
        motto = "放量下跌，跌破阳线"
        action = "马上跟空"
        color = "#FF3D00"
        should_speak = True
    elif vol_ratio > 2.0 and curr['high'] > resistance_level and curr['close'] < resistance_level:
        motto = "放量急涨，顶部不破"
        action = "假涨真空，这是机会"
        color = "#D50000"
        should_speak = True

    return motto, action, color, vol_ratio, support_level, resistance_level, should_speak

# 4. UI 界面展示
st.title("🤖 ETH 5分钟精准口诀 - 实时语音播报版")

df = get_realtime_data()
motto, action, color, v_ratio, sup, res, should_speak = analyze_logic(df)

# 触发语音播报
if should_speak:
    text_to_speech(action)

# 视觉看板
st.markdown(f"""
    <div style="background-color:{color}; padding:30px; border-radius:15px; text-align:center; border: 2px solid white;">
        <h1 style="color:white; font-size:45px; margin:0;">{action}</h1>
        <h2 style="color:white; opacity:0.9;">“{motto}”</h2>
        <p style="color:white;">实时量比: {v_ratio:.2f}x | 当前价: {df.iloc[-1]['close'] if not df.empty else 0}</p>
    </div>
""", unsafe_allow_html=True)

# K线图略（同前一版本）
fig = go.Figure(data=[go.Candlestick(x=df['ts'], open=df['open'], high=df['high'], low=df['low'], close=df['close'])])
fig.add_hline(y=sup, line_dash="dash", line_color="cyan")
fig.add_hline(y=res, line_dash="dash", line_color="magenta")
fig.update_layout(height=500, template="plotly_dark", xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

st.write(f"最后刷新: {datetime.now().strftime('%H:%M:%S')} (信号出现时将自动语音提示)")
