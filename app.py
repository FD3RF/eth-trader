import streamlit as st
import pandas as pd
import ccxt
import plotly.graph_objects as go
from datetime import datetime
import time

# --- 1. 系统核心配置 ---
st.set_page_config(page_title="ETH AI 终极播报系统", layout="wide")

# 信号记忆与冷却锁，确保播报不重叠
if "signal_memory" not in st.session_state:
    st.session_state.signal_memory = {"last_key": None, "last_time": 0}

def ai_voice_broadcast(text):
    """优化后的语音引擎：直接注入干净的 JS"""
    js = f"""<script>
    try {{
        window.speechSynthesis.cancel(); 
        var msg = new SpeechSynthesisUtterance("{text}");
        msg.lang='zh-CN'; msg.rate=1.2; msg.volume=1.0;
        window.speechSynthesis.speak(msg);
    }} catch(e) {{ console.log('Speech error'); }}
    </script>"""
    st.components.v1.html(js, height=0)

# --- 2. 稳健数据引擎 ---
@st.cache_resource
def init_exchange():
    return ccxt.okx({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})

def fetch_data():
    ex = init_exchange()
    try:
        # 获取 150 根确保分位数计算稳定
        bars = ex.fetch_ohlcv("ETH/USDT:USDT", timeframe="5m", limit=150)
        ticker = ex.fetch_ticker("ETH/USDT:USDT")
        df = pd.DataFrame(bars, columns=['ts','open','high','low','close','vol'])
        df['ts_dt'] = pd.to_datetime(df['ts'], unit='ms')
        return df.dropna(), ticker
    except:
        return None, None

# --- 3. 核心口诀判定引擎 (严格对齐) ---
def ai_engine(df, ticker):
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    # 量能基准判定
    avg_vol = df['vol'].iloc[-50:-1].median() 
    vol_ratio = curr['vol'] / (avg_vol if avg_vol > 0 else 1)
    
    # 动态锁定关键点位
    res = df['high'].iloc[-40:-1].quantile(0.95)
    sup = df['low'].iloc[-40:-1].quantile(0.05)
    h24, l24 = ticker['high'], ticker['low']
    
    status = {"action": "AI 扫描中", "motto": "静如处子，动如脱兔", "color": "#121212", "voice": "", "tri": None}

    # --- 做多逻辑判定 ---
    if vol_ratio < 0.6 and curr['low'] <= sup * 1.002 and curr['close'] > curr['low']:
        status.update({"action": "准备动手", "motto": "缩量回踩，低点不破", "color": "#0D47A1", "voice": "缩量回踩，低点不破，准备动手"})
    elif vol_ratio > 1.6 and curr['close'] > res:
        status.update({"action": "直接开多", "motto": "放量起涨，突破前高", "color": "#1B5E20", "voice": "放量起涨，突破前高，直接开多", "tri": "buy"})
    elif vol_ratio > 2.8 and curr['low'] >= sup and curr['close'] < prev['close'] * 0.99:
        status.update({"action": "这是机会", "motto": "放量急跌，底部不破", "color": "#2E7D32", "voice": "放量急跌，底部不破，这是机会"})
    
    # --- 做空逻辑判定 ---
    elif vol_ratio < 0.6 and curr['high'] >= res * 0.998 and curr['close'] < curr['high']:
        status.update({"action": "准备动手", "motto": "缩量反弹，高点不破", "color": "#E65100", "voice": "缩量反弹，高点不破，准备动手"})
    elif vol_ratio > 1.6 and curr['close'] < sup:
        status.update({"action": "直接开空", "motto": "放量下跌，跌破前低", "color": "#B71C1C", "voice": "放量下跌，跌破前低，直接开空", "tri": "sell"})
    elif vol_ratio > 2.8 and curr['high'] <= res and curr['close'] > prev['close'] * 1.01:
        status.update({"action": "这是机会", "motto": "放量急涨，顶部不破", "color": "#D32F2F", "voice": "放量急涨，顶部不破，这是机会"})

    # --- 埋伏/横盘 ---
    elif vol_ratio < 0.4:
        status.update({"action": "埋伏等涨", "motto": "缩量横盘，低点托住", "color": "#424242", "voice": "缩量横盘，低点托住，埋伏等涨"})

    return status, res, sup, h24, l24

# --- 4. 界面渲染 ---
def render():
    df, ticker = fetch_data()
    if df is None: return

    status, res, sup, h24, l24 = ai_engine(df, ticker)
    
    # 语音播报去重逻辑
    now = time.time()
    if (st.session_state.signal_memory["last_key"] != status["action"] or now - st.session_state.signal_memory["last_time"] > 25):
        if status["voice"]:
            ai_voice_broadcast(status["voice"])
            st.session_state.signal_memory["last_key"] = status["action"]
            st.session_state.signal_memory["last_time"] = now

    # 修复看板渲染（解决截图中的源码裸露问题）
    st.markdown(f"""
        <div style="background:{status['color']}; padding:30px; border-radius:15px; text-align:center; border: 3px solid #FFD700; color: white;">
            <h1 style="margin:0; font-size:55px;">{status['action']}</h1>
            <h3 style="color:#FFD700; margin:15px 0;">“{status['motto']}”</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; background: rgba(0,0,0,0.4); padding: 15px; border-radius: 10px;">
                <div><small style="color:#aaa;">5M压力</small><br><b style="color:#FF00FF;font-size:20px;">{res:.2f}</b></div>
                <div><small style="color:#aaa;">5M支撑</small><br><b style="color:#00FFFF;font-size:20px;">{sup:.2f}</b></div>
                <div><small style="color:#aaa;">24H高低</small><br><b>{ticker['high']:.1f}/{ticker['low']:.1f}</b></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # K线可视化与信号标记
        fig = go.Figure(data=[go.Candlestick(
        x=df['ts_dt'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        increasing_line_color='#00ff88', decreasing_line_color='#ff3344'
    )])

    if status['tri'] == "buy":
        fig.add_trace(go.Scatter(x=[df.iloc[-1]['ts_dt']], y=[df.iloc[-1]['low']*0.998], mode="markers", marker=dict(symbol="triangle-up", size=22, color="#00ff88"), name="直接开多"))
    elif status['tri'] == "sell":
        fig.add_trace(go.Scatter(x=[df.iloc[-1]['ts_dt']], y=[df.iloc[-1]['high']*1.002], mode="markers", marker=dict(symbol="triangle-down", size=22, color="#ff3344"), name="直接开空"))

    fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig, use_container_width=True)

render()
# 强制循环刷新
time.sleep(5)
st.rerun()
