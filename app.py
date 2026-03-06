import streamlit as st
import pandas as pd
import ccxt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from streamlit_autorefresh import st_autorefresh

# --- 1. 系统配置 ---
st.set_page_config(page_title="ETH AI 终极盯盘 V2.5", layout="wide")
st_autorefresh(interval=5000, key="refresh")

# 状态锁：确保播报精准不重报
if "signal_memory" not in st.session_state:
    st.session_state.signal_memory = {"last_key": None, "last_time": 0}

def ai_voice_broadcast(text):
    """注入干净的 JS 执行，彻底解决源码外泄"""
    js = f"""<script>
    try {{
        window.speechSynthesis.cancel();
        var msg = new SpeechSynthesisUtterance("{text}");
        msg.lang='zh-CN'; msg.rate=1.2;
        window.speechSynthesis.speak(msg);
    }} catch(e) {{}}
    </script>"""
    st.components.v1.html(js, height=0)

# --- 2. 稳健数据引擎 ---
@st.cache_resource
def init_exchange():
    return ccxt.okx({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})

def fetch_data():
    ex = init_exchange()
    try:
        # 获取 200 根 K 线确保 24H 采样深度
        bars = ex.fetch_ohlcv('ETH/USDT:USDT', '5m', limit=200)
        df = pd.DataFrame(bars, columns=['ts','open','high','low','close','vol'])
        df['ts_dt'] = pd.to_datetime(df['ts'], unit='ms')
        # 计算成交量进化：当前量 vs 过去 24 周期中位量
        df['vol_ma'] = df['vol'].rolling(window=24).median()
        df['vol_ratio'] = df['vol'] / df['vol_ma']
        return df.dropna(), ex.fetch_ticker('ETH/USDT:USDT')
    except:
        return None, None

# --- 3. 核心口诀判定 (完全对齐) ---
def ai_engine(df):
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    ratio = curr['vol_ratio']
    
    # 分位数支撑压力
    res = df['high'].iloc[-40:-1].quantile(0.95)
    sup = df['low'].iloc[-40:-1].quantile(0.05)
    
    status = {"action":"AI 扫描中","motto":"量价合一，顺势而为","color":"#121212","voice":None, "tri":None}

    # 做多口诀判定
    if ratio < 0.6 and curr['low'] <= sup * 1.001:
        status.update({"action":"准备多","motto":"缩量回踩，支撑有效","color":"#0D47A1","voice":"缩量回踩，低点不破"})
    elif ratio > 1.6 and curr['close'] > res:
        status.update({"action":"直接开多","motto":"爆量突破，猛龙过江","color":"#1B5E20","voice":"放量起涨，突破前高", "tri":"buy"})
    
    # 做空口诀判定
    elif ratio < 0.6 and curr['high'] >= res * 0.999:
        status.update({"action":"准备空","motto":"缩量反弹，压力明显","color":"#E65100","voice":"缩量反弹，高点不破"})
    elif ratio > 1.6 and curr['close'] < sup:
        status.update({"action":"直接开空","motto":"爆量跌破，趋势反转","color":"#B71C1C","voice":"放量下跌，跌破前低", "tri":"sell"})

    return status, ratio, res, sup

# --- 4. 界面渲染与进化曲线 ---
def render():
    df, ticker = fetch_data()
    if df is None: return

    status, ratio, res, sup = ai_engine(df)
    
    # 语音调度 (25s 冷却)
    now = time.time()
    if status["voice"] and st.session_state.signal_memory["last_key"] != status["action"] and now - st.session_state.last_voice_time > 25:
        ai_voice_broadcast(status["voice"])
        st.session_state.signal_memory["last_key"] = status["action"]
        st.session_state.last_voice_time = now

    # 顶部看板修复
    st.markdown(f"""
    <div style="background:{status['color']};padding:20px;border-radius:12px;text-align:center;border:2px solid #FFD700;color:white;">
        <h1 style="margin:0;font-size:42px;">{status['action']} ({ratio:.2f}x)</h1>
        <h3 style="color:#FFD700;margin:5px 0;">“{status['motto']}”</h3>
    </div>
    """, unsafe_allow_html=True)

    # 成交量进化双子图
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3])

    # K线与信号
    fig.add_trace(go.Candlestick(x=df['ts_dt'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="ETH"), row=1, col=1)
    fig.add_hline(y=res, line_dash="dash", line_color="magenta", row=1, col=1)
    fig.add_hline(y=sup, line_dash="dash", line_color="cyan", row=1, col=1)

    if status['tri'] == "buy":
        fig.add_trace(go.Scatter(x=[df.iloc[-1]['ts_dt']], y=[df.iloc[-1]['low']*0.999], mode="markers", marker=dict(symbol="triangle-up", size=18, color="#00ff88"), name="开多信号"), row=1, col=1)
    elif status['tri'] == "sell":
        fig.add_trace(go.Scatter(x=[df.iloc[-1]['ts_dt']], y=[df.iloc[-1]['high']*1.001], mode="markers", marker=dict(symbol="triangle-down", size=18, color="#ff3344"), name="开空信号"), row=1, col=1)

    # 进化曲线
    
    fig.add_trace(go.Scatter(x=df['ts_dt'], y=df['vol_ratio'], fill='tozeroy', line=dict(color='gold', width=2), name="量能倍数"), row=2, col=1)
    fig.add_hline(y=1.6, line_dash="dot", line_color="red", row=2, col=1)
    fig.add_hline(y=0.6, line_dash="dot", line_color="green", row=2, col=1)

    fig.update_layout(template="plotly_dark", height=700, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(fig, use_container_width=True)

render()
