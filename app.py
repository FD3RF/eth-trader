import streamlit as st
import pandas as pd
import ccxt
import plotly.graph_objects as go
from datetime import datetime
import time

st.set_page_config(page_title="ETH AI 终极播报", layout="wide")

# ====== 状态锁 ======
if "signal_memory" not in st.session_state:
    st.session_state.signal_memory = {"last_key": None}
if "auto" not in st.session_state:
    st.session_state.auto = False
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = datetime.now()

def ai_voice_broadcast(text):
    """语音播报"""
    js = f"""
    <script>
    try {{
        var msg = new SpeechSynthesisUtterance("{text}");
        msg.lang = 'zh-CN';
        msg.rate = 1.15;
        window.speechSynthesis.speak(msg);
    }} catch(e) {{}}
    </script>
    """
    st.components.v1.html(js, height=0)

# ====== 数据源（重试+安全）======
@st.cache_resource
def init_exchange():
    return ccxt.okx({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})

def fetch_data():
    ex = init_exchange()
    for _ in range(3):  # 数据重试
        try:
            bars = ex.fetch_ohlcv('ETH/USDT:USDT', timeframe='5m', limit=100)
            ticker = ex.fetch_ticker('ETH/USDT:USDT')

            df = pd.DataFrame(bars, columns=['ts','open','high','low','close','vol'])
            df['ts_dt'] = pd.to_datetime(df['ts'], unit='ms')
            return df, ticker
        except Exception:
            time.sleep(1)
    return None, None

# ====== 引擎（口诀对齐+均量保护）======
def ai_engine(df, ticker):
    curr = df.iloc[-1]
    price = curr['close']

    avg_vol = df['vol'].iloc[-21:-1].mean()
    vol_ratio = curr['vol'] / avg_vol if avg_vol > 0 else 1

    res_5m = df['high'].iloc[-31:-1].max()
    sup_5m = df['low'].iloc[-31:-1].min()
    h24, l24 = ticker['high'], ticker['low']

    lr = (res_5m - price) / (price - sup_5m) if (price - sup_5m) > 0.1 else 0
    sr = (price - sup_5m) / (res_5m - price) if (res_5m - price) > 0.1 else 0

    status = {
        "action": "AI 扫描中",
        "motto": "静观其变",
        "color": "#121212",
        "voice": None,
        "flash": None
    }

    # 放量突破（口诀）
    if vol_ratio > 1.6 and price > res_5m:
        status.update({
            "action": "直接开多",
            "motto": "放量起涨，突破前高",
            "color": "#1B5E20",
            "voice": "放量起涨，突破前高，直接开多",
            "flash": "gold"
        })
    # 放量跌破
    elif vol_ratio > 1.6 and price < sup_5m:
        status.update({
            "action": "直接开空",
            "motto": "放量下跌，跌破前低",
            "color": "#B71C1C",
            "voice": "放量下跌，跌破前低，直接开空",
            "flash": "purple"
        })
    # 缩量回踩
    elif vol_ratio < 0.5 and price <= sup_5m * 1.002 and price < curr['open']:
        status.update({
            "action": "准备动手(多)",
            "motto": "缩量回踩，低点不破",
            "color": "#0D47A1",
            "voice": "缩量回踩，低点不破，准备动手"
        })
    # 缩量反弹
    elif vol_ratio < 0.5 and price >= res_5m * 0.998 and price > curr['open']:
        status.update({
            "action": "准备动手(空)",
            "motto": "缩量反弹，高点不破",
            "color": "#E65100",
            "voice": "缩量反弹，高点不破，准备动手"
        })

    return status, vol_ratio, res_5m, sup_5m, lr, sr, h24, l24

# ====== UI渲染（安全）======
def render():
    df, ticker = fetch_data()
    if df is None or ticker is None:
        st.warning("数据暂不可用，正在重试…")
        return

    status, vr, res, sup, lr, sr, h24, l24 = ai_engine(df, ticker)

    # 防重复播报
    key = f"{status['action']}_{df.iloc[-1]['ts']}"
    if st.session_state.signal_memory["last_key"] != key and status.get("voice"):
        ai_voice_broadcast(status["voice"])
        st.session_state.signal_memory["last_key"] = key

    st.markdown(f"""
    <div style="background:{status['color']};padding:25px;border-radius:15px;text-align:center;">
        <h1 style="color:white;margin:0">{status['action']}</h1>
        <h3 style="color:#FFD700">{status['motto']}</h3>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("24H 最高", f"{h24:.2f}")
    with col2:
        st.metric("24H 最低", f"{l24:.2f}")

    fig = go.Figure(data=[go.Candlestick(
        x=df['ts_dt'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close']
    )])
    fig.update_layout(template="plotly_dark", height=450)
    st.plotly_chart(fig, use_container_width=True)

    st.caption(f"量比: {vr:.2f}x | 多盈亏比: {lr:.2f} | 空盈亏比: {sr:.2f}")

# ====== 控制 ======
if st.button("开始实时扫描"):
    st.session_state.auto = True

if st.button("停止扫描"):
    st.session_state.auto = False

# ====== 自动刷新 ======
if st.session_state.auto:
    render()
    now = datetime.now()
    if (now - st.session_state.last_refresh).total_seconds() < 5:
        time.sleep(5)
    st.session_state.last_refresh = datetime.now()
    st.rerun()
else:
    render()
