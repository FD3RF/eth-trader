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

# ====== 语音 ======
def ai_voice_broadcast(text):
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

# ====== 交易所 ======
@st.cache_resource
def init_exchange():
    return ccxt.okx({
        'enableRateLimit': True,
        'options': {'defaultType': 'swap'}
    })

# ====== 获取数据 ======
def fetch_data():

    ex = init_exchange()

    for _ in range(3):
        try:

            bars = ex.fetch_ohlcv(
                'ETH/USDT:USDT',
                timeframe='5m',
                limit=120
            )

            ticker = ex.fetch_ticker('ETH/USDT:USDT')

            df = pd.DataFrame(
                bars,
                columns=['ts','open','high','low','close','vol']
            )

            df['ts_dt'] = pd.to_datetime(df['ts'], unit='ms')

            return df, ticker

        except Exception:
            time.sleep(1)

    return None, None

# ====== 背离判定 ======
def is_bottom_divergence(df):

    if len(df) < 25:
        return False

    low1 = df['low'].iloc[-1]
    low2 = df['low'].iloc[-2]

    vol1 = df['vol'].iloc[-1]

    avg_vol = df['vol'].iloc[-21:-1].mean()

    if pd.isna(avg_vol) or avg_vol == 0:
        return False

    return (
        low1 < low2 and
        vol1 < avg_vol * 0.6 and
        df['close'].iloc[-1] > df['close'].iloc[-2]
    )

def is_top_divergence(df):

    if len(df) < 25:
        return False

    high1 = df['high'].iloc[-1]
    high2 = df['high'].iloc[-2]

    vol1 = df['vol'].iloc[-1]

    avg_vol = df['vol'].iloc[-21:-1].mean()

    if pd.isna(avg_vol) or avg_vol == 0:
        return False

    return (
        high1 > high2 and
        vol1 < avg_vol * 0.6 and
        df['close'].iloc[-1] < df['close'].iloc[-2]
    )

# ====== AI策略引擎 ======
def ai_engine(df, ticker):

    curr = df.iloc[-1]

    price = curr['close']

    avg_vol = df['vol'].iloc[-21:-1].mean()

    if pd.isna(avg_vol) or avg_vol == 0:
        avg_vol = 1

    vol_ratio = curr['vol'] / avg_vol

    res = df['high'].iloc[-31:-1].max()
    sup = df['low'].iloc[-31:-1].min()

    h24 = ticker['high']
    l24 = ticker['low']

    lr = (res - price) / (price - sup) if (price - sup) > 0.1 else 0
    sr = (price - sup) / (res - price) if (res - price) > 0.1 else 0

    status = {
        "action": "AI 扫描中",
        "motto": "静观其变",
        "color": "#121212",
        "voice": None
    }

    # ===== 口诀 =====

    if vol_ratio > 1.6 and price > res:

        status.update({
            "action": "直接开多",
            "motto": "放量起涨，突破前高",
            "color": "#1B5E20",
            "voice": "放量起涨，突破前高，直接开多"
        })

    elif vol_ratio > 1.6 and price < sup:

        status.update({
            "action": "直接开空",
            "motto": "放量下跌，跌破前低",
            "color": "#B71C1C",
            "voice": "放量下跌，跌破前低，直接开空"
        })

    elif vol_ratio < 0.5 and price <= sup * 1.002 and price < curr['open']:

        status.update({
            "action": "准备动手(多)",
            "motto": "缩量回踩，低点不破",
            "color": "#0D47A1",
            "voice": "缩量回踩，低点不破，准备动手"
        })

    elif vol_ratio < 0.5 and price >= res * 0.998 and price > curr['open']:

        status.update({
            "action": "准备动手(空)",
            "motto": "缩量反弹，高点不破",
            "color": "#E65100",
            "voice": "缩量反弹，高点不破，准备动手"
        })

    return status, vol_ratio, res, sup, lr, sr, h24, l24

# ====== UI渲染 ======
def render():

    df, ticker = fetch_data()

    if df is None:
        st.warning("数据异常")
        return

    status, vr, res, sup, lr, sr, h24, l24 = ai_engine(df, ticker)

    key = f"{status['action']}_{df.iloc[-1]['ts']}"

    if st.session_state.signal_memory["last_key"] != key and status["voice"]:

        ai_voice_broadcast(status["voice"])

        st.session_state.signal_memory["last_key"] = key

    st.markdown(f"""
    <div style="background:{status['color']};padding:25px;border-radius:15px;text-align:center;">
        <h1 style="color:white">{status['action']}</h1>
        <h3 style="color:#FFD700">{status['motto']}</h3>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    col1.metric("24H 最高", f"{h24:.2f}")
    col2.metric("24H 最低", f"{l24:.2f}")

    fig = go.Figure(data=[go.Candlestick(
        x=df['ts_dt'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close']
    )])

    # 压力
    fig.add_shape(
        type="line",
        x0=df['ts_dt'].min(),
        x1=df['ts_dt'].max(),
        y0=res,
        y1=res,
        line=dict(color="purple", dash="dash")
    )

    # 支撑
    fig.add_shape(
        type="line",
        x0=df['ts_dt'].min(),
        x1=df['ts_dt'].max(),
        y0=sup,
        y1=sup,
        line=dict(color="blue", dash="dash")
    )

    # 背离
    if is_bottom_divergence(df):

        idx = df['low'].idxmin()

        fig.add_trace(go.Scatter(
            x=[df.loc[idx,'ts_dt']],
            y=[df.loc[idx,'low']],
            mode="markers",
            marker=dict(symbol="triangle-up",size=16,color="green"),
            name="底背离"
        ))

    if is_top_divergence(df):

        idx = df['high'].idxmax()

        fig.add_trace(go.Scatter(
            x=[df.loc[idx,'ts_dt']],
            y=[df.loc[idx,'high']],
            mode="markers",
            marker=dict(symbol="triangle-down",size=16,color="red"),
            name="顶背离"
        ))

    fig.update_layout(template="plotly_dark",height=450)

    st.plotly_chart(fig,use_container_width=True)

    st.caption(f"量比: {vr:.2f}x | 多盈亏比: {lr:.2f} | 空盈亏比: {sr:.2f}")

render()

# 自动刷新
time.sleep(5)
st.rerun()
