import streamlit as st
import pandas as pd
import ccxt
import plotly.graph_objects as go
import time
import joblib
from sklearn.ensemble import RandomForestClassifier
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="ETH AI 终极盯盘", layout="wide")

# 自动刷新
st_autorefresh(interval=5000, key="refresh")

# 状态锁
if "signal_memory" not in st.session_state:
    st.session_state.signal_memory = {"last_key": None}

if "last_voice_time" not in st.session_state:
    st.session_state.last_voice_time = 0

MODEL_FILE = "ai_model.pkl"

# 交易参数
INSTRUMENT = "ETH/USDT:USDT"
TIMEFRAME = "5m"
LIMIT = 150


def ai_voice_broadcast(text):
    js = f"""
    <script>
    try {{
        var msg = new SpeechSynthesisUtterance("{text}");
        msg.lang='zh-CN';
        msg.rate=1.15;
        window.speechSynthesis.speak(msg);
    }} catch(e) {{}}
    </script>
    """
    st.components.v1.html(js, height=0)


@st.cache_resource
def init_exchange():
    return ccxt.okx({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})


def fetch_data():
    ex = init_exchange()
    for _ in range(3):
        try:
            bars = ex.fetch_ohlcv(INSTRUMENT, timeframe=TIMEFRAME, limit=LIMIT)
            if not bars or len(bars) < 60:
                continue
            df = pd.DataFrame(bars, columns=['ts','open','high','low','close','vol'])
            df['ts_dt'] = pd.to_datetime(df['ts'], unit='ms')
            df = df.dropna()
            return df
        except Exception:
            time.sleep(1)
    return None


def trend(df):
    ma20 = df['close'].rolling(20).mean()
    ma50 = df['close'].rolling(50).mean()
    if ma20.iloc[-1] > ma50.iloc[-1]:
        return "up"
    if ma20.iloc[-1] < ma50.iloc[-1]:
        return "down"
    return "side"


def trend_strength(df):
    ma20 = df['close'].rolling(20).mean()
    ma50 = df['close'].rolling(50).mean()
    slope = ma20.diff().iloc[-5:].mean()
    distance = abs(ma20.iloc[-1] - ma50.iloc[-1])
    volatility = df['close'].pct_change().rolling(20).std().iloc[-1]
    if pd.isna(volatility) or volatility == 0:
        volatility = 0.0001
    return round((distance / volatility) * abs(slope), 2)


def is_bottom_divergence(df):
    if len(df) < 60 or trend(df) != "down":
        return False
    low1 = df['low'].iloc[-1]
    low2 = df['low'].iloc[-2]
    avg = df['vol'].iloc[-50:-1].median()
    momentum = df['close'].diff().rolling(3).mean().iloc[-1]
    return low1 < low2 and df['vol'].iloc[-1] < avg * 0.6 and df['close'].iloc[-1] > df['close'].iloc[-2] and momentum > 0


def is_top_divergence(df):
    if len(df) < 60 or trend(df) != "up":
        return False
    high1 = df['high'].iloc[-1]
    high2 = df['high'].iloc[-2]
    avg = df['vol'].iloc[-50:-1].median()
    momentum = df['close'].diff().rolling(3).mean().iloc[-1]
    return high1 > high2 and df['vol'].iloc[-1] < avg * 0.6 and df['close'].iloc[-1] < df['close'].iloc[-2] and momentum < 0


def detect_fake_breakout(df, res, sup):
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    avg = df['vol'].iloc[-50:-1].median()
    if curr['close'] > res and curr['vol'] < avg * 0.8 and prev['close'] < res:
        return "假突破（无量上破）"
    if curr['close'] < sup and curr['vol'] < avg * 0.8 and prev['close'] > sup:
        return "假跌破（无量下破）"
    return None


def detect_accumulation(df):
    if len(df) < 80:
        return False
    price_range = df['high'].iloc[-30:].max() - df['low'].iloc[-30:].min()
    avg_vol = df['vol'].iloc[-50:-1].median()
    recent_vol = df['vol'].iloc[-10:].mean()
    return price_range / df['close'].iloc[-1] < 0.02 and recent_vol > avg_vol * 1.3


def detect_whale_pump(df):
    if len(df) < 60:
        return False
    return df['vol'].iloc[-3:].mean() > df['vol'].iloc[-50:-3].mean() * 2 and df['close'].iloc[-1] > df['close'].iloc[-5]


def detect_dump(df):
    avg = df['vol'].iloc[-50:-1].mean()
    return df['vol'].iloc[-1] > avg * 2 and df['close'].iloc[-1] < df['close'].iloc[-2] * 0.99


def ai_engine(df):
    curr = df.iloc[-1]
    price = curr['close']
    avg_vol = df['vol'].iloc[-50:-1].median()
    if pd.isna(avg_vol) or avg_vol == 0:
        avg_vol = 1
    vol_ratio = curr['vol'] / avg_vol

    res = df['high'].iloc[-40:-1].quantile(0.95)
    sup = df['low'].iloc[-40:-1].quantile(0.05)

    strength = trend_strength(df)
    fake = detect_fake_breakout(df, res, sup)
    resonance = trend(df)

    status = {"action": "静观其变", "motto": "等待信号", "color": "#121212", "voice": "AI扫描中"}

    if detect_whale_pump(df):
        status.update({"action": "庄家拉升", "motto": "资金异动", "color": "#1B5E20", "voice": "检测庄家拉升"})
    if detect_dump(df):
        status.update({"action": "砸盘预警", "motto": "注意风险", "color": "#B71C1C", "voice": "检测砸盘预警"})
    if fake:
        status.update({"action": fake, "motto": "无量信号", "color": "#4A148C", "voice": fake})
    if vol_ratio > 1.6 and price > res:
        status.update({"action": "直接开多", "motto": "放量突破", "color": "#1B5E20", "voice": "放量起涨做多"})
    elif vol_ratio > 1.6 and price < sup:
        status.update({"action": "直接开空", "motto": "放量跌破", "color": "#B71C1C", "voice": "放量跌破做空"})
    elif vol_ratio < 0.6 and price <= sup*1.002:
        status.update({"action": "准备多", "motto": "缩量回踩", "color": "#0D47A1", "voice": "缩量回踩观察"})
    elif vol_ratio < 0.6 and price >= res*0.998:
        status.update({"action": "准备空", "motto": "缩量反弹", "color": "#E65100", "voice": "缩量反弹观察"})

    return status, resonance, strength


def render():
    df = fetch_data()
    if df is None:
        st.warning("数据异常：无法获取K线")
        return

    status, resonance, strength = ai_engine(df)

    now = time.time()
    key = status["action"]

    if (
        st.session_state.signal_memory["last_key"] != key
        and status["voice"]
        and now - st.session_state.last_voice_time > 20
    ):
        ai_voice_broadcast(status["voice"])
        st.session_state.signal_memory["last_key"] = key
        st.session_state.last_voice_time = now

    st.markdown(f"""
    <div style="background:{status['color']};padding:25px;border-radius:15px;text-align:center;">
        <h1 style="color:white">{status['action']}</h1>
        <h3 style="color:#FFD700">{status['motto']}</h3>
    </div>
    """, unsafe_allow_html=True)

    st.write("多周期共振:", resonance)
    st.write("趋势强度:", strength)

    if detect_accumulation(df):
        st.success("检测主力吸筹")
    if detect_whale_pump(df):
        st.success("庄家拉升")
    if detect_dump(df):
        st.error("砸盘预警")

    fig = go.Figure(data=[go.Candlestick(
        x=df['ts_dt'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close']
    )])

    fig.add_hline(y=df['high'].iloc[-40:-1].quantile(0.95), line_dash="dash")
    fig.add_hline(y=df['low'].iloc[-40:-1].quantile(0.05), line_dash="dash")

    if is_bottom_divergence(df):
        fig.add_trace(go.Scatter(
            x=[df.iloc[-1]['ts_dt']],
            y=[df.iloc[-1]['low']],
            mode="markers",
            marker=dict(symbol="triangle-up", size=16),
            name="底背离"
        ))

    if is_top_divergence(df):
        fig.add_trace(go.Scatter(
            x=[df.iloc[-1]['ts_dt']],
            y=[df.iloc[-1]['high']],
            mode="markers",
            marker=dict(symbol="triangle-down", size=16),
            name="顶背离"
        ))

    fig.update_layout(template="plotly_dark", height=500)
    st.plotly_chart(fig, width="stretch")


render()
