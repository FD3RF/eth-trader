import streamlit as st
import pandas as pd
import ccxt
import plotly.graph_objects as go
import time
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from streamlit_autorefresh import st_autorefresh

# ========== 页面设置 ==========
st.set_page_config(page_title="ETH AI 终极盯盘", layout="wide")
st_autorefresh(interval=5000, key="refresh")

# ========== 状态初始化 ==========
if "signal_memory" not in st.session_state:
    st.session_state.signal_memory = {"last_key": None}

if "last_voice_time" not in st.session_state:
    st.session_state.last_voice_time = 0

MODEL_FILE = "ai_model.pkl"
INSTRUMENT = "ETH/USDT:USDT"
TIMEFRAME = "5m"
LIMIT = 150


# ========== 语音播报 ==========
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


# ========== 交易所 ==========
@st.cache_resource
def init_exchange():
    return ccxt.okx({"enableRateLimit": True, "options": {"defaultType": "swap"}})


def fetch_data():
    ex = init_exchange()
    for _ in range(3):
        try:
            bars = ex.fetch_ohlcv(INSTRUMENT, timeframe=TIMEFRAME, limit=LIMIT)
            ticker = ex.fetch_ticker(INSTRUMENT)
            if not bars or len(bars) < 60:
                continue

            df = pd.DataFrame(bars, columns=["ts", "open", "high", "low", "close", "vol"])
            df["ts_dt"] = pd.to_datetime(df["ts"], unit="ms")
            df = df.dropna()
            return df, ticker
        except Exception:
            time.sleep(1)
    return None, None


# ========== 趋势 ==========
def trend(df):
    ma20 = df["close"].rolling(20).mean()
    ma50 = df["close"].rolling(50).mean()
    if ma20.iloc[-1] > ma50.iloc[-1]:
        return "up"
    if ma20.iloc[-1] < ma50.iloc[-1]:
        return "down"
    return "side"


def trend_strength(df):
    ma20 = df["close"].rolling(20).mean()
    ma50 = df["close"].rolling(50).mean()
    slope = ma20.diff().iloc[-5:].mean()
    distance = abs(ma20.iloc[-1] - ma50.iloc[-1])
    volatility = df["close"].pct_change().rolling(20).std().iloc[-1]
    volatility = volatility if pd.notna(volatility) and volatility > 0 else 0.0001
    strength = (distance / volatility) * abs(slope)
    return round(strength, 2)


# ========== 背离 ==========
def is_bottom_divergence(df):
    if len(df) < 60 or trend(df) != "down":
        return False

    low1 = df["low"].iloc[-1]
    low2 = df["low"].iloc[-2]
    vol1 = df["vol"].iloc[-1]
    avg = df["vol"].iloc[-50:-1].median()
    momentum = df["close"].diff().rolling(3).mean().iloc[-1]

    return low1 < low2 and vol1 < avg * 0.6 and df["close"].iloc[-1] > df["close"].iloc[-2] and momentum > 0


def is_top_divergence(df):
    if len(df) < 60 or trend(df) != "up":
        return False

    high1 = df["high"].iloc[-1]
    high2 = df["high"].iloc[-2]
    vol1 = df["vol"].iloc[-1]
    avg = df["vol"].iloc[-50:-1].median()
    momentum = df["close"].diff().rolling(3).mean().iloc[-1]

    return high1 > high2 and vol1 < avg * 0.6 and df["close"].iloc[-1] < df["close"].iloc[-2] and momentum < 0


# ========== 假突破 ==========
def detect_fake_breakout(df, res, sup):
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    avg = df["vol"].iloc[-50:-1].median()

    if curr["close"] > res and curr["vol"] < avg * 0.8 and prev["close"] < res:
        return "fake_up"
    if curr["close"] < sup and curr["vol"] < avg * 0.8 and prev["close"] > sup:
        return "fake_down"
    return None


# ========== 主力与庄家 ==========
def detect_accumulation(df):
    if len(df) < 80:
        return False
    price_range = df["high"].iloc[-30:].max() - df["low"].iloc[-30:].min()
    avg_vol = df["vol"].iloc[-50:-1].median()
    recent_vol = df["vol"].iloc[-10:].mean()
    return (price_range / df["close"].iloc[-1] < 0.02) and (recent_vol > avg_vol * 1.3)


def detect_whale_pump(df):
    if len(df) < 60:
        return False
    return (
        df["vol"].iloc[-3:].mean() > df["vol"].iloc[-50:-3].mean() * 2
        and df["close"].iloc[-1] > df["close"].iloc[-5]
        and df["close"].rolling(20).mean().iloc[-1] > df["close"].rolling(20).mean().iloc[-3]
    )


def detect_dump(df):
    avg = df["vol"].iloc[-50:-1].mean()
    return (
        df["vol"].iloc[-1] > avg * 2
        and df["close"].iloc[-1] < df["close"].iloc[-2] * 0.99
        and df["close"].iloc[-1] < df["close"].rolling(20).mean().iloc[-1]
    )


# ========== 多周期 ==========
def fetch_multi_tf():
    ex = init_exchange()

    def build(b):
        return pd.DataFrame(b, columns=["ts", "open", "high", "low", "close", "vol"])

    df1 = build(ex.fetch_ohlcv(INSTRUMENT, "1m", limit=120))
    df5 = build(ex.fetch_ohlcv(INSTRUMENT, "5m", limit=120))
    df15 = build(ex.fetch_ohlcv(INSTRUMENT, "15m", limit=120))

    for d in (df1, df5, df15):
        d["ts_dt"] = pd.to_datetime(d["ts"], unit="ms")
    return df1, df5, df15


def tf_trend(df):
    ma20 = df["close"].rolling(20).mean()
    ma50 = df["close"].rolling(50).mean()
    if ma20.iloc[-1] > ma50.iloc[-1]:
        return "up"
    if ma20.iloc[-1] < ma50.iloc[-1]:
        return "down"
    return "side"


def multi_tf_resonance():
    df1, df5, df15 = fetch_multi_tf()
    t1, t5, t15 = tf_trend(df1), tf_trend(df5), tf_trend(df15)
    if t1 == t5 == t15 == "up":
        return "bull"
    if t1 == t5 == t15 == "down":
        return "bear"
    return "mixed"


# ========== AI 模型 ==========
def backtest_winrate(df):
    wins = trades = 0
    for i in range(60, len(df) - 5):
        price = df["close"].iloc[i]
        future = df["close"].iloc[i + 5]
        if df["vol"].iloc[i] > df["vol"].iloc[i - 50 : i].mean() * 1.5:
            trades += 1
            if future > price:
                wins += 1
    return round(wins / trades * 100, 2) if trades else 0


def train_ai(df):
    X, y = [], []
    for i in range(60, len(df) - 5):
        vol_ratio = df["vol"].iloc[i] / df["vol"].iloc[i - 50 : i].mean()
        momentum = df["close"].diff().iloc[i - 3 : i].sum()
        X.append([vol_ratio, momentum])
        y.append(1 if df["close"].iloc[i + 5] > df["close"].iloc[i] else 0)

    model = RandomForestClassifier()
    model.fit(X, y)
    joblib.dump(model, MODEL_FILE)
    return model


def load_model(df):
    try:
        return joblib.load(MODEL_FILE)
    except Exception:
        return train_ai(df)


def ai_predict(df):
    model = load_model(df)
    vol_ratio = df["vol"].iloc[-1] / df["vol"].iloc[-50:-1].median()
    momentum = df["close"].diff().iloc[-3:].sum()
    pred = model.predict([[vol_ratio, momentum]])[0]
    prob = model.predict_proba([[vol_ratio, momentum]])[0][1]
    return pred, prob


# ========== AI 引擎 ==========
def ai_engine(df):
    curr = df.iloc[-1]
    price = curr["close"]

    avg_vol = df["vol"].iloc[-50:-1].median()
    avg_vol = df["vol"].iloc[-100:-1].mean() if pd.isna(avg_vol) or avg_vol < 0.0001 else avg_vol

    vol_ratio = curr["vol"] / (avg_vol if avg_vol != 0 else 1)
    res = df["high"].iloc[-40:-1].quantile(0.95)
    sup = df["low"].iloc[-40:-1].quantile(0.05)

    strength = trend_strength(df)
    resonance = multi_tf_resonance()
    winrate = backtest_winrate(df)
    ai_pred, ai_prob = ai_predict(df)

    whale = detect_whale_pump(df)
    dump = detect_dump(df)
    fake = detect_fake_breakout(df, res, sup)
    reversal = trend(df) if trend(df) in ("up", "down") else None

    status = {"action": "AI 扫描中", "motto": "静观其变", "color": "#121212", "voice": None}

    if whale:
        status.update({"action": "庄家拉升", "motto": "资金异动", "color": "#1B5E20", "voice": "检测庄家拉升"})
    if dump:
        status.update({"action": "砸盘预警", "motto": "注意风险", "color": "#B71C1C", "voice": "检测砸盘预警"})
    if fake == "fake_up":
        status.update({"action": "假突破", "motto": "突破无量", "color": "#4A148C", "voice": "假突破警告"})
    if fake == "fake_down":
        status.update({"action": "假跌破", "motto": "跌破无量", "color": "#880E4F", "voice": "假跌破警告"})

    if vol_ratio > 1.6 and price > res:
        status.update({"action": "直接开多", "motto": "放量突破", "color": "#1B5E20", "voice": "放量起涨"})
    elif vol_ratio > 1.6 and price < sup:
        status.update({"action": "直接开空", "motto": "放量跌破", "color": "#B71C1C", "voice": "放量跌破"})
    elif vol_ratio < 0.6 and price <= sup * 1.002:
        status.update({"action": "准备多", "motto": "缩量回踩", "color": "#0D47A1", "voice": "缩量回踩"})
    elif vol_ratio < 0.6 and price >= res * 0.998:
        status.update({"action": "准备空", "motto": "缩量反弹", "color": "#E65100", "voice": "缩量反弹"})

    long_prob = round(ai_prob * 100, 2)
    short_prob = round(100 - long_prob, 2)
    score = min(int((vol_ratio > 1.5) * 25 + (strength > 3) * 25 + (long_prob > 60) * 25 + (resonance == "bull") * 25), 100)

    return status, vol_ratio, res, sup, strength, resonance, winrate, long_prob, short_prob, score, reversal


# ========== UI ==========
def render():
    df, _ = fetch_data()
    if df is None:
        st.warning("数据异常")
        return

    status, vr, res, sup, strength, resonance, winrate, long_prob, short_prob, score, reversal = ai_engine(df)

    now = time.time()
    key = status["action"]

    if st.session_state.signal_memory["last_key"] != key and status["voice"] and now - st.session_state.last_voice_time > 20:
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
    st.write("策略历史胜率:", winrate, "%")
    st.write("多头概率:", long_prob, "%")
    st.write("空头概率:", short_prob, "%")
    st.write("AI评分:", score, "/100")

    if detect_accumulation(df):
        st.success("检测主力吸筹")
    if detect_whale_pump(df):
        st.success("庄家拉升")
    if detect_dump(df):
        st.error("砸盘预警")
    if reversal:
        st.warning(f"趋势方向: {reversal}")

    fig = go.Figure(data=[go.Candlestick(
        x=df["ts_dt"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"]
    )])

    fig.add_hline(y=res, line_dash="dash")
    fig.add_hline(y=sup, line_dash="dash")

    if is_bottom_divergence(df):
        fig.add_trace(go.Scatter(
            x=[df.iloc[-1]["ts_dt"]],
            y=[df.iloc[-1]["low"]],
            mode="markers",
            marker=dict(symbol="triangle-up", size=16),
            name="底背离"
        ))

    if is_top_divergence(df):
        fig.add_trace(go.Scatter(
            x=[df.iloc[-1]["ts_dt"]],
            y=[df.iloc[-1]["high"]],
            mode="markers",
            marker=dict(symbol="triangle-down", size=16),
            name="顶背离"
        ))

    fig.update_layout(template="plotly_dark", height=500)
    st.plotly_chart(fig, width="stretch")


render()
