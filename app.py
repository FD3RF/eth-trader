import streamlit as st
import pandas as pd
import requests
import numpy as np
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

# ==================== 终极配置 ====================
SYMBOL = "ETH-USDT"
INTERVAL = "5m"
LIMIT = 100
VOL_MULTIPLIER = 1.2
PROB_THRESHOLD = 50
# =================================================

def safe_request(url):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None

def get_candles():
    url = f"https://www.okx.com/api/v5/market/candles?instId={SYMBOL}&bar={INTERVAL}&limit={LIMIT}"
    data = safe_request(url)
    if not data or data.get("code") != "0":
        return None

    df = pd.DataFrame(data["data"], columns=[
        "ts","o","h","l","c","v","volCcy","volCcyQuote","confirm"
    ])[::-1].reset_index(drop=True)

    df["time"] = pd.to_datetime(df["ts"].astype(float), unit="ms")
    for col in ["o","h","l","c","v"]:
        df[col] = pd.to_numeric(df[col])

    # 指标计算
    df["ema_fast"] = df["c"].ewm(span=12, adjust=False).mean()
    df["ema_slow"] = df["c"].ewm(span=26, adjust=False).mean()

    delta = df["c"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"] = df["rsi"].fillna(100)

    exp12 = df["c"].ewm(span=12, adjust=False).mean()
    exp26 = df["c"].ewm(span=26, adjust=False).mean()
    df["macd"] = exp12 - exp26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    df["bb_mid"] = df["c"].rolling(20).mean()
    df["bb_std"] = df["c"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]

    df["vol_ma"] = df["v"].rolling(10).mean()

    high_low = df["h"] - df["l"]
    high_close = (df["h"] - df["c"].shift()).abs()
    low_close = (df["l"] - df["c"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1, skipna=True)
    df["atr"] = tr.rolling(14).mean()
    df["atr"] = df["atr"].replace(0, np.nan)

    return df

def get_ls_ratio():
    url = "https://www.okx.com/api/v5/rubik/stat/contracts/long-short-account-ratio?instId=ETH-USDT&period=5m"
    data = safe_request(url)
    if data and data.get("code") == "0" and data["data"]:
        try:
            return float(data["data"][0][1])
        except:
            pass
    return 1.0

def generate_signal(df, ls_ratio):
    if df is None or len(df) < 50:
        return 50, 0, "数据不足", None, None, "数据不足"

    last = df.iloc[-1]
    required_cols = ["ema_fast", "ema_slow", "rsi", "macd_hist", "bb_upper", "bb_lower", "vol_ma", "v", "atr"]
    if any(pd.isna(last[col]) for col in required_cols):
        return 50, 0, "指标计算中", None, None, "指标暂未就绪"

    score = 50
    reasons = []
    trend = 1 if last["ema_fast"] > last["ema_slow"] else -1

    if last["ema_fast"] > last["ema_slow"]:
        score += 20
        reasons.append("EMA多")
    else:
        score -= 18
        reasons.append("EMA空")

    rsi = last["rsi"]
    if 30 < rsi < 70:
        score += 10
        reasons.append("RSI中性")
    elif rsi > 75:
        score -= 10
        reasons.append("RSI超买")
    elif rsi < 25:
        score += 5
        reasons.append("RSI超卖")

    if last["macd_hist"] > 0:
        score += 12
        reasons.append("MACD多头")
    else:
        score -= 12
        reasons.append("MACD空头")

    extreme = False
    if last["c"] > last["bb_upper"] and last["v"] > last["vol_ma"] * VOL_MULTIPLIER:
        extreme = True
        score += 20
        reasons.append("突破上轨放量")
    elif last["c"] < last["bb_lower"] and last["v"] > last["vol_ma"] * VOL_MULTIPLIER:
        extreme = True
        score += 20
        reasons.append("跌破下轨放量")

    if ls_ratio < 0.95:
        score += 8
        reasons.append("多空极空")
    elif ls_ratio > 1.05:
        score -= 8
        reasons.append("多空极多")

    prob = max(min(score, 95), 5)

    direction = 0
    if trend == 1 and extreme and prob > PROB_THRESHOLD:
        direction = 1
    elif trend == -1 and extreme and prob < PROB_THRESHOLD:
        direction = -1

    atr = last["atr"]
    if pd.isna(atr) or atr <= 0:
        sl = tp = None
        entry = "ATR无效"
        return prob, direction, entry, sl, tp, "ATR异常"

    if direction == 1:
        sl = last["c"] - atr * 1.5
        tp = last["c"] + atr * 2.5
        entry = f"{last['c'] - atr * 0.5:.1f} ~ {last['c'] + atr * 0.5:.1f}"
    elif direction == -1:
        sl = last["c"] + atr * 1.5
        tp = last["c"] - atr * 2.5
        entry = f"{last['c'] - atr * 0.5:.1f} ~ {last['c'] + atr * 0.5:.1f}"
    else:
        sl = tp = None
        entry = "观望"

    reason = " | ".join(reasons) if reasons else "无明显信号"
    return prob, direction, entry, sl, tp, reason

def main():
    st.set_page_config(
        layout="wide",
        page_title="ETH 5分钟波段信号",
        theme={
            "base": "dark",
            "primaryColor": "#c98bdb",
            "backgroundColor": "#0e1117",
            "secondaryBackgroundColor": "#262730",
            "textColor": "#fafafa",
            "font": "sans serif"
        }
    )
    st.title("📈 5分钟 ETH 波段信号（终极版）")

    with st.spinner("正在获取市场数据..."):
        df = get_candles()
        ls = get_ls_ratio()

    if df is None or len(df) < 50:
        st.error("❌ 无法获取足够K线数据，请检查网络或稍后重试")
        return

    prob, direction, entry, sl, tp, reason = generate_signal(df, ls)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("胜率评分", f"{prob:.1f}%")
    with col2:
        dir_text = "📈 多" if direction == 1 else "📉 空" if direction == -1 else "⚖️ 观望"
        st.metric("方向", dir_text)
    with col3:
        st.metric("入场区间", entry)

    st.caption(f"**信号理由**: {reason}")

    if sl is not None and tp is not None:
        col4, col5 = st.columns(2)
        with col4:
            st.metric("止损", f"{sl:.2f}")
        with col5:
            st.metric("止盈", f"{tp:.2f}")
    else:
        st.info("当前无明确交易信号，建议观望")

    current_price = df.iloc[-1]["c"]
    st.metric("当前价格", f"{current_price:.2f}")

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["time"], open=df["o"], high=df["h"], low=df["l"], close=df["c"], name="K线"
    ))
    fig.add_trace(go.Scatter(x=df["time"], y=df["bb_upper"], line=dict(color='gray', width=1), name="布林上轨"))
    fig.add_trace(go.Scatter(x=df["time"], y=df["bb_lower"], line=dict(color='gray', width=1), name="布林下轨", fill='tonexty', fillcolor='rgba(128,128,128,0.2)'))
    fig.add_trace(go.Scatter(x=df["time"], y=df["ema_fast"], line=dict(color='orange', width=1), name="EMA12"))
    fig.add_trace(go.Scatter(x=df["time"], y=df["ema_slow"], line=dict(color='blue', width=1), name="EMA26"))

    fig.update_layout(height=600, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    st_autorefresh(interval=60000, key="auto_refresh")

if __name__ == "__main__":
    main()
