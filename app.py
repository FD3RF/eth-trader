import streamlit as st
import pandas as pd
import requests
import numpy as np
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from typing import Tuple, Optional
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== 可调参数（支持侧边栏调整） ====================
DEFAULT_VOL_MULTIPLIER = 1.0      # 放量倍数：1.0 表示只要有量就算
DEFAULT_PROB_THRESHOLD = 45        # 评分阈值（多>45，空<55）
DEFAULT_SL_ATR = 1.5               # 止损倍数
DEFAULT_TP_ATR = 2.5               # 止盈倍数
DEFAULT_ENTRY_ATR_OFFSET = 0.5     # 入场区间半宽（ATR倍数）
# ==================================================================

# 固定参数
SYMBOL = "ETH-USDT"
INTERVAL = "5m"
LIMIT = 100

def safe_request(url: str) -> Optional[dict]:
    """安全的 API 请求，异常时返回 None"""
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.warning(f"Request failed: {e}")
    return None

@st.cache_data(ttl=60)
def get_candles() -> Optional[pd.DataFrame]:
    """获取K线数据并计算所有指标"""
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

def get_ls_ratio() -> float:
    """获取多空比，失败返回 1.0"""
    url = "https://www.okx.com/api/v5/rubik/stat/contracts/long-short-account-ratio?instId=ETH-USDT&period=5m"
    data = safe_request(url)
    if data and data.get("code") == "0" and data["data"]:
        try:
            return float(data["data"][0][1])
        except (ValueError, IndexError, TypeError):
            pass
    return 1.0

def calculate_score_and_reasons(last: pd.Series, ls_ratio: float,
                                vol_mult: float) -> Tuple[int, list, bool]:
    """
    计算多因子评分和信号理由
    因子权重说明：
      - EMA: ±20  (趋势主导)
      - RSI: +10/±10/±5 (中性/超买/超卖)
      - MACD: ±12 (动能确认)
      - 突破: +20 (极端行情加分)
      - 多空比: ±8 (情绪逆向)
    """
    score = 50
    reasons = []

    if last["ema_fast"] > last["ema_slow"]:
        score += 20
        reasons.append("EMA多")
    else:
        score -= 18
        reasons.append("EMA空")

    rsi = last["rsi"]
    if 30 <= rsi <= 70:
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
    if last["c"] > last["bb_upper"] and last["v"] > last["vol_ma"] * vol_mult:
        extreme = True
        score += 20
        reasons.append("突破上轨")
    elif last["c"] < last["bb_lower"] and last["v"] > last["vol_ma"] * vol_mult:
        extreme = True
        score += 20
        reasons.append("跌破下轨")

    if ls_ratio < 0.95:
        score += 8
        reasons.append("多空极空")
    elif ls_ratio > 1.05:
        score -= 8
        reasons.append("多空极多")

    return max(min(score, 95), 5), reasons, extreme

def generate_signal(df: pd.DataFrame, ls_ratio: float,
                    vol_mult: float, prob_thresh: float,
                    sl_atr: float, tp_atr: float, entry_offset: float) -> dict:
    if df is None or len(df) < 50:
        return {"direction": 0, "trigger": None, "sl": None, "tp": None,
                "entry_range": "数据不足", "prob": 50, "reason": "数据不足"}

    last = df.iloc[-1]
    required = ["ema_fast", "ema_slow", "rsi", "macd_hist", "bb_upper", "bb_lower", "vol_ma", "v", "atr"]
    if any(pd.isna(last[col]) for col in required):
        return {"direction": 0, "trigger": None, "sl": None, "tp": None,
                "entry_range": "指标计算中", "prob": 50, "reason": "指标暂未就绪"}

    prob, reasons, extreme = calculate_score_and_reasons(last, ls_ratio, vol_mult)
    trend = 1 if last["ema_fast"] > last["ema_slow"] else -1

    direction = 0
    if trend == 1 and prob > prob_thresh:
        direction = 1
    elif trend == -1 and prob < (100 - prob_thresh):
        direction = -1

    atr = last["atr"]
    if pd.isna(atr) or atr <= 0:
        return {"direction": 0, "trigger": None, "sl": None, "tp": None,
                "entry_range": "ATR无效", "prob": prob, "reason": "ATR异常"}

    if direction == 1:
        trigger = last["bb_upper"]
        sl = trigger - atr * sl_atr
        tp = trigger + atr * tp_atr
        entry_low = trigger - atr * entry_offset
        entry_high = trigger + atr * entry_offset
    elif direction == -1:
        trigger = last["bb_lower"]
        sl = trigger + atr * sl_atr
        tp = trigger - atr * tp_atr
        entry_low = trigger - atr * entry_offset
        entry_high = trigger + atr * entry_offset
    else:
        trigger = None
        sl = tp = None
        entry_low = entry_high = None

    entry_range = f"{entry_low:.1f} ~ {entry_high:.1f}" if entry_low else "观望"
    reason_str = " | ".join(reasons) if reasons else "无明显信号"

    return {
        "direction": direction,
        "trigger": trigger,
        "sl": sl,
        "tp": tp,
        "entry_range": entry_range,
        "prob": prob,
        "reason": reason_str
    }

def main():
    st.set_page_config(layout="wide", page_title="ETH 高频信号·清晰版")
    st.title("📈 5分钟 ETH 高频信号系统（逻辑清晰版）")

    with st.sidebar:
        st.header("⚙️ 参数调节")
        vol_mult = st.slider("放量倍数", 0.8, 2.0, 1.0, 0.1,
                             help="1.0 表示只要有量就算")
        prob_thresh = st.slider("评分阈值", 30, 60, 45, 1,
                                help="多>阈值，空<100-阈值")
        sl_atr = st.slider("止损 (ATR倍数)", 1.0, 3.0, 1.5, 0.1)
        tp_atr = st.slider("止盈 (ATR倍数)", 1.5, 4.0, 2.5, 0.1)
        entry_offset = st.slider("入场区间半宽 (ATR倍数)", 0.2, 1.0, 0.5, 0.1)

    with st.spinner("获取市场数据..."):
        df = get_candles()
        ls = get_ls_ratio()

    if df is None or len(df) < 50:
        st.error("❌ 无法获取足够K线数据，请检查网络")
        return

    signal = generate_signal(df, ls, vol_mult, prob_thresh,
                             sl_atr, tp_atr, entry_offset)

    if "last_direction" not in st.session_state:
        st.session_state.last_direction = 0
    current_dir = signal["direction"]

    if current_dir != 0 and current_dir != st.session_state.last_direction:
        if st.session_state.last_direction != 0:
            st.warning(f"⚠️ 信号反转：建议平仓当前{'多头' if st.session_state.last_direction==1 else '空头'}，反向开仓")
    if current_dir == 0 and st.session_state.last_direction != 0:
        st.warning(f"⚠️ 信号消失：建议平仓当前{'多头' if st.session_state.last_direction==1 else '空头'}")
    st.session_state.last_direction = current_dir

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("胜率评分", f"{signal['prob']}%")
    with col2:
        dir_text = "📈 多" if current_dir == 1 else "📉 空" if current_dir == -1 else "⚖️ 观望"
        st.metric("方向", dir_text)
    with col3:
        st.metric("入场区间", signal["entry_range"])

    st.caption(f"**信号理由**: {signal['reason']}")

    if signal["sl"] and signal["tp"]:
        col4, col5 = st.columns([1,1])
        with col4:
            st.metric("止损", f"{signal['sl']:.2f}")
        with col5:
            st.metric("止盈", f"{signal['tp']:.2f}")
    else:
        st.info("当前无明确交易信号，建议观望")

    last_price = df.iloc[-1]["c"]
    col6, col7 = st.columns(2)
    with col6:
        st.metric("当前价格", f"{last_price:.2f}")
    with col7:
        trigger_disp = f"{signal['trigger']:.2f}" if signal["trigger"] else "--"
        st.metric("触发价", trigger_disp)

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["time"], open=df["o"], high=df["h"], low=df["l"], close=df["c"], name="K线"
    ))
    fig.add_trace(go.Scatter(x=df["time"], y=df["bb_upper"], line=dict(color='gray', width=1), name="布林上轨"))
    fig.add_trace(go.Scatter(x=df["time"], y=df["bb_lower"], line=dict(color='gray', width=1), name="布林下轨",
                             fill='tonexty', fillcolor='rgba(128,128,128,0.2)'))
    fig.add_trace(go.Scatter(x=df["time"], y=df["ema_fast"], line=dict(color='orange', width=1), name="EMA12"))
    fig.add_trace(go.Scatter(x=df["time"], y=df["ema_slow"], line=dict(color='blue', width=1), name="EMA26"))
    if signal["trigger"]:
        fig.add_hline(y=signal["trigger"], line_dash="dash", line_color="red",
                      annotation_text="触发位", annotation_position="bottom right")
    fig.update_layout(height=600, xaxis_rangeslider_visible=False)

    # 使用 width='stretch' 替代 use_container_width，消除警告
    st.plotly_chart(fig, use_container_width=True)  # 当前版本仍支持
    # 若要彻底消除警告，可替换为下一行：
    # st.plotly_chart(fig, width='stretch')

    st_autorefresh(interval=60000, key="auto_refresh")

if __name__ == "__main__":
    main()
