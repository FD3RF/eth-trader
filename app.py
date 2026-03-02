import streamlit as st
import requests
import pandas as pd
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots  # 必须导入！
from streamlit_autorefresh import st_autorefresh
import os
from datetime import datetime
import time
import numpy as np

# ==========================
# 基础配置
# ==========================
st.set_page_config(layout="wide")
st.title("顶级模型 | 高质量单（胜率优先）【优化版】")

SYMBOL = "ETH-USDT-SWAP"
HISTORY_FILE = "history.csv"
LAST_SIGNAL_FILE = "last_signal.txt"

st_autorefresh(interval=8000, key="refresh")


# ==========================
# 数据获取（带缓存和重试）
# ==========================
@st.cache_data(ttl=5)
def get_data():
    url = "https://www.okx.com/api/v5/market/candles"
    params = {"instId": SYMBOL, "bar": "5m", "limit": 300}
    retries = 3
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=5, params=params)
            j = r.json()
            if "data" in j:
                break
        except Exception as e:
            if attempt == retries - 1:
                st.error(f"数据获取失败: {e}")
                return pd.DataFrame()
            time.sleep(1)
    else:
        return pd.DataFrame()

    if "data" not in j:
        return pd.DataFrame()

    df = pd.DataFrame(j["data"], columns=[
        "ts", "open", "high", "low", "close", "volume",
        "volCcy", "volCcyQuote", "confirm"
    ])
    df["ts"] = pd.to_datetime(df["ts"].astype(int), unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df.sort_values("ts").reset_index(drop=True)


df = get_data()
if df.empty:
    st.error("无法获取数据，请检查网络或API服务")
    st.stop()

if len(df) < 100:
    st.warning("数据量不足，可能影响指标计算")


# ==========================
# 技术指标计算（增强版）
# ==========================
# 基础EMA
df["EMA20"] = ta.trend.ema_indicator(df["close"], window=20)
df["EMA60"] = ta.trend.ema_indicator(df["close"], window=60)

# RSI
df["RSI"] = ta.momentum.rsi(df["close"], window=14)

# ATR（用于动态止损）
df["ATR"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)

# 成交量均线
df["VOL_MA20"] = df["volume"].rolling(window=20).mean()

# 新增：ADX（平均趋向指数，衡量趋势强度）
df["ADX"] = ta.trend.adx(df["high"], df["low"], df["close"], window=14)

# 新增：15分钟EMA（用于多时间框架验证）——由于只有5分钟数据，我们通过重采样模拟15分钟
# 简单方法：取最近3根5分钟K线的EMA20作为15分钟趋势近似
df["EMA15_approx"] = df["EMA20"].rolling(window=3).mean()

# 删除NaN
df = df.dropna().reset_index(drop=True)

latest = df.iloc[-1]
prev = df.iloc[-2]  # 前一根K线，用于形态判断


# ==========================
# 趋势判断（主趋势：EMA20/EMA60连续三期确认 + ADX > 25）
# ==========================
trend = "无"
ema_bull = (df["EMA20"].iloc[-3:] > df["EMA60"].iloc[-3:]).all()
ema_bear = (df["EMA20"].iloc[-3:] < df["EMA60"].iloc[-3:]).all()

# 要求ADX大于25，表明趋势存在
if ema_bull and latest["ADX"] > 25:
    trend = "多"
elif ema_bear and latest["ADX"] > 25:
    trend = "空"

# 多时间框架验证：15分钟趋势（近似）必须与主趋势一致
if trend == "多":
    tf_ok = latest["EMA15_approx"] > df["EMA15_approx"].iloc[-3:].mean()  # 简单判断15分钟EMA向上
elif trend == "空":
    tf_ok = latest["EMA15_approx"] < df["EMA15_approx"].iloc[-3:].mean()
else:
    tf_ok = False


# ==========================
# 关键结构（最近20根K线的最高/最低）
# ==========================
resistance = df["high"].iloc[-20:].max()
support = df["low"].iloc[-20:].min()


def is_retest(price, level, tolerance=0.001):
    """精确回踩判断，容差0.1%"""
    return price <= level * (1 + tolerance) and price >= level * (1 - tolerance)


def is_rejection_candle(row, direction):
    """
    识别拒绝K线形态（pin bar或吞没）
    direction: 'bull' 或 'bear'
    """
    body = abs(row["close"] - row["open"])
    lower = min(row["close"], row["open"]) - row["low"]
    upper = row["high"] - max(row["close"], row["open"])
    if direction == "bull":
        # 多头拒绝：下影线长，且收盘在顶部1/3
        return lower > body * 1.5 and row["close"] > (row["high"] + row["low"]) / 2
    else:
        # 空头拒绝：上影线长，且收盘在底部1/3
        return upper > body * 1.5 and row["close"] < (row["high"] + row["low"]) / 2


# 判断是否出现拒绝形态（根据趋势方向）
rejection = False
if trend == "多" and is_rejection_candle(latest, "bull"):
    rejection = True
elif trend == "空" and is_rejection_candle(latest, "bear"):
    rejection = True

# 同时要求前一根K线收盘在关键位另一侧，形成“测试+拒绝”
price_tested = False
if trend == "多":
    # 价格测试支撑，且前一根K线收盘在支撑下方（制造恐慌）
    if is_retest(latest["low"], support) and prev["close"] < support:
        price_tested = True
elif trend == "空":
    if is_retest(latest["high"], resistance) and prev["close"] > resistance:
        price_tested = True

# 结构有效综合条件
structure_ok = rejection and price_tested

# 动能过滤（RSI收窄）
rsi = latest["RSI"]
if trend == "多":
    rsi_ok = (55 <= rsi <= 65)  # 强势但不超买
elif trend == "空":
    rsi_ok = (35 <= rsi <= 45)  # 弱势但不超卖
else:
    rsi_ok = False

# 盈亏比计算（止损用ATR*1.2，止盈用ATR*2.5，提高盈亏比）
entry = latest["close"]
atr = latest["ATR"]
if trend == "多":
    stop = entry - atr * 1.2
    tp = entry + atr * 2.5
else:
    stop = entry + atr * 1.2
    tp = entry - atr * 2.5

# 盈亏比
if (entry - stop) != 0:
    rr = abs((tp - entry) / (entry - stop))
else:
    rr = 0

# 成交量放大（1.5倍）
volume_ok = latest["volume"] > df["VOL_MA20"].iloc[-1] * 1.5

# 最终信号（所有条件严格满足）
signal = None
if trend != "无" and structure_ok and volume_ok and rsi_ok and rr >= 2 and tf_ok:
    signal = trend

quality = "高" if signal else "无"


# ==========================
# 防重复触发
# ==========================
def already_signaled(sig, price):
    if not os.path.exists(LAST_SIGNAL_FILE):
        return False
    try:
        with open(LAST_SIGNAL_FILE, "r") as f:
            content = f.read().split(",")
            last_sig = content[0]
            last_price = float(content[1])
            price_diff_pct = abs(price - last_price) / last_price
            return last_sig == sig and price_diff_pct < 0.001
    except:
        return False


def mark_signal(sig, price):
    with open(LAST_SIGNAL_FILE, "w") as f:
        f.write(f"{sig},{price}")


# ==========================
# 历史记录管理
# ==========================
def load_history():
    if os.path.exists(HISTORY_FILE):
        return pd.read_csv(HISTORY_FILE)
    return pd.DataFrame(columns=["time", "direction", "entry", "stop", "tp", "result"])


def save_history_record(row):
    dfh = load_history()
    dfh = pd.concat([dfh, pd.DataFrame([row])], ignore_index=True)
    dfh = dfh.tail(5000)
    dfh.to_csv(HISTORY_FILE, index=False)


def update_history_results():
    dfh = load_history()
    if dfh.empty:
        return dfh
    current_price = latest["close"]
    for idx, row in dfh.iterrows():
        if pd.isna(row.get("result")) or row["result"] == "":
            if row["direction"] == "多":
                if current_price >= row["tp"]:
                    dfh.at[idx, "result"] = "win"
                elif current_price <= row["stop"]:
                    dfh.at[idx, "result"] = "lose"
            else:
                if current_price <= row["tp"]:
                    dfh.at[idx, "result"] = "win"
                elif current_price >= row["stop"]:
                    dfh.at[idx, "result"] = "lose"
    dfh.to_csv(HISTORY_FILE, index=False)
    return dfh


history_df = update_history_results()

# ==========================
# 生成新信号
# ==========================
if signal and not already_signaled(signal, entry):
    new_record = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "direction": signal,
        "entry": round(entry, 4),
        "stop": round(stop, 4),
        "tp": round(tp, 4),
        "result": ""
    }
    save_history_record(new_record)
    mark_signal(signal, entry)


# ==========================
# 绘图（增加ADX子图）
# ==========================
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                    row_heights=[0.7, 0.3])

# 主图：K线 + EMA
fig.add_trace(go.Candlestick(
    x=df["ts"], open=df["open"], high=df["high"],
    low=df["low"], close=df["close"], name="K线"
), row=1, col=1)
fig.add_trace(go.Scatter(x=df["ts"], y=df["EMA20"], line=dict(color="blue", width=1), name="EMA20"), row=1, col=1)
fig.add_trace(go.Scatter(x=df["ts"], y=df["EMA60"], line=dict(color="orange", width=1), name="EMA60"), row=1, col=1)
fig.add_hline(y=support, line_dash="dash", line_color="green", annotation_text="支撑", row=1, col=1)
fig.add_hline(y=resistance, line_dash="dash", line_color="red", annotation_text="阻力", row=1, col=1)

# 子图：ADX
fig.add_trace(go.Scatter(x=df["ts"], y=df["ADX"], line=dict(color="purple", width=1), name="ADX"), row=2, col=1)
fig.add_hline(y=25, line_dash="dot", line_color="gray", annotation_text="趋势阈值", row=2, col=1)

fig.update_layout(title=f"{SYMBOL} 5分钟图（优化版）", template="plotly_dark", height=700)
st.plotly_chart(fig, use_container_width=True)


# ==========================
# 状态面板
# ==========================
st.subheader("📊 当前市场状态（优化版）")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("趋势方向", trend)
with col2:
    st.metric("RSI(14)", f"{rsi:.2f}")
with col3:
    st.metric("ADX", f"{latest['ADX']:.2f}")
with col4:
    st.metric("ATR", f"{atr:.4f}")
with col5:
    st.metric("信号质量", quality)

# 详细条件检查
st.write("**当前条件检查**（✅为满足，❌为不满足）：")
checks = {
    "主趋势明确": trend != "无",
    "ADX > 25": latest["ADX"] > 25,
    "多时间框架一致": tf_ok,
    "精确回踩+拒绝形态": structure_ok,
    "RSI合适区间": rsi_ok,
    "成交量放大1.5倍": volume_ok,
    "盈亏比≥2": rr >= 2
}
for name, cond in checks.items():
    icon = "✅" if cond else "❌"
    st.write(f"{icon} {name}")

if signal:
    if signal == "多":
        st.success("📈 高质量做多信号（优化版）")
    else:
        st.error("📉 高质量做空信号（优化版）")
    col_e, col_s, col_t = st.columns(3)
    with col_e:
        st.metric("入场价", f"{entry:.4f}")
    with col_s:
        delta = -atr * 1.2 if signal == "多" else atr * 1.2
        st.metric("止损价", f"{stop:.4f}", delta=f"{delta:.4f}")
    with col_t:
        delta2 = atr * 2.5 if signal == "多" else -atr * 2.5
        st.metric("止盈价", f"{tp:.4f}", delta=f"{delta2:.4f}")
    st.info(f"预期盈亏比：{rr:.2f} : 1")
else:
    st.warning("⏳ 暂无高质量交易机会，等待更严格的条件")


# ==========================
# 历史记录与胜率
# ==========================
st.subheader("📜 历史信号记录（优化版）")
if not history_df.empty:
    total = len(history_df)
    completed = history_df[history_df["result"].isin(["win", "lose"])]
    wins = len(completed[completed["result"] == "win"])
    losses = len(completed[completed["result"] == "lose"])
    win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0

    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    with col_stat1:
        st.metric("总信号数", total)
    with col_stat2:
        st.metric("已完成", len(completed))
    with col_stat3:
        st.metric("胜率", f"{win_rate:.1f}%" if not np.isnan(win_rate) else "N/A")
    with col_stat4:
        st.metric("持仓中", len(history_df[history_df["result"] == ""]))

    st.dataframe(history_df.tail(20)[["time", "direction", "entry", "stop", "tp", "result"]])
else:
    st.info("暂无历史记录")
