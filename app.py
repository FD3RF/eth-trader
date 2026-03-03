"""
5分钟趋势合约系统（最终优化版·多空双向）
功能：
- 趋势持续确认（EMA均值上升）
- ADX上升条件
- 推动-回调比评分
- 成交量确认
- 加权评分
- 严格进场条件
- 止损基于结构
- 多空不同阈值
- 所有参数可调
"""

import streamlit as st
import pandas as pd
import ta
import plotly.graph_objects as go
from datetime import datetime
import requests
import time
import os
import json
import numpy as np
from streamlit_autorefresh import st_autorefresh

# ==========================
# 页面配置
# ==========================
st.set_page_config(layout="wide")
st.title("📈 5分钟趋势合约系统（最终优化版）")

SYMBOL = "ETH-USDT-SWAP"
CACHE_FILE_5M = "market_data_5m_cache.csv"
CACHE_FILE_15M = "market_data_15m_cache.csv"
SIGNAL_HISTORY_FILE = "signal_history.json"
ACCOUNT_FILE = "account.json"

st_autorefresh(interval=5000, key="refresh")

# ==========================
# 侧边栏参数
# ==========================
st.sidebar.header("⚙️ 策略参数")

# 基础指标
ema_fast = st.sidebar.slider("快线EMA", 10, 30, 20)
ema_slow = st.sidebar.slider("慢线EMA", 30, 100, 60)
adx_threshold = st.sidebar.slider("ADX阈值", 20, 40, 25)
atr_period = st.sidebar.slider("ATR周期", 7, 30, 14)

# 结构参数
structure_window = st.sidebar.slider("摆动点检测窗口", 2, 7, 3)
structure_score_threshold_long = st.sidebar.slider("多头结构评分阈值", 0, 100, 80)
structure_score_threshold_short = st.sidebar.slider("空头结构评分阈值", 0, 100, 75)
pullback_depth_max = st.sidebar.slider("最大回调深度(ATR倍数)", 1.0, 3.0, 2.0, 0.1)
min_push = st.sidebar.slider("最小推动幅度(ATR倍数)", 0.1, 0.5, 0.3, 0.05)

# 二次确认参数
use_second_confirm = st.sidebar.checkbox("启用二次确认", True)
second_volume_ma_period = st.sidebar.slider("成交量均线周期", 10, 50, 20)
second_max_shadow_ratio = st.sidebar.slider("最大影线比例", 0.3, 1.0, 0.6, 0.1)

# 多周期参数
use_multitf = st.sidebar.checkbox("启用15分钟EMA验证", True)
multitf_ema_period = st.sidebar.slider("15分钟EMA周期", 10, 30, 20)

# 风控参数
risk_reward = st.sidebar.slider("盈亏比", 1.0, 3.0, 2.0, 0.1)
risk_percent = st.sidebar.slider("单笔风险 (%)", 0.5, 5.0, 2.0, 0.5)
initial_balance = st.sidebar.number_input("初始资金 (USDT)", value=100.0, step=10.0)
stop_loss_buffer = st.sidebar.slider("止损缓冲(ATR倍数)", 0.0, 1.0, 0.5, 0.1)

# 功能开关
use_original_logic = st.sidebar.checkbox("使用原简单逻辑（EMA+ADX）", False)
use_account = st.sidebar.checkbox("启用模拟账户", True)
use_backtest = st.sidebar.checkbox("启用回测", True)
use_score = st.sidebar.checkbox("显示条件评分", True)

# 手动刷新
if st.sidebar.button("🔄 强制刷新数据"):
    st.cache_data.clear()
    st.rerun()

# ==========================
# 数据获取
# ==========================
@st.cache_data(ttl=3)
def fetch_okx_candles(bar="5m", limit=300):
    url = "https://www.okx.com/api/v5/market/candles"
    params = {"instId": SYMBOL, "bar": bar, "limit": limit}
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=5, params=params)
            if r.status_code == 200:
                j = r.json()
                if "data" in j:
                    return j["data"]
        except Exception as e:
            st.warning(f"数据获取失败（尝试 {attempt+1}）：{e}")
        time.sleep(2 ** attempt)
    return None

def get_5m_data():
    data = fetch_okx_candles("5m", 300)
    if data is not None:
        df = pd.DataFrame(data, columns=[
            "ts", "open", "high", "low", "close", "volume",
            "volCcy", "volCcyQuote", "confirm"
        ])
        df["ts"] = pd.to_datetime(df["ts"].astype(int), unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        df = df.sort_values("ts").reset_index(drop=True)
        df.to_csv(CACHE_FILE_5M, index=False)
        return df
    elif os.path.exists(CACHE_FILE_5M):
        st.warning("使用本地5分钟缓存数据")
        return pd.read_csv(CACHE_FILE_5M, parse_dates=["ts"])
    else:
        return pd.DataFrame()

def get_15m_data():
    data = fetch_okx_candles("15m", 100)
    if data is not None:
        df = pd.DataFrame(data, columns=[
            "ts", "open", "high", "low", "close", "volume",
            "volCcy", "volCcyQuote", "confirm"
        ])
        df["ts"] = pd.to_datetime(df["ts"].astype(int), unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        df = df.sort_values("ts").reset_index(drop=True)
        df.to_csv(CACHE_FILE_15M, index=False)
        return df
    elif os.path.exists(CACHE_FILE_15M):
        st.warning("使用本地15分钟缓存数据")
        return pd.read_csv(CACHE_FILE_15M, parse_dates=["ts"])
    else:
        return pd.DataFrame()

def get_ticker():
    url = f"https://www.okx.com/api/v5/market/ticker?instId={SYMBOL}"
    try:
        r = requests.get(url, timeout=3)
        j = r.json()
        if j.get("code") == "0":
            return float(j["data"][0]["last"])
    except:
        pass
    return None

df_5m = get_5m_data()
if df_5m.empty:
    st.error("无法获取5分钟数据")
    st.stop()

df_15m = get_15m_data()
real_price = get_ticker()

# ==========================
# 指标计算
# ==========================
df = df_5m.copy()
df["EMA_fast"] = ta.trend.ema_indicator(df["close"], window=ema_fast)
df["EMA_slow"] = ta.trend.ema_indicator(df["close"], window=ema_slow)
df["ADX"] = ta.trend.adx(df["high"], df["low"], df["close"], window=14)
df["ATR"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=atr_period)
df["volume_ma"] = df["volume"].rolling(window=second_volume_ma_period).mean()

df = df.dropna().reset_index(drop=True)

latest = df.iloc[-1]
prev = df.iloc[-2] if len(df) > 1 else latest

# ==========================
# 辅助函数：趋势持续确认
# ==========================
def trend_up_confirm(df):
    """判断多头趋势是否持续：EMA_fast > EMA_slow 且 EMA_fast 的3期均值上升"""
    ema_fast_3 = df["EMA_fast"].rolling(3).mean()
    return (df["EMA_fast"].iloc[-1] > df["EMA_slow"].iloc[-1] and
            ema_fast_3.iloc[-1] > ema_fast_3.iloc[-2])

def trend_down_confirm(df):
    """判断空头趋势是否持续：EMA_fast < EMA_slow 且 EMA_fast 的3期均值下降"""
    ema_fast_3 = df["EMA_fast"].rolling(3).mean()
    return (df["EMA_fast"].iloc[-1] < df["EMA_slow"].iloc[-1] and
            ema_fast_3.iloc[-1] < ema_fast_3.iloc[-2])

def adx_rising(df):
    """ADX上升（当前 > 前一根）"""
    return df["ADX"].iloc[-1] > df["ADX"].iloc[-2]

# ==========================
# 结构引擎（多头+空头，含推动-回调比）
# ==========================
def detect_swings(df, window=3):
    if len(df) < 2 * window + 1:
        return {"highs": [], "lows": []}
    highs, lows = [], []
    for i in range(window, len(df) - window):
        high_val = df["high"].iloc[i]
        low_val = df["low"].iloc[i]
        if high_val == df["high"].iloc[i-window:i+window+1].max():
            highs.append((df["ts"].iloc[i], high_val))
        if low_val == df["low"].iloc[i-window:i+window+1].min():
            lows.append((df["ts"].iloc[i], low_val))
    return {"highs": highs, "lows": lows}

def analyze_bullish_structure(df):
    result = {"entry_signal": None, "trend_score": 0, "swings": None}
    if not trend_up_confirm(df) or not adx_rising(df):
        return result

    swings = detect_swings(df, window=structure_window)
    result["swings"] = swings
    highs = swings["highs"]
    lows = swings["lows"]

    if len(highs) >= 2 and len(lows) >= 2:
        if (highs[-1][1] > highs[-2][1] and lows[-1][1] > lows[-2][1]):
            hl = lows[-1][1]
            hh = highs[-1][1]
            depth = abs(hh - hl) / (latest["ATR"] + 1e-10)

            # 推动-回调比计算
            push = highs[-1][1] - lows[-2][1]  # 推动段：从上一个低点到最近高点
            pullback = highs[-1][1] - latest["low"]  # 回调段：从最近高点到当前低点
            push_ok = push > min_push * latest["ATR"]
            pullback_ok = pullback < pullback_depth_max * latest["ATR"]

            score = 0
            if hl > latest["EMA_slow"]:
                score += 30
            if depth < pullback_depth_max:
                score += 30
            if push_ok and pullback_ok:
                score += 20
            if adx_rising(df):
                score += 20

            result["trend_score"] = score

            if score >= structure_score_threshold_long:
                recent_high = highs[-1][1]
                body = abs(latest["close"] - latest["open"])
                upper_shadow = latest["high"] - max(latest["close"], latest["open"])
                if latest["close"] > recent_high and body > upper_shadow:
                    result["entry_signal"] = "多"
    return result

def analyze_bearish_structure(df):
    result = {"entry_signal": None, "trend_score": 0, "swings": None}
    if not trend_down_confirm(df) or not adx_rising(df):
        return result

    swings = detect_swings(df, window=structure_window)
    result["swings"] = swings
    highs = swings["highs"]
    lows = swings["lows"]

    if len(highs) >= 2 and len(lows) >= 2:
        if (highs[-1][1] < highs[-2][1] and lows[-1][1] < lows[-2][1]):
            hh = highs[-1][1]
            ll = lows[-1][1]
            depth = abs(hh - ll) / (latest["ATR"] + 1e-10)

            # 推动-回调比（空头）
            push = highs[-2][1] - lows[-1][1]  # 推动段：从上一个高点到最近低点
            pullback = latest["high"] - lows[-1][1]  # 回调段：从当前高点到最近低点
            push_ok = push > min_push * latest["ATR"]
            pullback_ok = pullback < pullback_depth_max * latest["ATR"]

            score = 0
            if hh < latest["EMA_slow"]:
                score += 30
            if depth < pullback_depth_max:
                score += 30
            if push_ok and pullback_ok:
                score += 20
            if adx_rising(df):
                score += 20

            result["trend_score"] = score

            if score >= structure_score_threshold_short:
                recent_low = lows[-1][1]
                body = abs(latest["close"] - latest["open"])
                lower_shadow = min(latest["close"], latest["open"]) - latest["low"]
                if latest["close"] < recent_low and body > lower_shadow:
                    result["entry_signal"] = "空"
    return result

# ==========================
# 二次确认（多头+空头）
# ==========================
def detect_long_second_confirmation(df, swings):
    result = {"valid": False}
    if not swings or not swings["highs"] or len(df) < 2:
        return result

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    recent_high = swings["highs"][-1][1]

    first_break = (
        prev["close"] > recent_high and
        (prev["close"] - prev["open"]) > (prev["high"] - prev["close"])
    )
    if not first_break:
        return result

    second_confirm = (
        latest["close"] > prev["close"] and
        latest["close"] > recent_high and
        (latest["high"] - latest["close"]) < (latest["close"] - latest["open"]) * second_max_shadow_ratio
    )
    momentum_ok = (latest["ADX"] > adx_threshold and adx_rising(df))
    volume_ok = (latest["volume"] > latest["volume_ma"])

    result["valid"] = second_confirm and momentum_ok and volume_ok
    return result

def detect_short_second_confirmation(df, swings):
    result = {"valid": False}
    if not swings or not swings["lows"] or len(df) < 2:
        return result

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    recent_low = swings["lows"][-1][1]

    first_break = (
        prev["close"] < recent_low and
        (prev["open"] - prev["close"]) > (prev["close"] - prev["low"])
    )
    if not first_break:
        return result

    second_confirm = (
        latest["close"] < prev["close"] and
        latest["close"] < recent_low and
        (latest["close"] - latest["low"]) < (latest["open"] - latest["close"]) * second_max_shadow_ratio
    )
    momentum_ok = (latest["ADX"] > adx_threshold and adx_rising(df))
    volume_ok = (latest["volume"] > latest["volume_ma"])

    result["valid"] = second_confirm and momentum_ok and volume_ok
    return result

# ==========================
# 多周期共振
# ==========================
def check_multitf(df15, direction):
    if df15.empty or len(df15) < 2:
        return False
    df15 = df15.copy()
    df15["EMA"] = ta.trend.ema_indicator(df15["close"], window=multitf_ema_period)
    df15 = df15.dropna()
    if len(df15) < 2:
        return False
    latest_15 = df15.iloc[-1]
    prev_15 = df15.iloc[-2]
    if direction == "多":
        return latest_15["EMA"] > prev_15["EMA"]
    elif direction == "空":
        return latest_15["EMA"] < prev_15["EMA"]
    return False

# ==========================
# 信号生成（加权评分+严格条件）
# ==========================
signal = None
entry_price = None
stop_loss = None
take_profit = None

# 分析结构
bullish = analyze_bullish_structure(df)
bearish = analyze_bearish_structure(df)

# 二次确认
long_second = detect_long_second_confirmation(df, bullish["swings"]) if bullish["swings"] else {"valid": False}
short_second = detect_short_second_confirmation(df, bearish["swings"]) if bearish["swings"] else {"valid": False}

# 多周期状态
tf_ok_long = check_multitf(df_15m, "多")
tf_ok_short = check_multitf(df_15m, "空")

# 加权评分计算（满分100）
def weighted_score(base_score, second_ok, tf_ok):
    # 结构评分权重0.5，二次确认权重0.3，多周期权重0.2
    score = base_score * 0.5
    if second_ok:
        score += 40 * 0.3   # 二次确认满分40
    if tf_ok:
        score += 20 * 0.2   # 多周期满分20
    return int(score)

# 严格进场条件：加权分>=80, ADX上升, 成交量已内嵌于二次确认
if not use_original_logic:
    # 多头
    if bullish["entry_signal"] == "多":
        weighted = weighted_score(bullish["trend_score"], long_second["valid"], tf_ok_long)
        if weighted >= 80:
            signal = "多"
    # 空头
    if bearish["entry_signal"] == "空":
        weighted = weighted_score(bearish["trend_score"], short_second["valid"], tf_ok_short)
        if weighted >= 80:
            signal = "空"
else:
    # 原简单逻辑
    if latest["EMA_fast"] > latest["EMA_slow"] and latest["ADX"] > adx_threshold:
        signal = "多"
    elif latest["EMA_fast"] < latest["EMA_slow"] and latest["ADX"] > adx_threshold:
        signal = "空"

# ==========================
# 止损止盈计算（基于结构）
# ==========================
if signal == "多" and bullish["swings"] and bullish["swings"]["lows"]:
    recent_low = bullish["swings"]["lows"][-1][1]
    entry_price = latest["close"]
    stop_loss = recent_low - stop_loss_buffer * latest["ATR"]
    take_profit = entry_price + (entry_price - stop_loss) * risk_reward
elif signal == "空" and bearish["swings"] and bearish["swings"]["highs"]:
    recent_high = bearish["swings"]["highs"][-1][1]
    entry_price = latest["close"]
    stop_loss = recent_high + stop_loss_buffer * latest["ATR"]
    take_profit = entry_price - (stop_loss - entry_price) * risk_reward
else:
    # 回退
    if signal == "多":
        entry_price = latest["close"]
        stop_loss = entry_price - latest["ATR"] * 1.5
        take_profit = entry_price + (entry_price - stop_loss) * risk_reward
    elif signal == "空":
        entry_price = latest["close"]
        stop_loss = entry_price + latest["ATR"] * 1.5
        take_profit = entry_price - (stop_loss - entry_price) * risk_reward

# ==========================
# 策略提示
# ==========================
def strategy_prompt(signal, bullish, bearish, long_second, short_second, weighted):
    if not signal:
        return "⏳ 无进场条件"

    if signal == "多":
        if weighted < 80:
            return f"⚠ 加权分不足 ({weighted} < 80)"
        if not adx_rising(df):
            return "⚠ ADX未上升"
        if use_second_confirm and not long_second["valid"]:
            return "⚠ 多头二次确认未通过"
        if use_multitf and not tf_ok_long:
            return "⚠ 多周期不一致"
    elif signal == "空":
        if weighted < 80:
            return f"⚠ 加权分不足 ({weighted} < 80)"
        if not adx_rising(df):
            return "⚠ ADX未上升"
        if use_second_confirm and not short_second["valid"]:
            return "⚠ 空头二次确认未通过"
        if use_multitf and not tf_ok_short:
            return "⚠ 多周期不一致"
    return "✅ 满足进场条件"

# 计算当前信号的加权分
if signal == "多":
    current_weighted = weighted_score(bullish["trend_score"], long_second["valid"], tf_ok_long)
elif signal == "空":
    current_weighted = weighted_score(bearish["trend_score"], short_second["valid"], tf_ok_short)
else:
    current_weighted = 0

prompt = strategy_prompt(signal, bullish, bearish, long_second, short_second, current_weighted)

# ==========================
# 模拟账户
# ==========================
class TradingAccount:
    def __init__(self, initial):
        self.initial = initial
        self.file = ACCOUNT_FILE
        self.load()

    def load(self):
        if os.path.exists(self.file):
            with open(self.file, 'r') as f:
                data = json.load(f)
                self.balance = data.get('balance', self.initial)
                self.position = data.get('position', None)
                self.trades = data.get('trades', [])
        else:
            self.reset()

    def save(self):
        data = {
            'balance': self.balance,
            'position': self.position,
            'trades': self.trades
        }
        with open(self.file, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def reset(self):
        self.balance = self.initial
        self.position = None
        self.trades = []
        self.save()

    def can_open(self):
        return self.position is None

    def open_position(self, direction, price, stop_loss, take_profit):
        risk_amount = self.balance * (risk_percent / 100)
        stop_distance = abs(price - stop_loss)
        if stop_distance == 0:
            return False, "止损距离为零"
        raw_qty = risk_amount / stop_distance
        qty = round(raw_qty / 0.01) * 0.01
        if qty < 0.01:
            qty = 0.01
        fee = price * qty * 0.0005
        if self.balance < fee:
            return False, "余额不足"

        self.balance -= fee
        self.position = {
            'direction': direction,
            'entry': price,
            'qty': qty,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'open_time': datetime.now().isoformat()
        }
        self.save()
        return True, "开仓成功"

    def check_close(self, current_price):
        if not self.position:
            return None
        pos = self.position
        direction = pos['direction']
        entry = pos['entry']
        stop = pos['stop_loss']
        tp = pos['take_profit']
        qty = pos['qty']

        if direction == "多":
            if current_price <= stop:
                close_price = stop
                reason = "止损"
            elif current_price >= tp:
                close_price = tp
                reason = "止盈"
            else:
                return None
        elif direction == "空":
            if current_price >= stop:
                close_price = stop
                reason = "止损"
            elif current_price <= tp:
                close_price = tp
                reason = "止盈"
            else:
                return None
        else:
            return None

        pnl = (close_price - entry) * qty if direction == "多" else (entry - close_price) * qty
        fee = close_price * qty * 0.0005
        net_pnl = pnl - fee

        self.balance += net_pnl
        self.trades.append({
            **pos,
            'close_price': close_price,
            'close_time': datetime.now().isoformat(),
            'reason': reason,
            'pnl': pnl,
            'fee': fee,
            'net_pnl': net_pnl
        })
        self.position = None
        self.save()
        return reason, net_pnl

    def get_equity(self, current_price):
        if not self.position:
            return self.balance
        pos = self.position
        if pos['direction'] == "多":
            floating = (current_price - pos['entry']) * pos['qty']
        else:
            floating = (pos['entry'] - current_price) * pos['qty']
        return self.balance + floating

if use_account:
    account = TradingAccount(initial_balance)

    if signal and account.can_open():
        success, msg = account.open_position(signal, entry_price, stop_loss, take_profit)
        if success:
            st.sidebar.success(f"✅ 开仓 {signal} at {entry_price:.2f}")
        else:
            st.sidebar.warning(f"开仓失败: {msg}")

    check_price = real_price if real_price is not None else latest["close"]
    if account.position:
        result = account.check_close(check_price)
        if result:
            reason, net = result
            st.sidebar.info(f"📌 平仓: {reason}, 盈亏: {net:+.2f} USDT")

# ==========================
# 回测（简化版）
# ==========================
def run_backtest(df):
    if len(df) < 50:
        return {"trades": 0, "profit": 0, "balance": initial_balance}
    balance = initial_balance
    trades = []
    position = None
    for i in range(30, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        # 模拟信号：简单EMA交叉 + ADX
        if position is None:
            if row["EMA_fast"] > row["EMA_slow"] and row["ADX"] > adx_threshold:
                position = {
                    "direction": "多",
                    "entry": row["close"],
                    "stop": row["close"] - row["ATR"] * 1.5,
                    "tp": row["close"] + row["ATR"] * 1.5 * risk_reward
                }
            elif row["EMA_fast"] < row["EMA_slow"] and row["ADX"] > adx_threshold:
                position = {
                    "direction": "空",
                    "entry": row["close"],
                    "stop": row["close"] + row["ATR"] * 1.5,
                    "tp": row["close"] - row["ATR"] * 1.5 * risk_reward
                }
        else:
            # 检查止损止盈
            if position["direction"] == "多":
                if row["low"] <= position["stop"]:
                    close_price = position["stop"]
                    pnl = (close_price - position["entry"]) * 0.01
                    balance += pnl
                    trades.append(pnl)
                    position = None
                elif row["high"] >= position["tp"]:
                    close_price = position["tp"]
                    pnl = (close_price - position["entry"]) * 0.01
                    balance += pnl
                    trades.append(pnl)
                    position = None
            else:  # 空头
                if row["high"] >= position["stop"]:
                    close_price = position["stop"]
                    pnl = (position["entry"] - close_price) * 0.01
                    balance += pnl
                    trades.append(pnl)
                    position = None
                elif row["low"] <= position["tp"]:
                    close_price = position["tp"]
                    pnl = (position["entry"] - close_price) * 0.01
                    balance += pnl
                    trades.append(pnl)
                    position = None
    return {"trades": len(trades), "profit": sum(trades), "balance": balance}

# ==========================
# 图表绘制
# ==========================
fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=df["ts"],
    open=df["open"],
    high=df["high"],
    low=df["low"],
    close=df["close"],
    name="K线"
))

fig.add_trace(go.Scatter(x=df["ts"], y=df["EMA_fast"], line=dict(color="blue", width=1), name=f"EMA{ema_fast}"))
fig.add_trace(go.Scatter(x=df["ts"], y=df["EMA_slow"], line=dict(color="orange", width=1), name=f"EMA{ema_slow}"))

# 摆动点
if bullish["swings"]:
    swings = bullish["swings"]
    if swings["highs"]:
        hx, hy = zip(*swings["highs"])
        fig.add_trace(go.Scatter(x=hx, y=hy, mode='markers', marker=dict(symbol='triangle-down', size=8, color='red'), name='多头高点'))
    if swings["lows"]:
        lx, ly = zip(*swings["lows"])
        fig.add_trace(go.Scatter(x=lx, y=ly, mode='markers', marker=dict(symbol='triangle-up', size=8, color='green'), name='多头低点'))
if bearish["swings"]:
    swings = bearish["swings"]
    if swings["highs"]:
        hx, hy = zip(*swings["highs"])
        fig.add_trace(go.Scatter(x=hx, y=hy, mode='markers', marker=dict(symbol='triangle-down', size=8, color='orange'), name='空头高点'))
    if swings["lows"]:
        lx, ly = zip(*swings["lows"])
        fig.add_trace(go.Scatter(x=lx, y=ly, mode='markers', marker=dict(symbol='triangle-up', size=8, color='purple'), name='空头低点'))

# 信号标记
if signal:
    marker_symbol = "arrow-up" if signal == "多" else "arrow-down"
    fig.add_trace(go.Scatter(
        x=[latest["ts"]],
        y=[latest["close"]],
        mode="markers+text",
        marker=dict(symbol=marker_symbol, size=15, color="yellow"),
        text=signal,
        textposition="top center" if signal=="多" else "bottom center",
        name="信号"
    ))
    if stop_loss and take_profit:
        fig.add_hline(y=stop_loss, line_dash="dash", line_color="red", annotation_text="止损")
        fig.add_hline(y=take_profit, line_dash="dash", line_color="green", annotation_text="止盈")

fig.update_layout(
    title=f"{SYMBOL} 5分钟图（最终优化版）",
    template="plotly_dark",
    height=700,
    xaxis_rangeslider_visible=False
)
st.plotly_chart(fig, use_container_width=True)

# ==========================
# 信息面板
# ==========================
st.subheader("📊 当前状态")
col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
    st.metric("信号", signal if signal else "无")
with col2:
    st.metric("ADX", f"{latest['ADX']:.1f}")
with col3:
    st.metric("收盘价", f"{latest['close']:.2f}")
with col4:
    if real_price:
        st.metric("实时价", f"{real_price:.2f}", delta=f"{real_price - latest['close']:.2f}")
    else:
        st.metric("实时价", "N/A")
with col5:
    st.metric("ATR", f"{latest['ATR']:.4f}")
with col6:
    if signal == "多":
        st.metric("加权分", current_weighted)
    elif signal == "空":
        st.metric("加权分", current_weighted)
    else:
        st.metric("加权分", 0)

st.subheader("💡 策略提示")
st.info(prompt)

if use_score:
    st.metric("加权评分", current_weighted)

if signal:
    st.write(f"**入场价**: {entry_price:.2f} | **止损**: {stop_loss:.2f} | **止盈**: {take_profit:.2f}")

if use_multitf and not df_15m.empty:
    st.write(f"**15分钟EMA方向**: 多{'✅' if tf_ok_long else '❌'} 空{'✅' if tf_ok_short else '❌'}")

if use_second_confirm:
    if signal == "多":
        st.write(f"**二次确认**: {'✅ 通过' if long_second['valid'] else '❌ 未通过'}")
    elif signal == "空":
        st.write(f"**二次确认**: {'✅ 通过' if short_second['valid'] else '❌ 未通过'}")

st.caption(f"数据更新于: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ==========================
# 账户面板
# ==========================
if use_account:
    st.subheader("💰 模拟账户")
    col_acc1, col_acc2, col_acc3 = st.columns(3)
    with col_acc1:
        st.metric("余额", f"{account.balance:.2f} USDT")
    with col_acc2:
        equity = account.get_equity(real_price if real_price else latest["close"])
        st.metric("权益", f"{equity:.2f} USDT")
    with col_acc3:
        if account.position:
            st.metric("持仓", f"{account.position['direction']} {account.position['qty']:.2f}张")
        else:
            st.metric("持仓", "无")

    if account.position:
        pos = account.position
        st.write(f"开仓价: {pos['entry']:.2f} | 止损: {pos['stop_loss']:.2f} | 止盈: {pos['take_profit']:.2f}")

    if st.button("重置账户"):
        account.reset()
        st.rerun()

# ==========================
# 回测执行
# ==========================
if use_backtest and st.sidebar.button("运行回测"):
    with st.spinner("回测中..."):
        backtest_result = run_backtest(df)
    st.subheader("📈 回测结果")
    st.json(backtest_result)
