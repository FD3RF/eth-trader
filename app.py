"""
5分钟趋势合约系统（专业版）- 持久化账户 + 多时间框架验证
策略：EMA趋势 + ADX强度 + ATR自适应回调 + K线确认 + 结构支撑阻力
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

st.set_page_config(layout="wide")
st.title("📈 5分钟趋势合约系统（专业版）")

SYMBOL = "ETH-USDT-SWAP"
CACHE_FILE_5M = "market_data_5m_cache.csv"
CACHE_FILE_15M = "market_data_15m_cache.csv"
SIGNAL_HISTORY_FILE = "signal_history.json"
ACCOUNT_FILE = "account.json"

# ==========================
# 侧边栏参数
# ==========================
st.sidebar.header("⚙️ 策略参数")
adx_threshold = st.sidebar.slider("ADX趋势阈值", 20, 40, 25, help="ADX大于此值才认为有趋势")
ema_period_fast = st.sidebar.slider("快线EMA周期", 10, 30, 20)
ema_period_slow = st.sidebar.slider("慢线EMA周期", 30, 100, 60)
atr_period = st.sidebar.slider("ATR周期", 7, 30, 14)
lookback_sr = st.sidebar.slider("支撑/阻力周期（根K线）", 10, 50, 20, help="计算前高前低所用的K线数量")
callback_atr_mult = st.sidebar.slider("回调距离（ATR倍数）", 0.3, 1.0, 0.5, step=0.1, help="价格距离EMA20的最大ATR倍数")
risk_reward = st.sidebar.slider("盈亏比", 1.0, 3.0, 2.0, step=0.1)
risk_percent = st.sidebar.slider("单笔风险 (%)", 1.0, 5.0, 2.0, step=0.5)
initial_balance = st.sidebar.number_input("初始资金 (USDT)", value=100.0, step=10.0)

# 多时间框架验证开关
use_tf_filter = st.sidebar.checkbox("启用15分钟EMA验证", value=True, help="要求15分钟EMA方向与主趋势一致")

# 趋势持续确认周期
trend_confirm_bars = 3

# ==========================
# 带指数退避的数据获取（5分钟）
# ==========================
@st.cache_data(ttl=5)
def fetch_okx_candles(bar="5m", limit=300):
    url = "https://www.okx.com/api/v5/market/candles"
    params = {"instId": SYMBOL, "bar": bar, "limit": limit}
    max_retries = 5
    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=5, params=params)
            if r.status_code == 200:
                j = r.json()
                if "data" in j:
                    return j["data"]
            wait = 2 ** attempt
            time.sleep(wait)
        except:
            wait = 2 ** attempt
            time.sleep(wait)
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
    else:
        if os.path.exists(CACHE_FILE_5M):
            st.warning("使用本地5分钟缓存数据")
            df = pd.read_csv(CACHE_FILE_5M, parse_dates=["ts"])
            return df
        else:
            return pd.DataFrame()

def get_15m_data():
    data = fetch_okx_candles("15m", 100)  # 15分钟需要足够计算EMA20，100根足够
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
    else:
        if os.path.exists(CACHE_FILE_15M):
            st.warning("使用本地15分钟缓存数据")
            df = pd.read_csv(CACHE_FILE_15M, parse_dates=["ts"])
            return df
        else:
            return pd.DataFrame()

df_5m = get_5m_data()
if df_5m.empty:
    st.error("无法获取5分钟数据")
    st.stop()

df_15m = get_15m_data()
if df_15m.empty and use_tf_filter:
    st.warning("无法获取15分钟数据，多时间框架验证将禁用")
    use_tf_filter = False

# ==========================
# 5分钟指标计算
# ==========================
df = df_5m.copy()
df["EMA_fast"] = ta.trend.ema_indicator(df["close"], window=ema_period_fast)
df["EMA_slow"] = ta.trend.ema_indicator(df["close"], window=ema_period_slow)
df["ADX"] = ta.trend.adx(df["high"], df["low"], df["close"], window=14)
df["ATR"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=atr_period)

# 支撑阻力（滚动最高最低）
df["resistance_roll"] = df["high"].rolling(window=lookback_sr).max()
df["support_roll"] = df["low"].rolling(window=lookback_sr).min()

# 趋势持续确认条件
df["EMA_fast_up"] = df["EMA_fast"] > df["EMA_fast"].shift(1)
df["EMA_fast_down"] = df["EMA_fast"] < df["EMA_fast"].shift(1)
df["trend_up_confirm"] = (df["EMA_fast"] > df["EMA_slow"]) & df["EMA_fast_up"]
df["trend_down_confirm"] = (df["EMA_fast"] < df["EMA_slow"]) & df["EMA_fast_down"]
df["trend_up_streak"] = df["trend_up_confirm"].rolling(trend_confirm_bars).sum() == trend_confirm_bars
df["trend_down_streak"] = df["trend_down_confirm"].rolling(trend_confirm_bars).sum() == trend_confirm_bars

# 寻找结构高点/低点（真正的前高前低）
def find_swing_highs(df, window=5):
    highs = []
    for i in range(window, len(df)-window):
        left = df['high'].iloc[i-window:i].max()
        right = df['high'].iloc[i+1:i+window+1].max()
        if df['high'].iloc[i] > left and df['high'].iloc[i] > right:
            highs.append((df['ts'].iloc[i], df['high'].iloc[i]))
    return highs

def find_swing_lows(df, window=5):
    lows = []
    for i in range(window, len(df)-window):
        left = df['low'].iloc[i-window:i].min()
        right = df['low'].iloc[i+1:i+window+1].min()
        if df['low'].iloc[i] < left and df['low'].iloc[i] < right:
            lows.append((df['ts'].iloc[i], df['low'].iloc[i]))
    return lows

swing_highs = find_swing_highs(df, window=3)
swing_lows = find_swing_lows(df, window=3)

df = df.dropna().reset_index(drop=True)

latest = df.iloc[-1]
prev = df.iloc[-2] if len(df) > 1 else latest

# ==========================
# 15分钟EMA计算
# ==========================
tf_ok = True  # 默认通过
if use_tf_filter and not df_15m.empty:
    df_15m["EMA20"] = ta.trend.ema_indicator(df_15m["close"], window=20)
    df_15m = df_15m.dropna().reset_index(drop=True)
    if len(df_15m) > 1:
        latest_15m = df_15m.iloc[-1]
        prev_15m = df_15m.iloc[-2]
        # 多头要求15分钟EMA20上升，空头要求下降
        if latest["trend_up_streak"]:
            tf_ok = latest_15m["EMA20"] > prev_15m["EMA20"]
        elif latest["trend_down_streak"]:
            tf_ok = latest_15m["EMA20"] < prev_15m["EMA20"]
    else:
        tf_ok = False

# ==========================
# 趋势判断（加入持续确认）
# ==========================
trend = None
if latest["trend_up_streak"] and latest["ADX"] > adx_threshold and tf_ok:
    trend = "多"
elif latest["trend_down_streak"] and latest["ADX"] > adx_threshold and tf_ok:
    trend = "空"

# ==========================
# 回调入场信号（基于ATR + 方向约束）
# ==========================
signal = None
if trend == "多":
    # 方向约束：价格在慢线之上
    if latest["close"] > latest["EMA_slow"] and \
       abs(latest["close"] - latest["EMA_fast"]) < latest["ATR"] * callback_atr_mult and \
       prev["close"] < prev["EMA_fast"]:
        body = abs(latest["close"] - latest["open"])
        lower_shadow = min(latest["close"], latest["open"]) - latest["low"]
        if lower_shadow > body * 1.5 or latest["close"] > latest["open"]:
            signal = "多"
elif trend == "空":
    if latest["close"] < latest["EMA_slow"] and \
       abs(latest["close"] - latest["EMA_fast"]) < latest["ATR"] * callback_atr_mult and \
       prev["close"] > prev["EMA_fast"]:
        body = abs(latest["close"] - latest["open"])
        upper_shadow = latest["high"] - max(latest["close"], latest["open"])
        if upper_shadow > body * 1.5 or latest["close"] < latest["open"]:
            signal = "空"

# ==========================
# 信号持久化
# ==========================
def load_signal_history():
    if os.path.exists(SIGNAL_HISTORY_FILE):
        with open(SIGNAL_HISTORY_FILE, 'r') as f:
            data = json.load(f)
            # 将时间字符串转回datetime
            for item in data:
                item['time'] = datetime.fromisoformat(item['time'])
            return data
    return []

def save_signal_history(history):
    serializable = []
    for item in history:
        copy = item.copy()
        copy['time'] = item['time'].isoformat()
        serializable.append(copy)
    with open(SIGNAL_HISTORY_FILE, 'w') as f:
        json.dump(serializable, f, indent=2)

if "signal_history" not in st.session_state:
    st.session_state.signal_history = load_signal_history()

if signal:
    last_signal_time = st.session_state.signal_history[-1]["time"] if st.session_state.signal_history else None
    if last_signal_time != latest["ts"]:
        st.session_state.signal_history.append({
            "time": latest["ts"],
            "direction": signal,
            "price": latest["close"],
            "ema_fast": latest["EMA_fast"],
            "atr": latest["ATR"]
        })
        if len(st.session_state.signal_history) > 50:
            st.session_state.signal_history = st.session_state.signal_history[-50:]
        save_signal_history(st.session_state.signal_history)

# ==========================
# 模拟账户（持久化到文件）
# ==========================
class TradingAccount:
    def __init__(self, initial_balance):
        self.initial = initial_balance
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

    def open_position(self, direction, price, atr):
        if direction == "多":
            stop_loss = price - atr * 1.5
            take_profit = price + (price - stop_loss) * risk_reward
        else:
            stop_loss = price + atr * 1.5
            take_profit = price - (stop_loss - price) * risk_reward

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

    def check_stop_take(self, current_price):
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
                reason = "止损"
                close_price = stop
            elif current_price >= tp:
                reason = "止盈"
                close_price = tp
            else:
                return None
        else:
            if current_price >= stop:
                reason = "止损"
                close_price = stop
            elif current_price <= tp:
                reason = "止盈"
                close_price = tp
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

account = TradingAccount(initial_balance)

# 开仓逻辑
if signal and account.can_open():
    success, msg = account.open_position(signal, latest["close"], latest["ATR"])
    if success:
        st.sidebar.success(f"✅ 开仓 {signal}")
    else:
        st.sidebar.warning(f"开仓失败: {msg}")

# 检查止损止盈
if account.position:
    result = account.check_stop_take(latest["close"])
    if result:
        reason, net = result
        st.sidebar.info(f"📌 平仓: {reason}, 盈亏: {net:+.2f} USDT")

# ==========================
# 胜率统计
# ==========================
def calculate_performance(trades):
    if not trades:
        return {}
    df_t = pd.DataFrame(trades)
    wins = df_t[df_t['net_pnl'] > 0]
    losses = df_t[df_t['net_pnl'] < 0]
    total = len(df_t)
    win_rate = len(wins) / total * 100 if total > 0 else 0
    avg_win = wins['net_pnl'].mean() if len(wins) > 0 else 0
    avg_loss = abs(losses['net_pnl'].mean()) if len(losses) > 0 else 0
    profit_factor = (wins['net_pnl'].sum() / abs(losses['net_pnl'].sum())) if len(losses) > 0 and losses['net_pnl'].sum() != 0 else float('inf')
    return {
        'total_trades': total,
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'net_pnl_total': df_t['net_pnl'].sum()
    }

# ==========================
# 绘图
# ==========================
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df["ts"], open=df["open"], high=df["high"],
    low=df["low"], close=df["close"], name="K线"
))
fig.add_trace(go.Scatter(x=df["ts"], y=df["EMA_fast"], line=dict(color="blue", width=1), name=f"EMA{ema_period_fast}"))
fig.add_trace(go.Scatter(x=df["ts"], y=df["EMA_slow"], line=dict(color="orange", width=1), name=f"EMA{ema_period_slow}"))
fig.add_trace(go.Scatter(x=df["ts"], y=df["support_roll"], line=dict(color="green", width=1, dash="dash"), name="支撑(滚动)"))
fig.add_trace(go.Scatter(x=df["ts"], y=df["resistance_roll"], line=dict(color="red", width=1, dash="dash"), name="阻力(滚动)"))

# 结构高低点
if swing_highs:
    sh_x, sh_y = zip(*swing_highs)
    fig.add_trace(go.Scatter(x=sh_x, y=sh_y, mode='markers', marker=dict(symbol='triangle-down', size=8, color='red'), name='结构高点'))
if swing_lows:
    sl_x, sl_y = zip(*swing_lows)
    fig.add_trace(go.Scatter(x=sl_x, y=sl_y, mode='markers', marker=dict(symbol='triangle-up', size=8, color='green'), name='结构低点'))

# 信号点
if signal:
    fig.add_trace(go.Scatter(
        x=[latest["ts"]],
        y=[latest["close"]],
        mode="markers+text",
        marker=dict(symbol="arrow-up" if signal=="多" else "arrow-down", size=15, color="yellow"),
        text=signal,
        textposition="top center",
        name="新信号"
    ))

fig.update_layout(
    title=f"{SYMBOL} 5分钟图 (专业版)",
    template="plotly_dark",
    height=700,
    xaxis_rangeslider_visible=False
)
st.plotly_chart(fig, use_container_width=True)

# ==========================
# 状态面板
# ==========================
st.subheader("📊 当前状态")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("趋势方向", trend if trend else "无")
with col2:
    st.metric("ADX", f"{latest['ADX']:.1f}")
with col3:
    st.metric("最新价", f"{latest['close']:.2f}")
with col4:
    st.metric("ATR", f"{latest['ATR']:.4f}")

st.write(f"**滚动支撑** (最近{lookback_sr}根): {latest['support_roll']:.2f}")
st.write(f"**滚动阻力** (最近{lookback_sr}根): {latest['resistance_roll']:.2f}")
if use_tf_filter and not df_15m.empty:
    st.write(f"**15分钟EMA方向**: {'上升' if tf_ok else '下降或持平'}")
if signal:
    st.success(f"📈 当前信号: {signal}")
else:
    st.info("⏳ 无信号")

# ==========================
# 账户面板
# ==========================
st.subheader("💰 模拟账户")
col_acc1, col_acc2, col_acc3 = st.columns(3)
with col_acc1:
    st.metric("余额 (USDT)", f"{account.balance:.2f}")
with col_acc2:
    equity = account.get_equity(latest["close"])
    st.metric("权益 (USDT)", f"{equity:.2f}")
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
# 信号历史与绩效
# ==========================
st.subheader("📜 信号历史")
if st.session_state.signal_history:
    hist_df = pd.DataFrame(st.session_state.signal_history)
    hist_df['time'] = hist_df['time'].dt.strftime('%Y-%m-%d %H:%M')
    st.dataframe(hist_df[['time', 'direction', 'price', 'ema_fast', 'atr']], use_container_width=True)

st.subheader("📈 交易绩效统计")
if account.trades:
    perf = calculate_performance(account.trades)
    col_perf1, col_perf2, col_perf3, col_perf4 = st.columns(4)
    with col_perf1:
        st.metric("总交易", perf['total_trades'])
    with col_perf2:
        st.metric("胜率", f"{perf['win_rate']:.1f}%")
    with col_perf3:
        st.metric("总盈亏", f"{perf['net_pnl_total']:+.2f}")
    with col_perf4:
        pf = perf['profit_factor']
        pf_str = f"{pf:.2f}" if pf != float('inf') else "∞"
        st.metric("盈亏比", pf_str)
    st.dataframe(pd.DataFrame(account.trades).tail(10)[['open_time', 'direction', 'entry', 'close_price', 'reason', 'net_pnl']])
else:
    st.info("暂无交易记录")
