"""
5分钟布林带均值回归系统（优化版）
策略：价格假突破布林带后反向开单，结合趋势过滤和RSI确认
可选项：ADX趋势过滤、RSI超买超卖、成交量确认
"""

import streamlit as st
import requests
import pandas as pd
import ta
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
import os
from datetime import datetime
import time
import json

# ==========================
# 页面配置
# ==========================
st.set_page_config(layout="wide", page_title="5分钟布林带优化版")
st.title("🎯 5分钟布林带均值回归 · 优化版")
st.caption("策略：假突破确认 + 可选趋势过滤 | 初始资金 100 CNY")

SYMBOL = "ETH-USDT-SWAP"
ACCOUNT_FILE = "sim_account.json"
CACHE_FILE = "last_data_cache.csv"

st_autorefresh(interval=8000, key="refresh")

# ==========================
# 侧边栏参数设置
# ==========================
st.sidebar.header("⚙️ 策略参数")
bb_period = st.sidebar.slider("布林带周期", 10, 50, 20)
bb_std = st.sidebar.slider("布林带标准差", 1.5, 3.0, 2.0, step=0.1)
atr_period = st.sidebar.slider("ATR周期", 7, 30, 14)
risk_reward = st.sidebar.slider("盈亏比", 1.0, 3.0, 2.0, step=0.1)

# 优化选项开关
use_adx_filter = st.sidebar.checkbox("启用ADX趋势过滤", value=False)
use_rsi_filter = st.sidebar.checkbox("启用RSI超买超卖过滤", value=False)
use_volume_filter = st.sidebar.checkbox("启用成交量确认", value=False)
adx_threshold = st.sidebar.slider("ADX阈值", 20, 40, 25, disabled=not use_adx_filter)
rsi_oversold = st.sidebar.slider("RSI超卖线", 20, 40, 30, disabled=not use_rsi_filter)
rsi_overbought = st.sidebar.slider("RSI超买线", 60, 80, 70, disabled=not use_rsi_filter)
volume_multiplier = st.sidebar.slider("成交量倍数", 1.0, 3.0, 1.5, step=0.1, disabled=not use_volume_filter)

# ==========================
# 数据获取
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
                st.error(f"数据获取失败: {e}，使用缓存")
                if os.path.exists(CACHE_FILE):
                    return pd.read_csv(CACHE_FILE, parse_dates=["ts"])
                else:
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
    df = df.sort_values("ts").reset_index(drop=True)
    df.to_csv(CACHE_FILE, index=False)
    return df

df = get_data()
if df.empty:
    st.error("无法获取数据")
    st.stop()

if len(df) < max(bb_period, atr_period, 50):
    st.warning("数据量不足，请增加K线数量或减小周期")

# ==========================
# 计算指标
# ==========================
bb = ta.volatility.BollingerBands(df["close"], window=bb_period, window_dev=bb_std)
df["BB_upper"] = bb.bollinger_hband()
df["BB_middle"] = bb.bollinger_mavg()
df["BB_lower"] = bb.bollinger_lband()
df["ATR"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=atr_period)
df["RSI"] = ta.momentum.rsi(df["close"], window=14)
df["ADX"] = ta.trend.adx(df["high"], df["low"], df["close"], window=14)
df["VOL_MA20"] = df["volume"].rolling(window=20).mean()

df = df.dropna().reset_index(drop=True)
latest = df.iloc[-1]
prev = df.iloc[-2] if len(df) > 1 else latest

# ==========================
# 信号生成（优化版）
# ==========================
signal = None
signal_reason = ""

# 检查是否满足优化条件
adx_ok = (not use_adx_filter) or (latest["ADX"] > adx_threshold)
volume_ok = (not use_volume_filter) or (latest["volume"] > df["VOL_MA20"].iloc[-1] * volume_multiplier)

# 做多信号：价格跌破下轨后收盘价收回下轨上方
if prev["low"] <= prev["BB_lower"] and latest["close"] > latest["BB_lower"]:
    rsi_ok = (not use_rsi_filter) or (latest["RSI"] < rsi_oversold)
    if adx_ok and volume_ok and rsi_ok:
        signal = "多"
        signal_reason = "假跌破收回"

# 做空信号：价格突破上轨后收盘价回落至上轨下方
elif prev["high"] >= prev["BB_upper"] and latest["close"] < latest["BB_upper"]:
    rsi_ok = (not use_rsi_filter) or (latest["RSI"] > rsi_overbought)
    if adx_ok and volume_ok and rsi_ok:
        signal = "空"
        signal_reason = "假突破回落"

# ==========================
# 模拟账户（与之前相同，略作修改以支持动态盈亏比）
# ==========================
class SimpleAccount:
    def __init__(self, initial_balance_usdt=13.89):
        self.initial = initial_balance_usdt
        self.key = "account_opt"
        if self.key not in st.session_state:
            self.reset()

    def _get(self):
        return st.session_state[self.key]

    def _set(self, data):
        st.session_state[self.key] = data

    def reset(self):
        self._set({
            'balance': self.initial,
            'position': None,
            'trades': []
        })

    @property
    def balance(self):
        return self._get()['balance']

    @balance.setter
    def balance(self, value):
        d = self._get()
        d['balance'] = value
        self._set(d)

    @property
    def position(self):
        return self._get()['position']

    @position.setter
    def position(self, value):
        d = self._get()
        d['position'] = value
        self._set(d)

    @property
    def trades(self):
        return self._get()['trades']

    @trades.setter
    def trades(self, value):
        d = self._get()
        d['trades'] = value
        self._set(d)

    def can_open(self):
        return self.position is None

    def open_position(self, direction, price, atr):
        qty = 0.01
        if direction == "多":
            stop = price - 2 * atr
            tp = price + 2 * atr * risk_reward  # 使用用户设定的盈亏比
        else:
            stop = price + 2 * atr
            tp = price - 2 * atr * risk_reward

        fee = price * qty * 0.0005
        if self.balance < fee:
            return False, "余额不足"

        self.balance -= fee
        self.position = {
            'direction': direction,
            'entry': price,
            'qty': qty,
            'stop': stop,
            'tp': tp,
            'open_time': datetime.now().isoformat()
        }
        return True, "开仓成功"

    def check_stop_take(self, current_price):
        if not self.position:
            return None

        pos = self.position
        direction = pos['direction']
        entry = pos['entry']
        stop = pos['stop']
        tp = pos['tp']
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

        if direction == "多":
            pnl = (close_price - entry) * qty
        else:
            pnl = (entry - close_price) * qty

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

account = SimpleAccount()

# ==========================
# 处理信号开仓
# ==========================
if signal:
    st.info(f"信号触发: {signal} ({signal_reason})")
    if account.can_open():
        success, msg = account.open_position(signal, latest["close"], latest["ATR"])
        if success:
            st.sidebar.success(f"✅ 开仓 {signal} 成功")
        else:
            st.sidebar.warning(f"开仓失败: {msg}")

# 检查止损止盈
if account.position:
    result = account.check_stop_take(latest["close"])
    if result:
        reason, net = result
        st.sidebar.info(f"📌 平仓: {reason}, 盈亏: {net:+.2f} USDT")

# ==========================
# 侧边栏账户
# ==========================
with st.sidebar:
    st.header("💰 模拟账户")
    st.metric("余额 (USDT)", f"{account.balance:.2f}")
    equity = account.get_equity(latest["close"])
    st.metric("权益 (USDT)", f"{equity:.2f}")
    if account.position:
        pos = account.position
        st.write(f"**持仓**: {pos['direction']}")
        st.write(f"开仓价: {pos['entry']:.2f}")
        st.write(f"止损: {pos['stop']:.2f} | 止盈: {pos['tp']:.2f}")
        if pos['direction'] == "多":
            floating = (latest['close'] - pos['entry']) * pos['qty']
        else:
            floating = (pos['entry'] - latest['close']) * pos['qty']
        st.metric("浮动盈亏", f"{floating:.2f} USDT", delta=f"{floating:.2f}")
    else:
        st.info("无持仓")

    if st.button("重置账户"):
        account.reset()
        st.rerun()

    st.divider()
    st.write("**最近交易**")
    if account.trades:
        for t in account.trades[-5:]:
            st.caption(f"{t['close_time'][:16]} {t['direction']} {t['reason']} {t['net_pnl']:+.2f}")

# ==========================
# 绘图
# ==========================
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df["ts"], open=df["open"], high=df["high"],
    low=df["low"], close=df["close"], name="K线"
))
fig.add_trace(go.Scatter(x=df["ts"], y=df["BB_upper"], line=dict(color="gray", width=1, dash="dash"), name="上轨"))
fig.add_trace(go.Scatter(x=df["ts"], y=df["BB_middle"], line=dict(color="blue", width=1), name="中轨"))
fig.add_trace(go.Scatter(x=df["ts"], y=df["BB_lower"], line=dict(color="gray", width=1, dash="dash"), name="下轨"))

# 标记信号点
signal_df = df[df["close"] != df["close"].shift()]  # 简化，实际需记录信号位置，此处略
if signal:
    fig.add_trace(go.Scatter(
        x=[latest["ts"]],
        y=[latest["close"]],
        mode="markers",
        marker=dict(symbol="circle", size=10, color="yellow"),
        name="信号点"
    ))

fig.update_layout(title=f"{SYMBOL} 5分钟布林带 (优化版)", template="plotly_dark", height=600)
st.plotly_chart(fig, use_container_width=True)

# ==========================
# 状态面板
# ==========================
st.subheader("📊 当前状态")
st.write(f"最新价: {latest['close']:.2f}")
st.write(f"布林上轨: {latest['BB_upper']:.2f} | 中轨: {latest['BB_middle']:.2f} | 下轨: {latest['BB_lower']:.2f}")
st.write(f"ATR: {latest['ATR']:.4f} | RSI: {latest['RSI']:.2f} | ADX: {latest['ADX']:.2f}")
st.write(f"成交量: {latest['volume']:.0f} (20日均量: {df['VOL_MA20'].iloc[-1]:.0f})")

if signal:
    st.success(f"当前信号: {signal} ({signal_reason})")
else:
    st.info("无信号")

# 显示各优化条件状态
if use_adx_filter or use_rsi_filter or use_volume_filter:
    st.write("**优化条件状态**：")
    if use_adx_filter:
        st.write(f"ADX > {adx_threshold}: {'✅' if latest['ADX'] > adx_threshold else '❌'}")
    if use_rsi_filter:
        st.write(f"RSI超卖: {'✅' if latest['RSI'] < rsi_oversold else '❌'} / 超买: {'✅' if latest['RSI'] > rsi_overbought else '❌'}")
    if use_volume_filter:
        st.write(f"成交量 > {volume_multiplier}倍均量: {'✅' if latest['volume'] > df['VOL_MA20'].iloc[-1] * volume_multiplier else '❌'}")

# ==========================
# 历史交易记录
# ==========================
st.subheader("📜 交易记录")
if account.trades:
    df_trades = pd.DataFrame(account.trades)
    st.dataframe(df_trades.tail(20)[["open_time", "direction", "entry", "close_price", "reason", "net_pnl"]])
else:
    st.info("暂无交易")
