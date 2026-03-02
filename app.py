"""
5分钟布林带均值回归系统（低买高卖）
策略：价格触及或跌破下轨时做多，触及或突破上轨时做空
出场：固定止盈止损（基于ATR）
模拟交易：初始100 CNY（≈13.89 USDT），固定开仓数量0.01张
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
st.set_page_config(layout="wide", page_title="5分钟布林带低买高卖")
st.title("📉 5分钟布林带均值回归 · 低买高卖")
st.caption("策略：价格触及下轨做多，触及上轨做空 | 初始资金 100 CNY")

SYMBOL = "ETH-USDT-SWAP"
ACCOUNT_FILE = "sim_account.json"
CACHE_FILE = "last_data_cache.csv"

st_autorefresh(interval=8000, key="refresh")

# ==========================
# 数据获取
# ==========================
@st.cache_data(ttl=5)
def get_data():
    url = "https://www.okx.com/api/v5/market/candles"
    params = {"instId": SYMBOL, "bar": "5m", "limit": 200}
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

# ==========================
# 计算布林带和ATR
# ==========================
bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
df["BB_upper"] = bb.bollinger_hband()
df["BB_middle"] = bb.bollinger_mavg()
df["BB_lower"] = bb.bollinger_lband()
df["ATR"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)

df = df.dropna().reset_index(drop=True)
latest = df.iloc[-1]
prev = df.iloc[-2] if len(df) > 1 else latest

# ==========================
# 信号生成：触及布林带边界
# ==========================
signal = None
# 做多信号：最低价跌破或触及下轨，且收盘价在下轨附近（允许轻微刺穿）
if latest["low"] <= latest["BB_lower"]:
    signal = "多"
# 做空信号：最高价突破或触及上轨
elif latest["high"] >= latest["BB_upper"]:
    signal = "空"

# ==========================
# 简化模拟账户（与之前类似）
# ==========================
class SimpleAccount:
    def __init__(self, initial_balance_usdt=13.89):
        self.initial = initial_balance_usdt
        self.key = "account"
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
        """固定开仓0.01张，止损2倍ATR，止盈2倍ATR"""
        qty = 0.01
        if direction == "多":
            stop = price - 2 * atr
            tp = price + 2 * atr
        else:
            stop = price + 2 * atr
            tp = price - 2 * atr

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
    st.info(f"信号触发: {signal}")
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
# 绘图（K线+布林带）
# ==========================
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df["ts"], open=df["open"], high=df["high"],
    low=df["low"], close=df["close"], name="K线"
))
fig.add_trace(go.Scatter(x=df["ts"], y=df["BB_upper"], line=dict(color="gray", width=1, dash="dash"), name="上轨"))
fig.add_trace(go.Scatter(x=df["ts"], y=df["BB_middle"], line=dict(color="blue", width=1), name="中轨"))
fig.add_trace(go.Scatter(x=df["ts"], y=df["BB_lower"], line=dict(color="gray", width=1, dash="dash"), name="下轨"))

fig.update_layout(title=f"{SYMBOL} 5分钟布林带", template="plotly_dark", height=600)
st.plotly_chart(fig, use_container_width=True)

# ==========================
# 状态面板
# ==========================
st.subheader("📊 当前状态")
st.write(f"最新价: {latest['close']:.2f}")
st.write(f"布林上轨: {latest['BB_upper']:.2f} | 中轨: {latest['BB_middle']:.2f} | 下轨: {latest['BB_lower']:.2f}")
st.write(f"ATR: {latest['ATR']:.4f}")
if signal:
    st.success(f"当前信号: {signal} (触及布林边界)")
else:
    st.info("无信号")

# ==========================
# 历史交易记录
# ==========================
st.subheader("📜 交易记录")
if account.trades:
    df_trades = pd.DataFrame(account.trades)
    st.dataframe(df_trades.tail(20)[["open_time", "direction", "entry", "close_price", "reason", "net_pnl"]])
else:
    st.info("暂无交易")
