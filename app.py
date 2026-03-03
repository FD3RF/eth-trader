"""
ETH 战神 V2600 - 高胜率·盈亏优化版（最终完整版 + 交易计划展示）
核心策略：
  1. 15分钟EMA20方向确定主趋势（严格使用历史数据，无未来函数）
  2. 5分钟局部结构突破 (Higher Low / Lower High) —— 结构点延迟确认，避免未来数据
  3. 成交量确认 + 假突破过滤
  4. 小止损0.6ATR，小止盈0.8ATR (盈亏比1.33)
  5. 模拟账户自动交易（开仓、止损止盈、资金曲线）
  6. 回测模块展示历史表现（初始资金20 USDT）
  7. 实时信号提示及自动交易执行，并显示详细的交易计划
  8. K线图置顶，回测绩效放底部
"""

import streamlit as st
import pandas as pd
import ta
import plotly.graph_objects as go
import requests
import numpy as np
from datetime import datetime, timedelta
import threading
import json
import os

st.set_page_config(layout="wide", page_title="战神 V2600（最终版）")
st.title("🛡️ ETH 战神 V2600 (最终版)")

SYMBOL = "ETH-USDT-SWAP"
ACCOUNT_FILE = "sim_account.json"

# --------------------------
# 侧边栏参数
# --------------------------
with st.sidebar:
    st.header("⚙️ 盈利引擎参数")
    adx_threshold = st.slider("ADX 强度阈值", 20, 35, 25, help="低于此值不交易")
    ema_fast_period = st.slider("快线EMA周期", 5, 20, 8)
    ema_slow_period = st.slider("慢线EMA周期", 20, 60, 30)
    atr_period = st.slider("ATR周期", 7, 21, 10)
    swing_window = st.slider("结构点窗口", 2, 5, 3, help="识别高低点所需左右K线数")
    volume_mult = st.slider("突破放量倍数", 1.0, 2.0, 1.3, step=0.1)
    atr_sl_mult = st.number_input("止损ATR倍数", value=0.6, step=0.1)
    atr_tp_mult = st.number_input("止盈ATR倍数", value=0.8, step=0.1)
    risk_percent = st.slider("单笔风险 %", 0.5, 2.0, 1.0, step=0.1)
    slippage = st.number_input("滑点 (百分比)", value=0.05, step=0.01, format="%.2f", help="模拟成交价格偏差")
    initial_balance = st.number_input("初始资金 (USDT)", value=20.0, step=10.0)  # 默认20

    st.divider()
    st.info("💡 核心逻辑：15M趋势锁死 + 5M结构突破 + 量能确认 + 小止损小止盈")
    if st.button("🔄 强制刷新数据"):
        st.cache_data.clear()
        st.rerun()

# --------------------------
# 数据获取（缓存）
# --------------------------
@st.cache_data(ttl=5)
def get_klines(bar="5m", limit=500):
    """从OKX获取K线数据，返回DataFrame"""
    url = f"https://www.okx.com/api/v5/market/candles?instId={SYMBOL}&bar={bar}&limit={limit}"
    try:
        r = requests.get(url, timeout=5)
        data = r.json().get("data", [])
        df = pd.DataFrame(data, columns=["ts", "open", "high", "low", "close", "vol", "volCcy", "volCcyQuote", "confirm"])
        df["ts"] = pd.to_datetime(df["ts"].astype(int), unit="ms")
        for c in ["open", "high", "low", "close", "vol"]:
            df[c] = df[c].astype(float)
        return df.sort_values("ts").reset_index(drop=True)
    except Exception as e:
        st.error(f"数据获取失败: {e}")
        return pd.DataFrame()

# --------------------------
# 指标计算（5分钟）
# --------------------------
def compute_indicators(df_5m):
    df = df_5m.copy()
    df["EMA_fast"] = ta.trend.ema_indicator(df["close"], window=ema_fast_period)
    df["EMA_slow"] = ta.trend.ema_indicator(df["close"], window=ema_slow_period)
    df["ADX"] = ta.trend.adx(df["high"], df["low"], df["close"], window=14)
    df["ATR"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=atr_period)
    df["volume_ma"] = df["vol"].rolling(window=20).mean()
    # 趋势确认
    df["trend_up"] = (df["EMA_fast"] > df["EMA_slow"]) & (df["ADX"] > adx_threshold)
    df["trend_down"] = (df["EMA_fast"] < df["EMA_slow"]) & (df["ADX"] > adx_threshold)
    # 连续确认两期
    df["trend_up_streak"] = df["trend_up"].rolling(2).sum() == 2
    df["trend_down_streak"] = df["trend_down"].rolling(2).sum() == 2
    return df.dropna().reset_index(drop=True)

# --------------------------
# 结构点检测（预先计算，但回测时会延迟使用）
# --------------------------
def find_swing_points(df, window=3):
    """返回每个结构点的时间戳和价格，以及该点被确认的最早时间（=时间 + window*5分钟）"""
    highs, lows = [], []
    for i in range(window, len(df)-window):
        if df['high'].iloc[i] == max(df['high'].iloc[i-window:i+window+1]):
            confirm_time = df['ts'].iloc[i] + timedelta(minutes=5*window)
            highs.append((df['ts'].iloc[i], df['high'].iloc[i], confirm_time))
        if df['low'].iloc[i] == min(df['low'].iloc[i-window:i+window+1]):
            confirm_time = df['ts'].iloc[i] + timedelta(minutes=5*window)
            lows.append((df['ts'].iloc[i], df['low'].iloc[i], confirm_time))
    return highs, lows

# --------------------------
# 假突破检测
# --------------------------
def is_fake_break(row):
    body = abs(row["close"] - row["open"])
    if body == 0:
        return True
    upper = row["high"] - max(row["close"], row["open"])
    lower = min(row["close"], row["open"]) - row["low"]
    return upper > body * 1.5 or lower > body * 1.5

# --------------------------
# 模拟账户类（线程安全，持久化）
# --------------------------
class TradingAccount:
    def __init__(self, initial_balance):
        self.initial = initial_balance
        self.file = ACCOUNT_FILE
        self.lock = threading.Lock()
        self.load()

    def load(self):
        if os.path.exists(self.file):
            with self.lock:
                try:
                    with open(self.file, 'r') as f:
                        data = json.load(f)
                        self.balance = data.get('balance', self.initial)
                        self.position = data.get('position', None)
                        self.trades = data.get('trades', [])
                except:
                    self.reset()
        else:
            self.reset()

    def save(self):
        if len(self.trades) > 100:
            self.trades = self.trades[-100:]
        with self.lock:
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
            stop_loss = price - atr * atr_sl_mult
            take_profit = price + atr * atr_tp_mult
        else:
            stop_loss = price + atr * atr_sl_mult
            take_profit = price - atr * atr_tp_mult

        risk_amount = self.balance * (risk_percent / 100)
        stop_distance = abs(price - stop_loss)
        if stop_distance == 0:
            return False, "止损距离为零"

        raw_qty = risk_amount / stop_distance
        min_qty = 0.01
        qty = round(raw_qty / min_qty) * min_qty
        if qty < min_qty:
            qty = min_qty

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
                reason, close_price = "止损", stop
            elif current_price >= tp:
                reason, close_price = "止盈", tp
            else:
                return None
        else:
            if current_price >= stop:
                reason, close_price = "止损", stop
            elif current_price <= tp:
                reason, close_price = "止盈", tp
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
            return self.balance + (current_price - pos['entry']) * pos['qty']
        else:
            return self.balance + (pos['entry'] - current_price) * pos['qty']

# --------------------------
# 回测引擎（严格使用历史数据，初始资金与模拟账户一致）
# --------------------------
def run_backtest(df, df_15m_full, swing_highs, swing_lows, window):
    trades = []
    capital = initial_balance  # 使用侧边栏设定的初始资金
    position = None
    equity_curve = [capital]

    for i in range(1, len(df)-1):
        row = df.iloc[i]
        current_time = row['ts']

        # 15分钟趋势方向（只使用过去数据）
        past_15m = df_15m_full[df_15m_full['ts'] <= current_time - timedelta(minutes=15)]
        if len(past_15m) >= 2:
            last_ema = past_15m['EMA20'].iloc[-1]
            prev_ema = past_15m['EMA20'].iloc[-2]
            tf15_dir = "多" if last_ema > prev_ema else "空"
        else:
            tf15_dir = "无"

        # 已确认的结构点
        valid_highs = [(t, p) for t, p, ct in swing_highs if ct <= current_time]
        valid_lows = [(t, p) for t, p, ct in swing_lows if ct <= current_time]
        last_high = valid_highs[-1] if valid_highs else None
        last_low = valid_lows[-1] if valid_lows else None
        prev_high = valid_highs[-2] if len(valid_highs) >= 2 else None
        prev_low = valid_lows[-2] if len(valid_lows) >= 2 else None

        bull_structure = prev_low and last_low and last_low[1] > prev_low[1]
        bear_structure = prev_high and last_high and last_high[1] < prev_high[1]

        # 信号判断
        signal = None
        if (row["trend_up_streak"] and tf15_dir == "多" and bull_structure and not is_fake_break(row)):
            if last_high and row["close"] > last_high[1]:
                body = abs(row["close"] - row["open"])
                if body > (row["high"] - row["low"]) * 0.6:
                    if (row["close"] - last_high[1]) > row["ATR"] * 0.2:
                        if row["vol"] > row["volume_ma"] * volume_mult:
                            signal = "多"
        elif (row["trend_down_streak"] and tf15_dir == "空" and bear_structure and not is_fake_break(row)):
            if last_low and row["close"] < last_low[1]:
                body = abs(row["close"] - row["open"])
                if body > (row["high"] - row["low"]) * 0.6:
                    if (last_low[1] - row["close"]) > row["ATR"] * 0.2:
                        if row["vol"] > row["volume_ma"] * volume_mult:
                            signal = "空"

        # 开仓
        if signal and position is None:
            entry_price = df.iloc[i+1]["open"]
            if signal == "多":
                entry_price *= (1 + slippage/100)
            else:
                entry_price *= (1 - slippage/100)

            atr = row["ATR"]
            if signal == "多":
                sl = entry_price - atr * atr_sl_mult
                tp = entry_price + atr * atr_tp_mult
            else:
                sl = entry_price + atr * atr_sl_mult
                tp = entry_price - atr * atr_tp_mult

            risk_amount = capital * (risk_percent / 100)
            stop_dist = abs(entry_price - sl)
            if stop_dist > 0:
                qty = risk_amount / stop_dist
            else:
                qty = 0
            min_qty = 0.01
            qty = round(qty / min_qty) * min_qty
            if qty < min_qty:
                qty = min_qty

            fee = entry_price * qty * 0.0005
            if capital > fee:
                capital -= fee
                position = {
                    "direction": signal,
                    "entry": entry_price,
                    "qty": qty,
                    "sl": sl,
                    "tp": tp,
                    "open_idx": i+1,
                    "open_time": df.iloc[i+1]["ts"]
                }

        # 持仓管理
        if position is not None:
            exit_price = None
            reason = None
            for k in range(position["open_idx"]+1, min(position["open_idx"]+30, len(df))):
                cur = df.iloc[k]
                if position["direction"] == "多":
                    if cur["low"] <= position["sl"]:
                        exit_price = position["sl"]
                        reason = "止损"
                        break
                    elif cur["high"] >= position["tp"]:
                        exit_price = position["tp"]
                        reason = "止盈"
                        break
                else:
                    if cur["high"] >= position["sl"]:
                        exit_price = position["sl"]
                        reason = "止损"
                        break
                    elif cur["low"] <= position["tp"]:
                        exit_price = position["tp"]
                        reason = "止盈"
                        break
            if exit_price is None:
                exit_price = df.iloc[-1]["close"]
                reason = "时间平仓"

            if position["direction"] == "多":
                exit_price *= (1 - slippage/100)
            else:
                exit_price *= (1 + slippage/100)

            pnl = (exit_price - position["entry"]) * position["qty"] if position["direction"] == "多" \
                else (position["entry"] - exit_price) * position["qty"]
            fee = exit_price * position["qty"] * 0.0005
            net_pnl = pnl - fee
            capital += net_pnl
            trades.append({
                "时间": position["open_time"],
                "方向": position["direction"],
                "入场": round(position["entry"], 2),
                "离场": round(exit_price, 2),
                "盈亏": round(net_pnl, 2),
                "原因": reason
            })
            position = None

        equity_curve.append(capital)

    # 绩效计算
    if trades:
        df_t = pd.DataFrame(trades)
        wins = len(df_t[df_t["盈亏"] > 0])
        losses = len(df_t[df_t["盈亏"] < 0])
        total = wins + losses
        win_rate = wins / total * 100 if total > 0 else 0
        net_profit = capital - initial_balance
        if len(equity_curve) > 1:
            returns = np.diff(equity_curve) / equity_curve[:-1]
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(365*24*12) if np.std(returns) > 0 else 0
        else:
            sharpe = 0
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        max_dd = np.max(drawdown) * 100

        return trades, win_rate, net_profit, sharpe, max_dd, equity_curve
    else:
        return [], 0, 0, 0, 0, equity_curve

# --------------------------
# 主程序
# --------------------------
# 获取5分钟数据
df_5m = get_klines("5m", 500)
if df_5m.empty:
    st.error("无法获取行情数据")
    st.stop()

# 获取15分钟数据
df_15m_full = get_klines("15m", 200)
if not df_15m_full.empty:
    df_15m_full["EMA20"] = ta.trend.ema_indicator(df_15m_full["close"], window=20)
    df_15m_full = df_15m_full.dropna().reset_index(drop=True)
else:
    st.warning("15分钟数据获取失败，将使用5分钟自身趋势")
    df_15m_full = pd.DataFrame()

# 计算5分钟指标
df = compute_indicators(df_5m)

# 预先计算结构点
swing_highs, swing_lows = find_swing_points(df, window=swing_window)

# ---------- 实时信号判断 ----------
latest = df.iloc[-1]
current_time = latest['ts']

# 实时15分钟方向
past_15m = df_15m_full[df_15m_full['ts'] <= current_time - timedelta(minutes=15)]
if len(past_15m) >= 2:
    last_ema = past_15m['EMA20'].iloc[-1]
    prev_ema = past_15m['EMA20'].iloc[-2]
    tf15_dir_real = "多" if last_ema > prev_ema else "空"
else:
    tf15_dir_real = "无"

# 实时结构点
valid_highs = [(t, p) for t, p, ct in swing_highs if ct <= current_time]
valid_lows = [(t, p) for t, p, ct in swing_lows if ct <= current_time]
last_high = valid_highs[-1] if valid_highs else None
last_low = valid_lows[-1] if valid_lows else None
prev_high = valid_highs[-2] if len(valid_highs) >= 2 else None
prev_low = valid_lows[-2] if len(valid_lows) >= 2 else None

bull_structure = prev_low and last_low and last_low[1] > prev_low[1]
bear_structure = prev_high and last_high and last_high[1] < prev_high[1]

signal = None
if (latest["trend_up_streak"] and tf15_dir_real == "多" and bull_structure and not is_fake_break(latest)):
    if last_high and latest["close"] > last_high[1]:
        body = abs(latest["close"] - latest["open"])
        if body > (latest["high"] - latest["low"]) * 0.6:
            if (latest["close"] - last_high[1]) > latest["ATR"] * 0.2:
                if latest["vol"] > latest["volume_ma"] * volume_mult:
                    signal = "多"
elif (latest["trend_down_streak"] and tf15_dir_real == "空" and bear_structure and not is_fake_break(latest)):
    if last_low and latest["close"] < last_low[1]:
        body = abs(latest["close"] - latest["open"])
        if body > (latest["high"] - latest["low"]) * 0.6:
            if (last_low[1] - latest["close"]) > latest["ATR"] * 0.2:
                if latest["vol"] > latest["volume_ma"] * volume_mult:
                    signal = "空"

# ---------- 初始化模拟账户 ----------
if "account" not in st.session_state:
    st.session_state.account = TradingAccount(initial_balance=initial_balance)
account = st.session_state.account

# ---------- 计算交易计划（如果有信号且无持仓） ----------
plan = None
if signal and account.can_open():
    entry_price = latest["close"]
    atr = latest["ATR"]
    if signal == "多":
        stop_loss = entry_price - atr * atr_sl_mult
        take_profit = entry_price + atr * atr_tp_mult
    else:
        stop_loss = entry_price + atr * atr_sl_mult
        take_profit = entry_price - atr * atr_tp_mult

    risk_amount = account.balance * (risk_percent / 100)
    stop_distance = abs(entry_price - stop_loss)
    if stop_distance > 0:
        raw_qty = risk_amount / stop_distance
        min_qty = 0.01
        qty = round(raw_qty / min_qty) * min_qty
        if qty < min_qty:
            qty = min_qty
    else:
        qty = 0

    rr = abs(take_profit - entry_price) / abs(entry_price - stop_loss) if abs(entry_price - stop_loss) > 0 else 0

    plan = {
        "方向": signal,
        "入场价": round(entry_price, 2),
        "止损价": round(stop_loss, 2),
        "止盈价": round(take_profit, 2),
        "数量": round(qty, 4),
        "盈亏比": round(rr, 2),
        "理由": f"{tf15_dir_real}趋势 + 结构确认 + 放量突破"
    }

# ---------- 自动交易 ----------
if signal and account.can_open():
    success, msg = account.open_position(signal, latest["close"], latest["ATR"])
    if success:
        st.sidebar.success(f"✅ 自动开仓 {signal}")
    else:
        st.sidebar.warning(f"开仓失败: {msg}")

if account.position:
    result = account.check_stop_take(latest["close"])
    if result:
        reason, net = result
        st.sidebar.info(f"📌 平仓: {reason} 盈亏: {net:+.2f}")

# ---------- 顶部状态面板 ----------
c1, c2, c3, c4 = st.columns(4)
c1.metric("15分钟方向", tf15_dir_real)
c2.metric("ADX", f"{latest['ADX']:.1f}")
c3.metric("结构状态", f"多:{bull_structure} 空:{bear_structure}")
if signal:
    c4.success(f"📢 实时信号: {signal}")
else:
    c4.info("无信号")

# ---------- K线图（置顶）----------
fig = go.Figure(data=[go.Candlestick(
    x=df['ts'], open=df['open'], high=df['high'],
    low=df['low'], close=df['close'], name="ETH 5m"
)])
fig.add_trace(go.Scatter(x=df['ts'], y=df['EMA_fast'], line=dict(color='yellow', width=1), name=f"EMA{ema_fast_period}"))
fig.add_trace(go.Scatter(x=df['ts'], y=df['EMA_slow'], line=dict(color='orange', width=1), name=f"EMA{ema_slow_period}"))
# 绘制结构点
if swing_highs:
    sh_x, sh_y = zip(*[(t, p) for t, p, ct in swing_highs])
    fig.add_trace(go.Scatter(x=sh_x, y=sh_y, mode='markers', marker=dict(symbol='triangle-down', size=8, color='red'), name='结构高点'))
if swing_lows:
    sl_x, sl_y = zip(*[(t, p) for t, p, ct in swing_lows])
    fig.add_trace(go.Scatter(x=sl_x, y=sl_y, mode='markers', marker=dict(symbol='triangle-up', size=8, color='green'), name='结构低点'))
# 实时信号箭头
if signal:
    y_pos = latest['high'] * 1.002 if signal == "多" else latest['low'] * 0.998
    symbol = "arrow-up" if signal == "多" else "arrow-down"
    fig.add_trace(go.Scatter(
        x=[latest['ts']], y=[y_pos],
        mode='markers+text',
        marker=dict(symbol=symbol, size=15, color='yellow'),
        text=signal,
        textposition="top center" if signal=="多" else "bottom center",
        name='实时信号'
    ))
fig.update_layout(height=700, template="plotly_dark", xaxis_rangeslider_visible=False)
st.plotly_chart(fig, width='stretch')

# ---------- 交易计划展示 ----------
if plan:
    with st.container():
        st.markdown("---")
        st.subheader("📋 当前交易计划")
        col_p1, col_p2, col_p3, col_p4, col_p5 = st.columns(5)
        col_p1.metric("方向", plan["方向"])
        col_p2.metric("入场价", plan["入场价"])
        col_p3.metric("止损价", plan["止损价"])
        col_p4.metric("止盈价", plan["止盈价"])
        col_p5.metric("盈亏比", f"{plan['盈亏比']}:1")
        st.caption(f"计划开仓数量: {plan['数量']} 张 | 理由: {plan['理由']}")
        st.markdown("---")

# ---------- 侧边栏账户显示 ----------
with st.sidebar:
    st.divider()
    st.subheader("💰 模拟账户")
    st.write(f"余额: {account.balance:.2f} USDT")
    if account.position:
        pos = account.position
        st.write(f"持仓: {pos['direction']} {pos['qty']:.2f}张")
        st.write(f"入场: {pos['entry']:.2f} | 止损: {pos['stop_loss']:.2f} | 止盈: {pos['take_profit']:.2f}")
    else:
        st.write("无持仓")
    if st.button("重置账户"):
        account.reset()
        st.rerun()

# ---------- 回测结果（底部）----------
with st.spinner("回测进行中..."):
    trades, win_rate, net_profit, sharpe, max_dd, equity_curve = run_backtest(
        df, df_15m_full, swing_highs, swing_lows, swing_window
    )

if trades:
    st.subheader("📊 回测绩效 (20 USDT 初始资金)")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("总交易", len(trades))
    col2.metric("胜率", f"{win_rate:.1f}%")
    col3.metric("净利润", f"{net_profit:+.2f}")
    col4.metric("夏普比率", f"{sharpe:.2f}")
    col5.metric("最大回撤", f"{max_dd:.2f}%")
    st.dataframe(pd.DataFrame(trades).tail(20), width='stretch')

    # 资金曲线图
    fig_equity = go.Figure()
    equity_times = df['ts'].iloc[:len(equity_curve)]
    fig_equity.add_trace(go.Scatter(x=equity_times, y=equity_curve, mode='lines', name='资金曲线'))
    fig_equity.update_layout(title="资金曲线", xaxis_title="时间", yaxis_title="资金 (USDT)", height=400)
    st.plotly_chart(fig_equity, width='stretch')
else:
    st.warning("回测期间无交易")

# ---------- 信号历史 ----------
st.subheader("📜 最近信号记录")
if "signal_history" not in st.session_state:
    st.session_state.signal_history = []
if signal and (not st.session_state.signal_history or st.session_state.signal_history[-1]["时间"] != latest["ts"]):
    st.session_state.signal_history.append({
        "时间": latest["ts"],
        "方向": signal,
        "价格": round(latest["close"], 2),
        "ATR": round(latest["ATR"], 2)
    })
    if len(st.session_state.signal_history) > 50:
        st.session_state.signal_history = st.session_state.signal_history[-50:]

if st.session_state.signal_history:
    hist_df = pd.DataFrame(st.session_state.signal_history)
    hist_df['时间'] = hist_df['时间'].dt.strftime('%Y-%m-%d %H:%M')
    st.dataframe(hist_df, width='stretch')
else:
    st.info("暂无历史信号")
