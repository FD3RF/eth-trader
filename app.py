"""
小利润交易策略完整版（优化版）
- EMA趋势 + ADX过滤 + 结构突破
- 小止损小止盈 + 滑点 + 手续费
- 模拟账户 + 回测绩效指标（胜率、盈亏比、夏普、最大回撤）
- 数据缓存 + 增量更新
- 模块化对比 + 边际贡献分析
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import ta
import os
import time
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(layout="wide", page_title="小利润策略·专业版")
st.title("📈 小利润交易策略（专业版）")

SYMBOL = "ETH-USDT-SWAP"
DATA_DIR = "market_data"
os.makedirs(DATA_DIR, exist_ok=True)

# ==========================
# 数据层（CSV + 增量）
# ==========================
def get_filename(tf):
    return os.path.join(DATA_DIR, f"{SYMBOL}_{tf}.csv")

def load_local_data(tf):
    if os.path.exists(get_filename(tf)):
        return pd.read_csv(get_filename(tf), parse_dates=["ts"])
    return pd.DataFrame()

def save_local_data(df, tf):
    df = df.drop_duplicates(subset=["ts"]).sort_values("ts")
    df.to_csv(get_filename(tf), index=False)

def fetch_okx_chunk(tf, before, limit=1000):
    """拉取单批K线数据（返回DataFrame或空）"""
    url = "https://www.okx.com/api/v5/market/candles"
    params = {"instId": SYMBOL, "bar": tf, "limit": limit}
    if before:
        params["before"] = before
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json().get("data", [])
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data, columns=["ts", "open", "high", "low", "close", "vol", "volCcy", "volCcyQuote", "confirm"])
        df["ts"] = pd.to_datetime(df["ts"].astype(int), unit="ms")
        for c in ["open", "high", "low", "close", "vol"]:
            df[c] = df[c].astype(float)
        return df
    except Exception as e:
        st.error(f"请求失败: {e}")
        return pd.DataFrame()

def fetch_all_history(tf, days=365):
    """拉取全部历史数据（从当前向前追溯）"""
    end = datetime.now()
    start = end - timedelta(days=days)
    all_data = []
    current_end = end
    limit = 1000
    pbar = st.progress(0, text="正在拉取历史数据...")
    # 估算总批次数（粗略）
    total_batches = days // 3 + 1  # 每批约3天数据
    batch_count = 0

    while current_end > start:
        before = int(current_end.timestamp() * 1000)
        df_chunk = fetch_okx_chunk(tf, before, limit)
        if df_chunk.empty:
            break
        all_data.append(df_chunk)
        # 更新current_end为这批数据的最早时间
        current_end = df_chunk["ts"].min()
        batch_count += 1
        pbar.progress(min(batch_count / total_batches, 1.0))
        time.sleep(0.2)  # 礼貌性延迟

    pbar.empty()
    if all_data:
        full = pd.concat(all_data, ignore_index=True).sort_values("ts").drop_duplicates().reset_index(drop=True)
        full = full[full["ts"] >= start]
        return full
    return pd.DataFrame()

def update_local_data(tf, days=365):
    """增量更新本地数据"""
    local = load_local_data(tf)
    if local.empty:
        df = fetch_all_history(tf, days)
        if not df.empty:
            save_local_data(df, tf)
        return df
    else:
        # 只拉取最新数据
        last_ts = local["ts"].max()
        now = datetime.now()
        new_chunks = []
        current = now
        while current > last_ts:
            before = int(current.timestamp() * 1000)
            chunk = fetch_okx_chunk(tf, before, 1000)
            if chunk.empty:
                break
            chunk = chunk[chunk["ts"] > last_ts]
            if not chunk.empty:
                new_chunks.append(chunk)
            current = chunk["ts"].min() if not chunk.empty else last_ts
            time.sleep(0.2)
        if new_chunks:
            new_df = pd.concat(new_chunks, ignore_index=True)
            combined = pd.concat([local, new_df]).drop_duplicates(subset=["ts"]).sort_values("ts")
            save_local_data(combined, tf)
            return combined
        return local

# ==========================
# 策略参数
# ==========================
with st.sidebar:
    st.header("⚙️ 参数设置")
    tf = st.selectbox("周期", ["5m", "15m"], index=0)
    days = st.selectbox("回测长度", ["30天", "90天", "180天", "365天"], index=1)
    days_map = {"30天": 30, "90天": 90, "180天": 180, "365天": 365}
    lookback = days_map[days]

    st.divider()
    st.subheader("🧩 模块开关")
    use_structure = st.checkbox("结构突破", value=True)
    use_trend = st.checkbox("趋势过滤", value=False)
    use_volume = st.checkbox("成交量放大", value=False)
    use_fake = st.checkbox("假突破过滤", value=False)

    st.divider()
    st.subheader("📐 策略参数")
    ema_fast = st.slider("EMA快线", 5, 20, 8)
    ema_slow = st.slider("EMA慢线", 20, 60, 30)
    adx_thr = st.slider("ADX阈值", 18, 35, 22)
    vol_mult = st.slider("放量倍数", 1.0, 2.0, 1.3, step=0.1)
    sl_mult = st.number_input("止损ATR倍数", 0.4, 1.2, 0.6, step=0.1)
    tp_mult = st.number_input("止盈ATR倍数", 0.6, 2.0, 0.9, step=0.1)
    risk_pct = st.slider("单笔风险 %", 0.5, 2.0, 1.0, step=0.1)
    slippage = st.number_input("滑点 %", 0.0, 0.2, 0.05, step=0.01)

    update_btn = st.button("🔄 更新本地数据")
    run_btn = st.button("🚀 运行回测")

# ==========================
# 数据加载
# ==========================
if update_btn:
    with st.spinner("正在更新数据..."):
        df_raw = update_local_data(tf, days=lookback)
        st.success("数据更新完成")
else:
    df_raw = load_local_data(tf)
    if df_raw.empty:
        st.warning("本地无数据，请点击「更新本地数据」")
        st.stop()

# 截取指定天数
start_date = datetime.now() - timedelta(days=lookback)
df_raw = df_raw[df_raw["ts"] >= start_date].reset_index(drop=True)
if df_raw.empty:
    st.error(f"数据不足，请尝试更短周期或更新数据")
    st.stop()

st.success(f"数据范围: {df_raw['ts'].min()} 至 {df_raw['ts'].max()} ({len(df_raw)} 根K线)")

# ==========================
# 指标计算
# ==========================
df = df_raw.copy()
df["EMA_fast"] = ta.trend.ema_indicator(df["close"], window=ema_fast)
df["EMA_slow"] = ta.trend.ema_indicator(df["close"], window=ema_slow)
df["ADX"] = ta.trend.adx(df["high"], df["low"], df["close"], window=14)
df["ATR"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=10)
df["volume_ma"] = df["vol"].rolling(window=20).mean()
df = df.dropna().reset_index(drop=True)

# ==========================
# 结构点检测（延迟确认）
# ==========================
def find_swing_points(df, window=3):
    highs, lows = [], []
    for i in range(window, len(df)-window):
        if df['high'].iloc[i] == max(df['high'].iloc[i-window:i+window+1]):
            confirm_time = df['ts'].iloc[i] + timedelta(minutes=5*window)
            highs.append((df['ts'].iloc[i], df['high'].iloc[i], confirm_time))
        if df['low'].iloc[i] == min(df['low'].iloc[i-window:i+window+1]):
            confirm_time = df['ts'].iloc[i] + timedelta(minutes=5*window)
            lows.append((df['ts'].iloc[i], df['low'].iloc[i], confirm_time))
    return highs, lows

swing_highs, swing_lows = find_swing_points(df, window=3)

# ==========================
# 假突破检测
# ==========================
def is_fake(row):
    body = abs(row["close"] - row["open"])
    if body == 0:
        return True
    upper = row["high"] - max(row["close"], row["open"])
    lower = min(row["close"], row["open"]) - row["low"]
    return upper > body * 1.5 or lower > body * 1.5

# ==========================
# 回测引擎（模块化）
# ==========================
def run_backtest(df, swing_highs, swing_lows, modules):
    capital = 20.0
    position = None
    trades = []
    equity_curve = [capital]

    for i in range(1, len(df)-1):
        row = df.iloc[i]
        current_time = row['ts']

        # 获取截止当前时间的结构点
        valid_highs = [(t, p) for t, p, ct in swing_highs if ct <= current_time]
        valid_lows = [(t, p) for t, p, ct in swing_lows if ct <= current_time]
        last_high = valid_highs[-1] if valid_highs else None
        last_low = valid_lows[-1] if valid_lows else None
        prev_high = valid_highs[-2] if len(valid_highs) >= 2 else None
        prev_low = valid_lows[-2] if len(valid_lows) >= 2 else None

        bull_struct = prev_low and last_low and last_low[1] > prev_low[1]
        bear_struct = prev_high and last_high and last_high[1] < prev_high[1]

        # 基础信号：结构突破
        signal = None
        if modules["structure"]:
            if bull_struct and last_high and row["close"] > last_high[1]:
                signal = "多"
            elif bear_struct and last_low and row["close"] < last_low[1]:
                signal = "空"

        # 趋势过滤
        if signal and modules["trend"]:
            trend_up = row["EMA_fast"] > row["EMA_slow"]
            trend_down = row["EMA_fast"] < row["EMA_slow"]
            if signal == "多" and not trend_up:
                signal = None
            if signal == "空" and not trend_down:
                signal = None

        # ADX过滤（内置）
        if signal and modules.get("adx", True):  # 默认开启ADX
            if row["ADX"] <= adx_thr:
                signal = None

        # 成交量放大
        if signal and modules["volume"]:
            if row["vol"] <= row["volume_ma"] * vol_mult:
                signal = None

        # 假突破过滤
        if signal and modules["fake"]:
            if is_fake(row):
                signal = None

        # 开仓
        if signal and position is None:
            entry_price = df.iloc[i+1]["open"]
            # 滑点
            if signal == "多":
                entry_price *= (1 + slippage/100)
                sl = entry_price - row["ATR"] * sl_mult
                tp = entry_price + row["ATR"] * tp_mult
            else:
                entry_price *= (1 - slippage/100)
                sl = entry_price + row["ATR"] * sl_mult
                tp = entry_price - row["ATR"] * tp_mult

            risk = capital * (risk_pct / 100)
            stop_dist = abs(entry_price - sl)
            if stop_dist <= 0:
                continue
            qty = risk / stop_dist
            qty = round(qty / 0.01) * 0.01
            if qty < 0.01:
                qty = 0.01

            fee = entry_price * qty * 0.0005
            if capital > fee:
                capital -= fee
                position = {
                    "dir": signal,
                    "entry": entry_price,
                    "sl": sl,
                    "tp": tp,
                    "qty": qty,
                    "open_idx": i+1,
                    "open_time": df.iloc[i+1]["ts"]
                }

        # 持仓管理
        if position:
            exit_price = None
            reason = None
            for k in range(position["open_idx"]+1, min(position["open_idx"]+30, len(df))):
                cur = df.iloc[k]
                if position["dir"] == "多":
                    if cur["low"] <= position["sl"]:
                        exit_price = position["sl"]
                        reason = "止损"
                        break
                    if cur["high"] >= position["tp"]:
                        exit_price = position["tp"]
                        reason = "止盈"
                        break
                else:
                    if cur["high"] >= position["sl"]:
                        exit_price = position["sl"]
                        reason = "止损"
                        break
                    if cur["low"] <= position["tp"]:
                        exit_price = position["tp"]
                        reason = "止盈"
                        break
            if exit_price is None:
                exit_price = df.iloc[-1]["close"]
                reason = "时间"

            # 离场滑点
            if position["dir"] == "多":
                exit_price *= (1 - slippage/100)
            else:
                exit_price *= (1 + slippage/100)

            pnl = (exit_price - position["entry"]) * position["qty"] if position["dir"]=="多" else (position["entry"]-exit_price)*position["qty"]
            fee = exit_price * position["qty"] * 0.0005
            net = pnl - fee
            capital += net
            trades.append({
                "时间": position["open_time"],
                "方向": position["dir"],
                "入场": round(position["entry"], 2),
                "离场": round(exit_price, 2),
                "盈亏": round(net, 2),
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
        net_profit = capital - 20
        # 盈亏比
        avg_win = df_t[df_t["盈亏"]>0]["盈亏"].mean() if wins>0 else 0
        avg_loss = abs(df_t[df_t["盈亏"]<0]["盈亏"].mean()) if losses>0 else 0
        rr = avg_win / avg_loss if avg_loss>0 else 0

        # 夏普比率
        if len(equity_curve) > 1:
            returns = np.diff(equity_curve) / equity_curve[:-1]
            if np.std(returns) > 1e-9:
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(365*24*12)
            else:
                sharpe = 0
        else:
            sharpe = 0

        # 最大回撤
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        max_dd = np.max(drawdown) * 100

        return trades, win_rate, net_profit, sharpe, max_dd, rr, equity_curve, total
    else:
        return [], 0, 0, 0, 0, 0, equity_curve, 0

# ==========================
# 运行回测（当前模块组合）
# ==========================
if run_btn:
    modules = {
        "structure": use_structure,
        "trend": use_trend,
        "volume": use_volume,
        "fake": use_fake,
        "adx": True  # ADX始终启用（可改为可选）
    }
    with st.spinner("回测进行中..."):
        trades, win_rate, net_profit, sharpe, max_dd, rr, equity_curve, trade_cnt = run_backtest(
            df, swing_highs, swing_lows, modules
        )

    st.subheader(f"📊 回测结果 ({days})")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("交易次数", trade_cnt)
    col2.metric("胜率", f"{win_rate:.1f}%")
    col3.metric("净利润", f"{net_profit:+.2f}")
    col4.metric("盈亏比", f"{rr:.2f}")
    col5.metric("夏普", f"{sharpe:.2f}")
    col6.metric("最大回撤", f"{max_dd:.2f}%")

    if trades:
        # 资金曲线
        fig_eq = go.Figure()
        times = df['ts'].iloc[:len(equity_curve)]
        fig_eq.add_trace(go.Scatter(x=times, y=equity_curve, mode='lines', name='资金曲线'))
        fig_eq.update_layout(title="资金曲线", height=400)
        st.plotly_chart(fig_eq, use_container_width=True)

        st.dataframe(pd.DataFrame(trades).tail(20), use_container_width=True)
    else:
        st.info("该组合下无交易")

# ==========================
# 模块组合对比（自动运行）
# ==========================
st.divider()
st.subheader("📈 模块组合对比分析")

combinations = [
    {"name": "仅结构突破", "structure": True, "trend": False, "volume": False, "fake": False},
    {"name": "+趋势过滤", "structure": True, "trend": True, "volume": False, "fake": False},
    {"name": "+成交量", "structure": True, "trend": True, "volume": True, "fake": False},
    {"name": "+假突破过滤", "structure": True, "trend": True, "volume": True, "fake": True},
]

results = []
for comb in combinations:
    modules = {**comb, "adx": True}
    t, wr, npnl, shr, mdd, rr, _, cnt = run_backtest(df, swing_highs, swing_lows, modules)
    results.append({
        "组合": comb["name"],
        "交易次数": cnt,
        "胜率%": round(wr, 1),
        "净利润": round(npnl, 2),
        "盈亏比": round(rr, 2),
        "夏普": round(shr, 2),
        "最大回撤%": round(mdd, 2)
    })

df_comp = pd.DataFrame(results)
st.dataframe(df_comp, use_container_width=True)

# 边际贡献分析
base_row = df_comp[df_comp["组合"] == "仅结构突破"]
if not base_row.empty:
    base_profit = base_row["净利润"].values[0]
    base_win = base_row["胜率%"].values[0]
    base_cnt = base_row["交易次数"].values[0]
    st.subheader("📌 边际贡献（相对于基础组合）")
    contrib = []
    for _, row in df_comp.iterrows():
        if row["组合"] == "仅结构突破":
            continue
        contrib.append({
            "组合": row["组合"],
            "净利润变化": f"{row['净利润'] - base_profit:+.2f}",
            "交易次数变化": f"{row['交易次数'] - base_cnt:+d}",
            "胜率变化": f"{row['胜率%'] - base_win:+.1f}%"
        })
    st.dataframe(pd.DataFrame(contrib), use_container_width=True)

# ==========================
# K线图
# ==========================
st.subheader("📉 K线图（最后200根）")
df_plot = df.tail(200)
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df_plot["ts"], open=df_plot["open"], high=df_plot["high"],
    low=df_plot["low"], close=df_plot["close"], name="K线"
))
fig.add_trace(go.Scatter(x=df_plot["ts"], y=df_plot["EMA_fast"], name="EMA快", line=dict(color='yellow')))
fig.add_trace(go.Scatter(x=df_plot["ts"], y=df_plot["EMA_slow"], name="EMA慢", line=dict(color='orange')))

# 绘制结构点（最后部分）
swing_highs_plot = [(t, p) for t, p, ct in swing_highs if t >= df_plot['ts'].min()]
swing_lows_plot = [(t, p) for t, p, ct in swing_lows if t >= df_plot['ts'].min()]
if swing_highs_plot:
    sh_x, sh_y = zip(*swing_highs_plot)
    fig.add_trace(go.Scatter(x=sh_x, y=sh_y, mode='markers', marker=dict(symbol='triangle-down', size=8, color='red'), name='结构高点'))
if swing_lows_plot:
    sl_x, sl_y = zip(*swing_lows_plot)
    fig.add_trace(go.Scatter(x=sl_x, y=sl_y, mode='markers', marker=dict(symbol='triangle-up', size=8, color='green'), name='结构低点'))

fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)
