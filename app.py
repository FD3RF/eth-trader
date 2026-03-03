"""
策略模块化回测实验室（长周期稳定版）
- 本地CSV数据存储，避免API重复拉取
- 增量更新：只拉取最新K线，自动合并
- 多周期验证：5分钟/15分钟一键切换
- 模块边际贡献分析
- 交易频率稳定性提示
"""

import streamlit as st
import pandas as pd
import ta
import plotly.graph_objects as go
import requests
import numpy as np
from datetime import datetime, timedelta
import time
import os

st.set_page_config(layout="wide", page_title="策略模块化回测实验室")
st.title("📊 策略模块化回测实验室")

SYMBOL = "ETH-USDT-SWAP"
DATA_DIR = "market_data"
os.makedirs(DATA_DIR, exist_ok=True)

# --------------------------
# 侧边栏参数
# --------------------------
with st.sidebar:
    st.header("⚙️ 回测设置")
    
    # 数据管理
    st.subheader("📁 数据管理")
    if st.button("🔄 更新本地数据"):
        with st.spinner("正在从OKX拉取历史数据..."):
            # 这里会调用更新函数
            st.experimental_rerun()
    
    data_source = st.radio("数据源", ["本地缓存", "实时API"], index=0)
    if data_source == "实时API":
        st.warning("实时API仅能获取最近1000根K线，建议使用本地缓存")
    
    # 周期选择
    tf = st.selectbox("时间周期", ["5m", "15m"], index=0)
    period = st.selectbox("回测长度", ["7天", "30天", "90天", "180天", "365天"], index=2)
    days_map = {"7天": 7, "30天": 30, "90天": 90, "180天": 180, "365天": 365}
    lookback_days = days_map[period]
    
    st.divider()
    st.header("🧩 模块开关")
    use_structure = st.checkbox("结构突破", value=True)
    use_trend = st.checkbox("趋势过滤 (15M)", value=False)
    use_volume = st.checkbox("成交量放大", value=False)
    use_fake_filter = st.checkbox("假突破过滤", value=False)
    
    st.divider()
    st.subheader("📐 策略参数")
    adx_threshold = st.slider("ADX阈值", 20, 35, 25)
    volume_mult = st.slider("放量倍数", 1.0, 2.0, 1.3, step=0.1)
    atr_sl_mult = st.number_input("止损ATR倍数", value=0.6, step=0.1)
    atr_tp_mult = st.number_input("止盈ATR倍数", value=0.8, step=0.1)
    risk_percent = st.slider("单笔风险 %", 0.5, 2.0, 1.0, step=0.1)
    slippage = st.number_input("滑点 %", value=0.05, step=0.01)
    
    run_btn = st.button("🚀 运行回测")

# --------------------------
# 本地数据维护函数
# --------------------------
def get_local_filename(tf):
    return os.path.join(DATA_DIR, f"{SYMBOL}_{tf}.csv")

def load_local_data(tf):
    fname = get_local_filename(tf)
    if os.path.exists(fname):
        df = pd.read_csv(fname, parse_dates=["ts"])
        return df
    return pd.DataFrame()

def save_local_data(df, tf):
    fname = get_local_filename(tf)
    df.to_csv(fname, index=False)

def update_local_data(tf):
    """增量更新本地数据：只拉取最新的K线并合并"""
    local_df = load_local_data(tf)
    if local_df.empty:
        # 无本地数据，拉取全部历史
        st.info(f"首次拉取{tf}历史数据...")
        df = fetch_all_history(tf, days=365)  # 拉取尽可能多
        if not df.empty:
            save_local_data(df, tf)
        return df
    else:
        # 已有本地数据，只拉取最新部分
        last_ts = local_df["ts"].max()
        now = datetime.now()
        new_df = fetch_incremental(tf, last_ts, now)
        if not new_df.empty:
            combined = pd.concat([local_df, new_df]).drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
            save_local_data(combined, tf)
            return combined
        else:
            return local_df

def fetch_all_history(tf, days=365):
    """拉取所有历史数据（从当前向前追溯）"""
    end = datetime.now()
    start = end - timedelta(days=days)
    all_data = []
    current_end = end
    limit = 1000
    pbar = st.progress(0, text="正在拉取历史数据...")
    total_expected = days * (24*60//int(tf[:-1]))  # 粗略估计
    fetched = 0
    
    while current_end > start:
        before = int(current_end.timestamp() * 1000)
        url = f"https://www.okx.com/api/v5/market/candles?instId={SYMBOL}&bar={tf}&limit={limit}&before={before}"
        try:
            r = requests.get(url, timeout=10)
            data = r.json().get("data", [])
            if not data:
                break
            df_chunk = pd.DataFrame(data, columns=["ts", "open", "high", "low", "close", "vol", "volCcy", "volCcyQuote", "confirm"])
            df_chunk["ts"] = pd.to_datetime(df_chunk["ts"].astype(int), unit="ms")
            for c in ["open", "high", "low", "close", "vol"]:
                df_chunk[c] = df_chunk[c].astype(float)
            all_data.append(df_chunk)
            current_end = df_chunk["ts"].min()
            fetched += len(df_chunk)
            pbar.progress(min(fetched / total_expected, 1.0))
            time.sleep(0.2)
        except Exception as e:
            st.error(f"拉取失败: {e}")
            break
    pbar.empty()
    if all_data:
        full = pd.concat(all_data, ignore_index=True).sort_values("ts").drop_duplicates().reset_index(drop=True)
        return full[full["ts"] >= start]
    return pd.DataFrame()

def fetch_incremental(tf, from_ts, to_ts):
    """拉取从from_ts到to_ts之间的数据（用于增量更新）"""
    all_data = []
    current_end = to_ts
    limit = 1000
    while current_end > from_ts:
        before = int(current_end.timestamp() * 1000)
        url = f"https://www.okx.com/api/v5/market/candles?instId={SYMBOL}&bar={tf}&limit={limit}&before={before}"
        try:
            r = requests.get(url, timeout=10)
            data = r.json().get("data", [])
            if not data:
                break
            df_chunk = pd.DataFrame(data, columns=["ts", "open", "high", "low", "close", "vol", "volCcy", "volCcyQuote", "confirm"])
            df_chunk["ts"] = pd.to_datetime(df_chunk["ts"].astype(int), unit="ms")
            for c in ["open", "high", "low", "close", "vol"]:
                df_chunk[c] = df_chunk[c].astype(float)
            chunk_min = df_chunk["ts"].min()
            if chunk_min <= from_ts:
                # 只保留 from_ts 之后的部分
                df_chunk = df_chunk[df_chunk["ts"] > from_ts]
                all_data.append(df_chunk)
                break
            else:
                all_data.append(df_chunk)
                current_end = chunk_min
            time.sleep(0.2)
        except:
            break
    if all_data:
        return pd.concat(all_data, ignore_index=True).sort_values("ts").drop_duplicates().reset_index(drop=True)
    return pd.DataFrame()

# --------------------------
# 数据准备
# --------------------------
if data_source == "本地缓存":
    # 检查是否需要更新
    if "last_update" not in st.session_state or st.sidebar.button("🔄 检查并更新"):
        df_raw = update_local_data(tf)
        st.session_state["last_update"] = datetime.now()
    else:
        df_raw = load_local_data(tf)
else:
    # 实时API，仅获取最近1000根
    url = f"https://www.okx.com/api/v5/market/candles?instId={SYMBOL}&bar={tf}&limit=1000"
    try:
        r = requests.get(url, timeout=5)
        data = r.json().get("data", [])
        df_raw = pd.DataFrame(data, columns=["ts", "open", "high", "low", "close", "vol", "volCcy", "volCcyQuote", "confirm"])
        df_raw["ts"] = pd.to_datetime(df_raw["ts"].astype(int), unit="ms")
        for c in ["open", "high", "low", "close", "vol"]:
            df_raw[c] = df_raw[c].astype(float)
        df_raw = df_raw.sort_values("ts").reset_index(drop=True)
    except:
        st.error("实时API获取失败")
        st.stop()

if df_raw.empty:
    st.error("无数据")
    st.stop()

# 截取指定天数
start_date = datetime.now() - timedelta(days=lookback_days)
df_raw = df_raw[df_raw["ts"] >= start_date].reset_index(drop=True)

st.success(f"数据加载完成：{len(df_raw)} 根 {tf} K线，时间范围 {df_raw['ts'].min()} 至 {df_raw['ts'].max()}")

# --------------------------
# 指标计算
# --------------------------
df = df_raw.copy()
df["EMA_fast"] = ta.trend.ema_indicator(df["close"], window=8)
df["EMA_slow"] = ta.trend.ema_indicator(df["close"], window=30)
df["ADX"] = ta.trend.adx(df["high"], df["low"], df["close"], window=14)
df["ATR"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=10)
df["volume_ma"] = df["vol"].rolling(window=20).mean()
df = df.dropna().reset_index(drop=True)

# 如果需要15分钟趋势，则获取15分钟数据
df_15m = None
if use_trend:
    if data_source == "本地缓存":
        df_15m = load_local_data("15m")
        if df_15m.empty:
            st.warning("无15分钟本地数据，尝试实时拉取...")
            # 简化：实时拉取1000根
            url = f"https://www.okx.com/api/v5/market/candles?instId={SYMBOL}&bar=15m&limit=1000"
            r = requests.get(url).json()
            data = r.get("data", [])
            df_15m = pd.DataFrame(data, columns=["ts", "open", "high", "low", "close", "vol", "volCcy", "volCcyQuote", "confirm"])
            df_15m["ts"] = pd.to_datetime(df_15m["ts"].astype(int), unit="ms")
            for c in ["open", "high", "low", "close", "vol"]:
                df_15m[c] = df_15m[c].astype(float)
            df_15m = df_15m.sort_values("ts").reset_index(drop=True)
    else:
        # 实时API
        url = f"https://www.okx.com/api/v5/market/candles?instId={SYMBOL}&bar=15m&limit=1000"
        r = requests.get(url).json()
        data = r.get("data", [])
        df_15m = pd.DataFrame(data, columns=["ts", "open", "high", "low", "close", "vol", "volCcy", "volCcyQuote", "confirm"])
        df_15m["ts"] = pd.to_datetime(df_15m["ts"].astype(int), unit="ms")
        for c in ["open", "high", "low", "close", "vol"]:
            df_15m[c] = df_15m[c].astype(float)
        df_15m = df_15m.sort_values("ts").reset_index(drop=True)
    
    if not df_15m.empty:
        df_15m["EMA20"] = ta.trend.ema_indicator(df_15m["close"], window=20)
        df_15m = df_15m.dropna().reset_index(drop=True)
    else:
        st.warning("15分钟数据不可用，趋势过滤禁用")
        use_trend = False

# --------------------------
# 结构点检测
# --------------------------
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
# 回测函数（同前）
# --------------------------
def run_backtest_with_modules(df, df_15m, swing_highs, swing_lows,
                               use_structure, use_trend, use_volume, use_fake_filter):
    capital = 20.0
    position = None
    trades = []
    equity_curve = [capital]

    for i in range(1, len(df)-1):
        row = df.iloc[i]
        current_time = row['ts']

        # 15分钟趋势
        tf_dir = "无"
        if use_trend and df_15m is not None and not df_15m.empty:
            past_15m = df_15m[df_15m['ts'] <= current_time - timedelta(minutes=15)]
            if len(past_15m) >= 2:
                last_ema = past_15m['EMA20'].iloc[-1]
                prev_ema = past_15m['EMA20'].iloc[-2]
                tf_dir = "多" if last_ema > prev_ema else "空"

        # 结构点
        valid_highs = [(t, p) for t, p, ct in swing_highs if ct <= current_time]
        valid_lows = [(t, p) for t, p, ct in swing_lows if ct <= current_time]
        last_high = valid_highs[-1] if valid_highs else None
        last_low = valid_lows[-1] if valid_lows else None
        prev_high = valid_highs[-2] if len(valid_highs) >= 2 else None
        prev_low = valid_lows[-2] if len(valid_lows) >= 2 else None

        bull_structure = prev_low and last_low and last_low[1] > prev_low[1]
        bear_structure = prev_high and last_high and last_high[1] < prev_high[1]

        # 信号
        signal = None
        if use_structure:
            if bull_structure and last_high and row["close"] > last_high[1]:
                signal = "多"
            elif bear_structure and last_low and row["close"] < last_low[1]:
                signal = "空"

        if signal and use_trend:
            if signal == "多" and tf_dir != "多":
                signal = None
            elif signal == "空" and tf_dir != "空":
                signal = None

        if signal and use_volume:
            if row["vol"] <= row["volume_ma"] * volume_mult:
                signal = None

        if signal and use_fake_filter:
            if is_fake_break(row):
                signal = None

        # 开仓
        if signal and position is None:
            entry_price = df.iloc[i+1]["open"]
            if signal == "多":
                entry_price *= (1 + slippage/100)
                sl = entry_price - row["ATR"] * atr_sl_mult
                tp = entry_price + row["ATR"] * atr_tp_mult
            else:
                entry_price *= (1 - slippage/100)
                sl = entry_price + row["ATR"] * atr_sl_mult
                tp = entry_price - row["ATR"] * atr_tp_mult

            risk_amount = capital * (risk_percent / 100)
            stop_dist = abs(entry_price - sl)
            if stop_dist <= 0:
                continue
            qty = risk_amount / stop_dist
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

    # 绩效
    if trades:
        df_t = pd.DataFrame(trades)
        wins = len(df_t[df_t["盈亏"] > 0])
        losses = len(df_t[df_t["盈亏"] < 0])
        total = wins + losses
        win_rate = wins / total * 100 if total > 0 else 0
        net_profit = capital - 20
        if len(equity_curve) > 1:
            returns = np.diff(equity_curve) / equity_curve[:-1]
            if np.std(returns) > 1e-9:
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(365*24*12)
            else:
                sharpe = 0
        else:
            sharpe = 0
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        max_dd = np.max(drawdown) * 100
        return trades, win_rate, net_profit, sharpe, max_dd, equity_curve, total
    else:
        return [], 0, 0, 0, 0, equity_curve, 0

# --------------------------
# 运行回测
# --------------------------
if run_btn:
    with st.spinner("回测进行中..."):
        trades, win_rate, net_profit, sharpe, max_dd, equity_curve, trade_cnt = run_backtest_with_modules(
            df, df_15m, swing_highs, swing_lows,
            use_structure, use_trend, use_volume, use_fake_filter
        )

    st.subheader(f"📈 当前模块组合回测结果 ({period})")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("总交易", trade_cnt)
    col2.metric("胜率", f"{win_rate:.1f}%")
    col3.metric("净利润", f"{net_profit:+.2f} USDT")
    col4.metric("夏普比率", f"{sharpe:.2f}")
    col5.metric("最大回撤", f"{max_dd:.2f}%")
    col6.metric("平均每周", f"{trade_cnt/(lookback_days/7):.1f}")

    if len(equity_curve) > 1:
        fig_eq = go.Figure()
        times = df['ts'].iloc[:len(equity_curve)]
        fig_eq.add_trace(go.Scatter(x=times, y=equity_curve, mode='lines'))
        fig_eq.update_layout(title="资金曲线", height=400)
        st.plotly_chart(fig_eq, width='stretch')

    if trades:
        st.dataframe(pd.DataFrame(trades).tail(20), width='stretch')
    else:
        st.info("无交易")

# --------------------------
# 多组合对比（自动运行）
# --------------------------
st.divider()
st.subheader("📊 模块组合对比分析")

combinations = [
    {"name": "仅结构突破", "s": True, "t": False, "v": False, "f": False},
    {"name": "+趋势过滤", "s": True, "t": True, "v": False, "f": False},
    {"name": "+成交量", "s": True, "t": True, "v": True, "f": False},
    {"name": "+假突破过滤", "s": True, "t": True, "v": True, "f": True},
]

results = []
base_profit = None

for comb in combinations:
    t, wr, npnl, shr, mdd, _, cnt = run_backtest_with_modules(
        df, df_15m, swing_highs, swing_lows,
        comb["s"], comb["t"], comb["v"], comb["f"]
    )
    results.append({
        "组合": comb["name"],
        "交易次数": cnt,
        "胜率%": round(wr, 1),
        "净利润": round(npnl, 2),
        "夏普": round(shr, 2),
        "最大回撤%": round(mdd, 2)
    })

df_comp = pd.DataFrame(results)
st.dataframe(df_comp, width='stretch')

# 边际贡献
base_row = df_comp[df_comp["组合"] == "仅结构突破"]
if not base_row.empty:
    base_profit = base_row["净利润"].values[0]
    base_cnt = base_row["交易次数"].values[0]
    base_win = base_row["胜率%"].values[0]
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
    st.dataframe(pd.DataFrame(contrib), width='stretch')

# 稳定性提示
if len(df) < 1000:
    st.warning("数据量较少，回测结果可能不稳定。建议使用本地缓存获取更长周期。")
elif any(r["交易次数"] < 20 for r in results):
    st.warning("部分组合交易次数过少（<20），统计结论可能不可靠。")
