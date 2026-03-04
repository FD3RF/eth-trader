import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BASE_URL = "https://www.okx.com"

st.set_page_config(layout="wide", page_title="统一突破系统·专业版", page_icon="📈")
st.title("📊 统一突破系统（专业版：N周期突破 + 趋势过滤）")

# =========================
# 侧边栏参数
# =========================
with st.sidebar:
    st.header("策略参数")
    lookback = st.slider("突破观察周期", min_value=2, max_value=50, value=20, step=1,
                         help="计算过去N根K线的最高/最低，突破该区间才视为有效")
    body_threshold = st.slider("实体比例阈值", 0.1, 0.8, 0.4, 0.05,
                               help="K线实体高度占整个波动的比例阈值")
    vol_ma_period = st.slider("成交量均线周期", 5, 50, 20,
                              help="计算成交量均线的周期数")
    vol_multiplier = st.slider("成交量放大倍数", 1.0, 3.0, 1.5, 0.1,
                               help="成交量需大于均线的倍数")
    use_trend_filter = st.checkbox("启用趋势过滤 (EMA50)", value=True,
                                   help="只在价格位于EMA50上方时接受买入，下方时接受卖出")
    fee_rate = st.number_input("手续费率 (单边)", 
                               min_value=0.0, max_value=0.01, value=0.0005, step=0.0001, format="%.4f",
                               help="开平仓各收取一次")

# =========================
# 获取K线数据
# =========================
@st.cache_data(ttl=30)
def get_candles(limit=500):
    url = f"{BASE_URL}/api/v5/market/candles"
    r = requests.get(url, params={
        "instId": "ETH-USDT",
        "bar": "5m",
        "limit": limit
    }).json()

    df = pd.DataFrame(r["data"], columns=[
        "ts", "o", "h", "l", "c", "v", "volCcy", "volCcyQuote", "confirm"
    ])[::-1]

    df["time"] = pd.to_datetime(df["ts"].astype(float), unit="ms")
    df[["o", "h", "l", "c", "v"]] = df[["o", "h", "l", "c", "v"]].astype(float)
    return df

df = get_candles(limit=500)  # 获取更多数据，以便计算EMA和滚动窗口

# =========================
# 核心突破逻辑（N周期突破 + 趋势过滤）
# =========================
def breakout_logic(df):
    df = df.copy()
    # 过去N根K线的最高、最低（滚动窗口，shift(1)避免用到当前K线）
    df["rolling_high"] = df["h"].rolling(window=lookback).max().shift(1)
    df["rolling_low"] = df["l"].rolling(window=lookback).min().shift(1)

    # 实体比例
    df["body_ratio"] = abs(df["c"] - df["o"]) / (df["h"] - df["l"])
    df["body_ratio"] = df["body_ratio"].replace([np.inf, -np.inf], np.nan)

    # 成交量均线
    df["vol_ma"] = df["v"].rolling(vol_ma_period).mean()

    # 趋势过滤：EMA50
    df["ema50"] = df["c"].ewm(span=50).mean()

    # 买入条件：收盘价 > 过去N根最高，且为阳线，成交量放大，实体足够
    buy = (
        (df["c"] > df["rolling_high"]) &
        (df["c"] > df["o"]) &
        (df["v"] > df["vol_ma"] * vol_multiplier) &
        (df["body_ratio"] > body_threshold)
    )
    # 卖出条件：收盘价 < 过去N根最低，且为阴线，成交量放大，实体足够
    sell = (
        (df["c"] < df["rolling_low"]) &
        (df["c"] < df["o"]) &
        (df["v"] > df["vol_ma"] * vol_multiplier) &
        (df["body_ratio"] > body_threshold)
    )

    # 趋势过滤
    if use_trend_filter:
        buy = buy & (df["c"] > df["ema50"])      # 只在上升趋势中做多
        sell = sell & (df["c"] < df["ema50"])    # 只在下降趋势中做空

    df["signal"] = 0
    df.loc[buy, "signal"] = 1
    df.loc[sell, "signal"] = -1
    return df

df = breakout_logic(df)

# =========================
# 回测函数（修正Sharpe计算，采用资金曲线收益率）
# =========================
def run_backtest(df, fee_rate):
    initial_capital = 10000
    balance = initial_capital
    position = 0
    entry_price = 0.0
    trades = []               # 每笔交易的收益率（百分比）
    equity_curve = [balance]  # 资金曲线

    # 遍历K线，从足够长的索引开始（确保滚动窗口有效）
    start_idx = max(lookback, vol_ma_period, 50) + 1  # 确保所有指标都有有效值
    for i in range(start_idx, len(df) - 1):
        sig = df["signal"].iloc[i]

        if sig != 0 and sig != position:
            next_open = df["o"].iloc[i + 1]

            # 平仓
            if position != 0:
                if position == 1:
                    ret = (next_open / entry_price - 1)
                else:
                    ret = (entry_price / next_open - 1)
                ret -= fee_rate
                balance *= (1 + ret)
                trades.append(ret * 100)
                equity_curve.append(balance)

            # 开新仓
            if sig == 1:
                entry_price = next_open
                position = 1
                balance *= (1 - fee_rate)
                equity_curve.append(balance)
            elif sig == -1:
                entry_price = next_open
                position = -1
                balance *= (1 - fee_rate)
                equity_curve.append(balance)

    # 最后平仓
    if position != 0:
        last_close = df["c"].iloc[-1]
        if position == 1:
            ret = (last_close / entry_price - 1)
        else:
            ret = (entry_price / last_close - 1)
        ret -= fee_rate
        balance *= (1 + ret)
        trades.append(ret * 100)
        equity_curve.append(balance)
        position = 0

    # 计算各项指标
    total_return = (balance / initial_capital - 1) * 100
    num_trades = len(trades)

    if num_trades == 0:
        win_rate = 0.0
        avg_win = 0.0
        avg_loss = 0.0
        profit_factor = 0.0
        sharpe = 0.0
        max_drawdown = 0.0
    else:
        wins = [t for t in trades if t > 0]
        losses = [t for t in trades if t < 0]
        win_rate = len(wins) / num_trades * 100
        avg_win = np.mean(wins) if wins else 0
        avg_loss = abs(np.mean(losses)) if losses else 0
        profit_factor = (sum(wins) / abs(sum(losses))) if losses else np.inf

        # === 修正 Sharpe 计算：基于资金曲线每步收益率 ===
        equity = np.array(equity_curve)
        step_returns = np.diff(equity) / equity[:-1]   # 每步收益率（已扣除手续费）
        if len(step_returns) > 1 and np.std(step_returns) > 0:
            # 年化因子：5分钟K线，每天288根，每年365天
            sharpe = np.mean(step_returns) / np.std(step_returns) * np.sqrt(288 * 365)
        else:
            sharpe = 0.0

        # 计算最大回撤
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak * 100
        max_drawdown = np.max(drawdown)

    return {
        "final_balance": balance,
        "total_return": total_return,
        "num_trades": num_trades,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "equity_curve": equity_curve,
        "trades": trades
    }

backtest_results = run_backtest(df, fee_rate)

# =========================
# 显示回测指标
# =========================
st.subheader("📈 回测表现（基于历史数据）")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("总收益率", f"{backtest_results['total_return']:.2f}%")
    st.metric("交易次数", backtest_results['num_trades'])
with col2:
    st.metric("胜率", f"{backtest_results['win_rate']:.1f}%")
    # 盈亏比，防止除零
    rr = backtest_results['avg_win'] / backtest_results['avg_loss'] if backtest_results['avg_loss'] > 0 else np.nan
    st.metric("盈亏比", f"{rr:.2f}" if not np.isnan(rr) else "N/A")
with col3:
    st.metric("最大回撤", f"{backtest_results['max_drawdown']:.2f}%")
    st.metric("夏普比率", f"{backtest_results['sharpe']:.2f}")
with col4:
    st.metric("获利因子", f"{backtest_results['profit_factor']:.2f}")
    st.metric("手续费率", f"{fee_rate*100:.3f}%")

# =========================
# 资金曲线图
# =========================
if len(backtest_results['equity_curve']) > 1:
    fig_equity = go.Figure()
    fig_equity.add_trace(go.Scatter(
        y=backtest_results['equity_curve'],
        mode='lines',
        name='资金曲线',
        line=dict(color='blue', width=2)
    ))
    fig_equity.update_layout(
        title="资金曲线 (起始10000)",
        xaxis_title="交易步数",
        yaxis_title="账户余额",
        height=300,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig_equity, use_container_width=True)

# =========================
# 实时信号状态
# =========================
st.subheader("🔔 实时信号状态")
latest_signal = df["signal"].iloc[-1] if not df.empty else 0
if latest_signal == 1:
    st.success("📈 最新信号：买入")
elif latest_signal == -1:
    st.error("📉 最新信号：卖出")
else:
    st.info("⏸️ 最新信号：无")

# 最近信号时间
buy_signals = df[df["signal"] == 1]
sell_signals = df[df["signal"] == -1]
last_buy_time = buy_signals["time"].max() if not buy_signals.empty else None
last_sell_time = sell_signals["time"].max() if not sell_signals.empty else None
col1, col2 = st.columns(2)
with col1:
    if last_buy_time:
        st.success(f"最近买入信号: {last_buy_time.strftime('%Y-%m-%d %H:%M')}")
with col2:
    if last_sell_time:
        st.error(f"最近卖出信号: {last_sell_time.strftime('%Y-%m-%d %H:%M')}")

# =========================
# K线图（带买卖点）
# =========================
st.subheader("📊 最新K线图")
fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=df["time"],
    open=df["o"],
    high=df["h"],
    low=df["l"],
    close=df["c"],
    name="K线"
))

buy_points = df[df["signal"] == 1]
sell_points = df[df["signal"] == -1]
fig.add_trace(go.Scatter(
    x=buy_points["time"],
    y=buy_points["c"],
    mode="markers",
    marker=dict(symbol="triangle-up", size=12, color="green"),
    name="买入"
))
fig.add_trace(go.Scatter(
    x=sell_points["time"],
    y=sell_points["c"],
    mode="markers",
    marker=dict(symbol="triangle-down", size=12, color="red"),
    name="卖出"
))

# 可选：添加EMA50线
if use_trend_filter:
    fig.add_trace(go.Scatter(
        x=df["time"],
        y=df["ema50"],
        line=dict(color="purple", width=1),
        name="EMA50"
    ))

fig.update_layout(
    xaxis_rangeslider_visible=False,
    height=500,
    hovermode="x unified"
)
st.plotly_chart(fig, use_container_width=True)

st.caption("⚠️ 注意：回测已考虑手续费，使用下一根开盘价执行。本工具仅供参考，不构成投资建议。")
