import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import ccxt
import itertools
import re
from datetime import datetime

st.set_page_config(layout="wide", page_title="ATR区间突破系统", page_icon="📈")
st.title("📊 区间突破 + ATR风控 · 终极交互版")

# ========================= 侧边栏参数 =========================
with st.sidebar:
    st.header("⚙️ 策略参数")

    # 初始化session_state
    if 'lookback' not in st.session_state:
        st.session_state['lookback'] = 20
    if 'body_threshold' not in st.session_state:
        st.session_state['body_threshold'] = 0.004
    if 'vol_ma_period' not in st.session_state:
        st.session_state['vol_ma_period'] = 20
    if 'break_threshold' not in st.session_state:
        st.session_state['break_threshold'] = 0.001
    if 'atr_period' not in st.session_state:
        st.session_state['atr_period'] = 14
    if 'atr_multiplier' not in st.session_state:
        st.session_state['atr_multiplier'] = 2.0
    if 'risk_per_trade' not in st.session_state:
        st.session_state['risk_per_trade'] = 0.01
    if 'fee_rate' not in st.session_state:
        st.session_state['fee_rate'] = 0.0005

    lookback = st.number_input("突破周期 (lookback)", min_value=5, max_value=100, value=st.session_state['lookback'], step=1,
                               help="计算过去N根K线的最高/最低点作为突破参考")
    body_threshold = st.number_input("实体阈值 (比例)", min_value=0.0, max_value=0.05, value=st.session_state['body_threshold'], step=0.0005, format="%.4f",
                                     help="K线实体相对于开盘价的最小比例")
    vol_ma_period = st.number_input("成交量均线周期", min_value=5, max_value=50, value=st.session_state['vol_ma_period'], step=1)
    break_threshold = st.number_input("成交量突破阈值", min_value=0.0, max_value=0.01, value=st.session_state['break_threshold'], step=0.0001, format="%.4f",
                                      help="成交量需大于均线的 (1+阈值) 倍")
    atr_period = st.number_input("ATR周期", min_value=5, max_value=50, value=st.session_state['atr_period'], step=1)
    atr_multiplier = st.number_input("ATR止损倍数", min_value=1.0, max_value=5.0, value=st.session_state['atr_multiplier'], step=0.1,
                                     help="止损距离 = ATR * 倍数")
    risk_per_trade = st.number_input("每笔风险 (%)", min_value=0.001, max_value=0.05, value=st.session_state['risk_per_trade'], step=0.001, format="%.3f",
                                     help="每笔交易允许亏损占总资金的比例")
    fee_rate = st.number_input("手续费率 (单边)", min_value=0.0, max_value=0.01, value=st.session_state['fee_rate'], step=0.0001, format="%.4f")

    # 保存到session
    st.session_state['lookback'] = lookback
    st.session_state['body_threshold'] = body_threshold
    st.session_state['vol_ma_period'] = vol_ma_period
    st.session_state['break_threshold'] = break_threshold
    st.session_state['atr_period'] = atr_period
    st.session_state['atr_multiplier'] = atr_multiplier
    st.session_state['risk_per_trade'] = risk_per_trade
    st.session_state['fee_rate'] = fee_rate

    st.markdown("---")
    st.caption("数据源: ccxt OKX ETH/USDT 5分钟")

# ========================= 获取数据 =========================
@st.cache_data(ttl=60)
def load_data(limit=3000):
    exchange = ccxt.okx()
    bars = exchange.fetch_ohlcv("ETH/USDT", timeframe="5m", limit=limit)
    df = pd.DataFrame(bars, columns=['ts','o','h','l','c','v'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    return df

df_raw = load_data()
if df_raw.empty:
    st.error("无法获取数据")
    st.stop()

# ========================= 策略回测函数 =========================
def run_backtest(df, lookback, body_threshold, vol_ma_period, break_threshold,
                 atr_period, atr_multiplier, risk_per_trade, fee_rate):
    df = df.copy()
    # 指标计算
    df['body'] = abs(df['c'] - df['o']) / df['o']
    df['vol_ma'] = df['v'].rolling(vol_ma_period).mean()
    df['prev_high'] = df['h'].rolling(lookback).max().shift(1)
    df['prev_low'] = df['l'].rolling(lookback).min().shift(1)

    # ATR
    df['tr1'] = df['h'] - df['l']
    df['tr2'] = abs(df['h'] - df['c'].shift(1))
    df['tr3'] = abs(df['l'] - df['c'].shift(1))
    df['tr'] = df[['tr1','tr2','tr3']].max(axis=1)
    df['atr'] = df['tr'].rolling(atr_period).mean()

    # 信号
    df['long_signal'] = (
        (df['c'] > df['prev_high']) &
        (df['body'] > body_threshold) &
        (df['v'] > df['vol_ma'] * (1 + break_threshold))
    )
    df['short_signal'] = (
        (df['c'] < df['prev_low']) &
        (df['body'] > body_threshold) &
        (df['v'] > df['vol_ma'] * (1 + break_threshold))
    )

    # 回测
    balance = 10000
    position = 0
    entry_price = 0
    stop_price = 0
    position_size = 0

    equity_curve = []
    dates = []

    start_idx = max(lookback, vol_ma_period, atr_period) + 1
    if start_idx >= len(df) - 1:
        return None

    for i in range(start_idx, len(df)-1):
        open_price = df.iloc[i+1]['o']
        close_price = df.iloc[i]['c']
        atr = df.iloc[i]['atr']
        current_time = df.iloc[i]['ts']

        # 浮动权益
        if position != 0:
            floating_pnl = position * position_size * (close_price - entry_price)
            equity = balance + floating_pnl
        else:
            equity = balance
        equity_curve.append(equity)
        dates.append(current_time)

        # 平仓（止损）
        if position == 1 and close_price <= stop_price:
            pnl = position_size * (stop_price - entry_price)
            balance += pnl - abs(position_size * stop_price) * fee_rate
            position = 0
        elif position == -1 and close_price >= stop_price:
            pnl = position_size * (entry_price - stop_price)
            balance += pnl - abs(position_size * stop_price) * fee_rate
            position = 0

        # 开仓
        if position == 0:
            if df.iloc[i]['long_signal']:
                risk_amount = balance * risk_per_trade
                stop_price = open_price - atr_multiplier * atr
                stop_distance = open_price - stop_price
                if stop_distance <= 0:
                    continue
                position_size = risk_amount / stop_distance
                cost = position_size * open_price * fee_rate
                balance -= cost
                entry_price = open_price
                position = 1
            elif df.iloc[i]['short_signal']:
                risk_amount = balance * risk_per_trade
                stop_price = open_price + atr_multiplier * atr
                stop_distance = stop_price - open_price
                if stop_distance <= 0:
                    continue
                position_size = risk_amount / stop_distance
                cost = position_size * open_price * fee_rate
                balance -= cost
                entry_price = open_price
                position = -1

    # 强制平仓
    if position != 0:
        final_price = df.iloc[-1]['c']
        if position == 1:
            pnl = position_size * (final_price - entry_price)
        else:
            pnl = position_size * (entry_price - final_price)
        balance += pnl - abs(position_size * final_price) * fee_rate
    equity_curve.append(balance)
    dates.append(df.iloc[-1]['ts'])

    # 绩效计算
    equity_series = pd.Series(equity_curve, index=dates)
    total_return = (balance - 10000) / 10000
    days = (dates[-1] - dates[0]).days + 1
    years = days / 365
    cagr = (balance / 10000) ** (1 / years) - 1 if years > 0 else 0
    roll_max = equity_series.cummax()
    drawdown = equity_series / roll_max - 1
    max_dd = drawdown.min()
    returns = equity_series.pct_change().dropna()
    sharpe = (returns.mean() / returns.std()) * np.sqrt(365 * 24 * 12) if returns.std() != 0 else 0

    return {
        'balance': balance,
        'total_return': total_return,
        'cagr': cagr,
        'max_drawdown': max_dd,
        'sharpe': sharpe,
        'equity_series': equity_series,
        'df_signals': df  # 用于K线图
    }

# ========================= 运行当前参数回测 =========================
result = run_backtest(df_raw, lookback, body_threshold, vol_ma_period, break_threshold,
                      atr_period, atr_multiplier, risk_per_trade, fee_rate)

if result is None:
    st.error("数据不足，请减小参数或增加数据量")
    st.stop()

# ========================= 显示结果 =========================
st.subheader("📈 回测表现")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("最终资金", f"{result['balance']:.2f}")
    st.metric("总收益率", f"{result['total_return']*100:.2f}%")
with col2:
    st.metric("年化收益 (CAGR)", f"{result['cagr']*100:.2f}%")
    st.metric("最大回撤", f"{result['max_drawdown']*100:.2f}%")
with col3:
    st.metric("夏普比率", f"{result['sharpe']:.2f}")
    st.metric("交易手续费率", f"{fee_rate*100:.3f}%")
with col4:
    st.metric("数据起始", result['equity_series'].index[0].strftime('%Y-%m-%d'))
    st.metric("数据结束", result['equity_series'].index[-1].strftime('%Y-%m-%d'))

# ========================= 资金曲线 =========================
st.subheader("💰 资金曲线")
fig_equity = go.Figure()
fig_equity.add_trace(go.Scatter(
    x=result['equity_series'].index,
    y=result['equity_series'].values,
    mode='lines',
    name='资金曲线',
    line=dict(color='blue')
))
fig_equity.update_layout(height=400, xaxis_title="日期", yaxis_title="账户余额")
st.plotly_chart(fig_equity, use_container_width=True)

# ========================= 最新K线图（带信号点）=========================
st.subheader("📊 最新K线图（含买卖信号）")
df_plot = result['df_signals'].iloc[-100:].copy()  # 只显示最近100根
fig_candle = go.Figure(data=[go.Candlestick(
    x=df_plot['ts'],
    open=df_plot['o'],
    high=df_plot['h'],
    low=df_plot['l'],
    close=df_plot['c'],
    name='K线'
)])
# 信号点
longs = df_plot[df_plot['long_signal']]
shorts = df_plot[df_plot['short_signal']]
if not longs.empty:
    fig_candle.add_trace(go.Scatter(
        x=longs['ts'],
        y=longs['c'],
        mode='markers',
        marker=dict(symbol='triangle-up', size=12, color='green'),
        name='买入信号'
    ))
if not shorts.empty:
    fig_candle.add_trace(go.Scatter(
        x=shorts['ts'],
        y=shorts['c'],
        mode='markers',
        marker=dict(symbol='triangle-down', size=12, color='red'),
        name='卖出信号'
    ))
fig_candle.update_layout(xaxis_rangeslider_visible=False, height=500)
st.plotly_chart(fig_candle, use_container_width=True)

# ========================= 参数扫描 =========================
st.subheader("🔍 参数扫描")
if st.button("开始扫描（耗时较长，约30秒）"):
    # 定义扫描范围（可根据需要调整）
    lookback_range = [10, 20, 30]
    body_range = [0.002, 0.004, 0.006]
    vol_ma_range = [10, 20]
    break_range = [0.0005, 0.001, 0.002]
    atr_mult_range = [1.5, 2.0, 2.5]

    results = []
    total_combos = len(lookback_range) * len(body_range) * len(vol_ma_range) * len(break_range) * len(atr_mult_range)
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, (lb, bd, vp, br, am) in enumerate(itertools.product(
            lookback_range, body_range, vol_ma_range, break_range, atr_mult_range)):
        status_text.text(f"扫描中: lookback={lb}, body={bd}, vol_ma={vp}, break={br}, atr_mult={am} ({i+1}/{total_combos})")
        res = run_backtest(df_raw, lb, bd, vp, br, atr_period, am, risk_per_trade, fee_rate)
        if res is not None:
            results.append({
                'lookback': lb,
                'body': bd,
                'vol_ma': vp,
                'break': br,
                'atr_mult': am,
                '总收益率%': round(res['total_return']*100, 2),
                '夏普': round(res['sharpe'], 2),
                '最大回撤%': round(res['max_drawdown']*100, 2),
                '交易次数': 'N/A'  # 可以扩展记录交易次数
            })
        progress_bar.progress((i+1)/total_combos)

    progress_bar.empty()
    status_text.empty()
    result_df = pd.DataFrame(results).sort_values('夏普', ascending=False)
    st.session_state['scan_results'] = result_df
    st.success("扫描完成！")

if 'scan_results' in st.session_state:
    st.subheader("扫描结果（按夏普降序）")
    st.dataframe(st.session_state['scan_results'].head(20), use_container_width=True)

    # 应用扫描参数
    apply_options = []
    for idx, row in st.session_state['scan_results'].head(20).iterrows():
        apply_options.append(
            f"lookback={row['lookback']}, body={row['body']}, vol_ma={row['vol_ma']}, "
            f"break={row['break']}, atr_mult={row['atr_mult']}, 夏普={row['夏普']}"
        )
    selected = st.selectbox("选择参数组合应用", apply_options)
    if st.button("应用选中参数"):
        numbers = re.findall(r"[-+]?\d*\.?\d+", selected)
        if len(numbers) >= 5:
            try:
                st.session_state['lookback'] = int(float(numbers[0]))
                st.session_state['body_threshold'] = float(numbers[1])
                st.session_state['vol_ma_period'] = int(float(numbers[2]))
                st.session_state['break_threshold'] = float(numbers[3])
                st.session_state['atr_multiplier'] = float(numbers[4])
                st.rerun()
            except Exception as e:
                st.error(f"解析失败: {e}")
        else:
            st.error("无法提取参数")

    # 下载按钮
    csv = st.session_state['scan_results'].to_csv(index=False)
    st.download_button("📥 下载扫描结果", csv, "scan_results.csv", "text/csv")

st.caption("⚠️ 本工具基于历史数据回测，不构成投资建议。实盘需自行承担风险。")
