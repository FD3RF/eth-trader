import streamlit as st
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go

st.set_page_config(layout="wide")

st.title("5分钟高频趋势+震荡策略回测系统")

uploaded_file = st.file_uploader("上传5分钟K线CSV文件")

risk_per_trade = st.sidebar.slider("单笔风险 %", 0.1, 5.0, 1.0)
rr_ratio = st.sidebar.slider("止盈倍数 (R)", 0.3, 2.0, 0.6)
initial_capital = st.sidebar.number_input("初始资金", value=10000)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    # === 指标 ===
    df['ema9'] = ta.trend.ema_indicator(df['close'], window=9)
    df['ema21'] = ta.trend.ema_indicator(df['close'], window=21)
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    df['bb_upper'] = ta.volatility.BollingerBands(df['close']).bollinger_hband()
    df['bb_lower'] = ta.volatility.BollingerBands(df['close']).bollinger_lband()
    df['bb_width'] = df['bb_upper'] - df['bb_lower']

    df.dropna(inplace=True)

    capital = initial_capital
    equity_curve = []
    trades = []

    position = None

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]

        # === 趋势多单条件 ===
        trend_long = (
            row['ema9'] > row['ema21'] and
            row['rsi'] > 52 and
            prev['low'] <= prev['ema9'] and
            row['close'] > prev['high']
        )

        # === 趋势空单条件 ===
        trend_short = (
            row['ema9'] < row['ema21'] and
            row['rsi'] < 48 and
            prev['high'] >= prev['ema9'] and
            row['close'] < prev['low']
        )

        # === 震荡模块 ===
        range_long = (
            row['close'] <= row['bb_lower'] and
            row['bb_width'] < df['bb_width'].rolling(50).mean().iloc[i]
        )

        range_short = (
            row['close'] >= row['bb_upper'] and
            row['bb_width'] < df['bb_width'].rolling(50).mean().iloc[i]
        )

        if position is None:

            if trend_long or range_long:
                entry = row['close']
                stop = entry - row['atr']
                target = entry + row['atr'] * rr_ratio
                risk = capital * (risk_per_trade / 100)
                position = ("long", entry, stop, target, risk)

            elif trend_short or range_short:
                entry = row['close']
                stop = entry + row['atr']
                target = entry - row['atr'] * rr_ratio
                risk = capital * (risk_per_trade / 100)
                position = ("short", entry, stop, target, risk)

        else:
            side, entry, stop, target, risk = position

            if side == "long":
                if row['low'] <= stop:
                    capital -= risk
                    trades.append(-risk)
                    position = None
                elif row['high'] >= target:
                    profit = risk * rr_ratio
                    capital += profit
                    trades.append(profit)
                    position = None

            if side == "short":
                if row['high'] >= stop:
                    capital -= risk
                    trades.append(-risk)
                    position = None
                elif row['low'] <= target:
                    profit = risk * rr_ratio
                    capital += profit
                    trades.append(profit)
                    position = None

        equity_curve.append(capital)

    df = df.iloc[-len(equity_curve):]
    df['equity'] = equity_curve

    # === 输出结果 ===
    st.subheader("回测结果")

    total_trades = len(trades)
    win_rate = len([t for t in trades if t > 0]) / total_trades if total_trades > 0 else 0
    net_profit = capital - initial_capital

    st.write("总交易次数:", total_trades)
    st.write("胜率:", round(win_rate * 100, 2), "%")
    st.write("净利润:", round(net_profit, 2))
    st.write("最终资金:", round(capital, 2))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['equity'], name="Equity Curve"))
    st.plotly_chart(fig, use_container_width=True)
