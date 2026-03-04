# kline_strategy_app.py
import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 页面配置
st.set_page_config(layout="wide", page_title="纯K线策略回测")

st.title("📈 纯K线形态策略回测（吞没形态示例）")
st.markdown("本工具演示基于看涨/看跌吞没形态的简单策略（只做多头）")

# ---------- 数据获取（缓存10分钟）----------
@st.cache_data(ttl=600)
def fetch_ohlcv(symbol, timeframe, limit=500):
    """从Binance获取K线数据"""
    try:
        exchange = ccxt.binance()
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        st.error(f"数据获取失败: {e}")
        return pd.DataFrame()

# ---------- 形态识别函数 ----------
def detect_bullish_engulfing(df, body_ratio=1.0):
    """
    看涨吞没：
    1. 前一根为阴线 (close < open)
    2. 当前为阳线 (close > open)
    3. 当前阳线实体完全吞没前一根阴线实体 (open < prev_close 且 close > prev_open)
    可选实体比例要求：当前实体 >= 前实体 * body_ratio
    """
    prev_close = df['close'].shift(1)
    prev_open = df['open'].shift(1)
    prev_body = abs(prev_close - prev_open)
    curr_body = abs(df['close'] - df['open'])

    prev_is_bear = prev_close < prev_open
    curr_is_bull = df['close'] > df['open']
    engulf = (df['open'] < prev_close) & (df['close'] > prev_open)
    body_cond = curr_body >= prev_body * body_ratio

    return prev_is_bear & curr_is_bull & engulf & body_cond

def detect_bearish_engulfing(df, body_ratio=1.0):
    """
    看跌吞没：
    1. 前一根为阳线
    2. 当前为阴线
    3. 当前阴线实体完全吞没前一根阳线实体 (open > prev_close 且 close < prev_open)
    """
    prev_close = df['close'].shift(1)
    prev_open = df['open'].shift(1)
    prev_body = abs(prev_close - prev_open)
    curr_body = abs(df['close'] - df['open'])

    prev_is_bull = prev_close > prev_open
    curr_is_bear = df['close'] < df['open']
    engulf = (df['open'] > prev_close) & (df['close'] < prev_open)
    body_cond = curr_body >= prev_body * body_ratio

    return prev_is_bull & curr_is_bear & engulf & body_cond

# ---------- 简化回测（只做多）----------
def backtest(df, long_signals, short_signals, initial_capital=10000, commission=0.001):
    """
    回测逻辑：
    - 出现 long_signals 时全仓买入（开多）
    - 出现 short_signals 时平多（不考虑做空）
    - 手续费双边扣除（开平各一次）
    """
    df = df.copy()
    df['signal'] = 0
    df.loc[long_signals, 'signal'] = 1   # 开多信号
    df.loc[short_signals, 'signal'] = -1  # 平多信号

    # 生成持仓序列
    df['position'] = 0
    in_position = False
    for i in range(len(df)):
        if df['signal'].iloc[i] == 1 and not in_position:
            df.iloc[i, df.columns.get_loc('position')] = 1
            in_position = True
        elif df['signal'].iloc[i] == -1 and in_position:
            df.iloc[i, df.columns.get_loc('position')] = 0
            in_position = False
        else:
            df.iloc[i, df.columns.get_loc('position')] = 1 if in_position else 0

    # 计算收益率
    df['returns'] = df['close'].pct_change()
    df['strategy_returns'] = df['position'].shift(1) * df['returns']

    # 扣除手续费（信号变化时）
    trades = df['signal'].diff().fillna(0) != 0
    df.loc[trades, 'strategy_returns'] -= commission

    df['strategy_returns'] = df['strategy_returns'].fillna(0)
    df['equity'] = (1 + df['strategy_returns']).cumprod() * initial_capital

    return df

# ---------- 侧边栏参数 ----------
with st.sidebar:
    st.header("⚙️ 参数设置")
    symbol = st.selectbox("交易对", ["ETH/USDT", "BTC/USDT", "BNB/USDT"], index=0)
    timeframe = st.selectbox("时间周期", ["1m", "5m", "15m", "1h", "4h", "1d"], index=2)  # 默认15m
    limit = st.slider("K线数量", 100, 1000, 300, step=50)

    st.subheader("📐 吞没形态参数")
    body_ratio = st.slider("实体最小比例 (当前/前一根)", 0.5, 2.0, 1.0, 0.1)
    use_bull = st.checkbox("启用看涨吞没（开多）", value=True)
    use_bear = st.checkbox("启用看跌吞没（平多）", value=True)   # 简化：看跌信号作为平多

    st.subheader("💰 回测设置")
    initial_capital = st.number_input("初始资金", value=10000)
    commission = st.number_input("手续费率（双边）", value=0.001, format="%.4f")

    run_button = st.button("🚀 运行回测")

# ---------- 主逻辑 ----------
if run_button:
    with st.spinner("获取数据并回测中..."):
        df = fetch_ohlcv(symbol, timeframe, limit)
        if df.empty:
            st.stop()

        # 生成信号
        long_signal = pd.Series(False, index=df.index)
        short_signal = pd.Series(False, index=df.index)

        if use_bull:
            long_signal = detect_bullish_engulfing(df, body_ratio=body_ratio)
        if use_bear:
            short_signal = detect_bearish_engulfing(df, body_ratio=body_ratio)

        # 回测
        result_df = backtest(df, long_signal, short_signal, initial_capital, commission)

        # 计算绩效指标
        total_return = (result_df['equity'].iloc[-1] / initial_capital - 1) * 100

        # 年化周期数（粗略）
        if timeframe[-1] == 'm':
            periods_per_year = 252 * 24 * 60 / int(timeframe[:-1])
        elif timeframe[-1] == 'h':
            periods_per_year = 252 * 24 / int(timeframe[:-1])
        else:
            periods_per_year = 252  # 日线

        strategy_returns = result_df['strategy_returns']
        sharpe = np.sqrt(periods_per_year) * strategy_returns.mean() / strategy_returns.std() if strategy_returns.std() != 0 else 0
        max_drawdown = (result_df['equity'] / result_df['equity'].cummax() - 1).min()

        # 胜率（仅考虑有交易的周期）
        nonzero_returns = strategy_returns[strategy_returns != 0]
        win_rate = (nonzero_returns > 0).sum() / len(nonzero_returns) if len(nonzero_returns) > 0 else 0

        # 交易次数（信号变化次数/2，因为每次开平算两次信号，但我们统计的是信号变化点个数，近似为交易次数）
        trade_count = (result_df['signal'].diff() != 0).sum() // 2

        # ===== 可视化 =====
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=("K线图与信号", "持仓状态", "权益曲线")
        )

        # K线图
        fig.add_trace(go.Candlestick(
            x=result_df.index,
            open=result_df['open'],
            high=result_df['high'],
            low=result_df['low'],
            close=result_df['close'],
            name="K线",
            showlegend=False
        ), row=1, col=1)

        # 多头信号标记
        buy_points = result_df[long_signal]
        if not buy_points.empty:
            fig.add_trace(go.Scatter(
                x=buy_points.index,
                y=buy_points['low'] * 0.99,
                mode='markers',
                marker=dict(symbol='triangle-up', size=12, color='lime'),
                name='看涨吞没 (开多)'
            ), row=1, col=1)

        # 平多信号标记
        sell_points = result_df[short_signal]
        if not sell_points.empty:
            fig.add_trace(go.Scatter(
                x=sell_points.index,
                y=sell_points['high'] * 1.01,
                mode='markers',
                marker=dict(symbol='triangle-down', size=12, color='red'),
                name='看跌吞没 (平多)'
            ), row=1, col=1)

        # 持仓状态
        fig.add_trace(go.Scatter(
            x=result_df.index,
            y=result_df['position'],
            mode='lines',
            line=dict(color='blue', width=2),
            name='持仓'
        ), row=2, col=1)
        fig.update_yaxes(title_text="持仓", row=2, col=1, tickvals=[0, 1], ticktext=["空仓", "持仓"])

        # 权益曲线
        fig.add_trace(go.Scatter(
            x=result_df.index,
            y=result_df['equity'],
            mode='lines',
            line=dict(color='orange', width=2),
            name='权益'
        ), row=3, col=1)
        fig.update_yaxes(title_text="权益", row=3, col=1)

        fig.update_layout(height=900, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        # 绩效卡片
        st.subheader("📊 回测绩效")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("总收益率", f"{total_return:.2f}%")
        col2.metric("夏普比率", f"{sharpe:.2f}")
        col3.metric("最大回撤", f"{max_drawdown:.2%}")
        col4.metric("胜率", f"{win_rate:.2%}" if not np.isnan(win_rate) else "N/A")
        col5.metric("交易次数", trade_count)

        # 最近信号表
        st.subheader("📋 最近K线与信号")
        display_cols = ['open', 'high', 'low', 'close', 'signal']
        st.dataframe(result_df[display_cols].tail(20))

else:
    st.info("👈 在侧边栏设置参数后点击『运行回测』")
    st.markdown("""
    ### 📖 策略说明
    - **看涨吞没**：前一根为阴线，当前阳线实体完全覆盖前一根阴线实体（可调整实体比例）。
    - **看跌吞没**：前一根为阳线，当前阴线实体完全覆盖前一根阳线实体。
    - **简化规则**：出现看涨吞没时开多，出现看跌吞没时平多（暂不做空）。
    - **手续费**：双边费率（开仓+平仓），默认0.1%。
    """)
