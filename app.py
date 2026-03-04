# kline_strategy_local_complete.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 页面配置
st.set_page_config(layout="wide", page_title="纯K线策略回测（本地数据）")

st.title("📈 纯K线形态策略回测（本地CSV文件）")
st.markdown("上传您的K线CSV文件，回测基于吞没形态的简单策略。")

# ---------- 数据加载函数 ----------
@st.cache_data
def load_data_from_csv(uploaded_file):
    """
    从上传的CSV文件加载数据，尝试自动映射常见列名到标准列名。
    期望最终列名：timestamp, open, high, low, close, volume
    """
    if uploaded_file is None:
        return pd.DataFrame()

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"读取CSV文件失败: {e}")
        return pd.DataFrame()

    # 常见列名映射（可根据需要扩展）
    column_mapping = {
        'timestamp': ['timestamp', 'date', '时间', 'datetime', '日期'],
        'open': ['open', '开盘', '开盘价'],
        'high': ['high', '最高', '最高价'],
        'low': ['low', '最低', '最低价'],
        'close': ['close', '收盘', '收盘价'],
        'volume': ['volume', '成交量', '成交额', 'vol']
    }

    # 将现有列名统一为标准列名
    new_columns = {}
    for std_name, possible_names in column_mapping.items():
        for col in df.columns:
            if col.lower() in [p.lower() for p in possible_names]:
                new_columns[col] = std_name
                break

    if len(new_columns) < 6:
        st.error("无法识别必要的列（timestamp, open, high, low, close, volume）。请确保列名包含这些字段的常见名称。")
        st.write("当前文件列名：", list(df.columns))
        return pd.DataFrame()

    df.rename(columns=new_columns, inplace=True)

    # 只保留需要的列
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

    # 转换时间戳
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    except Exception as e:
        st.error(f"时间戳列转换失败: {e}")
        return pd.DataFrame()

    df.set_index('timestamp', inplace=True)
    # 确保数值列为浮点型
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)  # 删除无效行

    return df

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

# ---------- 回测函数（只做多）----------
def backtest(df, long_signals, short_signals, initial_capital=10000, commission=0.001):
    """
    简化回测：
    - 出现 long_signals 时全仓买入（开多）
    - 出现 short_signals 时平多
    - 手续费双边扣除
    """
    df = df.copy()
    df['signal'] = 0
    df.loc[long_signals, 'signal'] = 1
    df.loc[short_signals, 'signal'] = -1

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

    # 计算策略收益率
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
    
    uploaded_file = st.file_uploader("上传CSV文件", type=['csv'])
    
    st.subheader("📐 吞没形态参数")
    body_ratio = st.slider("实体最小比例 (当前/前一根)", 0.5, 2.0, 1.0, 0.1)
    use_bull = st.checkbox("启用看涨吞没（开多）", value=True)
    use_bear = st.checkbox("启用看跌吞没（平多）", value=True)

    st.subheader("💰 回测设置")
    initial_capital = st.number_input("初始资金", value=10000)
    commission = st.number_input("手续费率（双边）", value=0.001, format="%.4f")

    run_button = st.button("🚀 运行回测")

# ---------- 主逻辑 ----------
if run_button:
    if uploaded_file is None:
        st.warning("请先上传CSV文件。")
        st.stop()

    with st.spinner("加载数据并回测中..."):
        df = load_data_from_csv(uploaded_file)
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

        # 粗略年化（假设为日线，如果是其他周期请手动调整）
        periods_per_year = 252  # 默认按日线计算夏普比率
        strategy_returns = result_df['strategy_returns']
        sharpe = np.sqrt(periods_per_year) * strategy_returns.mean() / strategy_returns.std() if strategy_returns.std() != 0 else 0
        max_drawdown = (result_df['equity'] / result_df['equity'].cummax() - 1).min()

        nonzero_returns = strategy_returns[strategy_returns != 0]
        win_rate = (nonzero_returns > 0).sum() / len(nonzero_returns) if len(nonzero_returns) > 0 else 0
        trade_count = (result_df['signal'].diff() != 0).sum() // 2

        # ===== 绘图 =====
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

        # 买入信号
        buy_points = result_df[long_signal]
        if not buy_points.empty:
            fig.add_trace(go.Scatter(
                x=buy_points.index,
                y=buy_points['low'] * 0.99,
                mode='markers',
                marker=dict(symbol='triangle-up', size=12, color='lime'),
                name='看涨吞没 (开多)'
            ), row=1, col=1)

        # 卖出信号
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
    st.info("👈 请上传CSV文件并设置参数后点击『运行回测』")
    st.markdown("""
    ### 📖 使用说明
    1. **上传CSV文件**：文件需包含K线数据，常见列名（如 open, high, low, close, volume, timestamp/date）会被自动识别。
    2. **策略逻辑**：
       - 看涨吞没 → 开多
       - 看跌吞没 → 平多（暂不做空）
    3. **手续费**：双边费率（开平各一次），默认0.1%。
    """)
