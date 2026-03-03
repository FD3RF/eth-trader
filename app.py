import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import random
from streamlit_autorefresh import st_autorefresh

# ==================== 页面配置 ====================
st.set_page_config(page_title="ETH V57.0 实战终端 - 小利润策略", layout="wide")
st_autorefresh(interval=5000, key="auto_refresh")  # 每5秒自动刷新

# ==================== 模拟数据生成（可替换为真实数据）====================
def generate_simulated_data(minutes=200):
    """生成模拟的分钟级K线数据，包含一定的趋势和震荡特征"""
    now = datetime.now()
    times = [now - timedelta(minutes=i) for i in range(minutes, 0, -1)]
    
    # 构造价格序列：前半段震荡，后半段趋势
    base_price = 1920
    prices = []
    for i in range(minutes):
        if i < 100:
            # 震荡区
            change = random.uniform(-1.5, 1.5)
        else:
            # 趋势区（缓慢上升）
            change = random.uniform(0.2, 1.8)
        base_price += change
        prices.append(base_price)
    
    # 生成OHLC
    df = pd.DataFrame({
        'time': times,
        'open': prices[:-1] + [prices[-1]],
        'high': [p + random.uniform(0.5, 2.5) for p in prices],
        'low': [p - random.uniform(0.5, 2.5) for p in prices],
        'close': prices,
        'volume': [random.randint(800, 6000) for _ in range(minutes)]
    })
    df.set_index('time', inplace=True)
    return df

# ==================== 核心策略逻辑 ====================
def compute_indicators(df):
    """计算指标并生成交易信号（方案A + 方案B）"""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    open_ = df['open'].values
    volume = df['volume'].values
    
    # 基础指标
    current_price = close[-1]
    
    # 1分钟净流入（模拟）
    last_volume = volume[-1]
    price_change = close[-1] - close[-2]
    net_inflow = last_volume * (1 if price_change > 0 else -1) / 100
    
    # 买压占比（模拟）
    up_volume = volume[close > open_].sum()
    total_volume = volume.sum()
    buy_pressure = up_volume / total_volume * 100 if total_volume > 0 else 50
    
    # ---------- 计算技术指标 ----------
    # EMA
    ema_fast = pd.Series(close).ewm(span=8).mean().values
    ema_slow = pd.Series(close).ewm(span=21).mean().values
    
    # RSI (14)
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14).mean().values
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14).mean().values
    rsi = 100 - 100 / (1 + gain / (loss + 1e-9))
    
    # ADX (14) 简化计算
    tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
    tr = np.maximum(tr, np.abs(low - np.roll(close, 1)))
    atr = pd.Series(tr).rolling(14).mean().values
    
    plus_dm = np.maximum(high - np.roll(high, 1), 0)
    minus_dm = np.maximum(np.roll(low, 1) - low, 0)
    plus_di = 100 * (pd.Series(plus_dm).rolling(14).mean() / atr).values
    minus_di = 100 * (pd.Series(minus_dm).rolling(14).mean() / atr).values
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
    adx = pd.Series(dx).rolling(14).mean().values[-1]
    
    # ---------- 判断市场状态 ----------
    is_trend = adx > 25      # ADX > 25 认为有趋势
    is_oscillation = not is_trend
    
    # 趋势方向
    trend_up = ema_fast[-1] > ema_slow[-1]
    trend_down = ema_fast[-1] < ema_slow[-1]
    
    # 初始化信号变量
    signal = "观望中"
    entry = current_price
    stop_loss = current_price
    take_profit = current_price
    risk_reward = 1.0
    
    # ---------- 方案A：回调顺势（趋势模式）----------
    if is_trend and trend_up:
        # 多头趋势回调：价格回踩EMA8但不破EMA21，且前一根K线收阳
        if close[-1] < ema_fast[-1] * 1.002 and close[-1] > ema_slow[-1]:
            if close[-2] > open_[-2]:  # 前一根阳线确认
                signal = "回调做多"
                entry = current_price
                stop_loss = min(ema_slow[-1], close[-2] * 0.998)
                take_profit = entry + (entry - stop_loss) * 1.5  # 盈亏比1.5:1
    
    elif is_trend and trend_down:
        # 空头趋势回调
        if close[-1] > ema_fast[-1] * 0.998 and close[-1] < ema_slow[-1]:
            if close[-2] < open_[-2]:  # 前一根阴线确认
                signal = "回调做空"
                entry = current_price
                stop_loss = max(ema_slow[-1], close[-2] * 1.002)
                take_profit = entry - (stop_loss - entry) * 1.5
    
    # ---------- 方案B：均值回归（震荡模式）----------
    elif is_oscillation:
        if rsi[-1] < 25:
            signal = "超卖做多"
            entry = current_price
            stop_loss = current_price * 0.995
            take_profit = current_price * 1.005
        elif rsi[-1] > 75:
            signal = "超买做空"
            entry = current_price
            stop_loss = current_price * 1.005
            take_profit = current_price * 0.995
    
    # 如果信号触发，计算盈亏比
    if signal != "观望中":
        risk = abs(entry - stop_loss)
        reward = abs(take_profit - entry)
        risk_reward = reward / risk if risk > 0 else 1.0
    
    # 返回结果字典
    return {
        'current_price': current_price,
        'net_inflow': net_inflow,
        'buy_pressure': buy_pressure,
        'mode': "趋势模式" if is_trend else "震荡模式",
        'suggestion': "顺势回调" if is_trend else "高抛低吸",
        'signal': signal,
        'entry': entry,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'risk_reward': risk_reward
    }

# ==================== 主界面 ====================
st.title("🚀 ETH V57.0 实战终端 - 小利润策略")

# 获取数据
df = generate_simulated_data(minutes=200)

if df.empty:
    st.error("无法获取数据，请检查。")
    st.stop()

# 计算指标
ind = compute_indicators(df)

# ==================== 顶部自动识别与计划 ====================
with st.container():
    st.markdown("### 自动识别与计划")
    col_mode, _ = st.columns([2, 8])
    with col_mode:
        st.markdown(f"""
        <div style="background-color: #1e3a8a; padding: 20px; border-radius: 10px; text-align: center;">
            <h3 style="color: white;">建议：{ind['mode']}</h3>
            <p style="color: #ccc;">{ind['suggestion']}</p>
        </div>
        """, unsafe_allow_html=True)

# ==================== 关键指标卡片 ====================
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("现价", f"${ind['current_price']:.2f}", delta=f"{ind['current_price'] - df['close'].iloc[-2]:+.2f}")
with col2:
    st.metric("1min 净流入", f"{ind['net_inflow']:.2f} ETH", delta_color="inverse" if ind['net_inflow'] < 0 else "normal")
with col3:
    st.metric("买压占比", f"{ind['buy_pressure']:.1f}%")

# ==================== 交易信号卡片 ====================
st.markdown("---")
col_signal, col_detail = st.columns([2, 5])
with col_signal:
    # 根据信号类型设置卡片颜色
    if "观望" in ind['signal']:
        bg_color = "#f0b400"
    elif "做多" in ind['signal']:
        bg_color = "#00cc66"
    else:
        bg_color = "#ff4b4b"
    st.markdown(f"""
    <div style="background-color: {bg_color}; padding: 20px; border-radius: 10px; text-align: center;">
        <h2 style="color: white;">{ind['signal']}</h2>
    </div>
    """, unsafe_allow_html=True)

with col_detail:
    st.markdown(f"""
    **进场位:** ${ind['entry']:.2f}  
    **止损位:** ${ind['stop_loss']:.2f}  
    **止盈位:** ${ind['take_profit']:.2f}  
    **盈亏比:** {ind['risk_reward']:.2f}R
    """)

# ==================== 图表 ====================
st.markdown("---")
fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    row_heights=[0.7, 0.3], vertical_spacing=0.05)

# K线图
fig.add_trace(go.Candlestick(x=df.index,
                             open=df['open'],
                             high=df['high'],
                             low=df['low'],
                             close=df['close'],
                             name="ETH"), row=1, col=1)

# 均线
fig.add_trace(go.Scatter(x=df.index, y=df['close'].rolling(20).mean(),
                         line=dict(color='orange', width=1), name="MA20"), row=1, col=1)

# 水平线（进场、止损、止盈）
fig.add_hline(y=ind['entry'], line_dash="dash", line_color="white",
              annotation_text="进场", annotation_position="top left", row=1, col=1)
fig.add_hline(y=ind['stop_loss'], line_dash="dot", line_color="red",
              annotation_text="止损", annotation_position="bottom left", row=1, col=1)
fig.add_hline(y=ind['take_profit'], line_dash="dot", line_color="green",
              annotation_text="止盈", annotation_position="top left", row=1, col=1)

# 成交量
colors = ['red' if close < open else 'green' for close, open in zip(df['close'], df['open'])]
fig.add_trace(go.Bar(x=df.index, y=df['volume'], marker_color=colors, showlegend=False), row=2, col=1)

# 更新布局
fig.update_layout(
    height=600,
    template="plotly_dark",
    xaxis_rangeslider_visible=False,
    margin=dict(l=40, r=40, t=40, b=40)
)
fig.update_xaxes(title_text="时间", row=2, col=1)
fig.update_yaxes(title_text="价格 (USD)", row=1, col=1)
fig.update_yaxes(title_text="成交量", row=2, col=1)

st.plotly_chart(fig, use_container_width=True)

# 底部信息
st.caption(f"最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 数据源: 模拟数据 (可替换为真实数据) | 自动刷新每5秒")
