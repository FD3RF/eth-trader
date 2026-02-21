import streamlit as st
import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import time

# =============================
# 配置参数
# =============================
SYMBOL = "ETH/USDT:USDT"          # Bybit 永续合约格式
TIMEFRAMES = {"5m": 100, "15m": 100, "1h": 100}  # 周期及K线数量
REFRESH_INTERVAL = 5               # 刷新间隔（秒）
LEVERAGE = 100                      # 杠杆倍数（仅用于展示）
CIRCUIT_BREAKER_PCT = 0.005         # 0.5% 熔断阈值

# =============================
# 初始化交易所（Bybit 永续合约）
# =============================
exchange = ccxt.bybit({
    'enableRateLimit': True,
    'options': {
        'defaultType': 'linear'      # 永续合约
    }
})

# =============================
# Streamlit 页面设置
# =============================
st.set_page_config(layout="wide", page_title="ETH 永续合约监控")
st.title("🚀 ETH 永续合约 100x 多周期智能监控")

# 会话状态初始化
if 'last_price' not in st.session_state:
    st.session_state.last_price = 0
if 'system_halted' not in st.session_state:
    st.session_state.system_halted = False

# 侧边栏重置按钮
if st.sidebar.button("🔌 重置系统熔断"):
    st.session_state.system_halted = False
    st.session_state.last_price = 0

# =============================
# 核心函数
# =============================
def fetch_klines(timeframe, limit=100):
    """获取指定周期的K线数据并转换为DataFrame"""
    ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def add_indicators(df):
    """添加技术指标（使用 pandas_ta）"""
    df['ema9']   = ta.ema(df['close'], length=9)
    df['ema21']  = ta.ema(df['close'], length=21)
    df['rsi']    = ta.rsi(df['close'], length=14)
    df['adx']    = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14']
    df['atr']    = ta.atr(df['high'], df['low'], df['close'], length=14)
    macd = ta.macd(df['close'])
    df['macd']   = macd['MACD_12_26_9']
    df['signal'] = macd['MACDs_12_26_9']
    df['hist']   = macd['MACDh_12_26_9']
    return df

def detect_regime(df):
    """判断市场结构（趋势/震荡/高波动）"""
    df['adx'] = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14']
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['ema21'] = ta.ema(df['close'], length=21)

    adx_mean = df['adx'].tail(20).mean()
    atr_val = df['atr'].iloc[-1]
    slope = df['ema21'].iloc[-1] - df['ema21'].iloc[-5]

    if adx_mean > 25 and abs(slope) > 0.1:
        return "趋势", atr_val
    elif atr_val > (df['close'].iloc[-1] * 0.003):
        return "高波动", atr_val
    else:
        return "震荡", atr_val

def tf_score(df):
    """计算5分钟动能评分"""
    last = df.iloc[-1]
    score = 0

    # EMA 排列
    if last['ema9'] > last['ema21']:
        score += 20
    else:
        score -= 20

    # MACD 柱状图
    if last['hist'] > 0:
        score += 20
    else:
        score -= 20

    # ADX 强度
    if last['adx'] > 25:
        score += 25

    # RSI 过滤
    if last['rsi'] > 60:
        score += 15
    elif last['rsi'] < 40:
        score -= 15

    # 量能确认（最后一条K线的成交量是否大于20周期均值）
    vol_mean = df['volume'].tail(20).mean()
    if last['volume'] > vol_mean * 1.2:
        score += 20 if score > 0 else -20

    return score

def exhaustion_prob(df):
    """计算衰竭概率（基于ADX下降、MACD柱收缩、成交量萎缩）"""
    if len(df) < 5:
        return 0.0
    adx_drop = df['adx'].iloc[-1] < df['adx'].iloc[-3]
    hist_shrink = abs(df['hist'].iloc[-1]) < abs(df['hist'].iloc[-2])
    vol_mean = df['volume'].tail(20).mean()
    vol_drop = df['volume'].iloc[-1] < vol_mean

    # 归一化概率
    prob = (adx_drop + hist_shrink + vol_drop) / 3.0
    return prob

def get_signal():
    """获取三个周期的数据并生成交易信号"""
    try:
        df5  = fetch_klines('5m', TIMEFRAMES['5m'])
        df15 = fetch_klines('15m', TIMEFRAMES['15m'])
        df1h = fetch_klines('1h', TIMEFRAMES['1h'])

        # 添加指标
        df5  = add_indicators(df5)
        df15 = add_indicators(df15)
        df1h = add_indicators(df1h)

        # 判断15分钟结构
        regime, atr = detect_regime(df15)

        # 5分钟动能评分
        score_5 = tf_score(df5)

        # 衰竭概率
        exhaust = exhaustion_prob(df5)

        # 当前价格
        current_price = df5['close'].iloc[-1]

        # 信号方向
        direction = None
        if score_5 >= 50:
            direction = "LONG"
        elif score_5 <= -50:
            direction = "SHORT"

        # 计算止损和止盈（示例：基于ATR）
        if direction == "LONG":
            sl_dist = min(atr * 1.2, current_price * 0.003)  # 取ATR×1.2和0.3%中的较小者
            sl = current_price - sl_dist
            tp = current_price + (current_price - sl) * (1.2 + abs(score_5)/100 * 2.5)
        elif direction == "SHORT":
            sl_dist = min(atr * 1.2, current_price * 0.003)
            sl = current_price + sl_dist
            tp = current_price - (sl - current_price) * (1.2 + abs(score_5)/100 * 2.5)
        else:
            sl = tp = None

        # 如果衰竭概率过高，降低止盈倍数
        if exhaust > 0.66 and direction:
            tp = current_price + (current_price - sl) * 0.7 if direction == "LONG" else current_price - (sl - current_price) * 0.7

        return direction, current_price, sl, tp, score_5, exhaust, regime, df5, df15, df1h

    except Exception as e:
        st.error(f"数据获取失败：{e}")
        return None, None, None, None, None, None, None, None, None, None

# =============================
# 主循环
# =============================
placeholder = st.empty()

while True:
    # 熔断检测
    try:
        ticker = exchange.fetch_ticker(SYMBOL)
        current_ticker = ticker['last']
        if st.session_state.last_price != 0:
            change = abs(current_ticker - st.session_state.last_price) / st.session_state.last_price
            if change > CIRCUIT_BREAKER_PCT:
                st.session_state.system_halted = True
        st.session_state.last_price = current_ticker
    except Exception as e:
        st.sidebar.error(f"熔断检测异常：{e}")

    if st.session_state.system_halted:
        st.error("🚨 系统熔断！价格异常波动。请点击侧边栏重置按钮。")
        time.sleep(5)
        continue

    # 获取信号和数据
    direction, price, sl, tp, score, exhaust, regime, df5, df15, df1h = get_signal()

    with placeholder.container():
        if price is None:
            st.warning("正在等待数据...")
        else:
            # 第一行：实时指标
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("ETH 实时价", f"${price:.2f}")
            col2.metric("15m 结构", regime)
            col3.metric("5m 动能评分", f"{score} pt")
            col4.metric("衰竭概率", f"{exhaust*100:.1f}%")

            # 信号展示
            if direction:
                st.success(f"### 🎯 {direction} 信号触发")
                sc1, sc2, sc3, sc4 = st.columns(4)
                sc1.metric("入场位", f"${price:.2f}")
                sc2.metric("止损位", f"${sl:.2f}" if sl else "-")
                sc3.metric("止盈位", f"${tp:.2f}" if tp else "-")
                # 计算盈亏比
                if direction == "LONG" and sl:
                    rr = (tp - price) / (price - sl)
                elif direction == "SHORT" and sl:
                    rr = (price - tp) / (sl - price)
                else:
                    rr = None
                sc4.metric("盈亏比", f"1:{rr:.2f}" if rr else "-")
            else:
                st.info("💡 当前无明确信号，等待动能累积...")

            # 绘制5分钟K线图
            fig = go.Figure(data=[go.Candlestick(
                x=df5['timestamp'],
                open=df5['open'],
                high=df5['high'],
                low=df5['low'],
                close=df5['close'],
                name='5m K线'
            )])
            fig.update_layout(height=500, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig, width="stretch")   # 替代 use_container_width

            # 可选：显示15分钟和1小时的简略信息
            with st.expander("📊 多周期概览"):
                st.write("**15分钟**")
                st.dataframe(df15[['timestamp','close','ema9','ema21','rsi','adx']].tail(5))
                st.write("**1小时**")
                st.dataframe(df1h[['timestamp','close','ema9','ema21','rsi','adx']].tail(5))

    time.sleep(REFRESH_INTERVAL)
