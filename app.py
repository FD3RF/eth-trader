import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
from collections import deque

# ---------- 配置 ----------
SYMBOL = "ETH-USDT"          # 交易对（现货）
INTERVAL = "5m"               # K线周期
LIMIT = 100                   # 每次获取的K线数量
UPDATE_INTERVAL = 60          # 轮询间隔（秒），实际每5分钟数据才更新

# ---------- 全局数据缓存 ----------
candle_buffer = deque(maxlen=300)

# ---------- 获取K线数据 ----------
@st.cache_data(ttl=60)  # 缓存60秒，避免频繁请求
def fetch_klines():
    url = f"https://www.okx.com/api/v5/market/history-candles?instId={SYMBOL}&bar={INTERVAL}&limit={LIMIT}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()['data']
            # 转换为统一格式 [timestamp, open, high, low, close, volume]
            candles = [[int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])] for k in data]
            # 按时间正序排列（OKX返回的是倒序）
            candles.sort(key=lambda x: x[0])
            return candles
        else:
            st.error(f"API请求失败: {resp.status_code}")
            return None
    except Exception as e:
        st.error(f"请求异常: {e}")
        return None

# ---------- 指标计算 ----------
def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def detect_signal(df, fast=9, slow=21, rsi_period=14, buy_min=50, buy_max=70, sell_min=30, sell_max=50):
    if len(df) < slow + rsi_period + 5:
        return None
    last = df.iloc[-1]
    ema_fast_now = df['ema_fast'].iloc[-1]
    ema_slow_now = df['ema_slow'].iloc[-1]
    ema_fast_prev = df['ema_fast'].iloc[-2]
    ema_slow_prev = df['ema_slow'].iloc[-2]
    rsi_now = df['rsi'].iloc[-1]

    golden_cross = ema_fast_prev <= ema_slow_prev and ema_fast_now > ema_slow_now
    death_cross = ema_fast_prev >= ema_slow_prev and ema_fast_now < ema_slow_now

    if golden_cross and last['close'] > ema_fast_now and buy_min < rsi_now < buy_max:
        return ('BUY', last['close'], ema_fast_now, ema_slow_now, rsi_now)
    if death_cross and last['close'] < ema_fast_now and sell_min < rsi_now < sell_max:
        return ('SELL', last['close'], ema_fast_now, ema_slow_now, rsi_now)
    return None

# ---------- 计算止损止盈 ----------
def calculate_sltp(entry_price, side):
    if side == 'BUY':
        sl = entry_price * 0.994
        tp1 = entry_price * 1.006
        tp2 = entry_price * 1.012
    else:
        sl = entry_price * 1.006
        tp1 = entry_price * 0.994
        tp2 = entry_price * 0.988
    return sl, tp1, tp2

# ---------- Streamlit界面 ----------
st.set_page_config(page_title="ETH 5分钟策略 (REST版)", layout="wide")
st.title("📈 ETH 5分钟 EMA 剥头皮策略 (REST API 轮询)")

# 新手说明
with st.expander("📘 新手快速上手指南（点击展开）", expanded=True):
    st.markdown("""
    ### 如何根据信号下单？
    当顶部出现 **🟢 多头信号** 或 **🔴 空头信号** 时，请按以下步骤操作：
    1. **进场价格** = 信号卡片中的 **“进场价格”** 数值（例如 `1825.60`）
    2. **止损价格** = 信号卡片中的 **“止损价格”** 数值（例如 `1815.00`）
    3. 在交易所（如OKX）以市价单或限价单买入/卖出，价格尽量接近进场价。
    4. 同时设置止损单（止损价）和止盈单（可选，按TP1/TP2设置）。
    """)

# 侧边栏参数
st.sidebar.header("策略参数")
fast_ema = st.sidebar.number_input("快线 EMA", 1, 50, 9, 1)
slow_ema = st.sidebar.number_input("慢线 EMA", 2, 100, 21, 1)
rsi_period = st.sidebar.number_input("RSI 周期", 2, 50, 14, 1)
buy_min = st.sidebar.number_input("多头 RSI 下限", 0, 100, 50, 1)
buy_max = st.sidebar.number_input("多头 RSI 上限", 0, 100, 70, 1)
sell_min = st.sidebar.number_input("空头 RSI 下限", 0, 100, 30, 1)
sell_max = st.sidebar.number_input("空头 RSI 上限", 0, 100, 50, 1)
refresh_interval = st.sidebar.number_input("刷新间隔(秒)", 5, 300, 60, 5)

# 手动刷新按钮
if st.sidebar.button("立即刷新数据"):
    st.cache_data.clear()
    st.rerun()

# 显示最新更新时间
placeholder = st.empty()

# 主循环（自动刷新）
while True:
    # 获取数据
    candles = fetch_klines()
    if candles is None:
        time.sleep(10)
        continue

    # 更新缓冲区
    for c in candles:
        candle_buffer.append(c)

    if len(candle_buffer) < 30:
        placeholder.warning(f"正在等待数据积累... 当前 {len(candle_buffer)}/30 根")
    else:
        # 转为DataFrame
        df = pd.DataFrame(list(candle_buffer), columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
        df['time'] = pd.to_datetime(df['ts'], unit='ms')
        df.set_index('time', inplace=True)

        # 计算指标
        df['ema_fast'] = calculate_ema(df['close'], fast_ema)
        df['ema_slow'] = calculate_ema(df['close'], slow_ema)
        df['rsi'] = calculate_rsi(df['close'], rsi_period)

        # 检测信号
        signal = detect_signal(df, fast_ema, slow_ema, rsi_period,
                               buy_min, buy_max, sell_min, sell_max)

        # 创建显示区域
        with placeholder.container():
            st.caption(f"最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | K线数量: {len(df)}")

            if signal:
                side, price, ema_f, ema_s, rsi = signal
                sl, tp1, tp2 = calculate_sltp(price, side)

                # 信号标题
                if side == 'BUY':
                    st.success(f"### 🟢 多头信号 @ {df.index[-1].strftime('%Y-%m-%d %H:%M')}")
                else:
                    st.error(f"### 🔴 空头信号 @ {df.index[-1].strftime('%Y-%m-%d %H:%M')}")

                # 显示关键价格
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📌 进场价格", f"{price:.2f}")
                with col2:
                    st.metric("🛑 止损价格", f"{sl:.2f}", delta_color="inverse")
                with col3:
                    st.metric("🎯 第一目标 (TP1)", f"{tp1:.2f}")

                st.metric("🚀 第二目标 (TP2)", f"{tp2:.2f}")

                # 额外指标
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("EMA快线", f"{ema_f:.2f}")
                col2.metric("EMA慢线", f"{ema_s:.2f}")
                col3.metric("RSI", f"{rsi:.1f}")
                col4.metric("当前价格", f"{df['close'].iloc[-1]:.2f}")

                # 添加操作指引
                st.info(f"""
                **操作指引**：
                - **进场**：以市价单或限价单在 **{price:.2f}** 附近买入（多头）或卖出（空头）。
                - **止损**：立即设置止损单，价格为 **{sl:.2f}**。
                - **止盈**：可设置两个止盈单：TP1 = **{tp1:.2f}**（平半仓），TP2 = **{tp2:.2f}**（全平）。
                """)

            else:
                st.info("⏳ 等待信号出现...")

            # 绘制K线图
            fig = go.Figure(data=[go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='K线'
            )])
            fig.add_trace(go.Scatter(x=df.index, y=df['ema_fast'], name=f'EMA{fast_ema}', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=df.index, y=df['ema_slow'], name=f'EMA{slow_ema}', line=dict(color='blue')))
            fig.update_layout(height=500, template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)

            # 显示最近K线
            st.subheader("最近 10 根K线")
            display_df = df[['open', 'high', 'low', 'close', 'volume', 'ema_fast', 'ema_slow', 'rsi']].tail(10).round(2)
            st.dataframe(display_df)

    # 等待指定间隔
    time.sleep(refresh_interval)
    st.cache_data.clear()
    st.rerun()
