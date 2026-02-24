import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh

# ---------- 配置 ----------
SYMBOL = "ETH-USDT"          # 交易对（现货）
INTERVAL = "5m"               # K线周期
LIMIT = 100                   # 每次获取的K线数量

# ---------- 初始化 session_state ----------
if 'candle_buffer' not in st.session_state:
    st.session_state.candle_buffer = []  # 存储K线数据，格式 [timestamp, open, high, low, close, volume]
    st.session_state.last_fetch_time = 0

# ---------- 获取K线数据（带缓存合并）----------
def fetch_and_merge_klines():
    """从OKX API获取最新K线，并与现有缓存合并（去重、排序）"""
    url = f"https://www.okx.com/api/v5/market/history-candles?instId={SYMBOL}&bar={INTERVAL}&limit={LIMIT}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            st.error(f"API请求失败: {resp.status_code}")
            return False

        data = resp.json()['data']
        # 转换为统一格式 [timestamp, open, high, low, close, volume]
        new_candles = []
        for k in data:
            # OKX返回的时间戳为毫秒，直接使用
            ts = int(k[0])
            o = float(k[1])
            h = float(k[2])
            l = float(k[3])
            c = float(k[4])
            vol = float(k[5])
            new_candles.append([ts, o, h, l, c, vol])
        new_candles.sort(key=lambda x: x[0])  # 按时间正序

        if not st.session_state.candle_buffer:
            st.session_state.candle_buffer = new_candles
        else:
            # 合并去重
            all_candles = st.session_state.candle_buffer + new_candles
            ts_dict = {c[0]: c for c in all_candles}  # 保留最后一条
            merged = list(ts_dict.values())
            merged.sort(key=lambda x: x[0])
            if len(merged) > 300:
                merged = merged[-300:]
            st.session_state.candle_buffer = merged
        return True
    except Exception as e:
        st.error(f"请求异常: {e}")
        return False

# ---------- 指标计算 ----------
def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    # 前period个值填充为50
    rsi.iloc[:period] = 50.0
    return rsi

def detect_signal(df, fast, slow, rsi_period, buy_min, buy_max, sell_min, sell_max):
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

# ---------- Streamlit 页面配置 ----------
st.set_page_config(page_title="ETH 5分钟策略 (完美版)", layout="wide")
st.title("📈 ETH 5分钟 EMA 剥头皮策略 (REST API 轮询 · 完美版)")

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
    fetch_and_merge_klines()
    st.rerun()

# ---------- 自动刷新 ----------
st_autorefresh(interval=refresh_interval * 1000, key="auto_refresh")

# ---------- 拉取最新数据 ----------
fetch_and_merge_klines()

# ---------- 数据准备和指标计算 ----------
candles = st.session_state.candle_buffer
if len(candles) < 30:
    st.warning(f"正在等待数据积累... 当前 {len(candles)}/30 根")
    st.stop()

# 转为DataFrame
df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
# 时间戳处理：OKX返回的是毫秒，直接转换为UTC时间，再转为北京时间（UTC+8）
df['time'] = pd.to_datetime(df['timestamp'], unit='ms') + pd.Timedelta(hours=8)
df.set_index('time', inplace=True)
df = df[['open', 'high', 'low', 'close', 'volume']]  # 只保留价格和成交量

# 计算指标
df['ema_fast'] = calculate_ema(df['close'], fast_ema)
df['ema_slow'] = calculate_ema(df['close'], slow_ema)
df['rsi'] = calculate_rsi(df['close'], rsi_period)

# 检测信号
signal = detect_signal(df, fast_ema, slow_ema, rsi_period,
                       buy_min, buy_max, sell_min, sell_max)

# ---------- 显示信号卡片 ----------
st.caption(f"最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | K线数量: {len(df)}")

if signal:
    side, price, ema_f, ema_s, rsi = signal
    sl, tp1, tp2 = calculate_sltp(price, side)

    if side == 'BUY':
        st.success(f"### 🟢 多头信号 @ {df.index[-1].strftime('%Y-%m-%d %H:%M')}")
    else:
        st.error(f"### 🔴 空头信号 @ {df.index[-1].strftime('%Y-%m-%d %H:%M')}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📌 进场价格", f"{price:.2f}")
    with col2:
        st.metric("🛑 止损价格", f"{sl:.2f}", delta_color="inverse")
    with col3:
        st.metric("🎯 第一目标 (TP1)", f"{tp1:.2f}")
    st.metric("🚀 第二目标 (TP2)", f"{tp2:.2f}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("EMA快线", f"{ema_f:.2f}")
    col2.metric("EMA慢线", f"{ema_s:.2f}")
    col3.metric("RSI", f"{rsi:.1f}")
    col4.metric("当前价格", f"{df['close'].iloc[-1]:.2f}")

    st.info(f"""
    **操作指引**：
    - **进场**：以市价单或限价单在 **{price:.2f}** 附近买入（多头）或卖出（空头）。
    - **止损**：立即设置止损单，价格为 **{sl:.2f}**。
    - **止盈**：可设置两个止盈单：TP1 = **{tp1:.2f}**（平半仓），TP2 = **{tp2:.2f}**（全平）。
    """)
else:
    st.info("⏳ 等待信号出现...")

# ---------- 绘制K线图 + 成交量 ----------
fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    row_heights=[0.7, 0.3]
)

fig.add_trace(go.Candlestick(
    x=df.index,
    open=df['open'],
    high=df['high'],
    low=df['low'],
    close=df['close'],
    name='K线'
), row=1, col=1)

fig.add_trace(go.Scatter(x=df.index, y=df['ema_fast'], name=f'EMA{fast_ema}', line=dict(color='orange')), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['ema_slow'], name=f'EMA{slow_ema}', line=dict(color='blue')), row=1, col=1)

colors = ['red' if df['close'].iloc[i] < df['open'].iloc[i] else 'green' for i in range(len(df))]
fig.add_trace(go.Bar(x=df.index, y=df['volume'], name='成交量', marker_color=colors), row=2, col=1)

fig.update_layout(
    height=700,
    template='plotly_dark',
    xaxis_rangeslider_visible=False,
    showlegend=True
)
fig.update_yaxes(title_text="价格 (USDT)", row=1, col=1)
fig.update_yaxes(title_text="成交量", row=2, col=1)

st.plotly_chart(fig, use_container_width=True)

# ---------- 显示最近K线 ----------
st.subheader("最近 10 根K线")
# 确保列名正确
display_cols = ['open', 'high', 'low', 'close', 'volume', 'ema_fast', 'ema_slow', 'rsi']
display_df = df[display_cols].tail(10).round(2)
st.dataframe(display_df)
