import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from collections import deque
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh

# ---------- 配置 ----------
SYMBOL = "ETH-USDT"
INTERVAL = "5m"
LIMIT = 200

# ---------- 获取K线 ----------
@st.cache_data(ttl=15)
def fetch_klines():
    url = f"https://www.okx.com/api/v5/market/history-candles?instId={SYMBOL}&bar={INTERVAL}&limit={LIMIT}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json().get('data', [])
            candles = [[int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])] for k in data]
            candles.sort(key=lambda x: x[0])   # 正序
            return candles
        else:
            st.error(f"API请求失败: {resp.status_code}")
            return None
    except Exception as e:
        st.error(f"请求异常: {e}")
        return None

# ---------- 指标 ----------
def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(avg_loss != 0, 100.0)      # 避免除0
    rsi = rsi.where(avg_gain != 0, 0.0)
    return rsi

def detect_signal(df, fast, slow, rsi_period, buy_min, buy_max, sell_min, sell_max):
    if len(df) < slow + rsi_period + 5:
        return None
    last = df.iloc[-1]
    ema_f_now, ema_s_now = df['ema_fast'].iloc[-1], df['ema_slow'].iloc[-1]
    ema_f_prev, ema_s_prev = df['ema_fast'].iloc[-2], df['ema_slow'].iloc[-2]
    rsi_now = df['rsi'].iloc[-1]

    golden = ema_f_prev <= ema_s_prev and ema_f_now > ema_s_now
    death = ema_f_prev >= ema_s_prev and ema_f_now < ema_s_now

    if golden and last['close'] > ema_f_now and buy_min < rsi_now < buy_max:
        return ('BUY', last['close'], ema_f_now, ema_s_now, rsi_now)
    if death and last['close'] < ema_f_now and sell_min < rsi_now < sell_max:
        return ('SELL', last['close'], ema_f_now, ema_s_now, rsi_now)
    return None

def calculate_sltp(entry_price, side):
    if side == 'BUY':
        return entry_price * 0.994, entry_price * 1.006, entry_price * 1.012
    else:
        return entry_price * 1.006, entry_price * 0.994, entry_price * 0.988

# ---------- Streamlit 界面 ----------
st.set_page_config(page_title="ETH 5分钟策略", layout="wide")
st.title("📈 ETH 5分钟 EMA 剥头皮策略（自动刷新版）")

with st.expander("📘 新手快速上手指南", expanded=True):
    st.markdown("""
    ### 如何根据信号下单？
    1. **进场价格** = 信号卡片中的 **“进场价格”**
    2. **止损价格** = 信号卡片中的 **“止损价格”**
    3. 在 OKX 以市价/限价单进场（尽量贴近进场价）
    4. 同时挂止损单 + 两个止盈单（TP1 平半仓，TP2 全平）
    """)

# 侧边栏
st.sidebar.header("策略参数")
fast_ema = st.sidebar.number_input("快线 EMA", 1, 50, 9, 1)
slow_ema = st.sidebar.number_input("慢线 EMA", 2, 100, 21, 1)
rsi_period = st.sidebar.number_input("RSI 周期", 2, 50, 14, 1)
buy_min = st.sidebar.number_input("多头 RSI 下限", 0, 100, 50, 1)
buy_max = st.sidebar.number_input("多头 RSI 上限", 0, 100, 70, 1)
sell_min = st.sidebar.number_input("空头 RSI 下限", 0, 100, 30, 1)
sell_max = st.sidebar.number_input("空头 RSI 上限", 0, 100, 50, 1)
refresh_interval = st.sidebar.number_input("自动刷新间隔(秒)", 5, 300, 60, 5)

if st.sidebar.button("🔄 立即刷新"):
    st.rerun()

# 自动刷新（核心修复）
st_autorefresh(interval=refresh_interval * 1000, key="eth_scalp")

# ---------- 数据缓存 ----------
if 'candle_buffer' not in st.session_state:
    st.session_state.candle_buffer = deque(maxlen=400)
candle_buffer = st.session_state.candle_buffer

# 获取最新K线 + 智能合并
candles = fetch_klines()
if candles:
    if len(candle_buffer) == 0:
        for c in candles:
            candle_buffer.append(c)
    else:
        max_ts = candle_buffer[-1][0]
        new_candles = [c for c in candles if c[0] > max_ts]
        for c in new_candles:
            candle_buffer.append(c)

# ---------- 主界面 ----------
st.caption(f"⏰ 最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | K线数量: {len(candle_buffer)}")

if len(candle_buffer) < 30:
    st.warning(f"⏳ 数据积累中... 当前 {len(candle_buffer)}/30 根")
else:
    df = pd.DataFrame(list(candle_buffer), columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
    df['time'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('time', inplace=True)

    # 计算指标
    df['ema_fast'] = calculate_ema(df['close'], fast_ema)
    df['ema_slow'] = calculate_ema(df['close'], slow_ema)
    df['rsi'] = calculate_rsi(df['close'], rsi_period)

    signal = detect_signal(df, fast_ema, slow_ema, rsi_period, buy_min, buy_max, sell_min, sell_max)

    # ---------- 信号显示 ----------
    if signal:
        side, price, ema_f, ema_s, rsi = signal
        sl, tp1, tp2 = calculate_sltp(price, side)

        if side == 'BUY':
            st.success(f"### 🟢 多头信号 @ {df.index[-1].strftime('%H:%M')}")
        else:
            st.error(f"### 🔴 空头信号 @ {df.index[-1].strftime('%H:%M')}")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📌 进场价格", f"{price:.2f}")
        col2.metric("🛑 止损", f"{sl:.2f}", delta_color="inverse")
        col3.metric("🎯 TP1", f"{tp1:.2f}")
        col4.metric("🚀 TP2", f"{tp2:.2f}")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("EMA快线", f"{ema_f:.2f}")
        col2.metric("EMA慢线", f"{ema_s:.2f}")
        col3.metric("RSI", f"{rsi:.1f}")
        col4.metric("最新价格", f"{df['close'].iloc[-1]:.2f}")

        st.info(f"**操作指引**：以 **{price:.2f}** 附近进场，止损挂 **{sl:.2f}**，止盈挂 TP1/TP2")

    else:
        st.info("⏳ 等待信号出现...")

    # ---------- 升级版图表 ----------
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.06, row_heights=[0.75, 0.25],
                        subplot_titles=("K线 + EMA", "成交量"))

    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'],
                                 low=df['low'], close=df['close'], name='K线'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['ema_fast'], name=f'EMA{fast_ema}', line=dict(color='orange')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['ema_slow'], name=f'EMA{slow_ema}', line=dict(color='blue')), row=1, col=1)

    fig.add_trace(go.Bar(x=df.index, y=df['volume'], name='成交量', marker_color='rgba(255,255,255,0.3)'), row=2, col=1)

    fig.update_layout(height=650, template='plotly_dark', showlegend=True)
    fig.update_xaxes(rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # ---------- 最近K线 ----------
    st.subheader("最近 10 根K线")
    display_df = df[['open', 'high', 'low', 'close', 'volume', 'ema_fast', 'ema_slow', 'rsi']].tail(10).round(2)
    st.dataframe(display_df, use_container_width=True)

# 页脚
st.markdown("---")
st.caption("🔧 重构 by Grok • 自动刷新版 • 数据来自 OKX")
