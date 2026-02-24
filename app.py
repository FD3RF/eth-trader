import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from collections import deque
import plotly.graph_objects as go
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
            candles.sort(key=lambda x: x[0])
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
    return rsi.fillna(50)

def detect_signal(df, fast, slow, rsi_period, buy_min, buy_max, sell_min, sell_max):
    if len(df) < slow + rsi_period + 5:
        return None
    ema_f_now, ema_s_now = df['ema_fast'].iloc[-1], df['ema_slow'].iloc[-1]
    ema_f_prev, ema_s_prev = df['ema_fast'].iloc[-2], df['ema_slow'].iloc[-2]
    rsi_now = df['rsi'].iloc[-1]
    last_close = df['close'].iloc[-1]

    golden = ema_f_prev <= ema_s_prev and ema_f_now > ema_s_now
    death = ema_f_prev >= ema_s_prev and ema_f_now < ema_s_now

    if golden and last_close > ema_f_now and buy_min < rsi_now < buy_max:
        return ('BUY', last_close, ema_f_now, ema_s_now, rsi_now)
    if death and last_close < ema_f_now and sell_min < rsi_now < sell_max:
        return ('SELL', last_close, ema_f_now, ema_s_now, rsi_now)
    return None

def calculate_sltp(entry_price, side):
    if side == 'BUY':
        return entry_price * 0.994, entry_price * 1.006, entry_price * 1.012
    else:
        return entry_price * 1.006, entry_price * 0.994, entry_price * 0.988

# ---------- Streamlit 配置 ----------
st.set_page_config(page_title="ETH 5分钟策略", layout="wide")
st.title("📊 ETH 5分钟 EMA 剥头皮策略 (REST API 轮询)")

# 新手指南（完全和你截图一致）
with st.expander("📘 新手快速上手指南（点击展开）", expanded=True):
    st.markdown("""
    ### 如何根据信号下单？
    当顶部出现 🟢 多头信号 或 🔴 空头信号 时，请按以下步骤操作：
    1. **进场价格**=信号卡片中的“进场价格”数值（例如 1825.60）
    2. **止损价格**=信号卡片中的“止损价格”数值（例如 1815.00）
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

if st.sidebar.button("立即刷新新数据"):
    st.rerun()

# 自动刷新
st_autorefresh(interval=refresh_interval * 1000, key="eth_scalp_final")

# ---------- 数据持久化 ----------
if 'candle_buffer' not in st.session_state:
    st.session_state.candle_buffer = deque(maxlen=400)
candle_buffer = st.session_state.candle_buffer

# 获取并智能合并K线
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

st.caption(f"最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | K线数量: {len(candle_buffer)}")

if len(candle_buffer) < 30:
    st.warning(f"⏳ 数据积累中... 当前 {len(candle_buffer)}/30 根")
else:
    df = pd.DataFrame(list(candle_buffer), columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
    df['time'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('time', inplace=True)

    df['ema_fast'] = calculate_ema(df['close'], fast_ema)
    df['ema_slow'] = calculate_ema(df['close'], slow_ema)
    df['rsi'] = calculate_rsi(df['close'], rsi_period)

    signal = detect_signal(df, fast_ema, slow_ema, rsi_period, buy_min, buy_max, sell_min, sell_max)

    # ---------- 信号显示（完全匹配截图风格） ----------
    if signal:
        side, price, ema_f, ema_s, rsi = signal
        sl, tp1, tp2 = calculate_sltp(price, side)
        signal_time = df.index[-1].strftime('%Y-%m-%d %H:%M')

        if side == 'BUY':
            st.markdown(f"""
            <div style="background-color:#0a3d2a; padding:18px; border-radius:10px; border-left:8px solid #00ff9d; margin-bottom:15px;">
                <span style="font-size:28px; color:#00ff9d;">●</span>
                <strong style="color:#00ff9d; font-size:24px;"> 多头信号 @ {signal_time}</strong>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background-color:#3d0a0a; padding:18px; border-radius:10px; border-left:8px solid #ff4d4d; margin-bottom:15px;">
                <span style="font-size:28px; color:#ff4d4d;">●</span>
                <strong style="color:#ff4d4d; font-size:24px;"> 空头信号 @ {signal_time}</strong>
            </div>
            """, unsafe_allow_html=True)

        # 价格区域
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<span style="color:#ff4d4d;font-size:22px;">★</span> <strong>进场价格</strong><br><span style="font-size:36px;font-weight:bold;color:#ffffff;">{price:.2f}</span>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<span style="color:#ff99cc;font-size:22px;">●</span> <strong>止损价格</strong><br><span style="font-size:36px;font-weight:bold;color:#ffffff;">{sl:.2f}</span>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<span style="color:#ff99cc;font-size:22px;">●</span> <strong>第一目标 (TP1)</strong><br><span style="font-size:36px;font-weight:bold;color:#ffffff;">{tp1:.2f}</span>', unsafe_allow_html=True)

        st.markdown(f"""
        <div style="margin-top:10px;">
            <span style="color:#ffd700;font-size:22px;">✴</span> 
            <strong>第二目标 (TP2)</strong><br>
            <span style="font-size:32px;font-weight:bold;color:#ffd700;">{tp2:.2f}</span>
        </div>
        """, unsafe_allow_html=True)

        # 指标
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("EMA快线", f"{ema_f:.2f}")
        col2.metric("EMA慢线", f"{ema_s:.2f}")
        col3.metric("RSI", f"{rsi:.1f}")
        col4.metric("当前价格", f"{df['close'].iloc[-1]:.2f}")

        # 操作指引（蓝色框，和截图一致）
        st.markdown(f"""
        <div style="background-color:#1e3a5f; padding:18px; border-radius:10px; color:#a0d8ff; margin-top:15px;">
        <strong>操作指引：</strong>
        <ul>
            <li>进场：以市价单或限价单在 <strong>{price:.2f}</strong> 附近买入（多头）或卖出（空头）。</li>
            <li>止损：立即设置止损单，价格为 <strong>{sl:.2f}</strong>。</li>
            <li>止盈：可设置两个止盈单：TP1 = <strong>{tp1:.2f}</strong>（平半仓），TP2 = <strong>{tp2:.2f}</strong>（全平）。</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.info("⏳ 等待信号出现... 当前无有效信号")

    # ---------- K线图（和截图完全一致） ----------
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='K线'
    )])
    fig.add_trace(go.Scatter(x=df.index, y=df['ema_fast'], name=f'EMA{fast_ema}', line=dict(color='orange', width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df['ema_slow'], name=f'EMA{slow_ema}', line=dict(color='blue', width=2)))
    fig.update_layout(height=520, template='plotly_dark', xaxis_rangeslider_visible=False, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

    # 最近K线
    st.subheader("最近 10 根K线")
    display_df = df[['open', 'high', 'low', 'close', 'volume', 'ema_fast', 'ema_slow', 'rsi']].tail(10).round(2)
    st.dataframe(display_df, use_container_width=True)

st.markdown("---")
st.caption("🔧 最终完整整合版 • 已按你截图完美还原 • 祝交易顺利！💰")
