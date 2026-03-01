import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import threading
import time
from datetime import datetime
import plotly.graph_objects as go
from collections import deque
import websocket
from streamlit_autorefresh import st_autorefresh

# ===============================
# 配置参数
# ===============================
SYMBOL = "ETH-USDT"
TIMEFRAME = "5m"                # 5分钟K线
MAX_KEEP = 300                   # 保留K线数量
RISK_PER_TRADE = 0.01            # 单笔风险1%
MIN_RR = 1.5                     # 最小盈亏比
REQUEST_TIMEOUT = 10              # 网络请求超时
WS_RETRY_INTERVAL = 5             # WebSocket重连间隔

# ===============================
# 安全配置：必须使用secrets
# ===============================
try:
    API_KEY = st.secrets["API_KEY"]
    API_SECRET = st.secrets["API_SECRET"]
    PASSPHRASE = st.secrets["PASSPHRASE"]
except Exception:
    st.error("❌ 请在 .streamlit/secrets.toml 中配置您的OKX API密钥")
    st.stop()

# ===============================
# 全局状态（线程安全）
# ===============================
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()
if "data_queue" not in st.session_state:
    st.session_state.data_queue = deque(maxlen=20)
if "ws_status" not in st.session_state:
    st.session_state.ws_status = "连接中..."
if "last_signal" not in st.session_state:
    st.session_state.last_signal = None
if "signal_history" not in st.session_state:
    st.session_state.signal_history = deque(maxlen=20)

# ===============================
# WebSocket 实时K线（OKX，带自动重连）
# ===============================
def on_message(ws, message):
    try:
        data = json.loads(message)
        if data.get("arg", {}).get("channel") == f"candle{TIMEFRAME}":
            for item in data.get("data", []):
                candle = {
                    "time": pd.to_datetime(int(item[0]), unit='ms'),
                    "open": float(item[1]),
                    "high": float(item[2]),
                    "low": float(item[3]),
                    "close": float(item[4]),
                    "volume": float(item[5])
                }
                st.session_state.data_queue.append(candle)
    except Exception as e:
        print(f"消息解析错误: {e}")

def on_error(ws, error):
    st.session_state.ws_status = f"错误: {error}"
    print(f"WebSocket错误: {error}")

def on_close(ws, close_status_code, close_msg):
    st.session_state.ws_status = "断开，尝试重连..."
    print("WebSocket关闭")

def on_open(ws):
    subscribe_msg = {
        "op": "subscribe",
        "args": [{"channel": f"candle{TIMEFRAME}", "instId": SYMBOL}]
    }
    ws.send(json.dumps(subscribe_msg))
    st.session_state.ws_status = "已连接"

def start_ws():
    while True:
        try:
            ws = websocket.WebSocketApp(
                "wss://ws.okx.com:8443/ws/v5/public",
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            ws.run_forever(ping_interval=30, ping_timeout=10)
        except Exception as e:
            print(f"WebSocket异常: {e}")
        time.sleep(WS_RETRY_INTERVAL)

# 启动WebSocket线程（仅一次）
if not any(thread.name == "OKX_WS" for thread in threading.enumerate()):
    thread = threading.Thread(target=start_ws, daemon=True, name="OKX_WS")
    thread.start()

# ===============================
# 辅助函数：获取1小时趋势（用于多周期过滤）
# ===============================
def safe_request(url, timeout=REQUEST_TIMEOUT, retries=3):
    for i in range(retries + 1):
        try:
            res = requests.get(url, timeout=timeout)
            if res.status_code == 200:
                return res.json()
            elif res.status_code == 429:
                retry_after = res.headers.get('Retry-After')
                wait = int(retry_after) if retry_after else 2 ** i
                time.sleep(wait)
                continue
        except requests.exceptions.Timeout:
            print(f"请求超时，重试 {i+1}/{retries}")
        except Exception as e:
            print(f"请求异常: {e}，重试 {i+1}/{retries}")
        time.sleep(0.5)
    return None

@st.cache_data(ttl=300)
def get_1h_trend():
    try:
        url = f"https://www.okx.com/api/v5/market/candles?instId={SYMBOL}&bar=1H&limit=100"
        data = safe_request(url)
        if not data or data.get('code') != '0':
            return 0
        items = data['data']
        df = pd.DataFrame(items, columns=['ts','o','h','l','c','v','volCcy','volCcyQuote','confirm'])
        df = df[::-1].reset_index(drop=True)
        df['c'] = df['c'].astype(float)
        df['ema50'] = df['c'].ewm(span=50, adjust=False).mean()
        last = df.iloc[-1]
        prev = df.iloc[-2]
        if last['c'] > last['ema50'] and last['ema50'] > prev['ema50']:
            return 1
        elif last['c'] < last['ema50'] and last['ema50'] < prev['ema50']:
            return -1
        else:
            return 0
    except:
        return 0

# ===============================
# 数据处理
# ===============================
def update_dataframe():
    if not st.session_state.data_queue:
        return False
    new_rows = []
    while st.session_state.data_queue:
        new_rows.append(st.session_state.data_queue.popleft())
    if not new_rows:
        return False
    new_df = pd.DataFrame(new_rows).drop_duplicates(subset=["time"]).sort_values("time")
    if st.session_state.df.empty:
        st.session_state.df = new_df.tail(MAX_KEEP)
    else:
        combined = pd.concat([st.session_state.df, new_df]).drop_duplicates(subset=["time"]).sort_values("time")
        st.session_state.df = combined.tail(MAX_KEEP)
    return True

def compute_indicators(df):
    if len(df) < 50:
        return df
    df = df.copy()
    close = df['close']
    high = df['high']
    low = df['low']

    # EMA
    df['ema12'] = close.ewm(span=12, adjust=False).mean()
    df['ema26'] = close.ewm(span=26, adjust=False).mean()
    df['ema50'] = close.ewm(span=50, adjust=False).mean()

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))

    # ATR
    tr = np.maximum(
        high - low,
        np.maximum(abs(high - close.shift()), abs(low - close.shift()))
    )
    df['atr'] = tr.rolling(14).mean()

    # 布林带
    df['bb_mid'] = close.rolling(20).mean()
    df['bb_std'] = close.rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']

    # 成交量MA
    df['vol_ma'] = df['volume'].rolling(10).mean()

    return df

# ===============================
# 信号生成（多因子 + 趋势过滤 + 盈亏比检查）
# ===============================
def generate_signal(df, trend_1h, current_index):
    if len(df) < 50 or current_index < 50:
        return None

    last = df.iloc[current_index]
    prev = df.iloc[current_index-1]
    atr = last['atr'] if not np.isnan(last['atr']) else last['close'] * 0.01

    # 基础条件
    ema_bull = last['close'] > last['ema12']
    ema_bear = last['close'] < last['ema12']
    macd_bull = last['macd_hist'] > 0 and last['macd_hist'] > prev['macd_hist']
    macd_bear = last['macd_hist'] < 0 and last['macd_hist'] < prev['macd_hist']
    rsi_neutral = 30 < last['rsi'] < 70 if not np.isnan(last['rsi']) else False
    volume_confirm = last['volume'] > last['vol_ma'] * 1.2

    # 趋势过滤：只顺着大方向
    if trend_1h == 1 and last['close'] < last['ema50']:
        trend_ok = False
    elif trend_1h == -1 and last['close'] > last['ema50']:
        trend_ok = False
    else:
        trend_ok = True

    # 多头信号
    if ema_bull and macd_bull and rsi_neutral and volume_confirm and trend_ok:
        entry = last['close']
        stop = entry - atr * 1.5
        tp = entry + atr * 3.0
        rr = abs(tp - entry) / abs(entry - stop)
        if rr >= MIN_RR:
            return {
                "side": "LONG",
                "entry": entry,
                "stop": stop,
                "tp": tp,
                "time": last['time'],
                "rr": rr,
                "score": 70,
                "reason": "EMA多头|MACD金叉|RSI中性|放量"
            }

    # 空头信号
    if ema_bear and macd_bear and rsi_neutral and volume_confirm and trend_ok:
        entry = last['close']
        stop = entry + atr * 1.5
        tp = entry - atr * 3.0
        rr = abs(entry - tp) / abs(stop - entry)
        if rr >= MIN_RR:
            return {
                "side": "SHORT",
                "entry": entry,
                "stop": stop,
                "tp": tp,
                "time": last['time'],
                "rr": rr,
                "score": 30,
                "reason": "EMA空头|MACD死叉|RSI中性|放量"
            }

    return None

def is_signal_valid(signal, current_price, atr):
    """检查信号是否仍有效（价格未远离入场区）"""
    if signal is None:
        return None
    entry_low = signal['entry'] - atr * 0.5
    entry_high = signal['entry'] + atr * 0.5
    if current_price < entry_low - atr or current_price > entry_high + atr:
        return None
    return signal

# ===============================
# 仓位计算
# ===============================
def position_size(entry, stop, balance, risk_pct):
    risk_amount = balance * risk_pct / 100
    distance = abs(entry - stop)
    return risk_amount / distance if distance > 0 else 0

# ===============================
# 主界面
# ===============================
st.set_page_config(layout="wide", page_title="5分钟ETH高频信号·至尊版")
st.title("📈 5分钟ETH高频信号系统（至尊版）")
st.caption(f"WebSocket状态: {st.session_state.ws_status} | 数据实时更新")

# 侧边栏参数
with st.sidebar:
    st.header("⚙️ 账户参数")
    balance = st.number_input("账户余额 (USDT)", value=10000, min_value=100, step=100)
    risk = st.slider("单笔风险 %", 0.1, 5.0, 1.0, 0.1)

    st.header("🎛️ 策略开关")
    enable_trend_filter = st.checkbox("启用1小时趋势过滤", value=True)
    enable_volume_filter = st.checkbox("启用成交量过滤", value=True)

    st.header("📜 信号历史")
    if st.button("清空历史记录"):
        st.session_state.signal_history.clear()
        st.session_state.last_signal = None
        st.rerun()

# 更新数据
if update_dataframe():
    st.rerun()

df = st.session_state.df.copy()
if len(df) >= 50:
    df = compute_indicators(df)
    trend_1h = get_1h_trend() if enable_trend_filter else 0
    last_idx = len(df) - 1
    current_price = df['close'].iloc[-1]
    atr = df['atr'].iloc[-1] if not np.isnan(df['atr'].iloc[-1]) else current_price * 0.01

    # 生成新信号
    new_signal = generate_signal(df, trend_1h, last_idx)
    if new_signal:
        st.session_state.last_signal = new_signal
        st.session_state.signal_history.append(new_signal)

    # 检查已有信号是否有效
    valid_signal = is_signal_valid(st.session_state.last_signal, current_price, atr)

    # 显示最新行情
    last = df.iloc[-1]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("最新价格", f"${last['close']:.2f}")
    col2.metric("EMA12/26", f"{last['ema12']:.2f}/{last['ema26']:.2f}")
    col3.metric("RSI", f"{last['rsi']:.1f}" if not np.isnan(last['rsi']) else "N/A")
    col4.metric("ATR", f"{atr:.2f}")

    # 显示1小时趋势
    if enable_trend_filter:
        trend_text = {1: "多头", -1: "空头", 0: "震荡"}[trend_1h]
        st.info(f"1小时趋势: {trend_text}")

    # 信号展示
    if valid_signal:
        sig = valid_signal
        size = position_size(sig['entry'], sig['stop'], balance, risk)
        entry_zone = f"{sig['entry']-atr*0.5:.1f}~{sig['entry']+atr*0.5:.1f}"
        st.markdown(f"""
        <div style="background: linear-gradient(90deg, #1e2a3a, #0b1428); border-radius: 10px; padding: 20px; border: 2px solid {'#00cc77' if sig['side']=='LONG' else '#ff6b6b'};">
            <h2 style="color: {'#00cc77' if sig['side']=='LONG' else '#ff6b6b'}; margin:0;">{'🔥 多头' if sig['side']=='LONG' else '❄️ 空头'}</h2>
            <p style="color:#ccc;">胜率评分: {sig['score']}% | 信号理由: {sig['reason']}</p>
            <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:10px; margin-top:10px;">
                <div><span style="color:#aaa;">入场区间</span><br><b style="color:#00cc77;">{entry_zone}</b></div>
                <div><span style="color:#aaa;">止损</span><br><b style="color:#ff6b6b;">${sig['stop']:.2f}</b></div>
                <div><span style="color:#aaa;">止盈</span><br><b style="color:#00cc77;">${sig['tp']:.2f}</b></div>
            </div>
            <div style="margin-top:10px;">
                <span style="color:#aaa;">当前价格: ${current_price:.2f} | 触发价: ${sig['entry']:.2f} | 盈亏比: {sig['rr']:.2f}</span>
            </div>
            <div style="margin-top:10px; background:#333; height:4px; border-radius:2px;">
                <div style="width:{min(100, max(0, (current_price - (sig['entry']-atr))/(2*atr)*100))}%; height:100%; background:{'#00cc77' if sig['side']=='LONG' else '#ff6b6b'}; border-radius:2px;"></div>
            </div>
            <p style="color:#aaa; font-size:0.8rem;">信号有效范围: {sig['entry']-atr*0.5:.1f} ~ {sig['entry']+atr*0.5:.1f}</p>
        </div>
        """, unsafe_allow_html=True)

        # 建议仓位
        st.info(f"💰 建议仓位: **{size:.4f} ETH** (账户余额 {balance} USDT, 风险 {risk}%)")
    else:
        if st.session_state.last_signal:
            st.warning("⚠️ 当前无有效信号（价格已远离入场区或信号过期）")
        else:
            st.info("⏳ 等待信号...")

    # K线图
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df['time'].tail(100),
        open=df['open'].tail(100),
        high=df['high'].tail(100),
        low=df['low'].tail(100),
        close=df['close'].tail(100),
        name="K线"
    ))
    fig.add_trace(go.Scatter(x=df['time'].tail(100), y=df['ema12'].tail(100), line=dict(color='#00cc77'), name="EMA12"))
    fig.add_trace(go.Scatter(x=df['time'].tail(100), y=df['ema26'].tail(100), line=dict(color='#ff6b6b'), name="EMA26"))
    fig.add_trace(go.Scatter(x=df['time'].tail(100), y=df['bb_upper'].tail(100), line=dict(color='rgba(255,255,255,0.3)', dash='dot'), name="布林上轨"))
    fig.add_trace(go.Scatter(x=df['time'].tail(100), y=df['bb_lower'].tail(100), line=dict(color='rgba(255,255,255,0.3)', dash='dot'), name="布林下轨"))
    if valid_signal:
        fig.add_hline(y=valid_signal['entry'], line_dash="dash", line_color="orange", annotation_text="触发位")
    fig.update_layout(
        template="plotly_dark",
        height=500,
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )
    st.plotly_chart(fig, width='stretch')  # 替代 use_container_width=True

    # 信号历史
    if st.session_state.signal_history:
        st.subheader("最近信号记录")
        hist = list(st.session_state.signal_history)[-10:]
        hist_df = pd.DataFrame(hist)[['time','side','entry','stop','tp','rr']]
        hist_df['time'] = hist_df['time'].dt.strftime('%m-%d %H:%M')
        st.dataframe(hist_df, width='stretch')

else:
    st.info("等待数据接入...")

# 自动刷新（每5秒）
st_autorefresh(interval=5000, key="auto_refresh")
