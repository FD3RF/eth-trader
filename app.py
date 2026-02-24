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

# ---------- 获取K线数据 ----------
@st.cache_data(ttl=15)
def fetch_klines():
    """获取最新K线数据（历史接口，用于初始化）"""
    url = f"https://www.okx.com/api/v5/market/history-candles?instId={SYMBOL}&bar={INTERVAL}&limit={LIMIT}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json().get('data', [])
            candles = [[int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])] for k in data]
            candles.sort(key=lambda x: x[0])
            return candles
        return None
    except:
        return None

# ---------- 实时获取最新K线（用于更新） ----------
@st.cache_data(ttl=5)
def fetch_latest_candle():
    """获取最新一根K线（模拟实时）"""
    url = f"https://www.okx.com/api/v5/market/candles?instId={SYMBOL}&bar={INTERVAL}&limit=1"
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json().get('data', [])
            if data:
                k = data[0]
                return [int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])]
        return None
    except:
        return None

# ---------- 获取高周期趋势（1小时EMA200） ----------
@st.cache_data(ttl=300)
def get_higher_trend():
    """获取1小时趋势方向（up/down/neutral）"""
    try:
        url = f"https://www.okx.com/api/v5/market/history-candles?instId={SYMBOL}&bar=1H&limit=200"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json().get('data', [])
            closes = [float(k[4]) for k in data]
            if len(closes) < 200:
                return 'neutral'
            ema200 = pd.Series(closes).ewm(span=200, adjust=False).mean().iloc[-1]
            return 'up' if closes[-1] > ema200 else 'down'
    except:
        return 'neutral'
    return 'neutral'

# ---------- 指标计算 ----------
def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calculate_rsi_wilder(series, period=14):
    """Wilder 平滑 RSI（标准算法）"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calculate_atr(df, period=14):
    high, low, close = df['high'], df['low'], df['close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def volume_surge(df, vol_period=20, surge_mult=1.5):
    """判断当前成交量是否爆发"""
    if len(df) < vol_period + 1:
        return True
    avg_vol = df['volume'].rolling(window=vol_period).mean().iloc[-1]
    return df['volume'].iloc[-1] > avg_vol * surge_mult

def detect_signal_pro(df, fast, slow, rsi_period, buy_min, buy_max, sell_min, sell_max,
                      higher_trend, use_volume_filter, use_slope_filter, use_atr_filter):
    """高级信号检测（包含多种过滤）"""
    if len(df) < slow + rsi_period + 10:
        return None

    last = df.iloc[-1]
    ema_f_now, ema_s_now = df['ema_fast'].iloc[-1], df['ema_slow'].iloc[-1]
    ema_f_prev, ema_s_prev = df['ema_fast'].iloc[-2], df['ema_slow'].iloc[-2]
    rsi_now = df['rsi'].iloc[-1]
    atr_now = df['atr'].iloc[-1] if 'atr' in df.columns else 0

    # 基础交叉
    golden = ema_f_prev <= ema_s_prev and ema_f_now > ema_s_now
    death = ema_f_prev >= ema_s_prev and ema_f_now < ema_s_now

    # 斜率过滤
    slope_fast = (ema_f_now - df['ema_fast'].iloc[-4]) / 3 if len(df) >= 4 else 0

    # ATR过滤（波动过小不交易）
    atr_ok = atr_now > last['close'] * 0.002 if use_atr_filter else True

    # 成交量过滤
    vol_ok = volume_surge(df) if use_volume_filter else True

    if golden and last['close'] > ema_f_now and buy_min < rsi_now < buy_max:
        if use_slope_filter and slope_fast <= 0:
            return None
        if higher_trend == 'down':
            return None
        if not atr_ok or not vol_ok:
            return None
        return ('BUY', last['close'], ema_f_now, ema_s_now, rsi_now, atr_now)

    if death and last['close'] < ema_f_now and sell_min < rsi_now < sell_max:
        if use_slope_filter and slope_fast >= 0:
            return None
        if higher_trend == 'up':
            return None
        if not atr_ok or not vol_ok:
            return None
        return ('SELL', last['close'], ema_f_now, ema_s_now, rsi_now, atr_now)

    return None

def calculate_sltp(entry_price, side, atr=None, use_atr=False, atr_mult_sl=1.2, atr_mult_tp1=1.2, atr_mult_tp2=2.0):
    """计算止损止盈（支持ATR动态或固定百分比）"""
    if use_atr and atr and atr > 0:
        if side == 'BUY':
            sl = entry_price - atr * atr_mult_sl
            tp1 = entry_price + atr * atr_mult_tp1
            tp2 = entry_price + atr * atr_mult_tp2
        else:
            sl = entry_price + atr * atr_mult_sl
            tp1 = entry_price - atr * atr_mult_tp1
            tp2 = entry_price - atr * atr_mult_tp2
        return sl, tp1, tp2
    else:
        # 固定百分比
        if side == 'BUY':
            sl = entry_price * 0.994
            tp1 = entry_price * 1.006
            tp2 = entry_price * 1.012
        else:
            sl = entry_price * 1.006
            tp1 = entry_price * 0.994
            tp2 = entry_price * 0.988
        return sl, tp1, tp2

# ---------- 信号统计管理 ----------
def update_signal_stats():
    total = len(st.session_state.signal_history)
    if total == 0:
        st.session_state.signal_stats = {'total': 0, 'win': 0, 'loss': 0, 'pending': 0, 'win_rate': 0}
        return
    win = sum(1 for s in st.session_state.signal_history if s.get('result') == 'win')
    loss = sum(1 for s in st.session_state.signal_history if s.get('result') == 'loss')
    pending = sum(1 for s in st.session_state.signal_history if s.get('result') == 'pending')
    win_rate = round(win / (win + loss) * 100, 1) if (win + loss) > 0 else 0
    st.session_state.signal_stats = {
        'total': total, 'win': win, 'loss': loss, 'pending': pending, 'win_rate': win_rate
    }

def add_signal_to_history(signal, sl, tp1, tp2, atr=None):
    signal_time = datetime.now().strftime('%Y-%m-%d %H:%M')
    record = {
        'record_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'signal_time': signal_time,
        'side': signal[0],
        'price': signal[1],
        'ema_fast': signal[2],
        'ema_slow': signal[3],
        'rsi': signal[4],
        'atr': signal[5] if len(signal) > 5 else None,
        'sl': sl,
        'tp1': tp1,
        'tp2': tp2,
        'result': 'pending',
        'exit_price': None,
        'exit_time': None,
        'peak': signal[1],          # 用于移动止损
        'note': ''
    }
    st.session_state.signal_history.insert(0, record)
    if len(st.session_state.signal_history) > 200:
        st.session_state.signal_history = st.session_state.signal_history[:200]
    update_signal_stats()

def update_signal_result(index, result, exit_price=None, exit_reason='', note=''):
    if 0 <= index < len(st.session_state.signal_history):
        st.session_state.signal_history[index]['result'] = result
        if exit_price:
            st.session_state.signal_history[index]['exit_price'] = round(exit_price, 2)
            st.session_state.signal_history[index]['exit_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if exit_reason:
            st.session_state.signal_history[index]['exit_reason'] = exit_reason
        if note:
            st.session_state.signal_history[index]['note'] = note
        update_signal_stats()

def clear_signal_history():
    st.session_state.signal_history = []
    update_signal_stats()

# ---------- 检查移动止损触发 ----------
def check_trailing_stop(record, current_price, trailing_dist):
    side = record['side']
    peak = record['peak']
    if side == 'BUY':
        trailing_sl = peak * (1 - trailing_dist / 100)
        if current_price <= trailing_sl:
            return True, 'loss', trailing_sl, '移动止损触发'
    else:
        trailing_sl = peak * (1 + trailing_dist / 100)
        if current_price >= trailing_sl:
            return True, 'loss', trailing_sl, '移动止损触发'
    return False, None, None, None

# ---------- Streamlit 页面配置 ----------
st.set_page_config(page_title="ETH 5分钟策略 (职业版)", layout="wide")
st.title("📈 ETH 5分钟 EMA 剥头皮策略 (职业整合版)")

# 新手快速上手指南
with st.expander("📘 新手快速上手指南（点击展开）", expanded=True):
    st.markdown("""
    ### 如何根据信号下单？
    当顶部出现 🟢 多头信号 或 🔴 空头信号 时，请按以下步骤操作：
    1. **进场价格**=信号卡片中的“进场价格”数值（例如 1825.60）
    2. **止损价格**=信号卡片中的“止损价格”数值（例如 1815.00）
    3. 在交易所（如OKX）以市价单或限价单买入/卖出，价格尽量接近进场价。
    4. 同时设置止损单（止损价）和止盈单（可选，按TP1/TP2设置）。
    """)

# ---------- 侧边栏 ----------
st.sidebar.header("策略参数")
fast_ema = st.sidebar.number_input("快线 EMA", 1, 50, 9, 1)
slow_ema = st.sidebar.number_input("慢线 EMA", 2, 100, 21, 1)
rsi_period = st.sidebar.number_input("RSI 周期", 2, 50, 14, 1)
buy_min = st.sidebar.number_input("多头 RSI 下限", 0, 100, 50, 1)
buy_max = st.sidebar.number_input("多头 RSI 上限", 0, 100, 70, 1)
sell_min = st.sidebar.number_input("空头 RSI 下限", 0, 100, 30, 1)
sell_max = st.sidebar.number_input("空头 RSI 上限", 0, 100, 50, 1)
refresh_interval = st.sidebar.number_input("刷新间隔(秒)", 5, 300, 60, 5)

# ---------- 高级过滤设置 ----------
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ 高级过滤")
use_slope_filter = st.sidebar.checkbox("启用斜率过滤", value=True, help="要求快线EMA持续向上/向下")
use_volume_filter = st.sidebar.checkbox("启用成交量爆发过滤", value=True, help="当前成交量需大于均量1.5倍")
use_atr_filter = st.sidebar.checkbox("启用波动率过滤", value=True, help="ATR低于0.2%价格时不交易")
use_higher_tf_filter = st.sidebar.checkbox("启用高周期趋势过滤", value=True, help="1H EMA200方向过滤")

# ---------- ATR动态止损设置 ----------
st.sidebar.markdown("---")
st.sidebar.subheader("📊 动态止损")
use_atr_sl = st.sidebar.checkbox("启用ATR动态止损", value=False, help="用ATR计算止损止盈，替代固定百分比")
if use_atr_sl:
    atr_mult_sl = st.sidebar.slider("ATR止损倍数", 0.5, 3.0, 1.2, 0.1)
    atr_mult_tp1 = st.sidebar.slider("ATR TP1倍数", 0.5, 3.0, 1.2, 0.1)
    atr_mult_tp2 = st.sidebar.slider("ATR TP2倍数", 1.0, 5.0, 2.0, 0.1)
else:
    atr_mult_sl = atr_mult_tp1 = atr_mult_tp2 = 1.2

# ---------- 移动止损设置 ----------
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ 移动止损")
use_trailing = st.sidebar.checkbox("启用移动止损", value=False, help="开启后将在持仓期间动态调整止损建议，并自动标记结果")
if use_trailing:
    trailing_distance = st.sidebar.slider("回调距离 (%)", 0.1, 2.0, 0.3, 0.1)

if st.sidebar.button("立即刷新数据"):
    st.cache_data.clear()
    st.rerun()

if st.sidebar.button("🔄 重置所有状态"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# ---------- 初始化 session_state ----------
if 'candle_buffer' not in st.session_state:
    st.session_state.candle_buffer = deque(maxlen=500)
if 'signal_history' not in st.session_state:
    st.session_state.signal_history = []
if 'signal_stats' not in st.session_state:
    st.session_state.signal_stats = {'total': 0, 'win': 0, 'loss': 0, 'pending': 0, 'win_rate': 0}
if 'last_signal_time' not in st.session_state:
    st.session_state.last_signal_time = None

candle_buffer = st.session_state.candle_buffer
signal_history = st.session_state.signal_history

# ---------- 获取并合并K线 ----------
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

# 尝试获取最新一根K线（模拟实时更新）
latest = fetch_latest_candle()
if latest and (len(candle_buffer) == 0 or latest[0] > candle_buffer[-1][0]):
    candle_buffer.append(latest)

# ---------- 自动刷新 ----------
st_autorefresh(interval=refresh_interval * 1000, key="auto_refresh")

# ---------- 获取高周期趋势 ----------
higher_trend = get_higher_trend() if use_higher_tf_filter else 'neutral'

# ---------- 数据准备与指标计算 ----------
st.caption(f"最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | K线数量: {len(candle_buffer)} | 高周期趋势: {higher_trend.upper()}")

if len(candle_buffer) < 30:
    st.warning(f"⏳ 数据积累中... 当前 {len(candle_buffer)}/30 根")
else:
    df = pd.DataFrame(list(candle_buffer), columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
    df['time'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('time', inplace=True)

    df['ema_fast'] = calculate_ema(df['close'], fast_ema)
    df['ema_slow'] = calculate_ema(df['close'], slow_ema)
    df['rsi'] = calculate_rsi_wilder(df['close'], rsi_period)
    df['atr'] = calculate_atr(df, 14)

    # 检测信号（使用高级版本）
    signal = detect_signal_pro(
        df, fast_ema, slow_ema, rsi_period, buy_min, buy_max, sell_min, sell_max,
        higher_trend, use_volume_filter, use_slope_filter, use_atr_filter
    )

    # 记录新信号
    if signal:
        side, price, ema_f, ema_s, rsi, atr_val = signal
        sl, tp1, tp2 = calculate_sltp(price, side, atr_val, use_atr_sl, atr_mult_sl, atr_mult_tp1, atr_mult_tp2)
        current_kline_time = df.index[-1]
        if st.session_state.last_signal_time != current_kline_time:
            st.session_state.last_signal_time = current_kline_time
            add_signal_to_history(signal, sl, tp1, tp2, atr_val)

    # 更新最新信号的峰值（用于移动止损）
    if signal_history and signal_history[0]['result'] == 'pending':
        latest = signal_history[0]
        current_price = df['close'].iloc[-1]
        if latest['side'] == 'BUY':
            if current_price > latest['peak']:
                latest['peak'] = current_price
        else:
            if current_price < latest['peak']:
                latest['peak'] = current_price

    # ---------- 自动检查止损/止盈/移动止损 ----------
    current_price = df['close'].iloc[-1]
    for idx, rec in enumerate(signal_history):
        if rec['result'] != 'pending':
            continue
        side = rec['side']
        entry = rec['price']
        sl = rec['sl']
        tp1 = rec['tp1']
        tp2 = rec['tp2']
        # 固定止损/止盈
        if side == 'BUY':
            if current_price <= sl:
                update_signal_result(idx, 'loss', current_price, '止损触发')
                continue
            elif current_price >= tp2:
                update_signal_result(idx, 'win', current_price, 'TP2触发')
                continue
        else:
            if current_price >= sl:
                update_signal_result(idx, 'loss', current_price, '止损触发')
                continue
            elif current_price <= tp2:
                update_signal_result(idx, 'win', current_price, 'TP2触发')
                continue
        # 移动止损检查
        if use_trailing:
            should_exit, result, exit_price, reason = check_trailing_stop(rec, current_price, trailing_distance)
            if should_exit:
                update_signal_result(idx, result, exit_price, reason)

    # ---------- 信号卡片显示 ----------
    if signal:
        side, price, ema_f, ema_s, rsi, atr_val = signal
        sl, tp1, tp2 = calculate_sltp(price, side, atr_val, use_atr_sl, atr_mult_sl, atr_mult_tp1, atr_mult_tp2)
        signal_time = df.index[-1].strftime('%Y-%m-%d %H:%M')

        if side == 'BUY':
            st.markdown(f"""
            <div style="background-color:#0a3d2a;padding:18px;border-radius:10px;border-left:8px solid #00ff9d;margin-bottom:15px;">
                <span style="font-size:28px;color:#00ff9d;">●</span>
                <strong style="color:#00ff9d;font-size:24px;"> 多头信号 @ {signal_time}</strong>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background-color:#3d0a0a;padding:18px;border-radius:10px;border-left:8px solid #ff4d4d;margin-bottom:15px;">
                <span style="font-size:28px;color:#ff4d4d;">●</span>
                <strong style="color:#ff4d4d;font-size:24px;"> 空头信号 @ {signal_time}</strong>
            </div>
            """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<span style="color:#ff4d4d;font-size:22px;">★</span> <strong>进场价格</strong><br><span style="font-size:36px;font-weight:bold;">{price:.2f}</span>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<span style="color:#ff99cc;font-size:22px;">●</span> <strong>止损价格</strong><br><span style="font-size:36px;font-weight:bold;">{sl:.2f}</span>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<span style="color:#ff99cc;font-size:22px;">●</span> <strong>第一目标 (TP1)</strong><br><span style="font-size:36px;font-weight:bold;">{tp1:.2f}</span>', unsafe_allow_html=True)

        st.markdown(f'<span style="color:#ffd700;font-size:22px;">✴</span> <strong>第二目标 (TP2)</strong><br><span style="font-size:32px;font-weight:bold;color:#ffd700;">{tp2:.2f}</span>', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("EMA快线", f"{ema_f:.2f}")
        col2.metric("EMA慢线", f"{ema_s:.2f}")
        col3.metric("RSI", f"{rsi:.1f}")
        col4.metric("当前价格", f"{current_price:.2f}")

        # 移动止损信息
        if use_trailing and signal_history and signal_history[0]['result'] == 'pending':
            latest_rec = signal_history[0]
            peak = latest_rec['peak']
            if side == 'BUY':
                trailing_sl = max(sl, peak * (1 - trailing_distance / 100))
                st.info(f"📉 **移动止损建议**：当前最高价 {peak:.2f}，移动止损价应为 **{trailing_sl:.2f}**。")
            else:
                trailing_sl = min(sl, peak * (1 + trailing_distance / 100))
                st.info(f"📉 **移动止损建议**：当前最低价 {peak:.2f}，移动止损价应为 **{trailing_sl:.2f}**。")

        st.markdown(f"""
        <div style="background-color:#1e3a5f;padding:18px;border-radius:10px;color:#a0d8ff;margin-top:15px;">
        <strong>操作指引：</strong>
        <ul>
            <li>进场：以市价单或限价单在 <strong>{price:.2f}</strong> 附近买入（多头）或卖出（空头）。</li>
            <li>止损：立即设置止损单，价格为 <strong>{sl:.2f}</strong>。</li>
            <li>止盈：可设置两个止盈单：TP1 = <strong>{tp1:.2f}</strong>（平半仓），TP2 = <strong>{tp2:.2f}</strong>（全平）。</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background-color:#1e3a5f;padding:20px;border-radius:10px;text-align:center;font-size:20px;color:#89c2ff;">
            ⏳ 等待信号出现...
        </div>
        """, unsafe_allow_html=True)

    # ---------- 绘制K线图 + 成交量 ----------
    colors = np.where(df['close'] >= df['open'], '#00ff9d', '#ff4d4d')
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.06, row_heights=[0.72, 0.28],
                        subplot_titles=("价格 & EMA", "成交量"))
    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'],
                                 low=df['low'], close=df['close'], name='K线'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['ema_fast'], name=f'EMA{fast_ema}', line=dict(color='orange', width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['ema_slow'], name=f'EMA{slow_ema}', line=dict(color='blue', width=2.5)), row=1, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df['volume'], name='成交量',
                         marker_color=colors, opacity=0.85), row=2, col=1)
    fig.update_layout(height=720, template='plotly_dark', showlegend=True,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_xaxes(rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # ---------- 最近10根K线 ----------
    st.subheader("最近 10 根K线")
    display_df = df.reset_index()[['time', 'open', 'high', 'low', 'close', 'volume', 'ema_fast', 'ema_slow', 'rsi', 'atr']].tail(10)
    display_df['time'] = display_df['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    display_df = display_df.round(2)
    st.dataframe(display_df, use_container_width=True, hide_index=True)

# ---------- 侧边栏信号统计 ----------
st.sidebar.markdown("---")
st.sidebar.subheader("📊 信号统计")
update_signal_stats()
stats = st.session_state.signal_stats
st.sidebar.metric("总信号", stats['total'])
st.sidebar.metric("胜率", f"{stats['win_rate']}%")
col1, col2 = st.sidebar.columns(2)
col1.metric("盈利", stats['win'])
col2.metric("亏损", stats['loss'])
st.sidebar.markdown(f"""
<div style="background-color:#1e3a5f;padding:12px;border-radius:8px;text-align:center;margin-top:10px;">
📍 <strong>待定信号: {stats['pending']}</strong>
</div>
""", unsafe_allow_html=True)

if st.sidebar.button("🗑 清空历史信号"):
    clear_signal_history()
    st.rerun()

# ---------- 历史信号记录 ----------
st.markdown("---")
st.subheader("📜 历史信号记录")

if st.session_state.signal_history:
    hist_df = pd.DataFrame(st.session_state.signal_history)
    hist_cols = ['record_time', 'signal_time', 'side', 'price', 'sl', 'tp1', 'tp2', 'result', 'exit_price', 'exit_time', 'exit_reason', 'note', 'peak', 'atr']
    available_cols = [col for col in hist_cols if col in hist_df.columns]
    display_hist = hist_df[available_cols].copy()

    def color_result(val):
        if val == 'win':
            return 'background-color: #90ee90'
        elif val == 'loss':
            return 'background-color: #ffcccb'
        elif val == 'pending':
            return 'background-color: #fffacd'
        return ''
    if 'result' in display_hist.columns:
        styled_hist = display_hist.style.applymap(color_result, subset=['result'])
    else:
        styled_hist = display_hist
    st.dataframe(styled_hist, use_container_width=True, height=300)

    # 手动标记结果
    with st.expander("✏️ 手动标记信号结果", expanded=False):
        col1, col2, col3, col4 = st.columns([3, 1, 1, 2])
        with col1:
            options = [f"{i}. {s.get('record_time', '')} {s.get('side', '')} @{s.get('price', '')}" 
                       for i, s in enumerate(st.session_state.signal_history)]
            selected_idx = st.selectbox("选择信号", range(len(options)), format_func=lambda x: options[x])
        with col2:
            result = st.selectbox("结果", ["pending", "win", "loss"])
        with col3:
            exit_price = st.number_input("出场价", value=0.0, step=0.1)
        with col4:
            if st.button("更新结果"):
                update_signal_result(selected_idx, result, exit_price if exit_price > 0 else None)
                st.rerun()

    # 导出CSV
    csv = hist_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 导出历史信号 (CSV)",
        data=csv,
        file_name=f"signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
else:
    st.info("暂无历史信号，等待新信号出现...")

st.markdown("---")
st.caption("🚀 职业整合版 • 包含高级过滤与自动胜负判定 • 祝交易顺利！")
