import streamlit as st
import pandas as pd
import requests
from datetime import datetime
from collections import deque
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------- 配置 ----------
SYMBOL = "ETH"
CURRENCY = "USD"
INTERVAL = "5"
LIMIT = 200
INTERVAL_MINUTES = 5

# ---------- Session State 初始化（彻底解决 Warning） ----------
for key, default in {
    'history': deque(maxlen=200),
    'candles': deque(maxlen=500),
    'last_signal_time': None,
    'api_fail': 0,
    'last_error': "",
    'fast': 8, 'slow': 21, 'rsi_period': 14,
    'buy_min': 57, 'buy_max': 70, 'sell_min': 30, 'sell_max': 43,
    'refresh': 30, 'use_score': True, 'score_thresh': 70,
    'use_atr_sl': True, 'sl_m': 2.2, 'tp1_m': 0.8, 'tp2_m': 1.6
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ---------- 数据获取 ----------
@st.cache_data(ttl=10, show_spinner=False)
def fetch_klines_cached():
    return _fetch_klines_impl()

def _fetch_klines_impl():
    try:
        url = f"https://min-api.cryptocompare.com/data/v2/histominute?fsym={SYMBOL}&tsym={CURRENCY}&limit={LIMIT}&aggregate={INTERVAL}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()['Data']['Data']
        st.session_state.last_error = ""
        return [[x['time']*1000, x['open'], x['high'], x['low'], x['close'], x['volumefrom']] for x in data]
    except Exception as e:
        st.session_state.api_fail += 1
        st.session_state.last_error = f"CryptoCompare 失败: {str(e)[:50]}"
        try:
            url = f"https://api.binance.com/api/v3/klines?symbol=ETHUSDT&interval=5m&limit={LIMIT}"
            resp = requests.get(url, timeout=8)
            resp.raise_for_status()
            data = resp.json()
            st.session_state.last_error = ""
            return [[int(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5])] for x in data]
        except Exception as e2:
            st.session_state.last_error += f" | Binance 失败: {str(e2)[:50]}"
            return []

def fetch_latest():
    try:
        url = f"https://min-api.cryptocompare.com/data/v2/histominute?fsym={SYMBOL}&tsym={CURRENCY}&limit=1&aggregate={INTERVAL}"
        resp = requests.get(url, timeout=5)
        data = resp.json()['Data']['Data'][0]
        return [data['time']*1000, data['open'], data['high'], data['low'], data['close'], data['volumefrom']]
    except:
        try:
            url = "https://api.binance.com/api/v3/klines?symbol=ETHUSDT&interval=5m&limit=1"
            resp = requests.get(url, timeout=5)
            data = resp.json()[0]
            return [int(data[0]), float(data[1]), float(data[2]), float(data[3]), float(data[4]), float(data[5])]
        except:
            return None

# ---------- 指标 ----------
def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def rsi(s, n=14):
    d = s.diff()
    gain = d.clip(lower=0).ewm(alpha=1/n).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1/n).mean()
    return 100 - 100/(1+gain/loss)
def atr(df, n=14):
    tr = pd.concat([df['high']-df['low'], (df['high']-df['close'].shift()).abs(), (df['low']-df['close'].shift()).abs()], axis=1).max(1)
    return tr.ewm(alpha=1/n).mean()

# ---------- 评分 ----------
def get_score(side, df):
    last = df.iloc[-1]
    c, ef, es, r, a, v = last['close'], last['ema_fast'], last['ema_slow'], last['rsi'], last['atr'], last['volume']
    avg_v = df['volume'].rolling(20).mean().iloc[-1] if len(df)>=20 else v
    vr = v/avg_v if avg_v>0 else 1
    spread = abs(ef-es)/c*100
    slope = abs((ef - df['ema_fast'].iloc[-5])/5/c*100) if len(df)>=5 else 0
    atr_pct = a/c*100 if a>0 else 0

    s_ema = 20 if spread>=0.2 else 15 if spread>=0.15 else 10 if spread>=0.1 else 5 if spread>=0.05 else 0
    s_slope = 20 if slope>=0.04 else 15 if slope>=0.03 else 10 if slope>=0.02 else 5 if slope>=0.01 else 0
    s_vol = 20 if vr>=1.3 else 10 if vr>=0.8 else 0
    s_atr = 20 if atr_pct>=0.2 else 15 if atr_pct>=0.16 else 10 if atr_pct>=0.12 else 5 if atr_pct>=0.08 else 0

    if side=='BUY':
        s_rsi = 20 if 55<=r<=70 else 15 if 50<=r<55 or 70<r<=75 else 10 if 45<=r<50 or 75<r<=80 else 5 if 40<=r<45 or 80<r<=85 else 0
    else:
        s_rsi = 20 if 30<=r<=45 else 15 if 25<=r<30 or 45<r<=50 else 10 if 20<=r<25 or 50<r<=55 else 5 if 15<=r<20 or 55<r<=60 else 0

    total = s_ema + s_slope + s_vol + s_atr + s_rsi
    subs = {'EMA': s_ema, '斜率': s_slope, '量能': s_vol, 'ATR': s_atr, 'RSI': s_rsi}
    return total, subs

# ---------- 信号检测 ----------
def detect_signal(df, fast, slow, buy_range, sell_range, use_score, score_thresh):
    if len(df) < 50: return None
    last = df.iloc[-1]
    ef, es, r = last['ema_fast'], last['ema_slow'], last['rsi']
    is_bull = ef > es and last['close'] > ef*0.999
    is_bear = ef < es and last['close'] < ef*1.001
    if not (is_bull or is_bear): return None
    if is_bull and last['close'] <= df['close'].iloc[-2]: return None
    if is_bear and last['close'] >= df['close'].iloc[-2]: return None

    if use_score:
        total_score, _ = get_score('BUY' if is_bull else 'SELL', df)
        if total_score < score_thresh: return None
    else:
        if is_bull and not (buy_range[0] < r < buy_range[1]): return None
        if is_bear and not (sell_range[0] < r < sell_range[1]): return None
    return ('BUY' if is_bull else 'SELL', last['close'], ef, es, r, last['atr'])

# ---------- 止损止盈 ----------
def sltp(price, side, atr, use_atr, mult_sl=2.2, mult_tp1=0.8, mult_tp2=1.6):
    if use_atr and atr > 0:
        risk = atr * mult_sl
        if side == 'BUY':
            return price - risk, price + risk * mult_tp1, price + risk * mult_tp2
        return price + risk, price - risk * mult_tp1, price - risk * mult_tp2
    if side == 'BUY':
        return price*0.994, price*1.006, price*1.012
    return price*1.006, price*0.994, price*0.988

# ---------- 补K线 ----------
def fill_missing(buf, new):
    if not buf: return [new]
    last_ts = buf[-1][0]
    new_ts = new[0]
    expected = last_ts + INTERVAL_MINUTES*60*1000
    if new_ts > expected:
        missing = []
        ts = expected
        while ts < new_ts:
            missing.append([ts] + [buf[-1][4]]*4 + [0])
            ts += INTERVAL_MINUTES*60*1000
        return missing + [new]
    return [new]

# ---------- 重置默认 ----------
def reset_defaults():
    defaults = {'fast':8, 'slow':21, 'rsi_period':14, 'buy_min':57, 'buy_max':70,
                'sell_min':30, 'sell_max':43, 'refresh':30, 'use_score':True,
                'score_thresh':70, 'use_atr_sl':True, 'sl_m':2.2, 'tp1_m':0.8, 'tp2_m':1.6}
    for k, v in defaults.items():
        st.session_state[k] = v
    st.rerun()

# ---------- UI ----------
st.set_page_config(page_title="ETH 5分钟终极版", layout="wide")
st.markdown("""
<style>
    .stApp { background: #0a0e17; }
    .stApp .block-container { max-width: 100% !important; padding: 1rem 2rem; }
    .signal-card { background: linear-gradient(135deg, #0f2a1f, #0a1f33); border: 3px solid; border-radius: 18px; padding: 26px; margin: 16px 0; box-shadow: 0 0 30px rgba(0,255,157,0.3); animation: pulse-glow 2.8s ease-in-out infinite alternate; }
    @keyframes pulse-glow { 0% { box-shadow: 0 0 15px rgba(0,255,157,0.3), inset 0 0 10px rgba(0,255,157,0.15); } 50% { box-shadow: 0 0 45px rgba(0,255,157,0.7), inset 0 0 25px rgba(0,255,157,0.4); } 100% { box-shadow: 0 0 15px rgba(0,255,157,0.3), inset 0 0 10px rgba(0,255,157,0.15); } }
    .signal-card.sell-active { border-color: #ff4d88; animation: pulse-glow-sell 2.6s ease-in-out infinite alternate; }
    @keyframes pulse-glow-sell { 0% { box-shadow: 0 0 20px #ff4d8880; } 50% { box-shadow: 0 0 55px #ff4d88c0; } 100% { box-shadow: 0 0 20px #ff4d8880; } }
    .blink-title { animation: subtle-blink 4s infinite ease-in-out; }
    @keyframes subtle-blink { 0%,100%{opacity:1} 50%{opacity:0.75} }
    .signal-title { font-size:1.65rem; margin-bottom:18px; font-weight:bold; }
    .big-number { font-size:2.35rem; font-weight:bold; margin:6px 0; }
    .positive { color:#00ff9d; }
    .negative { color:#ff4d88; }
    .label { color:#a0b0c0; font-size:0.92rem; margin-bottom:6px; }
    .progress-container { background:#1e293b; border-radius:10px; height:14px; margin:14px 0; overflow:hidden; }
    .progress-bar { height:100%; background:linear-gradient(to right, #00ff9d, #00bfff); transition:width 0.5s ease; }
    .waiting-card { background: linear-gradient(135deg, #0f2a1f, #0a1f33); border: 3px solid; border-radius: 18px; padding: 42px; text-align: center; color: #ccd6e0; animation: pulse-wait 3s ease-in-out infinite alternate; }
    @keyframes pulse-wait { 0% { box-shadow: 0 0 12px #4da9ff80; } 50% { box-shadow: 0 0 35px #4da9ffc0; } 100% { box-shadow: 0 0 12px #4da9ff80; } }
    .trend-big { font-size:2rem; font-weight:bold; text-align:center; margin:10px 0; }
    .api-error { color:#ffaa00; font-size:0.9rem; margin-top:6px; }
    .stButton > button { width:100%; margin-bottom:8px; font-weight:600; }
</style>
""", unsafe_allow_html=True)

st.title("📈 ETH 5分钟 EMA 终极版 (双数据源 + 顶级优化)")

# ---------- 侧边栏 ----------
with st.sidebar:
    st.header("参数")
    fast = st.number_input("快线EMA", 1, 50, key='fast')
    slow = st.number_input("慢线EMA", 2, 100, key='slow')
    rsi_period = st.number_input("RSI周期", 2, 50, key='rsi_period')
    col1, col2 = st.columns(2)
    with col1:
        st.caption("多头RSI")
        buy_min = st.number_input("多头下限", 0, 100, key='buy_min')
        buy_max = st.number_input("多头上限", 0, 100, key='buy_max')
    with col2:
        st.caption("空头RSI")
        sell_min = st.number_input("空头下限", 0, 100, key='sell_min')
        sell_max = st.number_input("空头上限", 0, 100, key='sell_max')
    refresh = st.number_input("刷新秒数", 5, 300, key='refresh')
    st.caption(f"⏳ 下次刷新: {refresh}秒后 (手动)")

    use_score = st.checkbox("启用评分系统", key='use_score')
    score_thresh = st.slider("评分阈值", 0, 100, key='score_thresh', disabled=not use_score)
    use_atr_sl = st.checkbox("ATR动态止损", key='use_atr_sl')
    if use_atr_sl:
        sl_m = st.slider("止损倍数", 1.0, 5.0, key='sl_m', step=0.1)
        tp1_m = st.slider("TP1倍数", 0.2, 3.0, key='tp1_m', step=0.1)
        tp2_m = st.slider("TP2倍数", 0.5, 5.0, key='tp2_m', step=0.1)

    st.metric("📡 API失败", st.session_state.api_fail)
    if st.session_state.last_error:
        st.markdown(f'<div class="api-error">⚠️ {st.session_state.last_error}</div>', unsafe_allow_html=True)

    if st.session_state.history:
        total = len(st.session_state.history)
        wins = sum(1 for s in st.session_state.history if s.get('result') == 'win')
        losses = sum(1 for s in st.session_state.history if s.get('result') == 'loss')
        pending = sum(1 for s in st.session_state.history if s.get('result') == 'pending')
        win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
        st.sidebar.markdown("---")
        st.sidebar.markdown("**📈 信号统计**")
        st.sidebar.markdown(f"总信号: {total}")
        st.sidebar.markdown(f"✅ 盈利: {wins}")
        st.sidebar.markdown(f"❌ 亏损: {losses}")
        st.sidebar.markdown(f"⏳ 待定: {pending}")
        st.sidebar.markdown(f"🎯 胜率: {win_rate:.1f}%")

    col_btn1, col_btn2, col_btn3 = st.columns(3)
    with col_btn1:
        if st.button("🗑 清空历史"):
            st.session_state.history.clear()
    with col_btn2:
        if st.button("🔄 立即刷新"):
            st.cache_data.clear()
            st.rerun()
    with col_btn3:
        if st.button("⚙️ 重置默认"):
            reset_defaults()

# ---------- 数据加载 ----------
klines = fetch_klines_cached()
if klines:
    if not st.session_state.candles:
        st.session_state.candles.extend(klines)
    else:
        for k in klines:
            if k[0] > st.session_state.candles[-1][0]:
                for m in fill_missing(st.session_state.candles, k):
                    st.session_state.candles.append(m)

latest = fetch_latest()
if latest and (not st.session_state.candles or latest[0] > st.session_state.candles[-1][0]):
    for m in fill_missing(st.session_state.candles, latest):
        st.session_state.candles.append(m)

st.caption(f"最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | K线: {len(st.session_state.candles)}")

if len(st.session_state.candles) < 30:
    st.warning(f"⏳ 数据积累中... {len(st.session_state.candles)}/30")
else:
    df = pd.DataFrame(list(st.session_state.candles), columns=['ts','open','high','low','close','volume'])
    df['time'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('time', inplace=True)
    df['ema_fast'] = ema(df['close'], fast)
    df['ema_slow'] = ema(df['close'], slow)
    df['rsi'] = rsi(df['close'], rsi_period)
    df['atr'] = atr(df, 14)

    cp = df['close'].iloc[-1]

    # ---------- 趋势 ----------
    last = df.iloc[-1]
    ema_f, ema_s, price = last['ema_fast'], last['ema_slow'], last['close']
    if ema_f > ema_s and price > ema_f:
        trend = "🟢 强势多头"; trend_color = "#00ff9d"
    elif ema_f > ema_s:
        trend = "🟢 多头"; trend_color = "#00ff9d"
    elif ema_f < ema_s and price < ema_f:
        trend = "🔴 强势空头"; trend_color = "#ff4d88"
    elif ema_f < ema_s:
        trend = "🔴 空头"; trend_color = "#ff4d88"
    else:
        trend = "⚪ 震荡"; trend_color = "#aaa"
    st.markdown(f'<div class="trend-big" style="color:{trend_color};">{trend}</div>', unsafe_allow_html=True)
    st.markdown(f"快线 {ema_f:.2f} 慢线 {ema_s:.2f} 价格 {price:.2f}")

    # ---------- 信号检测 & 处理 ----------
    signal = detect_signal(df, fast, slow, (buy_min, buy_max), (sell_min, sell_max), use_score, score_thresh)

    for rec in st.session_state.history:
        if rec.get('result') != 'pending': continue
        if rec['side'] == 'BUY':
            if cp <= rec['sl']:
                rec['result'] = 'loss'; rec['exit_price'] = cp; rec['exit_reason'] = '止损'
            elif cp >= rec['tp2']:
                rec['result'] = 'win'; rec['exit_price'] = cp; rec['exit_reason'] = 'TP2'
        else:
            if cp >= rec['sl']:
                rec['result'] = 'loss'; rec['exit_price'] = cp; rec['exit_reason'] = '止损'
            elif cp <= rec['tp2']:
                rec['result'] = 'win'; rec['exit_price'] = cp; rec['exit_reason'] = 'TP2'

    if signal and st.session_state.last_signal_time != df.index[-1]:
        side, price, ef, es, r, a = signal
        sl, tp1, tp2 = sltp(price, side, a, use_atr_sl, sl_m if use_atr_sl else 2.2, tp1_m if use_atr_sl else 0.8, tp2_m if use_atr_sl else 1.6)
        rec = {'time': df.index[-1].strftime('%Y-%m-%d %H:%M'), 'side': side, 'price': price,
               'ema_fast': ef, 'ema_slow': es, 'rsi': r, 'atr': a,
               'sl': sl, 'tp1': tp1, 'tp2': tp2, 'result': 'pending', 'peak': price}
        st.session_state.history.appendleft(rec)
        st.session_state.last_signal_time = df.index[-1]

    # ---------- 信号卡片 / 等待卡片 ----------
    if st.session_state.history and st.session_state.history[0].get('result') == 'pending':
        r = st.session_state.history[0]
        cp = df['close'].iloc[-1]
        if r['side'] == 'BUY':
            pnl_pct = (cp - r['price']) / r['price'] * 100
            pnl_class = "positive" if pnl_pct >= 0 else "negative"
            pnl_sign = "+" if pnl_pct >= 0 else ""
            direction_emoji = "🟢 多头"
            risk_color = "#ff4d88"
            card_class = "signal-card buy-active"
        else:
            pnl_pct = (r['price'] - cp) / r['price'] * 100
            pnl_class = "positive" if pnl_pct >= 0 else "negative"
            pnl_sign = "+" if pnl_pct >= 0 else ""
            direction_emoji = "🔴 空头"
            risk_color = "#ff4d88"
            card_class = "signal-card sell-active"

        progress = max(0, min(1, (cp - r['price']) / (r['tp2'] - r['price']))) if r['side']=='BUY' else max(0, min(1, (r['price'] - cp) / (r['price'] - r['tp2'])))
        dist_sl = abs(cp - r['sl'])
        dist_sl_pct = dist_sl / r['price'] * 100

        title_class = "signal-title" + (" blink-title" if abs(pnl_pct)>3 else "")

        st.markdown(f"""
        <div class="{card_class}">
            <div class="{title_class}">{direction_emoji} 信号 @ {r['time']}</div>
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; text-align: center;">
                <div><div class="label">进场价格</div><div class="big-number">{r['price']:.2f}</div></div>
                <div><div class="label">目前盈亏</div><div class="big-number {pnl_class}">{pnl_sign}{pnl_pct:.2f}%</div></div>
                <div><div class="label">风险 / 距离止损</div><div class="big-number" style="color:{risk_color}">{dist_sl:.2f} ({dist_sl_pct:.2f}%)</div></div>
            </div>
            <div style="margin:22px 0">
                <div class="label">止损 / TP1 / TP2</div>
                <div style="display:flex;justify-content:space-between;font-size:1.12rem;margin-top:10px">
                    <span style="color:#ff4d88">SL {r['sl']:.2f}</span>
                    <span style="color:#4dff88">TP1 {r['tp1']:.2f}</span>
                    <span style="color:#ffd700">TP2 {r['tp2']:.2f}</span>
                </div>
                <div class="progress-container"><div class="progress-bar" style="width:{progress*100:.1f}%"></div></div>
            </div>
            <div style="display:flex;justify-content:space-between;font-size:1rem;color:#ccd6e0">
                <div>EMA快 {r['ema_fast']:.2f} | EMA慢 {r['ema_slow']:.2f}</div>
                <div>RSI {r['rsi']:.1f} | ATR {r['atr']:.2f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        side_guess = 'BUY' if last['ema_fast'] > last['ema_slow'] else 'SELL'
        total_score, _ = get_score(side_guess, df)
        title_color = trend_color if '多头' in trend else '#ff4d88' if '空头' in trend else '#aaa'
        high_score_hint = "✅ 评分已达标！等待EMA交叉与价格突破" if total_score >= score_thresh and use_score else ""
        st.markdown(f"""
        <div class="waiting-card">
            <h3 style="color:{title_color}; margin:0 0 18px 0; font-size:1.65rem;">等待下一个高质量信号...</h3>
            <div style="font-size:1.18rem; line-height:1.65;">
                <strong>当前趋势：</strong> {trend}<br>
                <strong>综合评分：</strong> {total_score}/100 （阈值 {score_thresh if use_score else '未启用'}）<br>
                <span style="color:#ffd700;font-weight:600;">{high_score_hint}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ---------- 图表 ----------
    plot_df = df.tail(200)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7,0.3])
    fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['open'], high=plot_df['high'], low=plot_df['low'], close=plot_df['close'], name='K线'), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['ema_fast'], name=f'EMA{fast}', line=dict(color='#ffd700')), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['ema_slow'], name=f'EMA{slow}', line=dict(color='#4da9ff')), row=1, col=1)
    colors = ['#26de81' if c>=o else '#fc5c65' for c,o in zip(plot_df['close'], plot_df['open'])]
    fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['volume'], marker_color=colors, showlegend=False), row=2, col=1)
    fig.add_hline(y=cp, line_dash="dash", line_color="#00ff9d", annotation_text=f"{cp:.2f}", annotation_position="top right", row=1, col=1)

    if st.session_state.history and st.session_state.history[0].get('result') == 'pending':
        r = st.session_state.history[0]
        fig.add_hline(y=r['price'], line_dash="dot", line_color="#ffffff", annotation_text="进场", annotation_position="top right", row=1, col=1)
        fig.add_hline(y=r['sl'], line_dash="dash", line_color="#ff4d88", annotation_text="SL", annotation_position="top right", annotation_font_color="#ff4d88", row=1, col=1)
        fig.add_hline(y=r['tp1'], line_dash="dash", line_color="#4dff88", annotation_text="TP1", annotation_position="top right", annotation_font_color="#4dff88", row=1, col=1)
        fig.add_hline(y=r['tp2'], line_dash="dash", line_color="#ffd700", annotation_text="TP2", annotation_position="top right", annotation_font_color="#ffd700", row=1, col=1)

    fig.update_xaxes(tickformat='%H:%M', tickangle=-45, nticks=10, tickfont_size=11, showgrid=True, gridcolor='rgba(80,80,80,0.3)', rangeslider_visible=False)
    fig.update_layout(height=720, template="plotly_dark", showlegend=False, xaxis_rangeslider_visible=False, margin=dict(l=40,r=40,t=40,b=110))
    st.plotly_chart(fig)

    # ---------- 最近K线 & 历史信号 ----------
    with st.expander("📊 最近10根K线（含指标）", expanded=False):
        show = df[['open','high','low','close','volume','ema_fast','ema_slow','rsi','atr']].tail(10).round(2)
        show.index = show.index.strftime('%Y-%m-%d %H:%M')
        show.columns = ['开盘','最高','最低','收盘','成交量','EMA快线','EMA慢线','RSI','ATR']
        st.dataframe(show)

    if st.session_state.history:
        st.subheader("📜 最近信号")
        hist = pd.DataFrame(list(st.session_state.history)[:10])
        if not hist.empty:
            for col in ['exit_price', 'exit_reason']:
                if col not in hist.columns:
                    hist[col] = None
        hist_display = hist[['time','side','price','result','exit_price','exit_reason']].copy()
        hist_display.columns = ['信号时间','方向','进场价','结果','出场价','出场原因']

        def calc_pnl(row):
            if row['结果'] == 'pending': return '待定'
            if pd.isna(row['出场价']): return '-'
            diff = row['出场价'] - row['进场价']
            pnl = (diff / row['进场价']) * 100 if row['方向'] == 'BUY' else (-diff / row['进场价']) * 100
            return f"{pnl:+.2f}%"
        hist_display['盈亏%'] = hist_display.apply(calc_pnl, axis=1)

        def style_result(val):
            if val == 'win': return 'background-color: #1a3c34; color: #00ff9d'
            if val == 'loss': return 'background-color: #3c1a1a; color: #ff4d4d'
            if val == 'pending': return 'background-color: #1e3a5f; color: #ffd700'
            return ''
        styled = hist_display.style.map(style_result, subset=['结果'])
        st.dataframe(styled)

        csv = hist_display.to_csv(index=False).encode('utf-8')
        st.download_button("📥 导出历史信号 (CSV)", csv, f"signals_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv")
    else:
        st.info("暂无历史信号")

st.markdown("---")
st.caption("🔥 顶级完美终极版 v5.1 • 零警告 • 双数据源 • 顶级动效 • 极致稳定 • 祝你交易大赚特赚！💰🚀")
