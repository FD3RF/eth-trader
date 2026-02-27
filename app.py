import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime
from collections import deque
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh

# ---------- 配置 ----------
SYMBOL = "ETH"
CURRENCY = "USD"
INTERVAL = "5"                     # 5分钟K线
LIMIT = 200
INTERVAL_MINUTES = 5

# ---------- Session ----------
if 'history' not in st.session_state:
    st.session_state.history = deque(maxlen=200)  # 仅内存记录
if 'candles' not in st.session_state:
    st.session_state.candles = deque(maxlen=500)
if 'last_signal_time' not in st.session_state:
    st.session_state.last_signal_time = None
if 'api_fail' not in st.session_state:
    st.session_state.api_fail = 0

# ---------- 数据获取 ----------
def fetch_klines():
    try:
        url = f"https://min-api.cryptocompare.com/data/v2/histominute?fsym={SYMBOL}&tsym={CURRENCY}&limit={LIMIT}&aggregate={INTERVAL}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()['Data']['Data']
        return [[x['time']*1000, x['open'], x['high'], x['low'], x['close'], x['volumefrom']] for x in data]
    except:
        st.session_state.api_fail += 1
        return []

def fetch_latest():
    try:
        url = f"https://min-api.cryptocompare.com/data/v2/histominute?fsym={SYMBOL}&tsym={CURRENCY}&limit=1&aggregate={INTERVAL}"
        resp = requests.get(url, timeout=5)
        return [resp.json()['Data']['Data'][0][k] for k in ['time','open','high','low','close','volumefrom']]
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

# ---------- 评分函数（简化版） ----------
def get_score(side, df):
    last = df.iloc[-1]
    c, ef, es, r, a, v = last['close'], last['ema_fast'], last['ema_slow'], last['rsi'], last['atr'], last['volume']
    avg_v = df['volume'].rolling(20).mean().iloc[-1] if len(df)>=20 else v
    vr = v/avg_v if avg_v>0 else 1
    spread = abs(ef-es)/c*100
    slope = abs((ef - df['ema_fast'].iloc[-5])/5/c*100) if len(df)>=5 else 0
    atr_pct = a/c*100 if a>0 else 0

    # 各项得分 (0-20)
    s_ema = 20 if spread>=0.2 else 15 if spread>=0.15 else 10 if spread>=0.1 else 5 if spread>=0.05 else 0
    s_slope = 20 if slope>=0.04 else 15 if slope>=0.03 else 10 if slope>=0.02 else 5 if slope>=0.01 else 0
    s_vol = 20 if vr>=1.3 else 10 if vr>=0.8 else 0
    s_atr = 20 if atr_pct>=0.2 else 15 if atr_pct>=0.16 else 10 if atr_pct>=0.12 else 5 if atr_pct>=0.08 else 0
    if side=='BUY':
        s_rsi = 20 if 55<=r<=70 else 15 if 50<=r<55 or 70<r<=75 else 10 if 45<=r<50 or 75<r<=80 else 5 if 40<=r<45 or 80<r<=85 else 0
    else:
        s_rsi = 20 if 30<=r<=45 else 15 if 25<=r<30 or 45<r<=50 else 10 if 20<=r<25 or 50<r<=55 else 5 if 15<=r<20 or 55<r<=60 else 0
    return s_ema + s_slope + s_vol + s_atr + s_rsi

# ---------- 信号检测 ----------
def detect_signal(df, fast, slow, buy_range, sell_range, use_score, score_thresh):
    if len(df) < 50: return None
    last = df.iloc[-1]
    ef, es, r = last['ema_fast'], last['ema_slow'], last['rsi']
    is_bull = ef > es and last['close'] > ef*0.999
    is_bear = ef < es and last['close'] < ef*1.001
    if not (is_bull or is_bear): return None

    # 价格必须突破前一根收盘
    if is_bull and last['close'] <= df['close'].iloc[-2]: return None
    if is_bear and last['close'] >= df['close'].iloc[-2]: return None

    # 评分或RSI范围
    if use_score:
        score = get_score('BUY' if is_bull else 'SELL', df)
        if score < score_thresh: return None
    else:
        if is_bull and not (buy_range[0] < r < buy_range[1]): return None
        if is_bear and not (sell_range[0] < r < sell_range[1]): return None
    return ('BUY' if is_bull else 'SELL', last['close'], ef, es, r, last['atr'])

# ---------- 止损止盈 ----------
def sltp(price, side, atr, use_atr, mult_sl=2.2, mult_tp1=0.8, mult_tp2=1.6):
    if use_atr and atr>0:
        risk = atr * mult_sl
        if side=='BUY':
            return price-risk, price+risk*mult_tp1, price+risk*mult_tp2
        else:
            return price+risk, price-risk*mult_tp1, price-risk*mult_tp2
    else:
        if side=='BUY':
            return price*0.994, price*1.006, price*1.012
        else:
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

# ---------- UI ----------
st.set_page_config(page_title="ETH 5分钟简化版", layout="wide")
st.markdown("""
<style>
    .stApp {background:#0f172a;}
    .signal-card {background:linear-gradient(135deg,#0a3d2a,#112233); border-radius:18px; padding:20px; border:2px solid #00ff9d;}
    .waiting-card {background:#1e3a5f; padding:30px; border-radius:18px; text-align:center;}
</style>
""", unsafe_allow_html=True)

st.title("📈 ETH 5分钟 EMA 策略 (简化版)")

# 侧边栏
with st.sidebar:
    st.header("参数")
    fast = st.number_input("快线EMA", 1, 50, 8)
    slow = st.number_input("慢线EMA", 2, 100, 21)
    rsi_period = st.number_input("RSI周期", 2, 50, 14)
    col1, col2 = st.columns(2)
    with col1:
        st.caption("多头RSI")
        buy_min = st.number_input("多头下限", 0, 100, 57, key='bmin')
        buy_max = st.number_input("多头上限", 0, 100, 70, key='bmax')
    with col2:
        st.caption("空头RSI")
        sell_min = st.number_input("空头下限", 0, 100, 30, key='smin')
        sell_max = st.number_input("空头上限", 0, 100, 43, key='smax')
    refresh = st.number_input("刷新秒数", 5, 300, 30)
    st.caption(f"下次刷新: {refresh}秒后")

    use_score = st.checkbox("启用评分系统", True)
    score_thresh = st.slider("评分阈值", 0, 100, 70, disabled=not use_score)
    use_atr_sl = st.checkbox("ATR动态止损", True)
    if use_atr_sl:
        sl_m = st.slider("止损倍数", 1.0, 5.0, 2.2, 0.1)
        tp1_m = st.slider("TP1倍数", 0.2, 3.0, 0.8, 0.1)
        tp2_m = st.slider("TP2倍数", 0.5, 5.0, 1.6, 0.1)

    st.metric("API失败", st.session_state.api_fail)
    if st.button("清空历史"):
        st.session_state.history.clear()

# 数据获取
klines = fetch_klines()
if klines:
    if not st.session_state.candles:
        st.session_state.candles.extend(klines)
    else:
        for k in klines:
            if k[0] > st.session_state.candles[-1][0]:
                for m in fill_missing(st.session_state.candles, k):
                    st.session_state.candles.append(m)
latest = fetch_latest()
if latest:
    if not st.session_state.candles or latest[0] > st.session_state.candles[-1][0]:
        for m in fill_missing(st.session_state.candles, latest):
            st.session_state.candles.append(m)

st_autorefresh(interval=refresh*1000, key='auto')
st.caption(f"最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | K线: {len(st.session_state.candles)}")

if len(st.session_state.candles) < 30:
    st.warning(f"数据积累中... {len(st.session_state.candles)}/30")
else:
    # 构建DataFrame
    df = pd.DataFrame(list(st.session_state.candles), columns=['ts','open','high','low','close','volume'])
    df['time'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('time', inplace=True)
    df['ema_fast'] = ema(df['close'], fast)
    df['ema_slow'] = ema(df['close'], slow)
    df['rsi'] = rsi(df['close'], rsi_period)
    df['atr'] = atr(df, 14)

    cp = df['close'].iloc[-1]
    signal = detect_signal(df, fast, slow, (buy_min,buy_max), (sell_min,sell_max), use_score, score_thresh)

    # 检查历史pending信号的止损止盈
    for i, rec in enumerate(st.session_state.history):
        if rec['result'] != 'pending': continue
        if rec['side'] == 'BUY':
            if cp <= rec['sl']:
                rec['result'] = 'loss'
                rec['exit_price'] = cp
                rec['exit_reason'] = '止损'
            elif cp >= rec['tp2']:
                rec['result'] = 'win'
                rec['exit_price'] = cp
                rec['exit_reason'] = 'TP2'
        else:
            if cp >= rec['sl']:
                rec['result'] = 'loss'
                rec['exit_price'] = cp
                rec['exit_reason'] = '止损'
            elif cp <= rec['tp2']:
                rec['result'] = 'win'
                rec['exit_price'] = cp
                rec['exit_reason'] = 'TP2'

    # 新信号处理
    if signal and st.session_state.last_signal_time != df.index[-1]:
        side, price, ef, es, r, a = signal
        sl, tp1, tp2 = sltp(price, side, a, use_atr_sl,
                            sl_m if use_atr_sl else 2.2,
                            tp1_m if use_atr_sl else 0.8,
                            tp2_m if use_atr_sl else 1.6)
        rec = {
            'time': df.index[-1].strftime('%Y-%m-%d %H:%M'),
            'side': side, 'price': price, 'ema_fast': ef, 'ema_slow': es,
            'rsi': r, 'atr': a, 'sl': sl, 'tp1': tp1, 'tp2': tp2,
            'result': 'pending', 'peak': price
        }
        st.session_state.history.appendleft(rec)
        st.session_state.last_signal_time = df.index[-1]

    # 显示信号卡片
    if st.session_state.history and st.session_state.history[0]['result'] == 'pending':
        r = st.session_state.history[0]
        risk = abs(r['price']-r['sl'])
        st.markdown(f"""
        <div class="signal-card">
            <h3 style="color:#00ff9d;">{'🟢 多头' if r['side']=='BUY' else '🔴 空头'} @ {r['time']}</h3>
            <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;">
                <div>进场<br><b>{r['price']:.2f}</b></div>
                <div>止损<br><b style="color:#ff99cc;">{r['sl']:.2f}</b><br>风险 {risk:.2f}</div>
                <div>TP1<br><b>{r['tp1']:.2f}</b></div>
                <div>TP2<br><b style="color:#ffd700;">{r['tp2']:.2f}</b></div>
            </div>
            <div style="display:flex;gap:10px;margin-top:10px;">
                <div>EMA快 {r['ema_fast']:.2f}</div>
                <div>EMA慢 {r['ema_slow']:.2f}</div>
                <div>RSI {r['rsi']:.1f}</div>
                <div>ATR {r['atr']:.2f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="waiting-card">⏳ 等待新信号...</div>', unsafe_allow_html=True)

    # 图表
    plot_df = df.tail(200)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7,0.3])
    fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['open'], high=plot_df['high'],
                                 low=plot_df['low'], close=plot_df['close'], name='K线'), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['ema_fast'], name=f'EMA{fast}', line=dict(color='#ffd700')), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['ema_slow'], name=f'EMA{slow}', line=dict(color='#4da9ff')), row=1, col=1)
    colors = ['#00ff9d' if c>=o else '#ff4d4d' for c,o in zip(plot_df['close'], plot_df['open'])]
    fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['volume'], marker_color=colors, showlegend=False), row=2, col=1)
    fig.add_hline(y=cp, line_dash="dash", line_color="#00ff9d", annotation_text=f"{cp:.2f}", row=1, col=1)
    fig.update_layout(height=600, template="plotly_dark", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # 市场诊断
    with st.expander("🔍 当前市场诊断", expanded=True):
        last = df.iloc[-1]
        spread = abs(last['ema_fast']-last['ema_slow'])/last['close']*100
        slope = (last['ema_fast']-df['ema_fast'].iloc[-5])/5/last['close'] if len(df)>=5 else 0
        range_pct = (df['high'].iloc[-20:].max()-df['low'].iloc[-20:].min())/last['close']*100
        vol_ratio = last['volume']/df['volume'].rolling(20).mean().iloc[-1] if len(df)>=20 else 0
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**EMA扩散** {spread:.3f}% {'✅' if spread>=0.1 else '❌'}")
            st.markdown(f"**EMA斜率** {slope:.5f} {'✅' if abs(slope)>=0.0002 else '❌'}")
        with col2:
            st.markdown(f"**20根波动** {range_pct:.2f}% {'✅' if range_pct>=0.3 else '❌'}")
            st.markdown(f"**成交量比** {vol_ratio:.2f}x {'✅' if vol_ratio>1.3 else '⏳'}")

    # 最近K线
    st.subheader("最近10根K线")
    show = df[['open','high','low','close','volume','ema_fast','ema_slow','rsi','atr']].tail(10).round(2)
    show.index = show.index.strftime('%Y-%m-%d %H:%M')
    st.dataframe(show, use_container_width=True)

    # 历史记录（仅显示最近5条）
    if st.session_state.history:
        st.subheader("📜 最近信号")
        hist = pd.DataFrame(list(st.session_state.history)[:5])
        st.dataframe(hist[['time','side','price','result','exit_price','exit_reason']], use_container_width=True)

st.markdown("---")
st.caption("🔥 简化版 v1.0 • 保留核心策略 • 祝你交易顺利！")
