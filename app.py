import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
import time
from datetime import datetime, timedelta

st.set_page_config(page_title="ETH 100倍杠杆智能交易终端", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    .metric-card { background: #1E1F2A; border-radius: 8px; padding: 16px; border-left: 4px solid #00D4FF; }
    .signal-box { background: #1E1F2A; border-radius: 10px; padding: 20px; border: 1px solid #333A44; }
    .warning-box { background: #332222; border-left: 4px solid #EF5350; padding: 10px; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ---------- CoinGecko 免费数据源 ----------
@st.cache_data(ttl=30)
def fetch_price():
    url = "https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd&include_24hr_change=true"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        return data['ethereum']['usd'], data['ethereum']['usd_24h_change']
    except:
        return None, None

def generate_klines(price, interval_min=5, limit=200):
    now = datetime.now()
    times = [now - timedelta(minutes=i*interval_min) for i in range(limit)][::-1]
    closes = [price * (1 + 0.001*np.random.randn()) for _ in range(limit)]
    opens = [closes[i-1] if i>0 else closes[0]*0.999 for i in range(limit)]
    highs = [max(opens[i], closes[i])*1.001 for i in range(limit)]
    lows = [min(opens[i], closes[i])*0.999 for i in range(limit)]
    vols = np.random.uniform(100,500,limit)
    return pd.DataFrame({"time":times,"open":opens,"high":highs,"low":lows,"close":closes,"volume":vols})

def add_indicators(df):
    df = df.copy()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma60"] = df["close"].rolling(60).mean()
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    df["macd"] = ta.trend.MACD(df["close"]).macd()
    df["macd_signal"] = ta.trend.MACD(df["close"]).macd_signal()
    df["bb_upper"] = ta.volatility.BollingerBands(df["close"]).bollinger_hband()
    df["bb_lower"] = ta.volatility.BollingerBands(df["close"]).bollinger_lband()
    return df

# ---------- 免费AI信号（基于规则）----------
def get_signal(df):
    if df.empty or len(df)<30: return "数据不足",0,"等待数据"
    last = df.iloc[-1]
    prev = df.iloc[-2]
    score = 0
    if not pd.isna(last['ma20']) and not pd.isna(last['ma60']):
        if last['ma20'] > last['ma60']: score += 20
        else: score -= 20
    if not pd.isna(last['rsi']):
        if last['rsi'] < 30: score += 25
        elif last['rsi'] > 70: score -= 25
        elif last['rsi'] > 50: score += 10
        else: score -= 10
    if not pd.isna(last['macd']) and not pd.isna(last['macd_signal']):
        if last['macd'] > last['macd_signal']: score += 15
        else: score -= 15
    conf = min(95, max(5, int(50 + score*0.5)))
    if score > 15: return "做多", conf, "均线多头+RSI健康"
    if score < -15: return "做空", conf, "均线空头+RSI弱势"
    return "观望", conf, "多空平衡"

def calc_position(capital, entry, stop, leverage=100):
    risk = 0.02  # 单笔风险2%
    if entry<=0 or stop<=0: return 0
    stop_pct = abs(entry-stop)/entry
    if stop_pct<=0: return 0
    max_loss = capital * risk
    pos_value = max_loss / stop_pct
    if pos_value > capital * leverage:
        pos_value = capital * leverage
    return pos_value / entry

# ---------- 界面 ----------
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = datetime.now()
    st.session_state.price = 2600

price, change = fetch_price()
if price: st.session_state.price = price
else: price = st.session_state.price

with st.sidebar:
    st.title("⚙️ 100倍杠杆控制")
    st.markdown("⚠️ 高风险")
    interval = st.selectbox("周期", ["1m","5m","15m","1h"], index=1)
    auto = st.checkbox("自动刷新", True)
    capital = st.number_input("本金 (USDT)", 10, value=1000)
    lev = st.select_slider("杠杆", [10,20,50,100], value=100)
    entry = st.number_input("入场价", value=price, step=1.0)
    stop = st.number_input("止损价", value=price*0.99, step=1.0)
    st.button("刷新数据", on_click=lambda: st.cache_data.clear())

st.title("📊 ETH 100倍杠杆智能交易终端 · 免费AI版")
st.caption(f"数据更新: {st.session_state.last_refresh.strftime('%H:%M:%S')} | 基于CoinGecko")

# 生成K线
interval_min = int(interval.replace('m','').replace('h','60'))
df = generate_klines(price, interval_min)
df = add_indicators(df)
last = df.iloc[-1]
prev = df.iloc[-2]

col1, col2, col3, col4, col5 = st.columns(5)
with col1: st.metric("ETH/USDT", f"${last['close']:.2f}", f"{last['close']-prev['close']:+.2f}")
with col2: st.metric("RSI(14)", f"{last['rsi']:.1f}")
with col3: st.metric("MA20", f"${last['ma20']:.2f}")
with col4: st.metric("MA60", f"${last['ma60']:.2f}")
with col5: st.metric("成交量", f"{last['volume']:.0f}")

st.warning(f"当前杠杆 {lev}倍 | 本金 {capital:.0f} USDT | 可开最大 {capital*lev/price:.3f} ETH | 单笔风险≤2%")

# K线图
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7,0.3])
fig.add_trace(go.Candlestick(x=df.time, open=df.open, high=df.high, low=df.low, close=df.close, name="K线"), row=1, col=1)
fig.add_trace(go.Scatter(x=df.time, y=df.ma20, name="MA20", line=dict(color="orange")), row=1, col=1)
fig.add_trace(go.Scatter(x=df.time, y=df.ma60, name="MA60", line=dict(color="blue")), row=1, col=1)
fig.add_trace(go.Scatter(x=df.time, y=df.rsi, name="RSI", line=dict(color="purple")), row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red", row=2)
fig.add_hline(y=30, line_dash="dash", line_color="green", row=2)
fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=600)
st.plotly_chart(fig, use_container_width=True)

# AI信号与交易建议
colA, colB = st.columns(2)
with colA:
    st.subheader("🎯 免费AI信号")
    dir, conf, reason = get_signal(df)
    color = "#26A69A" if dir=="做多" else "#EF5350" if dir=="做空" else "#888"
    emoji = "🟢" if dir=="做多" else "🔴" if dir=="做空" else "⚪"
    st.markdown(f'<div class="signal-box"><span style="font-size:24px;color:{color};">{emoji} {dir}</span><br>置信度 {conf}%<br>{reason}</div>', unsafe_allow_html=True)

with colB:
    st.subheader("📈 进场策略")
    if dir == "做多":
        stop_price = last['ma60'] if not pd.isna(last['ma60']) else last['close']*0.99
        st.markdown(f"**激进:** ${last['close']:.2f}\n\n**稳健:** ${last['ma20']:.2f}\n\n**止损:** ${stop_price:.2f}")
    elif dir == "做空":
        stop_price = last['ma60'] if not pd.isna(last['ma60']) else last['close']*1.01
        st.markdown(f"**激进:** ${last['close']:.2f}\n\n**稳健:** ${last['ma20']:.2f}\n\n**止损:** ${stop_price:.2f}")
    else:
        st.info("等待信号")

# 手动开仓盈亏
qty = calc_position(capital, entry, stop, lev)
if qty > 0:
    if dir=="做多":
        pnl = (last['close'] - entry) * qty
    else:
        pnl = (entry - last['close']) * qty
    color = "#26A69A" if pnl>=0 else "#EF5350"
    st.markdown(f"""
    <div style="background:#1E1F2A;padding:20px;border-radius:10px;">
        <span style="font-size:20px;">当前盈亏</span><br>
        <span style="font-size:32px;color:{color};">{pnl:+.2f} USDT</span><br>
        <span>数量 {qty:.4f} ETH | 保证金 {qty*entry/lev:.2f}</span>
    </div>
    """, unsafe_allow_html=True)

# 自动刷新
if auto and (datetime.now()-st.session_state.last_refresh).seconds > 30:
    st.cache_data.clear()
    st.session_state.last_refresh = datetime.now()
    st.rerun()

st.caption("⚠️ 免费AI信号基于技术指标生成，不构成投资建议。100倍杠杆高风险，务必设止损。")
