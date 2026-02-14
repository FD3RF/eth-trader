import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
import time
from datetime import datetime, timedelta
import os

st.set_page_config(page_title="全中文智能交易监控中心", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    .metric-card { background: #1E1F2A; border-radius: 8px; padding: 16px; border-left: 4px solid #00D4FF; }
    .signal-box { background: #1E1F2A; border-radius: 10px; padding: 20px; border: 1px solid #333A44; }
    .snapshot-item { background: #262730; padding: 8px 12px; border-radius: 6px; margin: 4px 0; font-size: 14px; }
</style>
""", unsafe_allow_html=True)

# 读取密钥
def get_secret(key):
    try:
        return st.secrets[key]
    except:
        return os.getenv(key)

BINANCE_API_KEY = get_secret("BINANCE_API_KEY")
AINFT_KEY = get_secret("AINFT_KEY")
if not AINFT_KEY:
    st.error("❌ 未找到 AINFT_KEY，请在 secrets 中配置")
    st.stop()

# ---------- 备用数据源：CoinGecko ----------
def fetch_coingecko_price():
    """从CoinGecko获取ETH/USDT实时价格（仅当前价格，用于回退）"""
    url = "https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data['ethereum']['usd']
    except:
        return None

# ---------- 币安K线获取（带备用）----------
@st.cache_data(ttl=60)
def fetch_klines(symbol="ETHUSDT", interval="5m", limit=200):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        headers = {"X-MBX-APIKEY": BINANCE_API_KEY} if BINANCE_API_KEY else {}
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        if resp.status_code == 451:
            # 币安被屏蔽，尝试CoinGecko获取当前价格，并生成模拟K线
            st.warning(f"⚠️ 币安API被屏蔽，使用CoinGecko备用数据")
            price = fetch_coingecko_price()
            if price:
                return generate_fallback_klines(price, interval, limit)
            else:
                return pd.DataFrame()
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        st.warning(f"获取K线失败，使用备用数据: {e}")
        price = fetch_coingecko_price()
        if price:
            return generate_fallback_klines(price, interval, limit)
        else:
            return pd.DataFrame()

    df = pd.DataFrame(data, columns=[
        "time","open","high","low","close","volume",
        "close_time","quote_volume","trades",
        "taker_buy_base","taker_buy_quote","ignore"
    ])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)
    return df

def generate_fallback_klines(current_price, interval, limit):
    """生成模拟K线（基于当前价格和随机波动）"""
    now = datetime.now()
    times = [now - timedelta(minutes=i*int(interval.replace('m','').replace('h','60').replace('d','1440'))) for i in range(limit)]
    times.reverse()
    closes = [current_price * (1 + np.random.normal(0, 0.001)) for _ in range(limit)]
    opens = [closes[i-1] if i>0 else closes[0]*0.999 for i in range(limit)]
    highs = [max(opens[i], closes[i]) * 1.001 for i in range(limit)]
    lows = [min(opens[i], closes[i]) * 0.999 for i in range(limit)]
    volumes = [np.random.uniform(100,500) for _ in range(limit)]
    df = pd.DataFrame({
        "time": times,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes
    })
    return df

def add_indicators(df):
    df = df.copy()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma60"] = df["close"].rolling(60).mean()
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    return df

@st.cache_data(ttl=60)
def fetch_all_periods():
    periods = ["1m","5m","15m","1h","4h","1d"]
    data = {}
    for p in periods:
        try:
            df = fetch_klines(interval=p, limit=200)
            if not df.empty:
                df = add_indicators(df)
                data[p] = df
            else:
                data[p] = pd.DataFrame()
        except Exception as e:
            st.warning(f"获取 {p} 数据失败: {e}")
            data[p] = pd.DataFrame()
    return data

def get_ai_signal(eth_df, btc_df=None):
    if eth_df.empty:
        return "数据不足", 0, ""
    e = eth_df.iloc[-1]
    time_str = e["time"].strftime("%Y-%m-%d %H:%M")
    btc_info = ""
    if btc_df is not None and not btc_df.empty:
        b = btc_df.iloc[-1]
        btc_info = f"【BTC参考】价格: {b['close']:.2f} RSI: {b['rsi']:.1f}"
    prompt = f"""
【ETH实时数据】时间:{time_str} 价格:{e['close']:.2f} MA20:{e['ma20']:.2f} MA60:{e['ma60']:.2f} RSI:{e['rsi']:.1f}
{btc_info}
请输出：方向（做多/做空/观望） 置信度（0-100） 理由（一句话）
"""
    url = "https://chat.ainft.com/webapi/chat/openai"
    headers = {"Authorization": f"Bearer {AINFT_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "gpt-5.2",
        "temperature": 0.3,
        "messages": [{"role": "system", "content": "你是专业加密货币交易员，输出简洁。"}, {"role": "user", "content": prompt}]
    }
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=15)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        if "做多" in content: direction = "做多"
        elif "做空" in content: direction = "做空"
        else: direction = "观望"
        import re
        conf = re.search(r'置信度[：:]\s*(\d+)', content)
        confidence = int(conf.group(1)) if conf else 50
        return direction, confidence, content
    except Exception as e:
        return "API错误", 0, str(e)

if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = datetime.now()

with st.sidebar:
    st.title("⚙️ 控制面板")
    interval = st.selectbox("选择K线周期", ["1m","5m","15m","1h","4h","1d"], index=1)
    auto_refresh = st.checkbox("自动刷新 (60秒)", value=True)
    use_simulated = st.checkbox("强制使用模拟数据", value=False)
    st.divider()
    st.subheader("📈 模拟交易")
    entry_price = st.number_input("入场价", 0.0, step=0.01)
    stop_price = st.number_input("止损价", 0.0, step=0.01)
    qty = st.number_input("数量(ETH)", 0.001, value=0.01, step=0.001, format="%.3f")
    if st.button("🚀 刷新数据", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.title("📊 全中文智能交易监控中心 · 备用数据版")
st.caption(f"数据更新: {st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')} | 当币安被屏蔽时自动使用CoinGecko价格")

if use_simulated:
    # 强制模拟
    dates = pd.date_range(end=datetime.now(), periods=200, freq='5min')
    sim_df = pd.DataFrame({"time": dates, "close": np.random.normal(2600,20,200).cumsum()+1800})
    sim_df["high"] = sim_df["close"]*1.002; sim_df["low"] = sim_df["close"]*0.998; sim_df["open"] = sim_df["close"].shift(1).fillna(sim_df["close"].iloc[0]); sim_df["volume"] = np.random.uniform(100,500,200)
    sim_df = add_indicators(sim_df)
    data_dict = {interval: sim_df}
else:
    data_dict = fetch_all_periods()
    if interval not in data_dict or data_dict[interval].empty:
        st.error(f"周期 {interval} 数据获取失败，请稍后重试")
        st.stop()

df = data_dict[interval]
latest = df.iloc[-1]
prev = df.iloc[-2] if len(df)>1 else latest

col1, col2, col3, col4, col5 = st.columns(5)
with col1: st.metric("ETH/USDT", f"${latest['close']:.2f}", f"{latest['close']-prev['close']:+.2f}")
with col2: st.metric("RSI(14)", f"{latest['rsi']:.1f}")
with col3: st.metric("MA20", f"${latest['ma20']:.2f}")
with col4: st.metric("MA60", f"${latest['ma60']:.2f}")
with col5: st.metric("成交量", f"{latest['volume']:.0f}")

st.subheader(f"{interval} K线图")
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7,0.3])
fig.add_trace(go.Candlestick(x=df["time"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="K线"), row=1, col=1)
fig.add_trace(go.Scatter(x=df["time"], y=df["ma20"], name="MA20", line=dict(color="orange")), row=1, col=1)
fig.add_trace(go.Scatter(x=df["time"], y=df["ma60"], name="MA60", line=dict(color="blue")), row=1, col=1)
fig.add_trace(go.Scatter(x=df["time"], y=df["rsi"], name="RSI", line=dict(color="purple")), row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red", row=2); fig.add_hline(y=30, line_dash="dash", line_color="green", row=2)
fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=500)
st.plotly_chart(fig, use_container_width=True)

col_left, col_right = st.columns([1,1])
with col_left:
    st.subheader("🎯 AI 信号")
    btc_df = data_dict.get("15m") if "15m" in data_dict else None
    direction, conf, content = get_ai_signal(df, btc_df)
    color = "#26A69A" if direction=="做多" else "#EF5350" if direction=="做空" else "#888"
    st.markdown(f'<div class="signal-box"><span style="font-size:24px;color:{color};">{"🟢" if direction=="做多" else "🔴" if direction=="做空" else "⚪"} {direction}</span><br>置信度:{conf}%<br>{content}</div>', unsafe_allow_html=True)

with col_right:
    st.subheader("💰 模拟盈亏")
    if entry_price>0:
        cur = latest["close"]
        if direction=="做多": profit = (cur-entry_price)*qty
        else: profit = (entry_price-cur)*qty
        color = "#26A69A" if profit>=0 else "#EF5350"
        st.markdown(f'<div style="background:#1E1F2A;padding:20px;border-radius:10px;"><span style="font-size:20px;">当前盈亏</span><br><span style="font-size:32px;color:{color};">{profit:+.2f} USDT</span></div>', unsafe_allow_html=True)
    else:
        st.info("输入入场价以计算盈亏")

st.subheader("📌 各周期快照")
cols = st.columns(3)
periods = ["1m","5m","15m","1h","4h","1d"]
for i,p in enumerate(periods):
    with cols[i%3]:
        if p in data_dict and not data_dict[p].empty and len(data_dict[p])>1:
            d = data_dict[p].iloc[-1]; d2 = data_dict[p].iloc[-2]
            arrow = "↑" if d["close"]>d2["close"] else "↓"
            color = "#26A69A" if arrow=="↑" else "#EF5350"
            st.markdown(f'<div class="snapshot-item"><span>{p}</span><span style="color:{color};margin-left:8px;">{arrow}</span><span style="float:right;">RSI {d["rsi"]:.1f} ${d["close"]:.2f}</span></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="snapshot-item">{p}: 数据获取中...</div>', unsafe_allow_html=True)

if auto_refresh and not use_simulated:
    if (datetime.now()-st.session_state.last_refresh).seconds>60:
        st.cache_data.clear()
        st.session_state.last_refresh = datetime.now()
        st.rerun()

st.caption("⚠️ 数据仅供参考，不构成投资建议。币安被屏蔽时自动使用CoinGecko价格，可能有延迟。")
