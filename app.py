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

# 页面配置
st.set_page_config(
    page_title="全中文智能交易监控中心",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义样式
st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    .metric-card {
        background: #1E1F2A;
        border-radius: 8px;
        padding: 16px;
        border-left: 4px solid #00D4FF;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .signal-box {
        background: #1E1F2A;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #333A44;
    }
    .snapshot-item {
        background: #262730;
        padding: 8px 12px;
        border-radius: 6px;
        margin: 4px 0;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# 从secrets读取密钥
def get_secret(key):
    try:
        return st.secrets[key]
    except:
        return os.getenv(key)

BINANCE_API_KEY = get_secret("BINANCE_API_KEY")
AINFT_KEY = get_secret("AINFT_KEY")
if not AINFT_KEY:
    st.error("❌ 未找到 AINFT_KEY，请在 secrets 或环境变量中配置")
    st.stop()

# ---------- 数据获取函数 ----------
@st.cache_data(ttl=60)
def fetch_klines(symbol="ETHUSDT", interval="5m", limit=200):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        headers = {"X-MBX-APIKEY": BINANCE_API_KEY} if BINANCE_API_KEY else {}
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        st.error(f"获取K线失败: {e}")
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

# ---------- 调用AINFT获取AI信号 ----------
def get_ai_signal(eth_df, btc_df=None):
    if eth_df.empty:
        return "数据不足", 0, ""
    e = eth_df.iloc[-1]
    time_str = e["time"].strftime("%Y-%m-%d %H:%M")

    btc_info = ""
    if btc_df is not None and not btc_df.empty:
        b = btc_df.iloc[-1]
        btc_info = f"""
【BTC 15分钟参考】
价格: {b['close']:.2f} USDT
RSI: {b['rsi']:.1f}
MA20: {b['ma20']:.2f} | MA60: {b['ma60']:.2f}
"""

    prompt = f"""
【ETH {eth_df['interval'] if 'interval' in eth_df.columns else '当前周期'}实时数据】
时间: {time_str}
价格: {e['close']:.2f} USDT
MA20: {e['ma20']:.2f} | MA60: {e['ma60']:.2f}
RSI: {e['rsi']:.1f}
成交量: {e['volume']:.2f}

{btc_info}

请输出简洁中文交易信号，格式如下：
方向：[做多/做空/观望]
置信度：[0-100的整数]
理由：[一句话]
"""

    url = "https://chat.ainft.com/webapi/chat/openai"   # ✅ 修正为正确域名
    headers = {
        "Authorization": f"Bearer {AINFT_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-5.2",
        "temperature": 0.3,
        "messages": [
            {"role": "system", "content": "你是专业加密货币交易员，输出必须简洁，只包含方向、置信度、理由。"},
            {"role": "user", "content": prompt}
        ]
    }
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=15)
        resp.raise_for_status()
        result = resp.json()
        content = result["choices"][0]["message"]["content"]
        if "做多" in content:
            direction = "做多"
        elif "做空" in content:
            direction = "做空"
        else:
            direction = "观望"
        import re
        conf_match = re.search(r'置信度[：:]\s*(\d+)', content)
        confidence = int(conf_match.group(1)) if conf_match else 50
        return direction, confidence, content
    except Exception as e:
        return "API错误", 0, str(e)

# ---------- 初始化session state ----------
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = datetime.now()
if "data" not in st.session_state:
    st.session_state.data = {}

# ---------- 侧边栏控制 ----------
with st.sidebar:
    st.title("⚙️ 控制面板")
    interval = st.selectbox(
        "选择K线周期",
        ["1m","5m","15m","1h","4h","1d"],
        index=1
    )
    auto_refresh = st.checkbox("自动刷新 (60秒)", value=True)
    use_simulated = st.checkbox("使用模拟数据（调试用）", value=False)
    st.divider()
    st.subheader("📈 模拟交易")
    col1, col2 = st.columns(2)
    with col1:
        entry_price = st.number_input("入场价 (USDT)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
    with col2:
        stop_price = st.number_input("止损价 (USDT)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
    qty = st.number_input("数量 (ETH)", min_value=0.001, value=0.01, step=0.001, format="%.3f")
    if st.button("🚀 刷新数据", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ---------- 主界面 ----------
st.title("📊 全中文智能交易监控中心 · 最终稳定版")
st.caption(f"数据 {st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')} 更新 | 多周期切换 | AI信号 | 模拟盈亏 | 微信提醒（需配置）")

# 获取数据
if use_simulated:
    # 模拟数据（调试用）
    dates = pd.date_range(end=datetime.now(), periods=200, freq='5min')
    sim_df = pd.DataFrame({
        "time": dates,
        "close": np.random.normal(2600, 20, 200).cumsum() + 1800,
        "high": 0,
        "low": 0,
        "open": 0,
        "volume": np.random.uniform(100, 500, 200)
    })
    sim_df["high"] = sim_df["close"] + np.random.uniform(5, 15, 200)
    sim_df["low"] = sim_df["close"] - np.random.uniform(5, 15, 200)
    sim_df["open"] = sim_df["close"].shift(1).fillna(sim_df["close"].iloc[0])
    sim_df = add_indicators(sim_df)
    data_dict = {interval: sim_df}
else:
    data_dict = fetch_all_periods()
    if interval not in data_dict or data_dict[interval].empty:
        st.error(f"周期 {interval} 数据获取失败")
        st.stop()

df = data_dict[interval]
latest = df.iloc[-1]
prev = df.iloc[-2] if len(df) > 1 else latest

# 顶部指标卡片
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    delta = latest["close"] - prev["close"]
    st.metric("ETH/USDT", f"${latest['close']:.2f}", f"{delta:+.2f}")
with col2:
    st.metric("RSI(14)", f"{latest['rsi']:.1f}" if not pd.isna(latest['rsi']) else "N/A")
with col3:
    st.metric("MA20", f"${latest['ma20']:.2f}" if not pd.isna(latest['ma20']) else "N/A")
with col4:
    st.metric("MA60", f"${latest['ma60']:.2f}" if not pd.isna(latest['ma60']) else "N/A")
with col5:
    st.metric("成交量", f"{latest['volume']:.0f}")

# K线图
st.subheader(f"{interval} K线图")

fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    row_heights=[0.7, 0.3]
)

fig.add_trace(go.Candlestick(
    x=df["time"],
    open=df["open"],
    high=df["high"],
    low=df["low"],
    close=df["close"],
    name="K线",
    increasing_line_color="#26A69A",
    decreasing_line_color="#EF5350"
), row=1, col=1)

fig.add_trace(go.Scatter(x=df["time"], y=df["ma20"], name="MA20", line=dict(color="orange", width=1)), row=1, col=1)
fig.add_trace(go.Scatter(x=df["time"], y=df["ma60"], name="MA60", line=dict(color="blue", width=1)), row=1, col=1)

fig.add_trace(go.Scatter(x=df["time"], y=df["rsi"], name="RSI", line=dict(color="purple")), row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)

fig.update_layout(
    template="plotly_dark",
    xaxis_rangeslider_visible=False,
    height=500,
    margin=dict(l=0, r=0, t=20, b=0),
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
fig.update_yaxes(title_text="价格 (USDT)", row=1, col=1)
fig.update_yaxes(title_text="RSI", row=2, col=1)

st.plotly_chart(fig, use_container_width=True)

# AI信号与模拟盈亏
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("🎯 AI 信号")
    btc_df = data_dict.get("15m") if "15m" in data_dict else None
    direction, confidence, full_content = get_ai_signal(df, btc_df)

    if direction == "做多":
        signal_color = "#26A69A"
        emoji = "🟢"
    elif direction == "做空":
        signal_color = "#EF5350"
        emoji = "🔴"
    else:
        signal_color = "#888888"
        emoji = "⚪"

    st.markdown(f"""
    <div class="signal-box">
        <span style="font-size: 24px; font-weight: bold; color: {signal_color};">{emoji} {direction}</span><br>
        <span style="font-size: 18px;">置信度: {confidence}%</span><br>
        <span style="color: #AAAAAA;">{full_content}</span>
    </div>
    """, unsafe_allow_html=True)

with col_right:
    st.subheader("💰 模拟盈亏")
    if entry_price > 0:
        current_price = latest["close"]
        if direction == "做多":
            profit_pct = (current_price - entry_price) / entry_price * 100
            profit_usd = (current_price - entry_price) * qty
        else:  # 做空
            profit_pct = (entry_price - current_price) / entry_price * 100
            profit_usd = (entry_price - current_price) * qty
        color = "#26A69A" if profit_usd >= 0 else "#EF5350"
        st.markdown(f"""
        <div style="background: #1E1F2A; padding: 20px; border-radius: 10px;">
            <span style="font-size: 20px;">当前盈亏</span><br>
            <span style="font-size: 32px; font-weight: bold; color: {color};">{profit_usd:+.2f} USDT</span><br>
            <span style="color: #AAAAAA;">({profit_pct:+.2f}%)</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("输入入场价以计算盈亏")

# 各周期快照
st.subheader("📌 各周期快照")

snapshot_cols = st.columns(3)
periods_list = ["1m","5m","15m","1h","4h","1d"]
for i, p in enumerate(periods_list):
    with snapshot_cols[i % 3]:
        if p in data_dict and not data_dict[p].empty and len(data_dict[p]) > 1:
            d = data_dict[p].iloc[-1]
            d_prev = data_dict[p].iloc[-2]
            arrow = "↑" if d["close"] > d_prev["close"] else "↓"
            color = "#26A69A" if arrow == "↑" else "#EF5350"
            st.markdown(f"""
            <div class="snapshot-item">
                <span style="font-weight: bold;">{p}</span>
                <span style="color: {color}; margin-left: 8px;">{arrow}</span>
                <span style="float: right;">RSI {d['rsi']:.1f}  ${d['close']:.2f}</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='snapshot-item'>{p}: 数据获取中...</div>", unsafe_allow_html=True)

# 自动刷新
if auto_refresh and not use_simulated:
    time_since = (datetime.now() - st.session_state.last_refresh).total_seconds()
    if time_since > 60:
        st.cache_data.clear()
        st.session_state.last_refresh = datetime.now()
        st.rerun()

st.divider()
st.caption("⚠️ 所有数据来自币安实时行情，AI信号仅供参考，不构成投资建议。杠杆交易风险极高，请自行控制仓位。")
