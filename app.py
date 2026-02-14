# -*- coding: utf-8 -*-
"""🔥 ETH 15m 精准交易终端 · Colab 一键版 🔥
   包含：实时数据、AI信号、专业图表
   使用前必做：在左侧🔑 Secrets 添加 AINFT_KEY（必需）和 PUSHPLUS_TOKEN（可选）
"""

!pip install -q streamlit pandas ta requests plotly pyngrok

import os
import threading
import time
from google.colab import userdata

# 读取密钥
try:
    AINFT_KEY = userdata.get('AINFT_KEY')
except:
    AINFT_KEY = None
    print("⚠️ 未找到 AINFT_KEY，AI 信号将使用规则代替")

try:
    PUSHPLUS_TOKEN = userdata.get('PUSHPLUS_TOKEN')
except:
    PUSHPLUS_TOKEN = None

# 写入 app.py
app_code = f'''
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
from datetime import datetime, timedelta

st.set_page_config(page_title="ETH 15m 交易终端", layout="wide")

st.markdown("""
<style>
    .stApp {{ background-color: #0E1117; color: white; }}
    .signal-box {{ background: #1E1F2A; border-radius: 10px; padding: 20px; border-left: 5px solid #00D4FF; }}
</style>
""", unsafe_allow_html=True)

# 从环境变量读取密钥（由 Colab 注入）
AINFT_KEY = os.environ.get("AINFT_KEY")
PUSHPLUS_TOKEN = os.environ.get("PUSHPLUS_TOKEN")

# 获取 Binance 数据（如果失败，使用模拟数据）
@st.cache_data(ttl=30)
def fetch_binance_klines(symbol, interval="15m", limit=200):
    url = "https://api.binance.com/api/v3/klines"
    try:
        r = requests.get(url, params={{"symbol": symbol, "interval": interval, "limit": limit}}, timeout=10)
        data = r.json()
        df = pd.DataFrame(data, columns=["time","o","h","l","c","v","ct","qv","n","tb","tq","i"])
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        for col in ["o","h","l","c","v"]: df[col] = df[col].astype(float)
        return df
    except:
        return None

def generate_simulated_klines(price, limit=200):
    now = datetime.now()
    times = [now - timedelta(minutes=15*i) for i in range(limit)][::-1]
    returns = np.random.randn(limit) * 0.002
    price_series = price * np.exp(np.cumsum(returns))
    price_series = price_series * (price / price_series[-1])
    closes = price_series
    opens = [closes[i-1] if i>0 else closes[0]*0.999 for i in range(limit)]
    highs = np.maximum(opens, closes) * 1.002
    lows = np.minimum(opens, closes) * 0.998
    vols = np.random.uniform(100, 500, limit)
    return pd.DataFrame({{"time": times, "o": opens, "h": highs, "l": lows, "c": closes, "v": vols}})

def add_indicators(df):
    df = df.copy()
    df["ma20"] = df["c"].rolling(20).mean()
    df["ma60"] = df["c"].rolling(60).mean()
    df["rsi"] = ta.momentum.RSIIndicator(df["c"], window=14).rsi()
    # 斐波那契
    if len(df) >= 100:
        recent_high = df["h"].rolling(100).max().iloc[-1]
        recent_low = df["l"].rolling(100).min().iloc[-1]
        diff = recent_high - recent_low
        df["fib_382"] = recent_high - diff * 0.382
        df["fib_500"] = recent_high - diff * 0.5
        df["fib_618"] = recent_high - diff * 0.618
    return df

def get_ai_signal(eth_df, btc_df, api_key):
    if not api_key:
        return "AI未配置，使用规则信号", ""
    e = eth_df.iloc[-1]
    b = btc_df.iloc[-1]
    prompt = f"""
【ETH 15m】价格:{e['c']:.2f} MA20:{e['ma20']:.2f} MA60:{e['ma60']:.2f} RSI:{e['rsi']:.1f}
斐波那契支撑:0.618={{e['fib_618']:.2f}} 0.5={{e['fib_500']:.2f}} 0.382={{e['fib_382']:.2f}}
【BTC 15m】价格:{b['c']:.2f} MA20:{b['ma20']:.2f} MA60:{b['ma60']:.2f} RSI:{b['rsi']:.1f}
请输出简洁交易计划：方向/进场/止损/止盈/BTC影响/风险。
"""
    url = "https://chat.ainft.com/webapi/chat/openai"
    headers = {{"Authorization": f"Bearer {{api_key}}", "Content-Type": "application/json"}}
    payload = {{
        "model": "gpt-5.2",
        "temperature": 0.2,
        "messages": [{{"role": "user", "content": prompt}}]
    }}
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=20)
        return resp.json()["choices"][0]["message"]["content"], ""
    except Exception as e:
        return "", f"AI调用失败: {{e}}"

def rule_signal(df):
    last = df.iloc[-1]
    if last['rsi'] < 30 and last['c'] > last['ma20']:
        return "做多", "RSI超卖+站上MA20"
    elif last['rsi'] > 70 and last['c'] < last['ma20']:
        return "做空", "RSI超买+跌破MA20"
    else:
        return "观望", "无明显信号"

st.title("🤖 ETH 15m 精准交易终端")

# 获取数据
eth_df = fetch_binance_klines("ETHUSDT", "15m", 200)
btc_df = fetch_binance_klines("BTCUSDT", "15m", 100)
if eth_df is None:
    st.warning("Binance数据获取失败，使用模拟数据")
    eth_df = generate_simulated_klines(3000, 200)
if btc_df is None:
    btc_df = generate_simulated_klines(65000, 100)

eth_df = add_indicators(eth_df)
btc_df = add_indicators(btc_df)

# 信号
if st.button("🚀 获取信号"):
    with st.spinner("分析中..."):
        if AINFT_KEY:
            signal, error = get_ai_signal(eth_df, btc_df, AINFT_KEY)
            if error:
                st.error(error)
                signal = "AI失败，改用规则"
        else:
            signal = "AI未配置，使用规则"
        if not signal:
            direction, reason = rule_signal(eth_df)
            signal = f"方向：{direction}\\n理由：{reason}"
        st.session_state.signal = signal
        st.session_state.eth = eth_df
        st.session_state.btc = btc_df

# 布局
col1, col2 = st.columns([2, 1.2])
with col1:
    st.subheader("📊 15分钟K线图")
    if "eth" in st.session_state:
        df = st.session_state.eth.tail(100)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7,0.3])
        fig.add_trace(go.Candlestick(x=df["time"], open=df["o"], high=df["h"], low=df["l"], close=df["c"]), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["time"], y=df["ma20"], name="MA20", line=dict(color="orange")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["time"], y=df["ma60"], name="MA60", line=dict(color="blue")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["time"], y=df["rsi"], name="RSI", line=dict(color="purple")), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=600)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("点击「获取信号」开始")

with col2:
    st.subheader("🎯 信号")
    if "signal" in st.session_state:
        st.markdown(f'<div class="signal-box">{st.session_state.signal.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)
        e = st.session_state.eth.iloc[-1]
        b = st.session_state.btc.iloc[-1]
        st.markdown("---")
        st.markdown("**📌 市场快照**")
        st.metric("ETH", f"${{e['c']:.2f}}", f"RSI {{e['rsi']:.1f}}")
        st.metric("BTC", f"${{b['c']:.0f}}", f"RSI {{b['rsi']:.1f}}")
        st.caption(f"⏱️ {{e['time'].strftime('%Y-%m-%d %H:%M')}}")
    else:
        st.info("点击「获取信号」")
'''

with open("app.py", "w") as f:
    f.write(app_code)

# 设置环境变量
if AINFT_KEY:
    os.environ["AINFT_KEY"] = AINFT_KEY
if PUSHPLUS_TOKEN:
    os.environ["PUSHPLUS_TOKEN"] = PUSHPLUS_TOKEN

# 启动 ngrok
from pyngrok import ngrok
ngrok.kill()
def run_streamlit():
    !streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true > streamlit.log 2>&1
thread = threading.Thread(target=run_streamlit)
thread.start()
time.sleep(5)
public_url = ngrok.connect(8501, "http")
print("\n" + "="*50)
print("✅ 应用已启动！点击下方链接访问：")
print("🌍", public_url)
print("="*50)
print("📱 手机电脑均可打开，点击「获取信号」")
