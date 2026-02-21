import streamlit as st
import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
import jobit # 如果你训练的模型是 pkl，确保环境有 joblib

# =============================
# 1. 核心参数 (100x 高响应配置)
# =============================
SYMBOL = "ETH/USDT:USDT"  # Bybit 永续合约标准符号
REFRESH_MS = 1000        # 1秒极速刷新
CIRCUIT_BREAKER_PCT = 0.005 

st.set_page_config(layout="wide", page_title="ETH 100x Pro (Bybit)", page_icon="📈")
st_autorefresh(interval=REFRESH_MS, key="bybit_update")

# =============================
# 2. 交易所初始化 (切换为 Bybit)
# =============================
@st.cache_resource
def get_exchange():
    # Bybit API 通常比 Binance 限制更少
    return ccxt.bybit({
        "enableRateLimit": True,
        "options": {
            "defaultType": "linear", # 线性合约
        },
        # 如果你本地需要代理，请取消下面两行的注释：
        # "proxies": {'http': 'http://127.0.0.1:7890', 'https': 'http://127.0.0.1:7890'},
    })

exchange = get_exchange()

# 状态管理
if 'last_price' not in st.session_state: st.session_state.last_price = 0
if 'system_halted' not in st.session_state: st.session_state.system_halted = False

# =============================
# 3. 核心算法 (兼容你的 train_model 指标)
# =============================

def fetch_signals(symbol):
    # 获取 5m 数据进行计算
    ohlcv = exchange.fetch_ohlcv(symbol, "5m", limit=100)
    df = pd.DataFrame(ohlcv, columns=["t","o","h","l","c","v"])
    
    # 匹配你 train_model.py 中的技术指标
    df["rsi"] = ta.rsi(df["c"], 14)
    df["ma20"] = ta.sma(df["c"], 20)
    df["ma60"] = ta.sma(df["c"], 60)
    macd = ta.macd(df["c"])
    df["hist"] = macd["MACDh_12_26_9"]
    df["atr"] = ta.atr(df["h"], df["l"], df["c"], 14)
    df["adx"] = ta.adx(df["h"], df["l"], df["c"], 14)["ADX_14"]
    
    last = df.iloc[-1]
    
    # 综合评分逻辑 (针对 100x 杠杆)
    score = 0
    score += 30 if last["c"] > last["ma20"] else -30
    score += 20 if last["hist"] > 0 else -20
    score += 25 if last["adx"] > 25 else 0
    if last["rsi"] > 60: score += 25
    elif last["rsi"] < 40: score -= 25
    
    return df, score

# =============================
# 4. 实时监控界面
# =============================
st.title("🛡️ ETH 100x Bybit Pro 监控系统")

if st.sidebar.button("🔌 重置系统"):
    st.session_state.system_halted = False
    st.session_state.last_price = 0

try:
    # 实时价格捕获
    ticker = exchange.fetch_ticker(SYMBOL)
    current_price = ticker['last']
    
    # 熔断检测
    if st.session_state.last_price != 0:
        change = abs(current_price - st.session_state.last_price) / st.session_state.last_price
        if change > CIRCUIT_BREAKER_PCT:
            st.session_state.system_halted = True
    st.session_state.last_price = current_price

    if st.session_state.system_halted:
        st.error(f"🚨 触发熔断保护！波动率过高。")
    else:
        # 数据与信号处理
        df, score = fetch_signals(SYMBOL)
        
        # 顶层看板
        c1, c2, c3 = st.columns(3)
        c1.metric("ETH Bybit Price", f"${current_price}")
        c2.metric("Trend Score", f"{score} pt", delta=f"{round(score,1)}")
        c3.metric("Execution Status", "READY" if abs(score) < 80 else "HIGH ALERT")

        # 信号输出
        if abs(score) >= 60:
            side = "LONG 🟢" if score > 0 else "SHORT 🔴"
            st.markdown(f"### 🎯 建议方向: {side}")
            
            # 计算 100x 的安全止损（基于 ATR）
            atr_val = df["atr"].iloc[-1]
            sl = current_price - (atr_val * 1.5) if score > 0 else current_price + (atr_val * 1.5)
            st.warning(f"100x 止损参考价: {round(sl, 2)}")
        else:
            st.info("📊 市场动能不足，等待信号中...")

        # 绘制实时 K 线
        fig = go.Figure(data=[go.Candlestick(
            x=pd.to_datetime(df['t'], unit='ms'),
            open=df['o'], high=df['h'], low=df['l'], close=df['c'],
            name="ETH 5m"
        )])
        fig.update_layout(height=500, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"❌ 链接异常: {e}")
    st.info("💡 建议：如果 Bybit 依然无法访问，请检查加速器是否开启了『TUN模式』或『全局模式』。")
