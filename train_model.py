import streamlit as st
import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

# =============================
# 1. 核心参数 (Bybit 优化)
# =============================
SYMBOL = "ETH/USDT:USDT" # Bybit 合约的标准符号格式
REFRESH_MS = 1500        # 本地部署响应速度
CIRCUIT_BREAKER_PCT = 0.005 

st.set_page_config(layout="wide", page_title="ETH 100x Pro (Bybit)", page_icon="📈")
st_autorefresh(interval=REFRESH_MS, key="local_update")

# =============================
# 2. 交易所切换逻辑
# =============================
@st.cache_resource
def get_exchange():
    # 这里切换为 Bybit
    # Bybit 通常不需要代理也能在很多地区直连 API
    return ccxt.bybit({
        "enableRateLimit": True,
        "options": {
            "defaultType": "linear", # 使用线性合约（USDT本位）
        }
        # 如果 Bybit 也报错，再取消下面代理的注释
        # "proxies": {'http': 'http://127.0.0.1:7890', 'https': 'http://127.0.0.1:7890'},
    })

exchange = get_exchange()

if 'last_price' not in st.session_state: st.session_state.last_price = 0
if 'system_halted' not in st.session_state: st.session_state.system_halted = False

# =============================
# 3. 核心算法 (10级 5m/15m 联动)
# =============================

def fetch_data(symbol, timeframe, limit=100):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["t","o","h","l","c","v"])
    return df

def get_pro_signals(df5, df15):
    # 15m 趋势基准
    ema21_15 = ta.ema(df15["c"], 21).iloc[-1]
    curr_15 = df15["c"].iloc[-1]
    
    # 5m 动能指标
    df5["ema9"] = ta.ema(df5["c"], 9)
    df5["rsi"] = ta.rsi(df5["c"], 14)
    df5["adx"] = ta.adx(df5["h"], df5["l"], df5["c"], 14)["ADX_14"]
    macd = ta.macd(df5["c"])
    df5["hist"] = macd["MACDh_12_26_9"]
    
    last = df5.iloc[-1]
    score = 0
    
    # 评分逻辑
    score += 30 if curr_15 > ema21_15 else -30       # 15m 结构趋势
    score += 20 if last["ema9"] > last["c"] else -20  # 短期乖离
    score += 25 if last["hist"] > 0 else -25          # MACD 柱状图
    score += 25 if last["adx"] > 25 else 0            # 趋势强度
    
    return score

# =============================
# 4. 实时监控界面
# =============================
st.title("🛡️ ETH 100x Bybit Pro 监控系统")

try:
    # 获取最新价
    ticker = exchange.fetch_ticker(SYMBOL)
    current_price = ticker['last']
    
    # 毫秒级熔断检测
    if st.session_state.last_price != 0:
        change = abs(current_price - st.session_state.last_price) / st.session_state.last_price
        if change > CIRCUIT_BREAKER_PCT:
            st.session_state.system_halted = True
    st.session_state.last_price = current_price

    if st.session_state.system_halted:
        st.error(f"🚨 触发熔断保护！检测到异常波动: {change:.4%}")
        if st.button("重启系统"):
            st.session_state.system_halted = False
    else:
        # 获取多周期数据
        df5 = fetch_data(SYMBOL, "5m")
        df15 = fetch_data(SYMBOL, "15m")
        
        score = get_pro_signals(df5, df15)
        
        # 仪表盘显示
        c1, c2, c3 = st.columns(3)
        c1.metric("ETH Bybit Price", f"${current_price}")
        c2.metric("Pro Score", f"{score} pt", delta=f"{score}")
        c3.metric("Leverage Risk", "100x ⚠️", delta_color="inverse")

        # 信号输出
        if abs(score) >= 60:
            side = "LONG 🟢" if score > 0 else "SHORT 🔴"
            st.markdown(f"## 建议操作: {side}")
            # 自动计算 100x 止损 (使用 5m ATR)
            atr = ta.atr(df5["h"], df5["l"], df5["c"], 14).iloc[-1]
            sl = current_price - (atr * 1.5) if score > 0 else current_price + (atr * 1.5)
            st.write(f"**建议止损位:** {round(sl, 2)}")
        else:
            st.info("📊 动能积蓄中... 结构分不足以支撑 100x 入场。")

        # 可视化 K 线
        fig = go.Figure(data=[go.Candlestick(
            x=pd.to_datetime(df5['t'], unit='ms'),
            open=df5['o'], high=df5['h'], low=df5['l'], close=df5['c']
        )])
        fig.update_layout(height=450, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Bybit 连接失败: {e}")
    st.info("提示: 如果 Bybit 也无法连接，请尝试开启加速器的『全局模式』。")
