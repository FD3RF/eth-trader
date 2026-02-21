# app.py
import streamlit as st
import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
import joblib
import os
from datetime import datetime

# 配置
st.set_page_config(layout="wide", page_title="ETH 100x AI-Pro (OKX)")
SYMBOL = st.sidebar.text_input("Trading Pair", "ETH/USDT:USDT", help="OKX swap symbol")
LEVERAGE = st.sidebar.slider("Leverage (1-100)", 1, 100, 100)
REFRESH_MS = st.sidebar.slider("Refresh (ms)", 1000, 5000, 2000)
CIRCUIT_BREAKER_PCT = 0.003
LONG_CONF_THRES = 0.78
SHORT_CONF_THRES = 0.82

st_autorefresh(interval=REFRESH_MS, key="okx_monitor")

@st.cache_resource
def init_system():
    exch = ccxt.okx({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    model = joblib.load("eth_ai_model.pkl") if os.path.exists("eth_ai_model.pkl") else None
    if model is None:
        st.sidebar.error("❌ 未找到模型文件 eth_ai_model.pkl")
    return exch, model

exchange, model = init_system()

# 会话状态
if 'last_price' not in st.session_state:
    st.session_state.last_price = 0
if 'system_halted' not in st.session_state:
    st.session_state.system_halted = False
if 'signal_log' not in st.session_state:
    st.session_state.signal_log = []

# 侧边栏（资金费率、信号日志）
with st.sidebar:
    st.header("📊 实时审计")
    try:
        funding = exchange.fetch_funding_rate(SYMBOL)
        f_rate = funding['fundingRate'] * 100
        f_time = datetime.fromtimestamp(funding['fundingTimestamp']/1000).strftime('%H:%M')
        f_color = "red" if abs(f_rate) > 0.03 else "green"
        st.markdown(f"**资金费率 ({SYMBOL})**")
        st.markdown(f"<h3 style='color:{f_color};'>{round(f_rate, 4)}%</h3>", unsafe_allow_html=True)
        st.caption(f"下次结算: {f_time}")
        if f_rate > 0.05:
            st.warning("⚠️ 多头成本极高，谨慎做多")
        elif f_rate < -0.05:
            st.warning("⚠️ 空头成本极高，谨慎做空")
    except Exception as e:
        st.error("资金费率获取失败")
    
    st.markdown("---")
    st.subheader("📝 历史信号")
    if st.session_state.signal_log:
        log_df = pd.DataFrame(st.session_state.signal_log).iloc[::-1]
        st.dataframe(log_df, use_container_width=True, height=400)
        if st.button("清除日志"):
            st.session_state.signal_log = []
            st.rerun()
    else:
        st.info("等待高置信度信号...")

# 特征工程
def get_analysis_data():
    try:
        ohlcv = exchange.fetch_ohlcv(SYMBOL, "5m", limit=100)
        df = pd.DataFrame(ohlcv, columns=["t", "o", "h", "l", "c", "v"])
        
        df["rsi"] = ta.rsi(df["c"], length=14)
        df["ma20"] = ta.sma(df["c"], length=20)
        df["ma60"] = ta.sma(df["c"], length=60)
        macd = ta.macd(df["c"])
        df["macd"] = macd["MACD_12_26_9"]
        df["macd_signal"] = macd["MACDs_12_26_9"]
        df["atr"] = ta.atr(df["h"], df["l"], df["c"], length=14)
        df["adx"] = ta.adx(df["h"], df["l"], df["c"], length=14)["ADX_14"]
        
        df = df.ffill().bfill()
        feat_cols = ['rsi', 'ma20', 'ma60', 'macd', 'macd_signal', 'atr', 'adx']
        return df, df[feat_cols].iloc[-1:]
    except Exception as e:
        st.error(f"数据获取失败: {e}")
        return None, None

# 主界面
st.title("⚔️ ETH 100x AI 实时监控 (OKX)")

if st.sidebar.button("🔌 重置熔断"):
    st.session_state.system_halted = False
    st.session_state.last_price = 0

try:
    ticker = exchange.fetch_ticker(SYMBOL)
    current_price = ticker['last']
    
    if st.session_state.last_price != 0:
        change = abs(current_price - st.session_state.last_price) / st.session_state.last_price
        if change > CIRCUIT_BREAKER_PCT:
            st.session_state.system_halted = True
    st.session_state.last_price = current_price

    if st.session_state.system_halted:
        st.error("🚨 触发系统熔断！价格剧烈波动。")
    else:
        df, current_feat = get_analysis_data()
        if df is None or current_feat is None:
            st.stop()
        
        if model is not None:
            prob = model.predict_proba(current_feat)[0]
            prob_l = prob[1]
            prob_s = prob[0]
        else:
            prob_l = prob_s = 0.5

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("ETH 实时价", f"${current_price}")
        col2.metric("多头置信度", f"{prob_l*100:.1f}%",
                    delta=f"{(prob_l-LONG_CONF_THRES)*100:.1f}%" if prob_l > LONG_CONF_THRES else None)
        col3.metric("空头置信度", f"{prob_s*100:.1f}%",
                    delta=f"{(prob_s-SHORT_CONF_THRES)*100:.1f}%" if prob_s > SHORT_CONF_THRES else None,
                    delta_color="inverse")
        col4.metric("ADX 强度", f"{df['adx'].iloc[-1]:.1f}")

        st.markdown("---")

        side = None
        if prob_l >= LONG_CONF_THRES and prob_l > prob_s:
            side = "LONG"
            st.success(f"🎯 **高置信度多单信号** (L:{prob_l:.2f} vs S:{prob_s:.2f})")
        elif prob_s >= SHORT_CONF_THRES and prob_s > prob_l:
            side = "SHORT"
            st.error(f"🎯 **高置信度空单信号** (S:{prob_s:.2f} vs L:{prob_l:.2f})")
        else:
            st.info("🔎 动能扫描中... AI 建议观望")

        if side:
            now_time = datetime.now().strftime("%H:%M:%S")
            if not st.session_state.signal_log or st.session_state.signal_log[-1]['时间'] != now_time:
                st.session_state.signal_log.append({
                    "时间": now_time,
                    "方向": side,
                    "价格": current_price,
                    "多头%": f"{prob_l*100:.1f}%",
                    "空头%": f"{prob_s*100:.1f}%"
                })

        if side:
            atr = df['atr'].iloc[-1]
            sl_dist = min(atr * 1.5, current_price * 0.003)
            if side == "LONG":
                sl = current_price - sl_dist
                tp = current_price + sl_dist * 2.5
            else:
                sl = current_price + sl_dist
                tp = current_price - sl_dist * 2.0
            sc1, sc2, sc3 = st.columns(3)
            sc1.write(f"**入场价:** {current_price}")
            sc2.write(f"**止损 (SL):** {round(sl, 2)}")
            sc3.write(f"**止盈 (TP):** {round(tp, 2)}")

        fig = go.Figure(data=[go.Candlestick(
            x=pd.to_datetime(df['t'], unit='ms'),
            open=df['o'], high=df['h'], low=df['l'], close=df['c']
        )])
        fig.update_layout(height=450, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.sidebar.error(f"运行异常: {e}")
