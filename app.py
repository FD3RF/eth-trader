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

# ================================
# 配置参数（可根据需要调整）
# ================================
st.set_page_config(layout="wide", page_title="ETH 100x AI-Pro (OKX)")

SYMBOL = st.sidebar.text_input("交易对", "ETH/USDT:USDT", help="OKX 永续合约格式")
LEVERAGE = st.sidebar.slider("杠杆 (1-100)", 1, 100, 100)
REFRESH_MS = st.sidebar.slider("刷新间隔 (毫秒)", 1000, 5000, 2000)
CIRCUIT_BREAKER_PCT = 0.003          # 0.3% 熔断阈值
LONG_CONF_THRES = 0.78                # 多头置信度门槛
SHORT_CONF_THRES = 0.82               # 空头置信度门槛

st_autorefresh(interval=REFRESH_MS, key="okx_monitor")

# ================================
# 初始化交易所和模型
# ================================
@st.cache_resource
def init_system():
    exch = ccxt.okx({
        "enableRateLimit": True,
        "options": {"defaultType": "swap"}
    })
    # 加载模型（必须存在 eth_ai_model.pkl）
    if os.path.exists("eth_ai_model.pkl"):
        model = joblib.load("eth_ai_model.pkl")
    else:
        model = None
        st.sidebar.error("❌ 未找到模型文件 eth_ai_model.pkl")
    return exch, model

exchange, model = init_system()

# ================================
# 会话状态管理
# ================================
if 'last_price' not in st.session_state:
    st.session_state.last_price = 0
if 'system_halted' not in st.session_state:
    st.session_state.system_halted = False
if 'signal_log' not in st.session_state:
    st.session_state.signal_log = []

# ================================
# 侧边栏：资金费率 + 信号日志
# ================================
with st.sidebar:
    st.header("📊 实时审计")
    
    # 资金费率
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
    
    # 信号日志
    st.subheader("📝 历史信号")
    if st.session_state.signal_log:
        log_df = pd.DataFrame(st.session_state.signal_log).iloc[::-1]  # 最新在上
        # 使用 width 参数替换即将弃用的 use_container_width
        st.dataframe(log_df, width='stretch', height=400)
        if st.button("清除日志"):
            st.session_state.signal_log = []
            st.rerun()
    else:
        st.info("等待高置信度信号...")

# ================================
# 核心特征工程（与训练脚本完全对齐）
# ================================
def get_analysis_data():
    """获取最新 K 线并计算特征（必须与训练时的特征一致）"""
    try:
        ohlcv = exchange.fetch_ohlcv(SYMBOL, "5m", limit=100)
        df = pd.DataFrame(ohlcv, columns=["t", "o", "h", "l", "c", "v"])
        
        # 计算技术指标（顺序和名称必须与训练时一致）
        df["rsi"] = ta.rsi(df["c"], length=14)
        df["ma20"] = ta.sma(df["c"], length=20)
        df["ma60"] = ta.sma(df["c"], length=60)
        macd = ta.macd(df["c"])
        df["macd"] = macd["MACD_12_26_9"]
        df["macd_signal"] = macd["MACD_12_26_9"]          # 注意：训练时用了 MACD 线作为信号线，保持原样
        df["atr"] = ta.atr(df["h"], df["l"], df["c"], length=14)
        df["atr_pct"] = df["atr"] / df["c"]               # 转换为百分比形式，与训练一致
        df["adx"] = ta.adx(df["h"], df["l"], df["c"], length=14)["ADX_14"]
        
        df = df.ffill().bfill()                            # 填充可能的 NaN
        # 特征列必须与训练时的 FEATURES 顺序完全一致
        feat_cols = ['rsi', 'ma20', 'ma60', 'macd', 'macd_signal', 'atr_pct', 'adx']
        return df, df[feat_cols].iloc[-1:]
    except Exception as e:
        st.error(f"数据获取失败: {e}")
        return None, None

# ================================
# 主界面
# ================================
st.title("⚔️ ETH 100x AI 实时监控 (OKX)")

if st.sidebar.button("🔌 重置熔断"):
    st.session_state.system_halted = False
    st.session_state.last_price = 0

try:
    # 获取最新价格
    ticker = exchange.fetch_ticker(SYMBOL)
    current_price = ticker['last']
    
    # 熔断检测
    if st.session_state.last_price != 0:
        change = abs(current_price - st.session_state.last_price) / st.session_state.last_price
        if change > CIRCUIT_BREAKER_PCT:
            st.session_state.system_halted = True
    st.session_state.last_price = current_price

    if st.session_state.system_halted:
        st.error("🚨 触发系统熔断！价格剧烈波动。")
    else:
        # 获取特征数据
        df, current_feat = get_analysis_data()
        if df is None or current_feat is None:
            st.stop()
        
        # 模型预测概率（假设模型输出二分类：0=空头，1=多头）
        if model is not None:
            prob = model.predict_proba(current_feat)[0]
            prob_l = prob[1]   # 多头概率
            prob_s = prob[0]   # 空头概率
        else:
            prob_l = prob_s = 0.5

        # 顶栏指标
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("ETH 实时价", f"${current_price}")
        col2.metric("多头置信度", f"{prob_l*100:.1f}%",
                    delta=f"{(prob_l-LONG_CONF_THRES)*100:.1f}%" if prob_l > LONG_CONF_THRES else None)
        col3.metric("空头置信度", f"{prob_s*100:.1f}%",
                    delta=f"{(prob_s-SHORT_CONF_THRES)*100:.1f}%" if prob_s > SHORT_CONF_THRES else None,
                    delta_color="inverse")
        col4.metric("ADX 强度", f"{df['adx'].iloc[-1]:.1f}")

        st.markdown("---")

        # 信号判断
        side = None
        if prob_l >= LONG_CONF_THRES and prob_l > prob_s:
            side = "LONG"
            st.success(f"🎯 **高置信度多单信号** (L:{prob_l:.2f} vs S:{prob_s:.2f})")
        elif prob_s >= SHORT_CONF_THRES and prob_s > prob_l:
            side = "SHORT"
            st.error(f"🎯 **高置信度空单信号** (S:{prob_s:.2f} vs L:{prob_l:.2f})")
        else:
            st.info("🔎 动能扫描中... AI 建议观望")

        # 记录日志
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

        # 止损止盈建议（基于 ATR）
        if side:
            atr = df['atr'].iloc[-1]               # 原始 ATR 用于止损距离
            sl_dist = min(atr * 1.5, current_price * 0.003)   # 止损距离（ATR倍数与0.3%取小）
            if side == "LONG":
                sl = current_price - sl_dist
                tp = current_price + sl_dist * 2.5   # 盈亏比 1:2.5
            else:
                sl = current_price + sl_dist
                tp = current_price - sl_dist * 2.0   # 空单盈亏比 1:2
            sc1, sc2, sc3 = st.columns(3)
            sc1.write(f"**入场价:** {current_price}")
            sc2.write(f"**止损 (SL):** {round(sl, 2)}")
            sc3.write(f"**止盈 (TP):** {round(tp, 2)}")

        # K线图
        fig = go.Figure(data=[go.Candlestick(
            x=pd.to_datetime(df['t'], unit='ms'),
            open=df['o'], high=df['h'], low=df['l'], close=df['c']
        )])
        fig.update_layout(height=450, template="plotly_dark", xaxis_rangeslider_visible=False)
        # 同样更新 use_container_width 为 width
        st.plotly_chart(fig, use_container_width=True)  # 此处 use_container_width 还未废弃，但为了统一可改为 width='stretch'，但 plotly_chart 的参数不同，暂时保留
        # 如果你也想消除 plotly_chart 的警告，可以改为 st.plotly_chart(fig, use_container_width=True) 目前没有警告，保持原样

except Exception as e:
    st.sidebar.error(f"运行异常: {e}")
