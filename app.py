import streamlit as st
import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
import joblib
import os

# =============================
# 1. 核心生产配置
# =============================
SYMBOL = "ETH/USDT"                # OKX 永续合约符号（USDT 本位）
REFRESH_MS = 3000                   # 刷新间隔 3 秒
CIRCUIT_BREAKER_PCT = 0.005         # 0.5% 熔断阈值
CONFIDENCE_THRESHOLD = 0.75         # 置信度阈值

st.set_page_config(layout="wide", page_title="ETH 100x AI Pro", page_icon="🤖")
st_autorefresh(interval=REFRESH_MS, key="prod_monitor")

@st.cache_resource
def init_system():
    """初始化交易所和模型"""
    exch = ccxt.okx({
        "enableRateLimit": True,
        "options": {"defaultType": "swap"}      # swap 表示永续合约
    })
    model = None
    model_path = "eth_ai_model.pkl"
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            st.sidebar.success(f"✅ 模型已加载：{model_path}")
        except Exception as e:
            st.sidebar.error(f"❌ 模型加载失败：{e}")
    else:
        st.sidebar.info("ℹ️ 未找到模型文件，AI 预测功能不可用")
    return exch, model

exchange, ai_model = init_system()

# 状态管理
if 'last_price' not in st.session_state:
    st.session_state.last_price = 0
if 'system_halted' not in st.session_state:
    st.session_state.system_halted = False

# 侧边栏重置按钮
if st.sidebar.button("🔌 重置系统熔断"):
    st.session_state.system_halted = False
    st.session_state.last_price = 0

# =============================
# 2. 生产级特征工程（与训练脚本严格对齐）
# =============================
def get_safe_analysis_data():
    """获取 K 线数据并计算特征"""
    try:
        ohlcv = exchange.fetch_ohlcv(SYMBOL, "5m", limit=150)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        
        # 重命名列以简化使用
        df.rename(columns={
            "timestamp": "t", "open": "o", "high": "h",
            "low": "l", "close": "c", "volume": "v"
        }, inplace=True)
        
        # 计算指标（与训练脚本完全一致）
        df["rsi"] = ta.rsi(df["c"], length=14)
        df["ma20"] = ta.sma(df["c"], length=20)
        df["ma60"] = ta.sma(df["c"], length=60)
        
        macd = ta.macd(df["c"])
        df["macd"] = macd["MACD_12_26_9"]
        df["macd_signal"] = macd["MACDs_12_26_9"]
        
        df["atr"] = ta.atr(df["h"], df["l"], df["c"], length=14)
        df["atr_pct"] = df["atr"] / df["c"] * 100    # 转换为百分比，与训练一致
        df["adx"] = ta.adx(df["h"], df["l"], df["c"], length=14)["ADX_14"]
        
        # 填充缺失值
        df = df.ffill().bfill()
        
        # 特征列（严格匹配训练脚本）
        feature_cols = ['rsi', 'ma20', 'ma60', 'macd', 'macd_signal', 'atr_pct', 'adx']
        # 取最新一行特征
        features = df[feature_cols].iloc[-1:].copy()
        
        return df, features
    except Exception as e:
        st.sidebar.error(f"数据获取失败: {e}")
        return None, None

# =============================
# 3. 实时交易逻辑
# =============================
st.title("🛡️ ETH 100x AI 生产级作战系统")

try:
    ticker = exchange.fetch_ticker(SYMBOL)
    current_price = ticker['last']
    
    # 熔断检测
    if st.session_state.last_price != 0:
        change = abs(current_price - st.session_state.last_price) / st.session_state.last_price
        if change > CIRCUIT_BREAKER_PCT:
            st.session_state.system_halted = True
    st.session_state.last_price = current_price

    if st.session_state.system_halted:
        st.error(f"🚨 触发系统熔断！价格异常波动。")
    else:
        df, current_feat = get_safe_analysis_data()
        
        pred, prob = 0, 0.0
        if ai_model is not None and current_feat is not None:
            try:
                # 预测概率
                proba = ai_model.predict_proba(current_feat)[0]
                pred = ai_model.predict(current_feat)[0]
                prob = proba[1]  # 假设类别1为看涨
            except Exception as e:
                st.sidebar.warning(f"预测失败: {e}")

        # 显示核心指标
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("ETH 实时价", f"${current_price:.2f}")
        col2.metric("AI 置信度", f"{prob*100:.1f}%")
        col3.metric("ADX 强度", f"{df['adx'].iloc[-1]:.1f}" if df is not None else "-")
        col4.metric("系统状态", "🔥 信号" if prob >= CONFIDENCE_THRESHOLD else "⏸️ 待机")

        st.markdown("---")

        # 信号触发
        if pred == 1 and prob >= CONFIDENCE_THRESHOLD:
            st.success(f"🎯 **高置信度多单信号 (置信度: {prob*100:.1f}%)**")
            atr = df["atr"].iloc[-1]
            # 止损距离 = min(ATR×1.5, 0.3% 价格)
            sl_dist = min(atr * 1.5, current_price * 0.003)
            sl = current_price - sl_dist
            tp = current_price + sl_dist * 2.5  # 盈亏比 1:2.5
            
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("入场位", f"${current_price:.2f}")
            sc2.metric("止损位", f"${sl:.2f}")
            sc3.metric("止盈位", f"${tp:.2f}")
        else:
            st.info("🔎 动能扫描中... AI 置信度未达标，禁止入场。")

        # 绘制 K 线图
        if df is not None:
            fig = go.Figure(data=[go.Candlestick(
                x=pd.to_datetime(df['t'], unit='ms'),
                open=df['o'], high=df['h'], low=df['l'], close=df['c'],
                name='K线'
            )])
            fig.update_layout(
                height=450,
                template="plotly_dark",
                xaxis_rangeslider_visible=False,
                margin=dict(l=0, r=0, t=0, b=0)
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.warning("暂无数据，请稍候...")

except Exception as e:
    st.error(f"系统运行异常: {e}")
    st.exception(e)  # 显示详细错误以便调试
