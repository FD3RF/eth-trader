import streamlit as st
import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
import joblib
import os
import time
from datetime import datetime

# ================================
# 1. 核心参数与看板设置
# ================================
st.set_page_config(layout="wide", page_title="ETH 100x 双向评分 AI (OKX)", page_icon="⚖️")

SYMBOL = "ETH/USDT:USDT"            # OKX 永续合约
REFRESH_MS = 2500                   # 2.5秒刷新
CIRCUIT_BREAKER_PCT = 0.003         # 0.3% 熔断
FINAL_CONF_THRES = 80                # 最终信心分门槛（满分100）

# 权重配置
TREND_WEIGHT = 0.5
MOMENTUM_WEIGHT = 0.3
MODEL_WEIGHT = 0.2

# 波动率过滤：ATR百分比 < 0.25% 时禁止交易
MIN_ATR_PCT = 0.0025

# 冷却时间：连续信号之间至少间隔 2 根 5m K 线（10分钟 = 600秒）
COOLDOWN_SECONDS = 600

st_autorefresh(interval=REFRESH_MS, key="bidirectional_ai")

# ================================
# 2. 初始化交易所和模型
# ================================
@st.cache_resource
def init_system():
    exch = ccxt.okx({
        "enableRateLimit": True,
        "options": {"defaultType": "swap"}
    })
    # 加载双模型（兼容通用模型）
    m_l = joblib.load("eth_ai_model_long.pkl") if os.path.exists("eth_ai_model_long.pkl") else None
    m_s = joblib.load("eth_ai_model_short.pkl") if os.path.exists("eth_ai_model_short.pkl") else None
    if m_l is None or m_s is None:
        generic = joblib.load("eth_ai_model.pkl") if os.path.exists("eth_ai_model.pkl") else None
        m_l = m_s = generic
        if generic:
            st.sidebar.info("💡 使用通用模型镜像多空")
    return exch, m_l, m_s

exchange, model_long, model_short = init_system()

# ================================
# 3. 状态管理
# ================================
if 'last_price' not in st.session_state:
    st.session_state.last_price = 0
if 'system_halted' not in st.session_state:
    st.session_state.system_halted = False
if 'signal_log' not in st.session_state:
    st.session_state.signal_log = []
if 'last_signal_time' not in st.session_state:
    st.session_state.last_signal_time = 0  # 上一次信号触发的时间戳（秒）

# ================================
# 4. 数据获取函数（多时间框架）
# ================================
def fetch_ohlcv(timeframe, limit=120):
    """获取指定周期的K线数据"""
    return exchange.fetch_ohlcv(SYMBOL, timeframe, limit=limit)

def get_multi_timeframe_data():
    """获取5m、15m、1h数据并计算指标"""
    ohlcv_5m = fetch_ohlcv("5m", 200)
    df_5m = pd.DataFrame(ohlcv_5m, columns=["t", "o", "h", "l", "c", "v"])
    
    ohlcv_15m = fetch_ohlcv("15m", 100)
    df_15m = pd.DataFrame(ohlcv_15m, columns=["t", "o", "h", "l", "c", "v"])
    
    ohlcv_1h = fetch_ohlcv("1h", 100)
    df_1h = pd.DataFrame(ohlcv_1h, columns=["t", "o", "h", "l", "c", "v"])
    
    return df_5m, df_15m, df_1h

# ================================
# 5. 指标计算函数
# ================================
def compute_features(df_5m, df_15m, df_1h):
    """计算所有需要的指标，返回DataFrame和最新特征向量"""
    # ----- 5m 指标（用于动量核 + 模型）-----
    df_5m["rsi"] = ta.rsi(df_5m["c"], length=14)
    df_5m["ma20"] = ta.sma(df_5m["c"], length=20)
    df_5m["ma60"] = ta.sma(df_5m["c"], length=60)
    macd = ta.macd(df_5m["c"])
    df_5m["macd"] = macd["MACD_12_26_9"]
    df_5m["macd_signal"] = macd["MACDs_12_26_9"]   # 标准信号线（请根据您的训练脚本调整）
    df_5m["atr"] = ta.atr(df_5m["h"], df_5m["l"], df_5m["c"], length=14)
    df_5m["atr_pct"] = df_5m["atr"] / df_5m["c"]
    df_5m["adx"] = ta.adx(df_5m["h"], df_5m["l"], df_5m["c"], length=14)["ADX_14"]
    
    # 动量核所需指标
    df_5m["ema9"] = ta.ema(df_5m["c"], length=9)
    df_5m["ema21"] = ta.ema(df_5m["c"], length=21)
    df_5m["vwap"] = ta.vwap(df_5m["h"], df_5m["l"], df_5m["c"], df_5m["v"])
    df_5m["volume_ma20"] = ta.sma(df_5m["v"], length=20)
    df_5m["atr_expand"] = df_5m["atr"] / df_5m["atr"].shift(1) - 1   # ATR扩张率
    
    # ----- 15m 指标（用于趋势核）-----
    df_15m["ema200"] = ta.ema(df_15m["c"], length=200)
    df_15m["adx"] = ta.adx(df_15m["h"], df_15m["l"], df_15m["c"], length=14)["ADX_14"]
    df_15m["vwap"] = ta.vwap(df_15m["h"], df_15m["l"], df_15m["c"], df_15m["v"])
    df_15m["hh"] = df_15m["h"].rolling(20).max()      # 20周期最高点
    df_15m["ll"] = df_15m["l"].rolling(20).min()      # 20周期最低点
    
    # ----- 1h 指标（用于趋势核）-----
    df_1h["ema200"] = ta.ema(df_1h["c"], length=200)
    df_1h["adx"] = ta.adx(df_1h["h"], df_1h["l"], df_1h["c"], length=14)["ADX_14"]
    df_1h["vwap"] = ta.vwap(df_1h["h"], df_1h["l"], df_1h["c"], df_1h["v"])
    df_1h["hh"] = df_1h["h"].rolling(20).max()
    df_1h["ll"] = df_1h["l"].rolling(20).min()
    
    # 填充NaN
    df_5m = df_5m.ffill().bfill()
    df_15m = df_15m.ffill().bfill()
    df_1h = df_1h.ffill().bfill()
    
    # 最新一行特征（用于模型预测）
    feat_cols = ['rsi', 'ma20', 'ma60', 'macd', 'macd_signal', 'atr_pct', 'adx']
    latest_feat = df_5m[feat_cols].iloc[-1:]
    
    return df_5m, df_15m, df_1h, latest_feat

# ================================
# 6. 双向评分函数
# ================================
def compute_trend_score(df_15m, df_1h):
    """计算趋势核的多空分数 (0-100)"""
    c15 = df_15m.iloc[-1]
    c1h = df_1h.iloc[-1]

    long_score = 0
    short_score = 0

    # EMA200 (每项15分)
    if c15['c'] > c15['ema200']:
        long_score += 15
    else:
        short_score += 15

    if c1h['c'] > c1h['ema200']:
        long_score += 15
    else:
        short_score += 15

    # ADX 强趋势加权 (每项10分，多空各加，因为趋势强对两方都有利)
    if c15['adx'] > 25:
        long_score += 10
        short_score += 10
    if c1h['adx'] > 25:
        long_score += 10
        short_score += 10

    # VWAP (每项10分)
    if c15['c'] > c15['vwap']:
        long_score += 10
    else:
        short_score += 10

    if c1h['c'] > c1h['vwap']:
        long_score += 10
    else:
        short_score += 10

    # 价格结构高低点 (每项10分)
    range_15 = c15['hh'] - c15['ll']
    if range_15 > 0:
        if (c15['c'] - c15['ll']) / range_15 > 0.5:
            long_score += 10
        else:
            short_score += 10

    range_1h = c1h['hh'] - c1h['ll']
    if range_1h > 0:
        if (c1h['c'] - c1h['ll']) / range_1h > 0.5:
            long_score += 10
        else:
            short_score += 10

    return min(long_score, 100), min(short_score, 100)

def compute_momentum_score(df_5m):
    """计算动量核的多空分数 (0-100)"""
    c = df_5m.iloc[-1]

    long_score = 0
    short_score = 0

    # EMA9 vs EMA21 (30分)
    if c['ema9'] > c['ema21']:
        long_score += 30
    else:
        short_score += 30

    # 价格 vs VWAP (20分)
    if c['c'] > c['vwap']:
        long_score += 20
    else:
        short_score += 20

    # 成交量放大 (25分，多空都加)
    if c['v'] > c['volume_ma20'] * 1.5:
        long_score += 25
        short_score += 25

    # ATR扩张 (25分，多空都加)
    if c['atr_expand'] > 0.1:
        long_score += 25
        short_score += 25

    return min(long_score, 100), min(short_score, 100)

def compute_model_prob(df_5m, latest_feat):
    """获取模型概率并转换为分数 (0-100)"""
    if model_long is None or model_short is None:
        return 50, 50
    prob_l = model_long.predict_proba(latest_feat)[0][1] * 100
    prob_s = model_short.predict_proba(latest_feat)[0][1] * 100
    return prob_l, prob_s

# ================================
# 7. 侧边栏（与之前一致）
# ================================
with st.sidebar:
    st.header("📊 实时审计")
    try:
        funding = exchange.fetch_funding_rate(SYMBOL)
        f_rate = funding['fundingRate'] * 100
        st.metric("OKX 资金费率", f"{f_rate:.4f}%", delta="看多成本高" if f_rate > 0.03 else "")
    except:
        st.write("费率加载中...")
    
    st.markdown("---")
    st.subheader("📝 历史信号")
    if st.session_state.signal_log:
        log_df = pd.DataFrame(st.session_state.signal_log).iloc[::-1]
        st.dataframe(log_df, use_container_width=True, height=350)
        if st.button("清除日志"):
            st.session_state.signal_log = []
            st.rerun()
    else:
        st.info("等待高置信度信号...")
    
    if st.button("🔌 重置熔断"):
        st.session_state.system_halted = False
        st.session_state.last_price = 0
        st.session_state.last_signal_time = 0

# ================================
# 8. 主界面
# ================================
st.title("⚖️ ETH 100x 双向评分 AI 决策终端 (趋势+动量+模型)")

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
        st.error("🚨 触发熔断保护！价格剧烈波动。")
    else:
        # 获取多周期数据并计算指标
        df_5m, df_15m, df_1h = get_multi_timeframe_data()
        df_5m, df_15m, df_1h, latest_feat = compute_features(df_5m, df_15m, df_1h)
        
        # 计算各项评分
        trend_long, trend_short = compute_trend_score(df_15m, df_1h)
        mom_long, mom_short = compute_momentum_score(df_5m)
        prob_l, prob_s = compute_model_prob(df_5m, latest_feat)
        
        # 计算最终多空信心分
        final_long = trend_long * TREND_WEIGHT + mom_long * MOMENTUM_WEIGHT + prob_l * MODEL_WEIGHT
        final_short = trend_short * TREND_WEIGHT + mom_short * MOMENTUM_WEIGHT + prob_s * MODEL_WEIGHT
        
        # 波动率过滤：ATR百分比 < 0.25% 时禁止交易
        atr_pct = df_5m['atr_pct'].iloc[-1]
        if atr_pct < MIN_ATR_PCT:
            st.warning(f"⚠️ 当前波动率过低 (ATR% = {atr_pct:.3%})，低于 {MIN_ATR_PCT:.2%}，禁止交易。")
            direction = None
            final_score = 0
        else:
            # 冷却时间检查
            current_time = time.time()
            time_since_last = current_time - st.session_state.last_signal_time
            if time_since_last < COOLDOWN_SECONDS:
                st.info(f"⏳ 冷却中，还需 {COOLDOWN_SECONDS - time_since_last:.0f} 秒才能产生新信号。")
                direction = None
                final_score = 0
            else:
                # 取高分方向
                if final_long > final_short and final_long >= FINAL_CONF_THRES:
                    direction = "LONG"
                    final_score = final_long
                elif final_short > final_long and final_short >= FINAL_CONF_THRES:
                    direction = "SHORT"
                    final_score = final_short
                else:
                    direction = None
                    final_score = 0
        
        # 顶部仪表盘（显示各核分数）
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("ETH 实时价", f"${current_price}")
        col2.metric("趋势核 (多/空)", f"{trend_long}/{trend_short}")
        col3.metric("动量核 (多/空)", f"{mom_long}/{mom_short}")
        col4.metric("模型 (多/空)", f"{prob_l:.0f}%/{prob_s:.0f}%")
        col5.metric("最终信心", f"{final_long:.0f}/{final_short:.0f}")
        
        st.markdown("---")
        
        # 显示最终信号（如果存在）
        if direction:
            # 记录信号时间
            st.session_state.last_signal_time = time.time()
            
            st.success(f"🎯 **高置信度交易信号：{direction}** (信心分 {final_score:.1f})")
            
            # 止损止盈计算（使用5m的ATR）
            atr_raw = df_5m['atr'].iloc[-1]
            sl_dist = min(atr_raw * 1.5, current_price * 0.004)  # 放宽至0.4%
            sl = current_price - sl_dist if direction == "LONG" else current_price + sl_dist
            tp = current_price + sl_dist * 2.5 if direction == "LONG" else current_price - sl_dist * 2.0
            
            sc1, sc2, sc3 = st.columns(3)
            sc1.write(f"**入场价:** {current_price}")
            sc2.write(f"**止损 (SL):** {round(sl, 2)}")
            sc3.write(f"**止盈 (TP):** {round(tp, 2)}")
            
            # 记录日志
            t_now = datetime.now().strftime("%H:%M:%S")
            if not st.session_state.signal_log or st.session_state.signal_log[-1]['时间'] != t_now:
                st.session_state.signal_log.append({
                    "时间": t_now,
                    "方向": direction,
                    "价格": current_price,
                    "信心分": f"{final_score:.1f}",
                    "趋势": f"{trend_long}/{trend_short}",
                    "动量": f"{mom_long}/{mom_short}",
                    "模型": f"{prob_l:.0f}%/{prob_s:.0f}%"
                })
        else:
            st.info("🔎 当前无符合要求的信号 (可能因波动率低、冷却中或信心不足)")
        
        # 显示K线图（5m）
        fig = go.Figure(data=[go.Candlestick(
            x=pd.to_datetime(df_5m['t'], unit='ms'),
            open=df_5m['o'], high=df_5m['h'], low=df_5m['l'], close=df_5m['c']
        )])
        fig.update_layout(height=450, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.sidebar.error(f"系统运行异常: {e}")
