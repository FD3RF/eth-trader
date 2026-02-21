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
# 1. 核心参数与看板设置
# ================================
st.set_page_config(layout="wide", page_title="ETH 100x Tri-Core AI (OKX)", page_icon="⚡")

SYMBOL = "ETH/USDT:USDT"            # OKX 永续合约
REFRESH_MS = 2500                   # 2.5秒刷新
CIRCUIT_BREAKER_PCT = 0.003         # 0.3% 熔断
FINAL_CONF_THRES = 80                # 最终信心分门槛（满分100）

# 权重配置
TREND_WEIGHT = 0.5
MOMENTUM_WEIGHT = 0.3
MODEL_WEIGHT = 0.2

st_autorefresh(interval=REFRESH_MS, key="tri_core_monitor")

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

# ================================
# 4. 数据获取函数（多时间框架）
# ================================
def fetch_ohlcv(timeframe, limit=120):
    """获取指定周期的K线数据"""
    return exchange.fetch_ohlcv(SYMBOL, timeframe, limit=limit)

def get_multi_timeframe_data():
    """获取5m、15m、1h数据并计算指标"""
    # 5m 数据（用于动量核和模型）
    ohlcv_5m = fetch_ohlcv("5m", 200)
    df_5m = pd.DataFrame(ohlcv_5m, columns=["t", "o", "h", "l", "c", "v"])
    
    # 15m 数据（用于趋势核）
    ohlcv_15m = fetch_ohlcv("15m", 100)
    df_15m = pd.DataFrame(ohlcv_15m, columns=["t", "o", "h", "l", "c", "v"])
    
    # 1h 数据（用于趋势核）
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
    df_5m["macd_signal"] = macd["MACDs_12_26_9"]   # 标准信号线（根据您训练脚本调整）
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
# 6. 评分函数
# ================================
def compute_trend_score(df_15m, df_1h):
    """计算趋势核评分 (0-100)"""
    # 取最新值
    c15 = df_15m.iloc[-1]
    c1h = df_1h.iloc[-1]
    
    score = 0
    reasons = []
    
    # EMA200方向（15m和1h各15分，共30分）
    if c15['c'] > c15['ema200']:
        score += 15
        reasons.append("15m价格>EMA200")
    if c1h['c'] > c1h['ema200']:
        score += 15
        reasons.append("1h价格>EMA200")
    
    # ADX强度（15m和1h各15分，共30分）
    if c15['adx'] > 25:
        score += 15
        reasons.append(f"15m ADX={c15['adx']:.1f}>25")
    if c1h['adx'] > 25:
        score += 15
        reasons.append(f"1h ADX={c1h['adx']:.1f}>25")
    
    # VWAP偏离（15m和1h各10分，共20分）
    if c15['c'] > c15['vwap']:
        score += 10
        reasons.append("15m价格>VWAP")
    if c1h['c'] > c1h['vwap']:
        score += 10
        reasons.append("1h价格>VWAP")
    
    # 价格结构高低点（15m和1h各10分，共20分）
    # 简单规则：价格处于近期区间上半部分加分
    range_15 = c15['hh'] - c15['ll']
    if range_15 > 0 and (c15['c'] - c15['ll']) / range_15 > 0.5:
        score += 10
        reasons.append("15m价格处于区间上半部")
    
    range_1h = c1h['hh'] - c1h['ll']
    if range_1h > 0 and (c1h['c'] - c1h['ll']) / range_1h > 0.5:
        score += 10
        reasons.append("1h价格处于区间上半部")
    
    return min(score, 100), reasons

def compute_momentum_score(df_5m):
    """计算动量核评分 (0-100)"""
    c = df_5m.iloc[-1]
    score = 0
    reasons = []
    
    # EMA9上穿EMA21 (30分)
    if c['ema9'] > c['ema21']:
        score += 30
        reasons.append("EMA9 > EMA21")
    
    # 价格在VWAP之上 (20分)
    if c['c'] > c['vwap']:
        score += 20
        reasons.append("价格 > VWAP")
    
    # 成交量放大 (25分)
    if c['v'] > c['volume_ma20'] * 1.5:
        score += 25
        reasons.append(f"成交量放大 {c['v']/c['volume_ma20']:.1f}倍")
    
    # ATR扩张 (25分)
    if c['atr_expand'] > 0.1:  # ATR扩张超过10%
        score += 25
        reasons.append(f"ATR扩张 {c['atr_expand']*100:.1f}%")
    
    return min(score, 100), reasons

def compute_model_prob(df_5m, latest_feat):
    """获取模型概率并转换为分数 (0-100)"""
    if model_long is None or model_short is None:
        return 50, 50, "无模型"
    
    prob_l = model_long.predict_proba(latest_feat)[0][1]
    prob_s = model_short.predict_proba(latest_feat)[0][1]
    return prob_l * 100, prob_s * 100, ""

# ================================
# 7. 侧边栏（与之前类似，略作调整）
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

# ================================
# 8. 主界面
# ================================
st.title("⚡ ETH 100x 三核 AI 决策终端 (趋势+动量+模型)")

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
        df_5m, df_15m, df_1h, latest_feat = get_multi_timeframe_data()
        df_5m, df_15m, df_1h, latest_feat = compute_features(df_5m, df_15m, df_1h)
        
        # 计算各项评分
        trend_score, trend_reasons = compute_trend_score(df_15m, df_1h)
        momentum_score, momentum_reasons = compute_momentum_score(df_5m)
        prob_l, prob_s, _ = compute_model_prob(df_5m, latest_feat)
        
        # 计算最终信心分（取多头概率作为模型分，因为趋势和动量已隐含方向）
        # 注意：此处我们取 prob_l 作为模型分，但实际方向由趋势和动量决定，最终信号应结合三者。
        # 简便处理：将 prob_l 作为模型分，但最终信号方向需根据趋势+动量判断。
        model_score = prob_l  # 0-100
        final_score = trend_score * TREND_WEIGHT + momentum_score * MOMENTUM_WEIGHT + model_score * MODEL_WEIGHT
        
        # 判断方向：趋势和动量都看多才算多头信号（严格一点）
        direction = None
        if trend_score >= 60 and momentum_score >= 60 and prob_l > 50:
            direction = "LONG"
        elif trend_score <= 40 and momentum_score <= 40 and prob_s > 50:
            direction = "SHORT"
        # 也可根据趋势和动量分数差值判断
        
        # 顶部仪表盘
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("ETH 实时价", f"${current_price}")
        col2.metric("趋势核", f"{trend_score}", help="\n".join(trend_reasons) if trend_reasons else "")
        col3.metric("动量核", f"{momentum_score}", help="\n".join(momentum_reasons) if momentum_reasons else "")
        col4.metric("模型置信", f"{model_score:.1f}%")
        
        st.markdown("---")
        
        # 显示最终信心分
        st.subheader(f"📊 最终 AI 信心分: **{final_score:.1f}** / 100 (门槛 {FINAL_CONF_THRES})")
        
        # 只有当最终信心分 > 门槛时，才显示交易计划
        if final_score >= FINAL_CONF_THRES and direction is not None:
            side = direction
            st.success(f"🎯 **高置信度交易信号：{side}** (信心分 {final_score:.1f})")
            
            # 止损止盈计算（使用5m的ATR）
            atr_raw = df_5m['atr'].iloc[-1]
            sl_dist = min(atr_raw * 1.5, current_price * 0.004)  # 放宽至0.4%
            sl = current_price - sl_dist if side == "LONG" else current_price + sl_dist
            tp = current_price + sl_dist * 2.5 if side == "LONG" else current_price - sl_dist * 2.0
            
            sc1, sc2, sc3 = st.columns(3)
            sc1.write(f"**入场价:** {current_price}")
            sc2.write(f"**止损 (SL):** {round(sl, 2)}")
            sc3.write(f"**止盈 (TP):** {round(tp, 2)}")
            
            # 记录日志
            t_now = datetime.now().strftime("%H:%M:%S")
            if not st.session_state.signal_log or st.session_state.signal_log[-1]['时间'] != t_now:
                st.session_state.signal_log.append({
                    "时间": t_now,
                    "方向": side,
                    "价格": current_price,
                    "信心分": f"{final_score:.1f}",
                    "趋势": trend_score,
                    "动量": momentum_score,
                    "模型": f"{model_score:.1f}%"
                })
        else:
            st.info("🔎 当前信心分未达阈值，等待高质量机会...")
        
        # 显示K线图（5m）
        fig = go.Figure(data=[go.Candlestick(
            x=pd.to_datetime(df_5m['t'], unit='ms'),
            open=df_5m['o'], high=df_5m['h'], low=df_5m['l'], close=df_5m['c']
        )])
        fig.update_layout(height=450, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.sidebar.error(f"系统运行异常: {e}")
