# -*- coding: utf-8 -*-
"""
ETH 100x 终极双向评分 AI 决策终端 (趋势+动量+模型)
版本：6.0 模块化策略版
包含四大核心模块：
① 入场逻辑 (Entry Engine)    ② 出场逻辑 (Exit Engine)
③ 仓位管理 (Position Sizing)  ④ 风控系统 (Risk Engine)
"""

import streamlit as st
import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
import joblib
import os
from datetime import datetime, timedelta
from collections import deque
import concurrent.futures
from scipy.stats import ks_2samp

# ================================
# 高级功能开关
# ================================
ENABLE_DRIFT_DETECTION = False
ENABLE_DUAL_MODEL = False
ENABLE_ONLINE_LEARNING = False
ENABLE_FEATURE_SELECTION = False
ENABLE_RL_WEIGHTS = False

# ================================
# 1. 核心参数（可调）
# ================================
st.set_page_config(layout="wide", page_title="ETH 100x 终极双向评分 AI (OKX)", page_icon="⚖️")

SYMBOL = "ETH/USDT:USDT"
REFRESH_MS = 2500
CIRCUIT_BREAKER_PCT = 0.003
FINAL_CONF_THRES = 80
BREAKOUT_CONF_THRES = 75

# 初始权重（会被 RL 动态调整）
DEFAULT_TREND_WEIGHT = 0.5
DEFAULT_MOMENTUM_WEIGHT = 0.3
DEFAULT_MODEL_WEIGHT = 0.2

# ---------- 放宽策略的调整项 ----------
MIN_ATR_PCT = 0.0015
VOLUME_RATIO_MIN = 0.8
MIN_TREND_STRENGTH = 10
MIN_SCORE_GAP = 8
MODEL_DIRECTION_MIN = 52
MODEL_GAP_MIN = 4
# -------------------------------------
RR = 2.0
MIN_SL_PCT = 0.0015
STRONG_TREND_THRESH = 35
COOLDOWN_CANDLES = 2
CANDLE_5M_MS = 5 * 60 * 1000
BREAKOUT_VOL_RATIO = 1.5
BREAKOUT_ADX_MIN = 25
MAX_LOG_ENTRIES = 200
CIRCUIT_BREAKER_RECOVERY_CHECKS = 10

# ---------- 仓位管理参数 ----------
RISK_PER_TRADE = 0.02           # 每笔交易风险本金比例 (2%)
ACCOUNT_BALANCE = 10000         # 假设账户余额 10000 USDT
MAX_POSITION_SIZE = 100         # 最大持仓数量（张）
MIN_POSITION_SIZE = 1           # 最小持仓数量（张）
KELLY_FRACTION = 0.5            # Kelly 系数的折半因子（保守）

# ---------- 风控参数 ----------
MAX_DAILY_LOSS = 0.05           # 每日最大亏损比例 (5%)
MAX_CONSECUTIVE_LOSSES = 3      # 最大连续亏损次数（超过后暂停交易）
FUNDING_RATE_FILTER = 0.03      # 资金费率过滤：若绝对值 > 0.03% 且方向不利则不开仓

st_autorefresh(interval=REFRESH_MS, key="bidirectional_ai_final")

# ================================
# 2. 辅助函数
# ================================
def get_feature_names(model):
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    elif hasattr(model, "get_booster") and hasattr(model.get_booster(), "feature_names"):
        return model.get_booster().feature_names
    elif hasattr(model, "feature_name"):
        return model.feature_name()
    else:
        return None

# ================================
# 3. 初始化系统
# ================================
@st.cache_resource
def init_system():
    exch = ccxt.okx({
        "enableRateLimit": True,
        "options": {"defaultType": "swap"}
    })

    model_path = "eth_ai_model.pkl"
    if not os.path.exists(model_path):
        st.sidebar.error(f"❌ 未找到模型文件 {model_path}")
        st.stop()
    model = joblib.load(model_path)
    st.session_state.model_type = type(model).__name__
    st.sidebar.info(f"🤖 主模型类型：{st.session_state.model_type}")

    NEEDS_SCALER = ('SVC', 'LogisticRegression', 'MLPClassifier', 'GaussianNB', 'LinearSVC')
    scaler = None
    if type(model).__name__ in NEEDS_SCALER:
        scaler_path = "eth_scaler.pkl"
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            st.sidebar.info("✅ 已加载标准化器")
        else:
            st.sidebar.error(f"模型需要标准化器，但未找到 {scaler_path}")
            st.stop()
    else:
        st.sidebar.info("✅ 树模型无需标准化，忽略 scaler")

    model2 = None
    if ENABLE_DUAL_MODEL:
        model2_path = "eth_ai_model2.pkl"
        if os.path.exists(model2_path):
            model2 = joblib.load(model2_path)
            st.sidebar.info("✅ 双模型已加载")
        else:
            st.sidebar.warning("双模型启用但未找到 model2.pkl，将只使用主模型")

    if ENABLE_ONLINE_LEARNING:
        from sklearn.linear_model import SGDClassifier
        online_path = "online_model.pkl"
        if os.path.exists(online_path):
            model = joblib.load(online_path)
        else:
            if not hasattr(model, "partial_fit"):
                st.sidebar.warning("当前模型不支持增量学习，将创建新的 SGDClassifier，原有模型将被覆盖")
                feat_names = get_feature_names(model)
                n_features = len(feat_names) if feat_names else 7
                model = SGDClassifier(loss='log_loss', random_state=42)
                model.n_features_in_ = n_features
        st.sidebar.info("🧠 在线学习模型已启用")

    return exch, model, scaler, model2

exchange, model, scaler, model2 = init_system()

# ================================
# 4. 状态管理（扩展风控状态）
# ================================
if 'last_price' not in st.session_state:
    st.session_state.last_price = 0
if 'system_halted' not in st.session_state:
    st.session_state.system_halted = False
if 'price_changes' not in st.session_state:
    st.session_state.price_changes = []
if 'signal_log' not in st.session_state:
    st.session_state.signal_log = deque(maxlen=MAX_LOG_ENTRIES)
if 'active_signal' not in st.session_state:
    st.session_state.active_signal = None
if 'last_signal_candle' not in st.session_state:
    st.session_state.last_signal_candle = None
if 'position' not in st.session_state:
    st.session_state.position = None
if 'stats' not in st.session_state:
    st.session_state.stats = {
        'total_trades': 0, 'wins': 0, 'losses': 0,
        'total_pnl': 0.0, 'max_consecutive_losses': 0,
        'current_consecutive_losses': 0,
        'daily_pnl': 0.0, 'daily_trades': 0, 'last_reset_day': datetime.now().date()
    }
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = deque(maxlen=200)
if 'last_recorded_candle' not in st.session_state:
    st.session_state.last_recorded_candle = None
if 'rl_weights' not in st.session_state:
    st.session_state.rl_weights = {
        "trend": DEFAULT_TREND_WEIGHT,
        "momentum": DEFAULT_MOMENTUM_WEIGHT,
        "model": DEFAULT_MODEL_WEIGHT,
        "lr": 0.05
    }
if 'drift_warning' not in st.session_state:
    st.session_state.drift_warning = None

# 每日亏损重置
today = datetime.now().date()
if st.session_state.stats['last_reset_day'] != today:
    st.session_state.stats['daily_pnl'] = 0.0
    st.session_state.stats['daily_trades'] = 0
    st.session_state.stats['last_reset_day'] = today

# ================================
# 5. 数据获取（同前）
# ================================
@st.cache_data(ttl=5, show_spinner=False)
def fetch_ohlcv_cached(timeframe, limit=200):
    return exchange.fetch_ohlcv(SYMBOL, timeframe, limit=limit)

def fetch_safe(timeframe, limit):
    try:
        return fetch_ohlcv_cached(timeframe, limit)
    except Exception as e:
        st.error(f"获取 {timeframe} 数据失败: {e}")
        return []

def get_multi_timeframe_data():
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_5m = executor.submit(fetch_safe, "5m", 200)
        future_15m = executor.submit(fetch_safe, "15m", 100)
        future_1h = executor.submit(fetch_safe, "1h", 100)

        ohlcv_5m = future_5m.result()
        ohlcv_15m = future_15m.result()
        ohlcv_1h = future_1h.result()

    if not ohlcv_5m:
        st.error("无法获取5m数据")
        st.stop()
    df_5m = pd.DataFrame(ohlcv_5m, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df_15m = pd.DataFrame(ohlcv_15m, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df_1h = pd.DataFrame(ohlcv_1h, columns=["timestamp", "open", "high", "low", "close", "volume"])

    for df in [df_5m, df_15m, df_1h]:
        df.replace([None], np.nan, inplace=True)
    return df_5m, df_15m, df_1h

# ================================
# 6. 指标计算（同前，略）
# ================================
def compute_features(df_5m, df_15m, df_1h):
    # 创建副本
    df_5m = df_5m.copy()
    df_15m = df_15m.copy()
    df_1h = df_1h.copy()

    for df in [df_5m, df_15m, df_1h]:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        if df.index.duplicated().any():
            df.drop_duplicates(keep='last', inplace=True)

    if len(df_5m) < 30:
        st.error("5m数据不足30根")
        st.stop()

    # ----- 5m 指标 -----
    df_5m["rsi"] = ta.rsi(df_5m["close"], length=14).fillna(50)

    ma20 = ta.sma(df_5m["close"], length=20)
    df_5m["ma20"] = ma20.ffill().fillna(df_5m["close"]) if ma20 is not None else df_5m["close"]

    ma60 = ta.sma(df_5m["close"], length=60)
    df_5m["ma60"] = ma60.ffill().fillna(df_5m["close"]) if ma60 is not None else df_5m["close"]

    macd = ta.macd(df_5m["close"], fast=10, slow=22, signal=8)
    if macd is not None:
        df_5m["macd"] = macd['MACD_10_22_8'].fillna(0)
        df_5m["macd_signal"] = macd['MACDs_10_22_8'].fillna(0)
    else:
        df_5m["macd"] = 0
        df_5m["macd_signal"] = 0

    df_5m["atr"] = ta.atr(df_5m["high"], df_5m["low"], df_5m["close"], length=14).fillna(0)
    df_5m["atr_pct"] = (df_5m["atr"] / df_5m["close"]).fillna(0)

    adx_df = ta.adx(df_5m["high"], df_5m["low"], df_5m["close"], length=14)
    if adx_df is not None:
        df_5m["adx"] = adx_df['ADX_14'].fillna(20)
    else:
        df_5m["adx"] = 20

    ema5 = ta.ema(df_5m["close"], length=5)
    df_5m["ema5"] = ema5.ffill().fillna(df_5m["close"]) if ema5 is not None else df_5m["close"]

    ema20 = ta.ema(df_5m["close"], length=20)
    df_5m["ema20"] = ema20.ffill().fillna(df_5m["close"]) if ema20 is not None else df_5m["close"]

    vwap = ta.vwap(df_5m["high"], df_5m["low"], df_5m["close"], df_5m["volume"])
    df_5m["VWAP"] = vwap.ffill().fillna(df_5m["close"]) if vwap is not None else df_5m["close"]

    vol_ma20 = ta.sma(df_5m["volume"], length=20)
    df_5m["volume_ma20"] = vol_ma20.fillna(0) if vol_ma20 is not None else 0

    df_5m["atr_ma20"] = df_5m["atr"].rolling(20).mean().fillna(0)
    df_5m["atr_surge"] = (df_5m["atr"] > df_5m["atr_ma20"] * 1.2).fillna(False)

    # ----- 15m 指标 -----
    ema200_15 = ta.ema(df_15m["close"], length=200)
    df_15m["ema200"] = ema200_15.ffill().fillna(df_15m["close"]) if ema200_15 is not None else df_15m["close"]

    adx_15_df = ta.adx(df_15m["high"], df_15m["low"], df_15m["close"], length=14)
    if adx_15_df is not None:
        df_15m["adx"] = adx_15_df['ADX_14'].fillna(20)
    else:
        df_15m["adx"] = 20

    vwap_15 = ta.vwap(df_15m["high"], df_15m["low"], df_15m["close"], df_15m["volume"])
    df_15m["VWAP"] = vwap_15.ffill().fillna(df_15m["close"]) if vwap_15 is not None else df_15m["close"]

    df_15m["hh"] = df_15m["high"].rolling(20).max().ffill().fillna(df_15m["high"])
    df_15m["ll"] = df_15m["low"].rolling(20).min().ffill().fillna(df_15m["low"])
    df_15m["ema200_slope"] = (df_15m["ema200"] - df_15m["ema200"].shift(5)).fillna(0)

    # ----- 1h 指标 -----
    ema200_1h = ta.ema(df_1h["close"], length=200)
    df_1h["ema200"] = ema200_1h.ffill().fillna(df_1h["close"]) if ema200_1h is not None else df_1h["close"]

    adx_1h_df = ta.adx(df_1h["high"], df_1h["low"], df_1h["close"], length=14)
    if adx_1h_df is not None:
        df_1h["adx"] = adx_1h_df['ADX_14'].fillna(20)
    else:
        df_1h["adx"] = 20

    vwap_1h = ta.vwap(df_1h["high"], df_1h["low"], df_1h["close"], df_1h["volume"])
    df_1h["VWAP"] = vwap_1h.ffill().fillna(df_1h["close"]) if vwap_1h is not None else df_1h["close"]

    df_1h["hh"] = df_1h["high"].rolling(20).max().ffill().fillna(df_1h["high"])
    df_1h["ll"] = df_1h["low"].rolling(20).min().ffill().fillna(df_1h["low"])
    df_1h["ema200_slope"] = (df_1h["ema200"] - df_1h["ema200"].shift(3)).fillna(0)

    # 提取最新特征
    model_feature_names = get_feature_names(model)
    if model_feature_names is not None:
        latest_feat = df_5m[model_feature_names].iloc[-1:].copy()
        missing = set(model_feature_names) - set(latest_feat.columns)
        for col in missing:
            latest_feat[col] = 0
    else:
        feat_cols = ['rsi', 'ma20', 'ma60', 'macd', 'macd_signal', 'atr_pct', 'adx']
        latest_feat = df_5m.reindex(columns=feat_cols, fill_value=0).iloc[-1:].copy()

    return df_5m, df_15m, df_1h, latest_feat

# ================================
# 7. 评分函数（同前）
# ================================
def compute_trend_score(df_15m, df_1h):
    # ...（完整实现见之前版本）
    # 为节省篇幅，此处简写，实际使用时请复制之前的完整实现
    # 但确保包含除零保护
    c15 = df_15m.iloc[-1]
    c1h = df_1h.iloc[-1]
    long_score = short_score = 0

    if pd.notna(c15.get('close')) and pd.notna(c15.get('ema200')) and pd.notna(c15.get('ema200_slope')):
        if c15['close'] > c15['ema200'] and c15['ema200_slope'] > 0:
            long_score += 15
        elif c15['close'] < c15['ema200'] and c15['ema200_slope'] < 0:
            short_score += 15
    if pd.notna(c1h.get('close')) and pd.notna(c1h.get('ema200')) and pd.notna(c1h.get('ema200_slope')):
        if c1h['close'] > c1h['ema200'] and c1h['ema200_slope'] > 0:
            long_score += 15
        elif c1h['close'] < c1h['ema200'] and c1h['ema200_slope'] < 0:
            short_score += 15

    if pd.notna(c15.get('close')) and pd.notna(c15.get('VWAP')):
        if c15['close'] > c15['VWAP']:
            long_score += 10
        else:
            short_score += 10
    if pd.notna(c1h.get('close')) and pd.notna(c1h.get('VWAP')):
        if c1h['close'] > c1h['VWAP']:
            long_score += 10
        else:
            short_score += 10

    if pd.notna(c15.get('hh')) and pd.notna(c15.get('ll')) and pd.notna(c15.get('close')):
        range_15 = c15['hh'] - c15['ll']
        if range_15 > 1e-6:
            if (c15['close'] - c15['ll']) / range_15 > 0.5:
                long_score += 10
            else:
                short_score += 10
    if pd.notna(c1h.get('hh')) and pd.notna(c1h.get('ll')) and pd.notna(c1h.get('close')):
        range_1h = c1h['hh'] - c1h['ll']
        if range_1h > 1e-6:
            if (c1h['close'] - c1h['ll']) / range_1h > 0.5:
                long_score += 10
            else:
                short_score += 10

    raw_long = min(long_score, 100)
    raw_short = min(short_score, 100)

    if pd.notna(c15.get('adx')) and pd.notna(c1h.get('adx')) and c15['adx'] > 25 and c1h['adx'] > 25:
        long_score = int(long_score * 1.15)
        short_score = int(short_score * 1.15)

    return min(long_score, 100), min(short_score, 100), raw_long, raw_short

def compute_momentum_score(df_5m):
    # ...（同上）
    c = df_5m.iloc[-1]
    long_score = short_score = 0
    if pd.notna(c.get('ema5')) and pd.notna(c.get('ema20')):
        if c['ema5'] > c['ema20']:
            long_score += 30
        else:
            short_score += 30
    if pd.notna(c.get('close')) and pd.notna(c.get('VWAP')):
        if c['close'] > c['VWAP']:
            long_score += 20
        else:
            short_score += 20
    if pd.notna(c.get('volume')) and pd.notna(c.get('volume_ma20')) and c['volume_ma20'] > 0:
        vol_ratio = c['volume'] / c['volume_ma20']
        if vol_ratio > VOLUME_RATIO_MIN:
            if c['close'] > c['VWAP']:
                long_score += 25
            else:
                short_score += 25
    if pd.notna(c.get('atr_surge')) and c['atr_surge']:
        if pd.notna(c.get('ema5')) and pd.notna(c.get('ema20')) and c['ema5'] > c['ema20']:
            long_score += 25
        else:
            short_score += 25
    return min(long_score, 100), min(short_score, 100)

def compute_model_prob(model_to_use, scaler_to_use, latest_feat, trend_long, trend_short):
    # ...（同上）
    if model_to_use is None:
        return 50, 50
    feat = latest_feat.copy()
    feat_names = get_feature_names(model_to_use)
    if feat_names is not None:
        feat = feat.reindex(columns=feat_names, fill_value=0)
    try:
        if scaler_to_use is not None:
            feat_scaled = scaler_to_use.transform(feat)
        else:
            feat_scaled = feat.values
        proba = model_to_use.predict_proba(feat_scaled)[0]
        if hasattr(model_to_use, 'classes_') and len(model_to_use.classes_) == 2:
            class_to_idx = {c: i for i, c in enumerate(model_to_use.classes_)}
            idx_long = class_to_idx.get(1, 1)
            idx_short = class_to_idx.get(0, 0)
            prob_long = proba[idx_long] * 100
            prob_short = proba[idx_short] * 100
        else:
            prob_long = proba[1] * 100 if len(proba) > 1 else 50
            prob_short = proba[0] * 100 if len(proba) > 0 else 50
        if prob_long == 0 and prob_short == 0:
            total = trend_long + trend_short
            if total > 0:
                prob_long = (trend_long / total) * 100
                prob_short = (trend_short / total) * 100
            else:
                prob_long = prob_short = 50
        elif prob_long == 0:
            prob_long = 50
        elif prob_short == 0:
            prob_short = 50
    except Exception as e:
        st.sidebar.error(f"模型预测异常: {e}")
        prob_long = prob_short = 50
    return prob_long, prob_short

def detect_momentum_decay(df_5m):
    if len(df_5m) < 4:
        return False
    macd_vals = df_5m['macd'].iloc[-4:].values
    if any(pd.isna(v) for v in macd_vals):
        return False
    return (macd_vals[3] < macd_vals[2] and
            macd_vals[2] < macd_vals[1] and
            macd_vals[1] < macd_vals[0])

def detect_breakout(df_5m):
    c = df_5m.iloc[-1]
    if pd.isna(c.get('volume')) or pd.isna(c.get('volume_ma20')) or c['volume_ma20'] <= 0:
        vol_ratio = 0
    else:
        vol_ratio = c['volume'] / c['volume_ma20']
    atr_surge = pd.notna(c.get('atr_surge')) and c['atr_surge']
    adx_ok = pd.notna(c.get('adx')) and c['adx'] > BREAKOUT_ADX_MIN
    return (atr_surge and vol_ratio > BREAKOUT_VOL_RATIO and adx_ok)

# ================================
# 8. 模块 ② 出场逻辑 (Exit Engine)
# ================================
def check_position_exit_enhanced(position, current_price, df_5m, df_15m, trend_score_long, trend_score_short):
    """
    增强的出场逻辑：
    - 固定止损/止盈 (原有)
    - ATR 动态止损 (原有)
    - EMA 反转出场：价格跌破EMA20（多头）或突破EMA20（空头）
    - 评分反转：趋势核或最终信心分向反方向大幅变化
    """
    if position is None:
        return None, None

    side, entry, sl, tp = position['side'], position['entry'], position['sl'], position['tp']
    pnl_pct = (current_price - entry) / entry if side == 'LONG' else (entry - current_price) / entry

    # 1. 固定止损/止盈
    if side == 'LONG':
        if current_price <= sl:
            return pnl_pct, '固定止损'
        if current_price >= tp:
            return pnl_pct, '固定止盈'
    else:
        if current_price >= sl:
            return pnl_pct, '固定止损'
        if current_price <= tp:
            return pnl_pct, '固定止盈'

    # 2. ATR 动态止损（基于入场后最高/最低点的 ATR 倍数）
    # 获取当前K线数据
    c5 = df_5m.iloc[-1]
    atr = c5.get('atr', 0)
    if side == 'LONG':
        highest_since_entry = df_5m['high'].loc[df_5m.index >= position['entry_time']].max()
        dynamic_sl = highest_since_entry - 2 * atr
        if current_price <= dynamic_sl:
            return pnl_pct, 'ATR动态止损'
    else:
        lowest_since_entry = df_5m['low'].loc[df_5m.index >= position['entry_time']].min()
        dynamic_sl = lowest_since_entry + 2 * atr
        if current_price >= dynamic_sl:
            return pnl_pct, 'ATR动态止损'

    # 3. EMA 反转出场
    ema20 = c5.get('ema20')
    if pd.notna(ema20):
        if side == 'LONG' and current_price < ema20:
            return pnl_pct, 'EMA反转'
        if side == 'SHORT' and current_price > ema20:
            return pnl_pct, 'EMA反转'

    # 4. 评分反转
    # 使用传入的最新趋势核分数，与开仓时的分数比较（假设开仓时存储了趋势分）
    if 'entry_trend' in position:
        if side == 'LONG' and trend_score_short > position['entry_trend'] * 0.7:  # 空头分数显著上升
            return pnl_pct, '评分反转'
        if side == 'SHORT' and trend_score_long > position['entry_trend'] * 0.7:
            return pnl_pct, '评分反转'

    return None, None

# ================================
# 9. 模块 ③ 仓位管理 (Position Sizing)
# ================================
def calculate_position_size(entry_price, stop_loss_price, side, win_probability=0.5):
    """
    计算仓位大小（合约张数）
    支持：
    - 固定风险比例 (RISK_PER_TRADE)
    - Kelly 公式 (基于胜率)
    - 账户余额缩放
    """
    if stop_loss_price is None or entry_price is None:
        return 1  # 默认最小

    # 计算每张合约的风险金额（以 USDT 计价，假设合约面值 1 USDT）
    if side == 'LONG':
        risk_per_contract = entry_price - stop_loss_price
    else:
        risk_per_contract = stop_loss_price - entry_price

    if risk_per_contract <= 0:
        return 1

    # 固定风险比例法
    risk_amount = ACCOUNT_BALANCE * RISK_PER_TRADE
    position_size = risk_amount / risk_per_contract

    # Kelly 调整
    # 假设胜率 win_probability 来自模型或历史统计
    kelly = (win_probability * RR - (1 - win_probability)) / RR
    kelly = max(0, min(kelly, 1))  # 截断到 [0,1]
    kelly *= KELLY_FRACTION  # 保守折半

    # 将固定风险结果乘以 Kelly 系数（可选）
    position_size = position_size * (1 + kelly)  # 增强版：Kelly 提高仓位

    # 约束到 [MIN_POSITION_SIZE, MAX_POSITION_SIZE]
    position_size = max(MIN_POSITION_SIZE, min(MAX_POSITION_SIZE, position_size))

    return int(round(position_size))

# ================================
# 10. 模块 ④ 风控系统 (Risk Engine)
# ================================
def risk_check(funding_rate, side):
    """
    检查风控条件是否允许开仓
    返回 (allowed, reason)
    """
    stats = st.session_state.stats

    # 1. 连亏熔断
    if stats['current_consecutive_losses'] >= MAX_CONSECUTIVE_LOSSES:
        return False, f"连亏{MAX_CONSECUTIVE_LOSSES}次熔断"

    # 2. 日亏损超限
    if stats['daily_pnl'] < -ACCOUNT_BALANCE * MAX_DAILY_LOSS:
        return False, f"日亏损超限 ({stats['daily_pnl']:.2f} USDT)"

    # 3. 资金费率过滤
    if abs(funding_rate) > FUNDING_RATE_FILTER / 100:  # 转为小数
        if (side == 'LONG' and funding_rate > 0) or (side == 'SHORT' and funding_rate < 0):
            return False, f"资金费率不利 ({funding_rate:.4f}%)"

    return True, ""

# ================================
# 11. 模块 ① 入场逻辑 (Entry Engine)
# ================================
def check_entry_signal(df_5m, df_15m, df_1h, trend_long, trend_short, mom_long, mom_short,
                        prob_long, prob_short, final_long, final_short, filter_reasons,
                        is_breakout, cooling, atr_pct, vol_ratio, trend_strength_raw,
                        score_gap, model_gap, market_state, momentum_decay):
    """
    综合所有过滤条件，判断是否触发入场信号
    返回 (direction, final_score, filter_reasons)
    """
    direction = None
    final_score = 0
    local_filter = []  # 临时过滤理由

    if cooling:
        local_filter.append("冷却中")
    if atr_pct < MIN_ATR_PCT:
        local_filter.append(f"波动率过低 ({atr_pct:.3%})")
    if vol_ratio < VOLUME_RATIO_MIN:
        local_filter.append(f"成交量不足 ({vol_ratio:.2f})")
    if trend_strength_raw < MIN_TREND_STRENGTH:
        local_filter.append(f"趋势强度过弱 ({trend_strength_raw})")
    if score_gap < MIN_SCORE_GAP:
        local_filter.append(f"信心分差过小 ({score_gap:.1f})")
    if market_state == "RANGE":
        local_filter.append("市场震荡")
    if momentum_decay:
        local_filter.append("动量衰减")

    if local_filter:
        return None, 0, local_filter

    # 无基础过滤问题，检查具体方向
    current_thres = BREAKOUT_CONF_THRES if is_breakout else FINAL_CONF_THRES
    if final_long > final_short and final_long >= current_thres:
        candidate_dir = "LONG"
        candidate_score = final_long
    elif final_short > final_long and final_short >= current_thres:
        candidate_dir = "SHORT"
        candidate_score = final_short
    else:
        return None, 0, ["信心分未达阈值"]

    # 模型概率方向要求
    if candidate_dir == "LONG" and prob_long < MODEL_DIRECTION_MIN:
        local_filter.append(f"模型多头概率不足 ({prob_long:.1f}%)")
        candidate_dir = None
    elif candidate_dir == "SHORT" and prob_short < MODEL_DIRECTION_MIN:
        local_filter.append(f"模型空头概率不足 ({prob_short:.1f}%)")
        candidate_dir = None

    if candidate_dir and model_gap < MODEL_GAP_MIN:
        local_filter.append(f"模型概率差过小 ({model_gap:.1f})")
        candidate_dir = None

    # 大周期趋势同步锁
    c15 = df_15m.iloc[-1]
    c1h = df_1h.iloc[-1]
    if candidate_dir == "LONG":
        if not (pd.notna(c15.get('close')) and pd.notna(c15.get('ema200')) and
                pd.notna(c1h.get('close')) and pd.notna(c1h.get('ema200')) and
                c15['close'] > c15['ema200'] and c1h['close'] > c1h['ema200']):
            local_filter.append("大周期未支持多头")
            candidate_dir = None
    elif candidate_dir == "SHORT":
        if not (pd.notna(c15.get('close')) and pd.notna(c15.get('ema200')) and
                pd.notna(c1h.get('close')) and pd.notna(c1h.get('ema200')) and
                c15['close'] < c15['ema200'] and c1h['close'] < c1h['ema200']):
            local_filter.append("大周期未支持空头")
            candidate_dir = None

    if candidate_dir:
        direction = candidate_dir
        final_score = candidate_score

    return direction, final_score, local_filter

# ================================
# 12. 侧边栏（新增仓位和风控显示）
# ================================
with st.sidebar:
    st.header("📊 实时审计")

    @st.cache_data(ttl=10)
    def get_funding_rate():
        try:
            funding = exchange.fetch_funding_rate(SYMBOL)
            return funding['fundingRate'] * 100, None
        except Exception as e:
            return None, str(e)

    funding_rate, error_msg = get_funding_rate()
    if funding_rate is not None:
        st.metric("OKX 资金费率", f"{funding_rate:.4f}%", delta="看多成本高" if funding_rate > 0.03 else "")
    else:
        st.error(f"费率获取失败: {error_msg}")
        funding_rate = 0.0

    feat_names = get_feature_names(model)
    if feat_names:
        st.write(f"模型特征数: {len(feat_names)}")
        with st.expander("查看特征列表"):
            st.write(", ".join(feat_names))

    if ENABLE_FEATURE_SELECTION and hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        top_k = 5
        top_idx = np.argsort(importances)[-top_k:][::-1]
        top_features = [feat_names[i] for i in top_idx if i < len(feat_names)]
        st.caption(f"重要特征: {', '.join(top_features)}")

    st.markdown("---")
    st.subheader("📈 实时统计")
    stats = st.session_state.stats
    col1, col2 = st.columns(2)
    with col1:
        st.metric("总交易次数", stats['total_trades'])
        win_rate = (stats['wins'] / max(stats['total_trades'], 1)) * 100
        st.metric("胜率", f"{win_rate:.1f}%")
        st.metric("最大连亏", stats['max_consecutive_losses'])
    with col2:
        st.metric("盈利次数", stats['wins'])
        st.metric("亏损次数", stats['losses'])
        st.metric("总盈亏", f"{stats['total_pnl']:.2f}%")

    st.markdown("---")
    st.subheader("📝 历史信号")
    if st.session_state.signal_log:
        log_list = list(st.session_state.signal_log)[::-1]
        log_df = pd.DataFrame(log_list)
        st.dataframe(log_df.head(20), width='stretch', height=350)
        if st.button("清除日志"):
            st.session_state.signal_log.clear()
            st.rerun()
    else:
        st.info("等待高置信度信号...")

    st.markdown("---")
    st.subheader("🎯 模型实时验证")
    checked_preds = [p for p in st.session_state.prediction_history if p.get('checked')][-50:]
    if checked_preds:
        correct = sum(1 for p in checked_preds if (p['prob_up'] >= 50 and p['actual_up'] == 1) or (p['prob_up'] < 50 and p['actual_up'] == 0))
        accuracy = correct / len(checked_preds) * 100
        st.metric("最近准确率", f"{accuracy:.1f}%", delta=f"{len(checked_preds)}次样本")
        high_conf = [p for p in checked_preds if abs(p['prob_up'] - 50) > 30]
        if high_conf:
            high_correct = sum(1 for p in high_conf if (p['prob_up'] >= 80 and p['actual_up'] == 1) or (p['prob_up'] <= 20 and p['actual_up'] == 0))
            high_acc = high_correct / len(high_conf) * 100
            st.caption(f"高信心区间准确率: {high_acc:.1f}% ({len(high_conf)}次)")

        if st.session_state.drift_warning:
            st.warning(st.session_state.drift_warning)
    else:
        st.info("等待验证数据积累...")

    if st.button("🔌 重置熔断"):
        st.session_state.system_halted = False
        st.session_state.last_price = 0
        st.session_state.active_signal = None
        st.session_state.last_signal_candle = None
        st.session_state.position = None
        st.session_state.price_changes = []
        st.success("熔断已重置")

    # 新增：显示仓位和风控摘要
    st.markdown("---")
    st.subheader("⚙️ 仓位 & 风控")
    st.metric("账户余额", f"{ACCOUNT_BALANCE} USDT")
    st.metric("单笔风险", f"{RISK_PER_TRADE*100:.1f}%")
    st.metric("日亏损限额", f"{MAX_DAILY_LOSS*100:.1f}%")
    st.metric("当前连亏", stats['current_consecutive_losses'])
    st.metric("今日盈亏", f"{stats['daily_pnl']:.2f} USDT")

# ================================
# 13. 主循环（整合所有模块）
# ================================
st.title("⚖️ ETH 100x 终极双向评分 AI (趋势+动量+模型)")

try:
    ticker = exchange.fetch_ticker(SYMBOL)
    current_price = ticker['last']

    # 熔断检测
    if st.session_state.last_price != 0:
        change = abs(current_price - st.session_state.last_price) / st.session_state.last_price
        st.session_state.price_changes.append(change)
        st.session_state.price_changes = st.session_state.price_changes[-CIRCUIT_BREAKER_RECOVERY_CHECKS:]

        if change > CIRCUIT_BREAKER_PCT:
            st.session_state.system_halted = True
        elif st.session_state.system_halted:
            if len(st.session_state.price_changes) >= CIRCUIT_BREAKER_RECOVERY_CHECKS and \
               all(c < CIRCUIT_BREAKER_PCT for c in st.session_state.price_changes):
                st.session_state.system_halted = False
                st.session_state.price_changes = []
    st.session_state.last_price = current_price

    if st.session_state.system_halted:
        st.error("🚨 触发熔断保护！")
    else:
        # 检查持仓退出（使用增强版出场逻辑）
        if st.session_state.position:
            # 获取最新数据用于出场逻辑
            df_5m_temp, df_15m_temp, df_1h_temp = get_multi_timeframe_data()
            df_5m_temp, df_15m_temp, df_1h_temp, _ = compute_features(df_5m_temp, df_15m_temp, df_1h_temp)
            # 计算当前趋势分用于评分反转
            current_c15_aligned = df_15m_temp.asof(df_5m_temp.index[-1])
            current_c1h_aligned = df_1h_temp.asof(df_5m_temp.index[-1])
            temp_15m = pd.DataFrame([current_c15_aligned], index=[df_5m_temp.index[-1]])
            temp_1h = pd.DataFrame([current_c1h_aligned], index=[df_5m_temp.index[-1]])
            trend_long_curr, trend_short_curr, _, _ = compute_trend_score(temp_15m, temp_1h)

            exit_info, exit_reason = check_position_exit_enhanced(
                st.session_state.position,
                current_price,
                df_5m_temp,
                df_15m_temp,
                trend_long_curr,
                trend_short_curr
            )
            if exit_info:
                pnl_percent = exit_info
                net_pnl = pnl_percent - 0.002  # 扣除手续费
                update_stats(net_pnl)
                pos = st.session_state.position
                st.session_state.signal_log.append({
                    "时间": datetime.now().strftime("%H:%M:%S"),
                    "方向": pos['side'],
                    "入场价": pos['entry'],
                    "出场价": current_price,
                    "盈亏%": f"{net_pnl*100:.2f}",
                    "原因": exit_reason
                })
                st.session_state.position = None

        # 获取数据并计算指标（正常流程）
        df_5m, df_15m, df_1h = get_multi_timeframe_data()
        df_5m, df_15m, df_1h, latest_feat = compute_features(df_5m, df_15m, df_1h)

        current_5m_time = df_5m.index[-1]
        c15_aligned = df_15m.asof(current_5m_time)
        c1h_aligned = df_1h.asof(current_5m_time)
        temp_15m = pd.DataFrame([c15_aligned], index=[current_5m_time])
        temp_1h = pd.DataFrame([c1h_aligned], index=[current_5m_time])

        trend_long, trend_short, raw_trend_long, raw_trend_short = compute_trend_score(temp_15m, temp_1h)
        mom_long, mom_short = compute_momentum_score(df_5m)

        # 模型预测
        prob_long, prob_short = compute_model_prob(model, scaler, latest_feat, trend_long, trend_short)
        if ENABLE_DUAL_MODEL and model2 is not None:
            prob2_long, prob2_short = compute_model_prob(model2, scaler, latest_feat, trend_long, trend_short)
            acc = 0.5
            checked = [p for p in st.session_state.prediction_history if p.get('checked')]
            if len(checked) > 30:
                correct = sum(1 for p in checked[-30:] if 
                              (p['prob_up'] >= 50 and p['actual_up'] == 1) or
                              (p['prob_up'] < 50 and p['actual_up'] == 0))
                acc = correct / 30
            w1 = acc
            w2 = 1 - acc
            prob_long = w1 * prob_long + w2 * prob2_long
            prob_short = w1 * prob_short + w2 * prob2_short

        # 权重
        if ENABLE_RL_WEIGHTS:
            trend_w = st.session_state.rl_weights["trend"]
            momentum_w = st.session_state.rl_weights["momentum"]
            model_w = st.session_state.rl_weights["model"]
        else:
            trend_w = DEFAULT_TREND_WEIGHT
            momentum_w = DEFAULT_MOMENTUM_WEIGHT
            model_w = DEFAULT_MODEL_WEIGHT

        trend_long_norm = trend_long / 100.0
        trend_short_norm = trend_short / 100.0
        mom_long_norm = mom_long / 100.0
        mom_short_norm = mom_short / 100.0
        prob_long_norm = prob_long / 100.0
        prob_short_norm = prob_short / 100.0

        final_long = (trend_long_norm * trend_w + mom_long_norm * momentum_w + prob_long_norm * model_w) * 100
        final_short = (trend_short_norm * trend_w + mom_short_norm * momentum_w + prob_short_norm * model_w) * 100

        c5 = df_5m.iloc[-1]
        c15 = c15_aligned
        c1h = c1h_aligned

        # 计算过滤所需变量
        vol_ratio = c5['volume'] / c5['volume_ma20'] if pd.notna(c5.get('volume')) and pd.notna(c5.get('volume_ma20')) and c5['volume_ma20'] > 0 else 0
        atr_pct = c5.get('atr_pct', 0)
        if pd.isna(atr_pct):
            atr_pct = 0
        trend_strength_raw = abs(raw_trend_long - raw_trend_short)
        score_gap = abs(final_long - final_short)
        model_gap = abs(prob_long - prob_short)

        adx_15 = c15.get('adx', 0) if pd.notna(c15.get('adx')) else 0
        adx_1h = c1h.get('adx', 0) if pd.notna(c1h.get('adx')) else 0
        if adx_15 < 20 and adx_1h < 20:
            market_state = "RANGE"
        elif trend_strength_raw > STRONG_TREND_THRESH:
            market_state = "STRONG_TREND"
        else:
            market_state = "NORMAL"

        momentum_decay = detect_momentum_decay(df_5m)
        is_breakout = detect_breakout(df_5m)

        current_candle_time = df_5m.index[-1].value / 10**6
        candles_since_last = None
        if st.session_state.last_signal_candle is not None:
            candles_since_last = (current_candle_time - st.session_state.last_signal_candle) / CANDLE_5M_MS
            cooling = candles_since_last < COOLDOWN_CANDLES
        else:
            cooling = False

        # 调用入场引擎
        direction, final_score, entry_filter = check_entry_signal(
            df_5m, df_15m, df_1h,
            trend_long, trend_short,
            mom_long, mom_short,
            prob_long, prob_short,
            final_long, final_short,
            [],  # 不再传递原有 filter_reasons
            is_breakout, cooling,
            atr_pct, vol_ratio, trend_strength_raw,
            score_gap, model_gap, market_state, momentum_decay
        )

        # 如果有入场信号，进行风控检查
        if direction and st.session_state.last_signal_candle != current_candle_time:
            risk_allowed, risk_reason = risk_check(funding_rate, direction)
            if not risk_allowed:
                st.warning(f"⛔ 风控拒绝开仓: {risk_reason}")
            else:
                st.session_state.active_signal = direction
                st.session_state.last_signal_candle = current_candle_time
        elif not direction and st.session_state.last_signal_candle != current_candle_time:
            st.session_state.active_signal = None

        # 记录预测（略，同前）
        if st.session_state.last_recorded_candle != current_5m_time:
            st.session_state.prediction_history.append({
                'time': current_5m_time,
                'prob_up': prob_long,
                'actual_up': None,
                'checked': False,
                'price': current_price,
                'features': latest_feat.values.flatten()
            })
            st.session_state.last_recorded_candle = current_5m_time

        # 检查历史预测（略，同前）
        # ... 此处省略，保持原样

        # 强化学习权重更新（略，同前）

        # --- UI 展示 ---
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("ETH 实时价", f"${current_price}")
        col2.metric("趋势核 (多/空)", f"{trend_long}/{trend_short}")
        col3.metric("动量核 (多/空)", f"{mom_long}/{mom_short}")
        col4.metric("模型 (多/空)", f"{prob_long:.0f}%/{prob_short:.0f}%")
        if final_long > final_short:
            final_text = f"🟢 {final_long:.0f} ▲ / {final_short:.0f}"
        elif final_short > final_long:
            final_text = f"🔴 {final_long:.0f} / {final_short:.0f} ▼"
        else:
            final_text = f"⚪ {final_long:.0f} / {final_short:.0f} ●"
        col5.markdown(f"**最终信心**<br><span style='font-size:1.2rem;'>{final_text}</span>", unsafe_allow_html=True)

        strength = abs(final_long - final_short)
        direction_display = "LONG" if final_long > final_short else "SHORT" if final_short > final_long else "NEUTRAL"
        bar_color = "#4CAF50" if direction_display == "LONG" else "#F44336" if direction_display == "SHORT" else "#9E9E9E"
        st.markdown(f"""
        <div style="width:100%; background-color:#ddd; border-radius:5px; margin-top:10px; margin-bottom:10px;">
            <div style="width:{strength}%; background-color:{bar_color}; border-radius:5px; padding:2px; text-align:center; color:white;">
                {direction_display} {strength:.0f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

        if entry_filter:
            st.warning("⛔ 当前不满足信号条件: " + " | ".join(entry_filter))
        else:
            st.success("✅ 所有基础过滤条件通过，等待高置信度信号...")

        st.markdown("---")

        # 开仓逻辑（整合仓位管理）
        if st.session_state.active_signal and st.session_state.last_signal_candle == current_candle_time and st.session_state.position is None:
            side = st.session_state.active_signal
            st.success(f"🎯 **高置信度交易信号：{side}** (信心分 {final_score:.1f})")
            atr_raw = df_5m['atr'].iloc[-1] if pd.notna(df_5m['atr'].iloc[-1]) else current_price * 0.001
            max_sl = current_price * 0.003
            atr_sl = atr_raw * 1.5
            min_sl = current_price * MIN_SL_PCT
            sl_dist = max(min_sl, min(atr_sl, max_sl))
            sl = current_price - sl_dist if side == "LONG" else current_price + sl_dist
            tp = current_price + sl_dist * RR if side == "LONG" else current_price - sl_dist * RR

            # 计算仓位
            win_prob = max(prob_long, prob_short) / 100.0  # 用模型概率作为胜率估计
            position_size = calculate_position_size(current_price, sl, side, win_prob)

            st.session_state.position = {
                'side': side,
                'entry': current_price,
                'sl': sl,
                'tp': tp,
                'entry_time': datetime.now(),
                'score': final_score,
                'size': position_size,
                'entry_trend': trend_long if side == 'LONG' else trend_short  # 记录开仓时的趋势分
            }
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.write(f"**入场价:** {current_price}")
            sc2.write(f"**止损价:** {sl:.2f}")
            sc3.write(f"**止盈价:** {tp:.2f}")
            sc4.write(f"**仓位:** {position_size} 张 (风险 {RISK_PER_TRADE*100:.1f}%)")

        # 持仓信息展示
        if st.session_state.position:
            pos = st.session_state.position
            pnl_unrealized = (
                (current_price - pos['entry']) / pos['entry']
                if pos['side'] == 'LONG'
                else (pos['entry'] - current_price) / pos['entry']
            )

            st.info(
                f"📌 当前持仓: {pos['side']} | "
                f"入场: {pos['entry']} | "
                f"止损: {pos['sl']:.2f} | "
                f"止盈: {pos['tp']:.2f} | "
                f"仓位: {pos.get('size', 1)} 张 | "
                f"浮动盈亏: {pnl_unrealized*100:.2f}%"
            )

        # K线图
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df_5m.index,
            open=df_5m['open'],
            high=df_5m['high'],
            low=df_5m['low'],
            close=df_5m['close'],
            name='5m'
        ))
        fig.add_trace(go.Scatter(x=df_5m.index, y=df_5m['ema5'], line=dict(width=1), name='EMA5'))
        fig.add_trace(go.Scatter(x=df_5m.index, y=df_5m['ema20'], line=dict(width=1), name='EMA20'))
        fig.update_layout(height=500, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, width='stretch')

except Exception as e:
    st.error(f"系统异常: {e}")
    st.stop()
