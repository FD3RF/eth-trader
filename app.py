# -*- coding: utf-8 -*-

"""

ETH 100x 终极双向评分 AI 决策终端 (趋势+动量+模型)

版本：2.0 最终优化版

特性：

- 多周期趋势/动量/模型融合

- 熔断、冷却、风控过滤

- 实时模型准确率验证

- 并发数据获取提升性能

- 自动识别模型类型（树模型无需标准化）

- 指标计算与训练代码完全一致（列名对齐）

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

import time

from datetime import datetime

from collections import deque

import concurrent.futures



# ================================

# 1. 全局配置参数（集中管理）

# ================================

st.set_page_config(layout="wide", page_title="ETH 100x 终极双向评分 AI (OKX)", page_icon="⚖️")



# 交易对设置

SYMBOL = "ETH/USDT:USDT"            # OKX 永续合约

REFRESH_MS = 2500                   # 2.5秒刷新



# 风控参数

CIRCUIT_BREAKER_PCT = 0.003         # 0.3% 熔断阈值

FINAL_CONF_THRES = 80                # 最终信心分门槛（满分100）

BREAKOUT_CONF_THRES = 75             # 爆发行情下的降低门槛

MIN_ATR_PCT = 0.0025                 # 最小波动率（ATR百分比）

MIN_SCORE_GAP = 10                    # 多空最小分差

VOLUME_RATIO_MIN = 1.2                # 最小成交量放大倍数

MODEL_DIRECTION_MIN = 55              # 模型概率方向确认门槛（%）

MODEL_GAP_MIN = 5                      # 模型概率最小差值

RR = 2.0                               # 风险收益比

MIN_SL_PCT = 0.0015                    # 最小止损距离（防止过小）

MIN_TREND_STRENGTH = 15                # 最小趋势强度

STRONG_TREND_THRESH = 35               # 强趋势阈值

COOLDOWN_CANDLES = 2                    # 冷却K线数量

CANDLE_5M_MS = 5 * 60 * 1000            # 5分钟毫秒数

BREAKOUT_VOL_RATIO = 1.5                # 爆发成交量倍数

BREAKOUT_ADX_MIN = 25                    # 爆发ADX最小值

MAX_LOG_ENTRIES = 200                    # 日志最大条数

CIRCUIT_BREAKER_RECOVERY_CHECKS = 10     # 熔断恢复所需连续稳定次数



# 权重配置

TREND_WEIGHT = 0.5

MOMENTUM_WEIGHT = 0.3

MODEL_WEIGHT = 0.2



# 自动刷新

st_autorefresh(interval=REFRESH_MS, key="bidirectional_ai_final")



# ================================

# 2. 辅助函数：获取模型特征名称

# ================================

def get_feature_names(model):

    """获取模型的特征名称列表，支持多种库"""

    if hasattr(model, "feature_names_in_"):

        return list(model.feature_names_in_)

    elif hasattr(model, "get_booster") and hasattr(model.get_booster(), "feature_names"):

        return model.get_booster().feature_names

    elif hasattr(model, "feature_name"):

        return model.feature_name()

    else:

        return None



# ================================

# 3. 初始化系统（交易所、模型、scaler）

# ================================

@st.cache_resource

def init_system():

    exch = ccxt.okx({

        "enableRateLimit": True,

        "options": {"defaultType": "swap"}

    })



    # 加载模型

    model_path = "eth_ai_model.pkl"

    if not os.path.exists(model_path):

        st.sidebar.error(f"❌ 未找到模型文件 {model_path}，请上传至应用根目录。")

        st.stop()

    model = joblib.load(model_path)



    # 识别模型类型

    model_type = type(model).__name__

    st.session_state.model_type = model_type

    st.sidebar.info(f"🤖 模型类型：{model_type}")



    # 需要标准化的模型列表（可根据实际情况扩展）

    NEEDS_SCALER = ('SVC', 'LogisticRegression', 'MLPClassifier', 'GaussianNB', 'LinearSVC')

    needs_scaler = model_type in NEEDS_SCALER



    scaler = None

    if needs_scaler:

        scaler_path = "eth_scaler.pkl"

        if os.path.exists(scaler_path):

            scaler = joblib.load(scaler_path)

            st.sidebar.info("✅ 已加载标准化器")

        else:

            st.sidebar.error(f"模型 {model_type} 需要标准化器，但未找到 {scaler_path}。请上传或重新训练。")

            st.stop()

    else:

        st.sidebar.info("✅ 树模型无需标准化，忽略 scaler")



    return exch, model, scaler



exchange, model, scaler = init_system()



# ================================

# 4. 状态管理（使用 deque 自动管理）

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

        'total_trades': 0,

        'wins': 0,

        'losses': 0,

        'total_pnl': 0.0,

        'max_consecutive_losses': 0,

        'current_consecutive_losses': 0,

        'last_update': None

    }

# 实时验证历史

if 'prediction_history' not in st.session_state:

    st.session_state.prediction_history = deque(maxlen=100)



# ================================

# 5. 数据获取（并发 + 缓存）

# ================================

@st.cache_data(ttl=5, show_spinner=False)

def fetch_ohlcv_cached(timeframe, limit=200):

    """获取指定周期K线（缓存5秒）"""

    return exchange.fetch_ohlcv(SYMBOL, timeframe, limit=limit)



def get_multi_timeframe_data():

    """并发获取5m/15m/1h数据"""

    with concurrent.futures.ThreadPoolExecutor() as executor:

        future_5m = executor.submit(fetch_ohlcv_cached, "5m", 200)

        future_15m = executor.submit(fetch_ohlcv_cached, "15m", 100)

        future_1h = executor.submit(fetch_ohlcv_cached, "1h", 100)



        ohlcv_5m = future_5m.result()

        ohlcv_15m = future_15m.result()

        ohlcv_1h = future_1h.result()



    if not ohlcv_5m:

        st.error("无法获取5m数据，请检查网络或交易所状态。")

        st.stop()

    df_5m = pd.DataFrame(ohlcv_5m, columns=["timestamp", "open", "high", "low", "close", "volume"])



    if not ohlcv_15m:

        st.error("无法获取15m数据，请检查网络或交易所状态。")

        st.stop()

    df_15m = pd.DataFrame(ohlcv_15m, columns=["timestamp", "open", "high", "low", "close", "volume"])



    if not ohlcv_1h:

        st.error("无法获取1h数据，请检查网络或交易所状态。")

        st.stop()

    df_1h = pd.DataFrame(ohlcv_1h, columns=["timestamp", "open", "high", "low", "close", "volume"])



    # 统一处理缺失值

    for df in [df_5m, df_15m, df_1h]:

        df.replace([None], np.nan, inplace=True)



    return df_5m, df_15m, df_1h



# ================================

# 6. 指标计算（与训练代码完全对齐）

# ================================

def compute_features(df_5m, df_15m, df_1h):

    """计算所有指标，返回带特征的DataFrame和最新特征向量"""

    if len(df_5m) < 30:

        st.error("5m数据不足30根，无法可靠计算。")

        st.stop()



    # 时间索引处理

    for df in [df_5m, df_15m, df_1h]:

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        df.set_index('timestamp', inplace=True)

        df.sort_index(inplace=True)

        if df.index.duplicated().any():

            df = df[~df.index.duplicated(keep='last')]



    # ----- 5m 指标 -----

    df_5m["rsi"] = ta.rsi(df_5m["close"], length=14)

    df_5m["ma20"] = ta.sma(df_5m["close"], length=20)

    df_5m["ma60"] = ta.sma(df_5m["close"], length=60)



    # MACD（使用列名）

    macd = ta.macd(df_5m["close"], fast=10, slow=22, signal=8)

    if macd is not None:

        df_5m["macd"] = macd['MACD_10_22_8']

        df_5m["macd_signal"] = macd['MACDs_10_22_8']

    else:

        df_5m["macd"] = 0

        df_5m["macd_signal"] = 0



    # ATR

    df_5m["atr"] = ta.atr(df_5m["high"], df_5m["low"], df_5m["close"], length=14)

    df_5m["atr_pct"] = df_5m["atr"] / df_5m["close"]



    # ADX（使用列名）

    adx_df = ta.adx(df_5m["high"], df_5m["low"], df_5m["close"], length=14)

    if adx_df is not None:

        df_5m["adx"] = adx_df['ADX_14']

    else:

        df_5m["adx"] = 20



    # 动量核所需

    df_5m["ema5"] = ta.ema(df_5m["close"], length=5)

    df_5m["ema20"] = ta.ema(df_5m["close"], length=20)

    vwap = ta.vwap(df_5m["high"], df_5m["low"], df_5m["close"], df_5m["volume"])

    df_5m["VWAP"] = vwap

    df_5m["volume_ma20"] = ta.sma(df_5m["volume"], length=20)

    df_5m["atr_ma20"] = df_5m["atr"].rolling(20).mean()

    df_5m["atr_surge"] = (df_5m["atr"] > df_5m["atr_ma20"] * 1.2).fillna(False)



    # ----- 15m 指标 -----

    df_15m["ema200"] = ta.ema(df_15m["close"], length=200)

    adx_15_df = ta.adx(df_15m["high"], df_15m["low"], df_15m["close"], length=14)

    if adx_15_df is not None:

        df_15m["adx"] = adx_15_df['ADX_14']

    else:

        df_15m["adx"] = 20

    vwap_15 = ta.vwap(df_15m["high"], df_15m["low"], df_15m["close"], df_15m["volume"])

    df_15m["VWAP"] = vwap_15

    df_15m["hh"] = df_15m["high"].rolling(20).max()

    df_15m["ll"] = df_15m["low"].rolling(20).min()

    df_15m["ema200_slope"] = df_15m["ema200"] - df_15m["ema200"].shift(5)



    # ----- 1h 指标 -----

    df_1h["ema200"] = ta.ema(df_1h["close"], length=200)

    adx_1h_df = ta.adx(df_1h["high"], df_1h["low"], df_1h["close"], length=14)

    if adx_1h_df is not None:

        df_1h["adx"] = adx_1h_df['ADX_14']

    else:

        df_1h["adx"] = 20

    vwap_1h = ta.vwap(df_1h["high"], df_1h["low"], df_1h["close"], df_1h["volume"])

    df_1h["VWAP"] = vwap_1h

    df_1h["hh"] = df_1h["high"].rolling(20).max()

    df_1h["ll"] = df_1h["low"].rolling(20).min()

    df_1h["ema200_slope"] = df_1h["ema200"] - df_1h["ema200"].shift(3)



    # 填充缺失值

    default_close_5m = df_5m['close'].iloc[-1] if len(df_5m) > 0 else 0

    default_close_15m = df_15m['close'].iloc[-1] if len(df_15m) > 0 else 0

    default_close_1h = df_1h['close'].iloc[-1] if len(df_1h) > 0 else 0



    default_values_5m = {

        'rsi': 50.0, 'adx': 20.0, 'macd': 0.0, 'macd_signal': 0.0, 'atr_pct': 0.0,

        'ma20': default_close_5m, 'ma60': default_close_5m, 'ema5': default_close_5m,

        'ema20': default_close_5m, 'VWAP': default_close_5m, 'volume_ma20': 0,

        'atr_ma20': 0, 'atr': 0,

    }

    default_values_15m = {

        'adx': 20.0, 'VWAP': default_close_15m, 'hh': default_close_15m,

        'll': default_close_15m, 'ema200': default_close_15m, 'ema200_slope': 0,

    }

    default_values_1h = {

        'adx': 20.0, 'VWAP': default_close_1h, 'hh': default_close_1h,

        'll': default_close_1h, 'ema200': default_close_1h, 'ema200_slope': 0,

    }



    df_5m = df_5m.ffill().fillna(value=default_values_5m).infer_objects(copy=False)

    df_15m = df_15m.ffill().fillna(value=default_values_15m).infer_objects(copy=False)

    df_1h = df_1h.ffill().fillna(value=default_values_1h).infer_objects(copy=False)

    df_5m["atr_surge"] = df_5m["atr_surge"].fillna(False)



    # 提取最新特征用于模型预测

    model_feature_names = get_feature_names(model)

    if model_feature_names is not None:

        missing = set(model_feature_names) - set(df_5m.columns)

        for col in missing:

            df_5m[col] = 0

        latest_feat = df_5m[model_feature_names].iloc[-1:].copy()

    else:

        feat_cols = ['rsi', 'ma20', 'ma60', 'macd', 'macd_signal', 'atr_pct', 'adx']

        latest_feat = df_5m[feat_cols].iloc[-1:].copy()



    return df_5m, df_15m, df_1h, latest_feat



# ================================

# 7. 评分函数（趋势/动量/模型）

# ================================

def compute_trend_score(df_15m, df_1h):

    """趋势核评分"""

    c15 = df_15m.iloc[-1]

    c1h = df_1h.iloc[-1]



    long_score = short_score = 0



    # EMA200 + 斜率

    if pd.notna(c15['close']) and pd.notna(c15['ema200']) and pd.notna(c15['ema200_slope']):

        if c15['close'] > c15['ema200'] and c15['ema200_slope'] > 0:

            long_score += 15

        elif c15['close'] < c15['ema200'] and c15['ema200_slope'] < 0:

            short_score += 15



    if pd.notna(c1h['close']) and pd.notna(c1h['ema200']) and pd.notna(c1h['ema200_slope']):

        if c1h['close'] > c1h['ema200'] and c1h['ema200_slope'] > 0:

            long_score += 15

        elif c1h['close'] < c1h['ema200'] and c1h['ema200_slope'] < 0:

            short_score += 15



    # VWAP

    if pd.notna(c15['close']) and pd.notna(c15['VWAP']):

        if c15['close'] > c15['VWAP']:

            long_score += 10

        else:

            short_score += 10



    if pd.notna(c1h['close']) and pd.notna(c1h['VWAP']):

        if c1h['close'] > c1h['VWAP']:

            long_score += 10

        else:

            short_score += 10



    # 价格结构

    range_15 = c15['hh'] - c15['ll'] if pd.notna(c15['hh']) and pd.notna(c15['ll']) else 0

    if range_15 > 0 and pd.notna(c15['close']):

        if (c15['close'] - c15['ll']) / range_15 > 0.5:

            long_score += 10

        else:

            short_score += 10



    range_1h = c1h['hh'] - c1h['ll'] if pd.notna(c1h['hh']) and pd.notna(c1h['ll']) else 0

    if range_1h > 0 and pd.notna(c1h['close']):

        if (c1h['close'] - c1h['ll']) / range_1h > 0.5:

            long_score += 10

        else:

            short_score += 10



    raw_long = min(long_score, 100)

    raw_short = min(short_score, 100)



    # ADX倍率

    if pd.notna(c15['adx']) and pd.notna(c1h['adx']) and c15['adx'] > 25 and c1h['adx'] > 25:

        long_score = int(long_score * 1.15)

        short_score = int(short_score * 1.15)



    return min(long_score, 100), min(short_score, 100), raw_long, raw_short



def compute_momentum_score(df_5m):

    """动量核评分"""

    c = df_5m.iloc[-1]

    long_score = short_score = 0



    # EMA5 vs EMA20

    if pd.notna(c['ema5']) and pd.notna(c['ema20']):

        if c['ema5'] > c['ema20']:

            long_score += 30

        else:

            short_score += 30



    # 价格 vs VWAP

    if pd.notna(c['close']) and pd.notna(c['VWAP']):

        if c['close'] > c['VWAP']:

            long_score += 20

        else:

            short_score += 20



    # 成交量放大

    if pd.notna(c['volume']) and pd.notna(c['volume_ma20']) and c['volume_ma20'] > 0:

        vol_ratio = c['volume'] / c['volume_ma20']

        if vol_ratio > VOLUME_RATIO_MIN:

            if c['close'] > c['VWAP']:

                long_score += 25

            else:

                short_score += 25



    # ATR扩张

    if pd.notna(c['atr_surge']) and c['atr_surge']:

        if pd.notna(c['ema5']) and pd.notna(c['ema20']) and c['ema5'] > c['ema20']:

            long_score += 25

        else:

            short_score += 25



    return min(long_score, 100), min(short_score, 100)



def compute_model_prob(df_5m, latest_feat, trend_long, trend_short):

    """模型概率预测"""

    if model is None:

        return 50, 50



    feat_for_model = latest_feat.copy()

    model_feature_names = get_feature_names(model)

    if model_feature_names is not None:

        feat_for_model = feat_for_model.reindex(columns=model_feature_names, fill_value=0)



    try:

        if scaler is not None:

            feat_scaled = scaler.transform(feat_for_model)

        else:

            feat_scaled = feat_for_model.values



        proba = model.predict_proba(feat_scaled)[0]



        # 根据 classes_ 确定多头/空头索引

        if hasattr(model, 'classes_') and len(model.classes_) == 2:

            # 假设训练时多头标记为1

            if model.classes_[1] == 1:

                prob_long = proba[1] * 100

                prob_short = proba[0] * 100

            else:

                prob_long = proba[0] * 100

                prob_short = proba[1] * 100

        else:

            prob_long = proba[1] * 100 if len(proba) > 1 else 50

            prob_short = proba[0] * 100 if len(proba) > 0 else 50



        # 异常回退

        if prob_long == 0 and prob_short == 0:

            total_trend = trend_long + trend_short

            if total_trend > 0:

                prob_long = (trend_long / total_trend) * 100

                prob_short = (trend_short / total_trend) * 100

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



# ================================

# 8. 辅助检测函数

# ================================

def detect_momentum_decay(df_5m):

    """动量衰减：MACD连续3根下降"""

    if len(df_5m) < 4:

        return False

    macd_vals = df_5m['macd'].iloc[-4:].values

    if any(pd.isna(v) for v in macd_vals):

        return False

    return (macd_vals[3] < macd_vals[2] and

            macd_vals[2] < macd_vals[1] and

            macd_vals[1] < macd_vals[0])



def detect_breakout(df_5m):

    """爆发结构检测"""

    c = df_5m.iloc[-1]

    if pd.isna(c['volume']) or pd.isna(c['volume_ma20']) or c['volume_ma20'] <= 0:

        vol_ratio = 0

    else:

        vol_ratio = c['volume'] / c['volume_ma20']

    atr_surge = pd.notna(c['atr_surge']) and c['atr_surge']

    adx_ok = pd.notna(c['adx']) and c['adx'] > BREAKOUT_ADX_MIN

    return (atr_surge and vol_ratio > BREAKOUT_VOL_RATIO and adx_ok)



# ================================

# 9. 盈亏统计

# ================================

def check_position_exit(position, current_price):

    if position is None:

        return None

    side, entry, sl, tp = position['side'], position['entry'], position['sl'], position['tp']

    if side == 'LONG':

        if current_price <= sl:

            return (sl - entry) / entry, '止损'

        elif current_price >= tp:

            return (tp - entry) / entry, '止盈'

    else:

        if current_price >= sl:

            return (entry - sl) / entry, '止损'

        elif current_price <= tp:

            return (entry - tp) / entry, '止盈'

    return None



def update_stats(pnl):

    stats = st.session_state.stats

    stats['total_trades'] += 1

    stats['total_pnl'] += pnl * 100

    if pnl > 0:

        stats['wins'] += 1

        stats['current_consecutive_losses'] = 0

    else:

        stats['losses'] += 1

        stats['current_consecutive_losses'] += 1

        if stats['current_consecutive_losses'] > stats['max_consecutive_losses']:

            stats['max_consecutive_losses'] = stats['current_consecutive_losses']

    stats['last_update'] = datetime.now()



# ================================

# 10. 侧边栏面板

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



    feat_names = get_feature_names(model)

    if feat_names:

        st.write(f"模型特征数: {len(feat_names)}")

        with st.expander("查看特征列表"):

            st.write(", ".join(feat_names))



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

    checked_preds = [p for p in st.session_state.prediction_history if p['checked']][-50:]

    if checked_preds:

        correct = sum(1 for p in checked_preds if (p['prob_up'] >= 50 and p['actual_up'] == 1) or (p['prob_up'] < 50 and p['actual_up'] == 0))

        accuracy = correct / len(checked_preds) * 100

        st.metric("最近准确率", f"{accuracy:.1f}%", delta=f"{len(checked_preds)}次样本")

        high_conf = [p for p in checked_preds if abs(p['prob_up'] - 50) > 30]

        if high_conf:

            high_correct = sum(1 for p in high_conf if (p['prob_up'] >= 80 and p['actual_up'] == 1) or (p['prob_up'] <= 20 and p['actual_up'] == 0))

            high_acc = high_correct / len(high_conf) * 100

            st.caption(f"高信心区间准确率: {high_acc:.1f}% ({len(high_conf)}次)")

    else:

        st.info("等待验证数据积累...")



    if st.button("🔌 重置熔断（注意：冷却状态也会重置）"):

        st.session_state.system_halted = False

        st.session_state.last_price = 0

        st.session_state.active_signal = None

        st.session_state.last_signal_candle = None

        st.session_state.position = None

        st.session_state.price_changes = []

        st.success("熔断已重置，冷却已清除")



# ================================

# 11. 主界面逻辑

# ================================

st.title("⚖️ ETH 100x 终极双向评分 AI 决策终端 (趋势+动量+模型)")



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

        st.error("🚨 触发熔断保护！价格剧烈波动。")

    else:

        # 检查持仓退出

        if st.session_state.position:

            exit_info = check_position_exit(st.session_state.position, current_price)

            if exit_info:

                pnl_percent, reason = exit_info

                net_pnl = pnl_percent - 0.002  # 扣除手续费滑点

                update_stats(net_pnl)

                pos = st.session_state.position

                st.session_state.signal_log.append({

                    "时间": datetime.now().strftime("%H:%M:%S"),

                    "方向": pos['side'],

                    "入场价": pos['entry'],

                    "出场价": current_price,

                    "盈亏%": f"{net_pnl*100:.2f}",

                    "原因": reason

                })

                st.session_state.position = None



        # 获取数据并计算指标

        df_5m, df_15m, df_1h = get_multi_timeframe_data()

        df_5m, df_15m, df_1h, latest_feat = compute_features(df_5m, df_15m, df_1h)



        # 对齐15m/1h到当前5m时间戳

        current_5m_time = df_5m.index[-1]

        c15_aligned = df_15m.asof(current_5m_time)

        c1h_aligned = df_1h.asof(current_5m_time)

        temp_15m = pd.DataFrame([c15_aligned], index=[current_5m_time])

        temp_1h = pd.DataFrame([c1h_aligned], index=[current_5m_time])



        trend_long, trend_short, raw_trend_long, raw_trend_short = compute_trend_score(temp_15m, temp_1h)

        mom_long, mom_short = compute_momentum_score(df_5m)

        prob_long, prob_short = compute_model_prob(df_5m, latest_feat, trend_long, trend_short)



        # 归一化

        trend_long_norm = trend_long / 100.0

        trend_short_norm = trend_short / 100.0

        mom_long_norm = mom_long / 100.0

        mom_short_norm = mom_short / 100.0

        prob_long_norm = prob_long / 100.0

        prob_short_norm = prob_short / 100.0



        final_long = (trend_long_norm * TREND_WEIGHT + mom_long_norm * MOMENTUM_WEIGHT + prob_long_norm * MODEL_WEIGHT) * 100

        final_short = (trend_short_norm * TREND_WEIGHT + mom_short_norm * MOMENTUM_WEIGHT + prob_short_norm * MODEL_WEIGHT) * 100



        c5 = df_5m.iloc[-1]

        c15 = c15_aligned

        c1h = c1h_aligned



        # 过滤条件计算

        vol_ratio = c5['volume'] / c5['volume_ma20'] if pd.notna(c5['volume']) and pd.notna(c5['volume_ma20']) and c5['volume_ma20'] > 0 else 0

        atr_pct = c5['atr_pct'] if pd.notna(c5['atr_pct']) else 0

        trend_strength_raw = abs(raw_trend_long - raw_trend_short)

        score_gap = abs(final_long - final_short)

        model_gap = abs(prob_long - prob_short)



        adx_15 = c15['adx'] if pd.notna(c15['adx']) else 0

        adx_1h = c1h['adx'] if pd.notna(c1h['adx']) else 0

        if adx_15 < 20 and adx_1h < 20:

            market_state = "RANGE"

        elif trend_strength_raw > STRONG_TREND_THRESH:

            market_state = "STRONG_TREND"

        else:

            market_state = "NORMAL"



        momentum_decay = detect_momentum_decay(df_5m)

        is_breakout = detect_breakout(df_5m)



        current_candle_time = df_5m.index[-1].value / 10**6



        # 冷却检查

        if st.session_state.last_signal_candle is not None:

            candles_since_last = (current_candle_time - st.session_state.last_signal_candle) / CANDLE_5M_MS

            cooling = candles_since_last < COOLDOWN_CANDLES

        else:

            cooling = False



        direction = None

        final_score = 0

        filter_reasons = []



        if cooling:

            filter_reasons.append(f"冷却中，还需 {COOLDOWN_CANDLES - candles_since_last:.1f} 根K线")

        if atr_pct < MIN_ATR_PCT:

            filter_reasons.append(f"波动率过低 (ATR% = {atr_pct:.3%})")

        if vol_ratio < VOLUME_RATIO_MIN:

            filter_reasons.append(f"成交量不足 (倍数 {vol_ratio:.2f})")

        if trend_strength_raw < MIN_TREND_STRENGTH:

            filter_reasons.append(f"趋势强度过弱 ({trend_strength_raw} < {MIN_TREND_STRENGTH})")

        if score_gap < MIN_SCORE_GAP:

            filter_reasons.append(f"多空信心分差过小 ({score_gap:.1f} < {MIN_SCORE_GAP})")

        if market_state == "RANGE":

            filter_reasons.append("市场处于震荡期 (双ADX<20)")

        if momentum_decay:

            filter_reasons.append("动量衰减 (MACD连续下降)")



        if not filter_reasons:

            current_thres = BREAKOUT_CONF_THRES if is_breakout else FINAL_CONF_THRES

            if final_long > final_short and final_long >= current_thres:

                candidate_dir = "LONG"

                candidate_score = final_long

            elif final_short > final_long and final_short >= current_thres:

                candidate_dir = "SHORT"

                candidate_score = final_short

            else:

                candidate_dir = None



            # 模型方向确认

            if candidate_dir == "LONG" and prob_long < MODEL_DIRECTION_MIN:

                filter_reasons.append(f"模型多头概率不足 ({prob_long:.1f}% < {MODEL_DIRECTION_MIN}%)")

                candidate_dir = None

            elif candidate_dir == "SHORT" and prob_short < MODEL_DIRECTION_MIN:

                filter_reasons.append(f"模型空头概率不足 ({prob_short:.1f}% < {MODEL_DIRECTION_MIN}%)")

                candidate_dir = None



            if candidate_dir and model_gap < MODEL_GAP_MIN:

                filter_reasons.append(f"模型概率差过小 ({model_gap:.1f} < {MODEL_GAP_MIN})")

                candidate_dir = None



            # 趋势同步锁

            if candidate_dir == "LONG":

                if not (pd.notna(c15['close']) and pd.notna(c15['ema200']) and

                        pd.notna(c1h['close']) and pd.notna(c1h['ema200']) and

                        c15['close'] > c15['ema200'] and c1h['close'] > c1h['ema200']):

                    filter_reasons.append("大周期未支持多头趋势 (15m或1h价格低于EMA200)")

                    candidate_dir = None

            elif candidate_dir == "SHORT":

                if not (pd.notna(c15['close']) and pd.notna(c15['ema200']) and

                        pd.notna(c1h['close']) and pd.notna(c1h['ema200']) and

                        c15['close'] < c15['ema200'] and c1h['close'] < c1h['ema200']):

                    filter_reasons.append("大周期未支持空头趋势 (15m或1h价格高于EMA200)")

                    candidate_dir = None



            if candidate_dir:

                direction = candidate_dir

                final_score = candidate_score



        # 更新信号锁

        if direction and st.session_state.last_signal_candle != current_candle_time:

            st.session_state.active_signal = direction

            st.session_state.last_signal_candle = current_candle_time

        elif not direction and st.session_state.last_signal_candle != current_candle_time:

            st.session_state.active_signal = None



        # 记录预测用于实时验证

        st.session_state.prediction_history.append({

            'time': current_5m_time,

            'prob_up': prob_long,

            'actual_up': None,

            'checked': False,

            'price': current_price

        })



        # 检查历史预测

        for pred in st.session_state.prediction_history:

            if not pred['checked']:

                try:

                    idx = df_5m.index.get_loc(pred['time'])

                    if idx + 3 < len(df_5m):

                        future_close = df_5m.iloc[idx + 3]['close']

                        entry_close = df_5m.iloc[idx]['close']

                        pred['actual_up'] = 1 if future_close > entry_close * 1.002 else 0

                        pred['checked'] = True

                except KeyError:

                    continue



        # UI 展示

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



        if filter_reasons:

            st.warning("⛔ 当前不满足信号条件: " + " | ".join(filter_reasons))

        else:

            st.success("✅ 所有基础过滤条件通过，等待高置信度信号...")



        st.markdown("---")



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



            st.session_state.position = {

                'side': side,

                'entry': current_price,

                'sl': sl,

                'tp': tp,

                'entry_time': datetime.now(),

                'score': final_score

            }



            sc1, sc2, sc3 = st.columns(3)

            sc1.write(f"**入场价:** {current_price}")

            sc2.write(f"**止损 (SL):** {round(sl, 2)}")

            sc3.write(f"**止盈 (TP):** {round(tp, 2)}")

        else:

            st.info("🔎 当前无符合要求的信号")



        # K线图

        fig = go.Figure(data=[go.Candlestick(

            x=df_5m.index,

            open=df_5m['open'], high=df_5m['high'], low=df_5m['low'], close=df_5m['close']

        )])

        fig.update_layout(height=450, template="plotly_dark", xaxis_rangeslider_visible=False)

        st.plotly_chart(fig, use_conta
