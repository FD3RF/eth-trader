# -*- coding: utf-8 -*-
"""
ETH 100x 智能做单策略终端 v7.1（最终优化版）
特性：
- 激进入场参数：信心阈值 70，模型权重 70%，大幅放宽过滤
- 默认杠杆 30 倍，兼顾胜率与机会
- UI 卡片化、紧凑布局、超大信号卡
- 完整策略模块与风控
"""

import streamlit as st
import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from streamlit_autorefresh import st_autorefresh
import joblib
import os
from datetime import datetime, timedelta, timezone
from collections import deque
import random
from scipy import stats

# ================================
# 1. 全局配置（激进参数）
# ================================
st.set_page_config(layout="wide", page_title="ETH 100x 做单策略 v7.1", page_icon="📈", initial_sidebar_state="collapsed")

# 自定义CSS（卡片风格、颜色、布局）
st.markdown("""
<style>
    /* 全局字体与背景 */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    /* 卡片通用样式 */
    .card {
        background: #1e1e1e;
        border-radius: 20px;
        padding: 20px 18px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.5);
        transition: transform 0.2s, box-shadow 0.2s;
        border: 1px solid #333;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 28px rgba(0,255,157,0.15);
    }
    .card-label {
        font-size: 0.85rem;
        color: #aaa;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    .card-value {
        font-size: 2.2rem;
        font-weight: 700;
        line-height: 1.2;
        text-align: right;
    }
    .card-value-small {
        font-size: 1.4rem;
        font-weight: 600;
    }
    /* 多空颜色 */
    .bull { color: #00ff9d; }
    .bear { color: #ff3b5c; }
    .neutral { color: #9aa0a6; }

    /* 超大信号卡 */
    .signal-card {
        background: linear-gradient(145deg, #252525 0%, #1a1a1a 100%);
        border-radius: 24px;
        padding: 20px 30px;
        margin: 20px 0 10px 0;
        border-left: 10px solid;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 8px 20px rgba(0,0,0,0.6);
    }
    .signal-card.bull { border-left-color: #00ff9d; }
    .signal-card.bear { border-left-color: #ff3b5c; }
    .signal-card.neutral { border-left-color: #9aa0a6; }

    .signal-label {
        font-size: 1.2rem;
        color: #ccc;
        font-weight: 500;
    }
    .signal-value {
        font-size: 3.2rem;
        font-weight: 800;
    }

    /* 杠杆滑块卡片内调整 */
    .leverage-card {
        background: #252525;
        padding: 16px 18px;
    }
    .stSlider > div > div > div {
        background: #00ff9d !important;
    }

    /* 侧边栏折叠 */
    .css-1d391kg { padding-top: 1rem; }

    /* K线图容器 */
    .chart-container {
        border-radius: 20px;
        overflow: hidden;
        border: 1px solid #333;
        margin-top: 10px;
    }

    /* 最后更新时间小标签 */
    .update-time {
        text-align: right;
        color: #666;
        font-size: 0.75rem;
        margin-top: 4px;
        font-family: monospace;
    }

    /* 紧凑布局 */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1300px;
    }
</style>
""", unsafe_allow_html=True)

SYMBOL = "ETH/USDT:USDT"
REFRESH_MS = 2500

# 激进入场阈值
FINAL_CONF_THRES = 70           # 原75，降低到70增加信号

# 固定止损止盈
STOP_LOSS_PCT = 0.01
TAKE_PROFIT_PCT = 0.02

# 风险参数
RISK_PER_TRADE = 0.01
ACCOUNT_BALANCE = 10000

# 合约参数
CONTRACT_SIZE = 0.01

# 滑点与手续费
DEFAULT_ENTRY_SLIPPAGE = 0.0003
DEFAULT_EXIT_SLIPPAGE = 0.0003
TAKER_FEE = 0.0005

# 维持保证金率
MAINTENANCE_MARGIN_RATE = 0.004

# 过滤参数（大幅放宽，几乎不设限）
MIN_ATR_PCT = 0.0001            # 允许极低波动
MAX_ATR_PCT = 0.1               # 允许极高波动
VOLUME_RATIO_MIN = 0.8          # 成交量只需达到0.8倍均量
MIN_TREND_STRENGTH = 0          # 取消趋势强度要求
MIN_SCORE_GAP = 0                # 取消多空分差要求
MODEL_DIRECTION_MIN = 0          # 取消模型方向最低要求
COOLDOWN_CANDLES = 0             # 取消冷却期

# 权重（模型主导）
TREND_WEIGHT = 0.1
MOMENTUM_WEIGHT = 0.2
MODEL_WEIGHT = 0.7

# 资金费结算时间
FUNDING_HOURS = [0, 8, 16]

# 标记价格计算参数
MARK_PRICE_EMA_PERIOD = 10

# 风控默认值
DEFAULT_CONSECUTIVE_LOSS_LIMIT = 3
DEFAULT_DAILY_LOSS_LIMIT = 500
DEFAULT_DAILY_RISK_PCT = 0.025

# 出场策略开关（默认开启）
USE_ATR_STOP = True
USE_EMA_REVERSE = True
USE_SCORE_REVERSE = True

st_autorefresh(interval=REFRESH_MS, key="quant_v71")

# ================================
# 2. 初始化系统与状态
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

    return exch, model, scaler

exchange, model, scaler = init_system()

def init_session_state():
    """集中初始化所有 session_state 变量"""
    today = datetime.now().date()
    defaults = {
        'last_price': 0,
        'system_halted': False,
        'price_changes': [],
        'signal_log': deque(maxlen=500),
        'active_signal': None,
        'last_signal_candle': None,
        'position': None,
        'stats': {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'current_consecutive_losses': 0,
            'max_consecutive_losses': 0,
            'last_reset_day': today,
            'daily_pnl': 0.0,
            'daily_trades': 0,
        },
        'prediction_history': deque(maxlen=200),
        'last_recorded_candle': None,
        'last_5m_candle_time': None,
        'last_date_recorded': datetime.now().date() - timedelta(days=1),
        'equity_history': [],
        'initial_equity': ACCOUNT_BALANCE,
        'current_equity': ACCOUNT_BALANCE,
        'daily_equity': [],
        'last_funding_check': datetime.now(timezone.utc),
        'latest_funding_rate': 0.0,
        'pnl_list': [],
        'entry_slippage': DEFAULT_ENTRY_SLIPPAGE,
        'exit_slippage': DEFAULT_EXIT_SLIPPAGE,
        'consecutive_loss_limit': DEFAULT_CONSECUTIVE_LOSS_LIMIT,
        'daily_loss_limit': DEFAULT_DAILY_LOSS_LIMIT,
        'daily_risk_pct': DEFAULT_DAILY_RISK_PCT,
        'today_pnl': 0.0,
        'daily_peak_equity': ACCOUNT_BALANCE,
        'use_atr_stop': USE_ATR_STOP,
        'use_ema_reverse': USE_EMA_REVERSE,
        'use_score_reverse': USE_SCORE_REVERSE,
        'use_dynamic_position': False,
        'leverage': 30,                     # 默认30倍杠杆
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ================================
# 3. 辅助函数
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

def get_mark_price(df_5m):
    if len(df_5m) < MARK_PRICE_EMA_PERIOD:
        return df_5m['close'].iloc[-1]
    ema = ta.ema(df_5m['close'], length=MARK_PRICE_EMA_PERIOD)
    return ema.iloc[-1] if not pd.isna(ema.iloc[-1]) else df_5m['close'].iloc[-1]

# ================================
# 4. 数据获取
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
    ohlcv_5m = fetch_safe("5m", 200)
    ohlcv_15m = fetch_safe("15m", 100)
    ohlcv_1h = fetch_safe("1h", 100)

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
# 5. 指标计算
# ================================
def compute_features(df_5m, df_15m, df_1h):
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

    # 5m指标
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
    df_5m["adx"] = adx_df['ADX_14'].fillna(20) if adx_df is not None else 20
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

    # 15m指标
    ema200_15 = ta.ema(df_15m["close"], length=200)
    df_15m["ema200"] = ema200_15.ffill().fillna(df_15m["close"]) if ema200_15 is not None else df_15m["close"]
    adx_15_df = ta.adx(df_15m["high"], df_15m["low"], df_15m["close"], length=14)
    df_15m["adx"] = adx_15_df['ADX_14'].fillna(20) if adx_15_df is not None else 20
    vwap_15 = ta.vwap(df_15m["high"], df_15m["low"], df_15m["close"], df_15m["volume"])
    df_15m["VWAP"] = vwap_15.ffill().fillna(df_15m["close"]) if vwap_15 is not None else df_15m["close"]
    df_15m["hh"] = df_15m["high"].rolling(20).max().ffill().fillna(df_15m["high"])
    df_15m["ll"] = df_15m["low"].rolling(20).min().ffill().fillna(df_15m["low"])
    df_15m["ema200_slope"] = (df_15m["ema200"] - df_15m["ema200"].shift(5)).fillna(0)

    # 1h指标
    ema200_1h = ta.ema(df_1h["close"], length=200)
    df_1h["ema200"] = ema200_1h.ffill().fillna(df_1h["close"]) if ema200_1h is not None else df_1h["close"]
    adx_1h_df = ta.adx(df_1h["high"], df_1h["low"], df_1h["close"], length=14)
    df_1h["adx"] = adx_1h_df['ADX_14'].fillna(20) if adx_1h_df is not None else 20
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
# 6. 评分函数
# ================================
def compute_trend_score(df_15m, df_1h):
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

# ================================
# 7. 四大策略模块
# ================================

class EntryEngine:
    def __init__(self, config):
        self.config = config

    def generate_signal(self, df_5m, df_15m, df_1h, latest_feat, model, scaler, last_signal_candle=None, current_candle_time=None):
        trend_long, trend_short, raw_trend_long, raw_trend_short = compute_trend_score(df_15m, df_1h)
        mom_long, mom_short = compute_momentum_score(df_5m)
        prob_long, prob_short = compute_model_prob(model, scaler, latest_feat, trend_long, trend_short)

        final_long = (trend_long * self.config['trend_weight'] + mom_long * self.config['momentum_weight'] + prob_long * self.config['model_weight']) / 100
        final_short = (trend_short * self.config['trend_weight'] + mom_short * self.config['momentum_weight'] + prob_short * self.config['model_weight']) / 100

        c5 = df_5m.iloc[-1]
        c15 = df_15m.iloc[-1] if len(df_15m) > 0 else None
        c1h = df_1h.iloc[-1] if len(df_1h) > 0 else None

        filter_reasons = []
        trigger_details = []

        atr_pct = c5.get('atr_pct', 0)
        if pd.isna(atr_pct):
            atr_pct = 0
        if atr_pct < self.config['min_atr_pct']:
            filter_reasons.append(f"波动率过低 ({atr_pct:.3%})")
        if atr_pct > self.config['max_atr_pct']:
            filter_reasons.append(f"波动率过高 ({atr_pct:.3%})")

        vol_ratio = c5['volume'] / c5['volume_ma20'] if pd.notna(c5.get('volume')) and pd.notna(c5.get('volume_ma20')) and c5['volume_ma20'] > 0 else 0
        if vol_ratio < self.config['volume_ratio_min']:
            filter_reasons.append(f"成交量不足 ({vol_ratio:.2f})")
        else:
            trigger_details.append(f"成交量 {vol_ratio:.2f}")

        trend_strength_raw = abs(raw_trend_long - raw_trend_short)
        if trend_strength_raw < self.config['min_trend_strength']:
            filter_reasons.append(f"趋势强度过弱 ({trend_strength_raw} < {self.config['min_trend_strength']})")
        else:
            trigger_details.append(f"趋势强度 {trend_strength_raw}")

        score_gap = abs(final_long - final_short)
        if score_gap < self.config['min_score_gap']:
            filter_reasons.append(f"多空分差过小 ({score_gap:.1f} < {self.config['min_score_gap']})")

        adx_15 = c15.get('adx', 0) if c15 is not None and pd.notna(c15.get('adx')) else 0
        adx_1h = c1h.get('adx', 0) if c1h is not None and pd.notna(c1h.get('adx')) else 0
        if adx_15 < 20 and adx_1h < 20:
            filter_reasons.append("市场处于震荡期 (双ADX<20)")

        if detect_momentum_decay(df_5m):
            filter_reasons.append("动量衰减")

        cooling = False
        if last_signal_candle is not None and current_candle_time is not None:
            if last_signal_candle in df_5m.index:
                idx_last = df_5m.index.get_loc(last_signal_candle)
                idx_current = df_5m.index.get_loc(current_candle_time)
                candles_since = idx_current - idx_last
                cooling = candles_since < COOLDOWN_CANDLES
        if cooling:
            filter_reasons.append("冷却中")

        if abs(st.session_state.latest_funding_rate) > 0.0005:
            filter_reasons.append("资金费过高")

        reason = " | ".join(trigger_details) if trigger_details else "无触发条件"

        if filter_reasons:
            return None, filter_reasons, (final_long, final_short, prob_long, prob_short, trend_long, trend_short, mom_long, mom_short), reason

        if final_long >= self.config['conf_thres'] and final_long > final_short:
            return 'LONG', filter_reasons, (final_long, final_short, prob_long, prob_short, trend_long, trend_short, mom_long, mom_short), reason
        elif final_short >= self.config['conf_thres'] and final_short > final_long:
            return 'SHORT', filter_reasons, (final_long, final_short, prob_long, prob_short, trend_long, trend_short, mom_long, mom_short), reason
        else:
            return None, filter_reasons, (final_long, final_short, prob_long, prob_short, trend_long, trend_short, mom_long, mom_short), reason


class ExitEngine:
    def __init__(self, sl_pct, tp_pct):
        self.sl_pct = sl_pct
        self.tp_pct = tp_pct

    def check_exit(self, position, current_candle, current_price, df_5m, final_long, final_short):
        exit_price, reason = self._check_fixed_by_candle(position, current_candle)
        if reason:
            return exit_price, reason

        exit_price, reason = self._check_fixed_by_ticker(position, current_price)
        if reason:
            return exit_price, reason

        if st.session_state.use_atr_stop:
            exit_price, reason = self._check_atr_stop(position, df_5m)
            if reason:
                return exit_price, reason

        if st.session_state.use_ema_reverse:
            exit_price, reason = self._check_ema_reverse(position, df_5m)
            if reason:
                return exit_price, reason

        if st.session_state.use_score_reverse:
            exit_price, reason = self._check_score_reverse(position, final_long, final_short)
            if reason:
                return exit_price, reason

        return None, None

    def _check_fixed_by_candle(self, position, candle):
        side = position['side']
        sl = position['sl']
        tp = position['tp']

        if side == 'LONG':
            if candle['low'] <= sl:
                return sl, '止损(触及K线低点)'
            elif candle['high'] >= tp:
                return tp, '止盈(触及K线高点)'
        else:
            if candle['high'] >= sl:
                return sl, '止损(触及K线高点)'
            elif candle['low'] <= tp:
                return tp, '止盈(触及K线低点)'
        return None, None

    def _check_fixed_by_ticker(self, position, price):
        side = position['side']
        sl = position['sl']
        tp = position['tp']

        if side == 'LONG':
            if price <= sl:
                return sl, '止损(盘口)'
            elif price >= tp:
                return tp, '止盈(盘口)'
        else:
            if price >= sl:
                return sl, '止损(盘口)'
            elif price <= tp:
                return tp, '止盈(盘口)'
        return None, None

    def _check_atr_stop(self, position, df_5m):
        if len(df_5m) < 2:
            return None, None
        atr = df_5m['atr'].iloc[-1]
        if pd.isna(atr):
            return None, None
        entry = position['entry']
        side = position['side']
        if side == 'LONG':
            stop = entry - (atr * 2)
            if df_5m['low'].iloc[-1] <= stop:
                return stop, 'ATR动态止损'
        else:
            stop = entry + (atr * 2)
            if df_5m['high'].iloc[-1] >= stop:
                return stop, 'ATR动态止损'
        return None, None

    def _check_ema_reverse(self, position, df_5m):
        c = df_5m.iloc[-1]
        if position['side'] == 'LONG' and c['close'] < c['ema20']:
            return c['close'], 'EMA反转出场'
        if position['side'] == 'SHORT' and c['close'] > c['ema20']:
            return c['close'], 'EMA反转出场'
        return None, None

    def _check_score_reverse(self, position, final_long, final_short):
        if position['side'] == 'LONG' and final_short > final_long:
            return st.session_state.last_price, '评分反转出场'
        if position['side'] == 'SHORT' and final_long > final_short:
            return st.session_state.last_price, '评分反转出场'
        return None, None


class PositionSizer:
    def __init__(self, risk_per_trade, contract_size, stop_loss_pct):
        self.risk_per_trade = risk_per_trade
        self.contract_size = contract_size
        self.stop_loss_pct = stop_loss_pct

    def calculate_size(self, equity, entry_price, final_score=None):
        risk_amount = equity * self.risk_per_trade
        risk_per_contract = entry_price * self.contract_size * self.stop_loss_pct
        if risk_per_contract <= 0:
            return 0.01
        size = risk_amount / risk_per_contract

        if st.session_state.use_dynamic_position and final_score is not None:
            scale = 1 + (final_score - 80) * 0.01
            scale = max(scale, 0.5)
            size *= scale

        return round(size, 2)


class RiskEngine:
    def __init__(self, consecutive_loss_limit, daily_loss_limit):
        self.consecutive_loss_limit = consecutive_loss_limit
        self.daily_loss_limit = daily_loss_limit

    def is_halted(self, current_consecutive_losses, today_pnl):
        if self.consecutive_loss_limit and current_consecutive_losses >= self.consecutive_loss_limit:
            return True, f"连亏{current_consecutive_losses}次触发熔断"
        if self.daily_loss_limit and today_pnl <= -self.daily_loss_limit:
            return True, f"当日亏损{today_pnl:.2f}超过限额{self.daily_loss_limit}"
        return False, ""

# ================================
# 8. 开平仓函数（使用动态杠杆）
# ================================
def open_position(entry_price, side, final_score):
    entry_slip = st.session_state.entry_slippage
    if side == 'LONG':
        actual_entry = entry_price * (1 + entry_slip)
    else:
        actual_entry = entry_price * (1 - entry_slip)

    sizer = PositionSizer(RISK_PER_TRADE, CONTRACT_SIZE, STOP_LOSS_PCT)
    size = sizer.calculate_size(st.session_state.current_equity, actual_entry, final_score)

    sl_dist = actual_entry * STOP_LOSS_PCT
    tp_dist = actual_entry * TAKE_PROFIT_PCT
    if side == "LONG":
        sl = actual_entry - sl_dist
        tp = actual_entry + tp_dist
    else:
        sl = actual_entry + sl_dist
        tp = actual_entry - tp_dist

    entry_fee = actual_entry * CONTRACT_SIZE * size * TAKER_FEE
    st.session_state.current_equity -= entry_fee

    st.session_state.position = {
        'side': side,
        'entry': actual_entry,
        'sl': sl,
        'tp': tp,
        'entry_time': datetime.now(),
        'score': final_score,
        'size': size
    }

    st.session_state.signal_log.append({
        "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "事件": "开仓",
        "方向": side,
        "入场价": actual_entry,
        "仓位": size,
        "手续费": f"{entry_fee:.2f}"
    })

    return actual_entry, sl, tp, size

def calculate_exit_price_and_fee(exit_indicated, side):
    exit_slip = st.session_state.exit_slippage
    if side == 'LONG':
        exit_price = exit_indicated * (1 - exit_slip)
    else:
        exit_price = exit_indicated * (1 + exit_slip)

    fee = exit_price * CONTRACT_SIZE * st.session_state.position['size'] * TAKER_FEE
    return exit_price, fee

def apply_liquidation():
    pos = st.session_state.position
    if pos is None:
        return

    leverage = st.session_state.leverage
    notional = pos['entry'] * CONTRACT_SIZE * pos['size']
    margin = notional / leverage
    loss = -margin
    st.session_state.current_equity += loss

    st.session_state.signal_log.append({
        "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "事件": "爆仓",
        "方向": pos['side'],
        "入场价": pos['entry'],
        "损失保证金": f"{margin:.2f}",
        "剩余权益": f"{st.session_state.current_equity:.2f}"
    })

    st.session_state.stats['total_trades'] += 1
    st.session_state.stats['losses'] += 1
    st.session_state.stats['current_consecutive_losses'] += 1
    if st.session_state.stats['current_consecutive_losses'] > st.session_state.stats['max_consecutive_losses']:
        st.session_state.stats['max_consecutive_losses'] = st.session_state.stats['current_consecutive_losses']

    st.session_state.pnl_list.append(-margin)
    st.session_state.today_pnl += loss

    st.session_state.position = None
    st.session_state.equity_history.append((datetime.now(), st.session_state.current_equity))

def check_liquidation(mark_price):
    if st.session_state.position is None:
        return False
    pos = st.session_state.position
    entry = pos['entry']
    side = pos['side']
    leverage = st.session_state.leverage

    if side == 'LONG':
        liquidation_price = entry * (1 - (1/leverage) + MAINTENANCE_MARGIN_RATE)
        if mark_price <= liquidation_price:
            apply_liquidation()
            return True
    else:
        liquidation_price = entry * (1 + (1/leverage) - MAINTENANCE_MARGIN_RATE)
        if mark_price >= liquidation_price:
            apply_liquidation()
            return True
    return False

def apply_funding_if_needed():
    now_utc = datetime.now(timezone.utc)
    for hour in FUNDING_HOURS:
        funding_time = now_utc.replace(hour=hour, minute=0, second=0, microsecond=0)
        if st.session_state.last_funding_check < funding_time <= now_utc:
            if st.session_state.position is not None:
                pos = st.session_state.position
                size = pos['size']
                entry = pos['entry']
                side = pos['side']

                notional = entry * CONTRACT_SIZE * size
                funding_rate = st.session_state.latest_funding_rate
                if side == 'LONG':
                    funding_cost = notional * funding_rate
                else:
                    funding_cost = -notional * funding_rate

                st.session_state.current_equity -= funding_cost
                st.session_state.signal_log.append({
                    "时间": funding_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "事件": "资金费",
                    "方向": side,
                    "金额": f"{funding_cost:.2f}"
                })
            st.session_state.last_funding_check = funding_time

def record_trade(exit_price, reason):
    pos = st.session_state.position
    if pos is None:
        return

    side = pos['side']
    entry = pos['entry']
    size = pos['size']

    actual_exit, fee = calculate_exit_price_and_fee(exit_price, side)

    if side == 'LONG':
        price_diff = actual_exit - entry
    else:
        price_diff = entry - actual_exit
    gross_pnl = price_diff * CONTRACT_SIZE * size
    net_pnl = gross_pnl - fee

    st.session_state.current_equity += net_pnl
    st.session_state.pnl_list.append(net_pnl)

    st.session_state.stats['total_trades'] += 1
    if net_pnl > 0:
        st.session_state.stats['wins'] += 1
        st.session_state.stats['current_consecutive_losses'] = 0
    else:
        st.session_state.stats['losses'] += 1
        st.session_state.stats['current_consecutive_losses'] += 1
        if st.session_state.stats['current_consecutive_losses'] > st.session_state.stats['max_consecutive_losses']:
            st.session_state.stats['max_consecutive_losses'] = st.session_state.stats['current_consecutive_losses']

    st.session_state.today_pnl += net_pnl

    st.session_state.signal_log.append({
        "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "事件": "平仓",
        "方向": side,
        "入场价": entry,
        "出场价": actual_exit,
        "毛盈亏": f"{gross_pnl:.2f}",
        "手续费": f"{fee:.2f}",
        "净盈亏": f"{net_pnl:.2f}",
        "原因": reason
    })

    st.session_state.position = None
    st.session_state.equity_history.append((datetime.now(), st.session_state.current_equity))

def record_daily_equity_force():
    now = datetime.now()
    today = now.date()
    if len(st.session_state.daily_equity) == 0 or st.session_state.daily_equity[-1][0] != today:
        st.session_state.today_pnl = 0.0
        st.session_state.daily_peak_equity = st.session_state.current_equity
        st.session_state.daily_equity.append((today, st.session_state.current_equity))
    if now.hour == 0 and now.minute < 5:
        if len(st.session_state.daily_equity) == 0 or st.session_state.daily_equity[-1][0] != today:
            st.session_state.today_pnl = 0.0
            st.session_state.daily_peak_equity = st.session_state.current_equity
            st.session_state.daily_equity.append((today, st.session_state.current_equity))

# ================================
# 9. 统计与风险分析
# ================================
def compute_pnl_distribution_metrics(pnl_list):
    if len(pnl_list) < 5:
        return {}
    pnl_array = np.array(pnl_list)
    metrics = {
        '总笔数': len(pnl_array),
        '平均盈亏': np.mean(pnl_array),
        '盈亏中位数': np.median(pnl_array),
        '标准差': np.std(pnl_array),
        '最大盈利': np.max(pnl_array),
        '最大亏损': np.min(pnl_array),
        '盈利次数': np.sum(pnl_array > 0),
        '亏损次数': np.sum(pnl_array < 0),
        '胜率': np.mean(pnl_array > 0) * 100,
        '盈亏比': np.mean(pnl_array[pnl_array > 0]) / abs(np.mean(pnl_array[pnl_array < 0])) if np.any(pnl_array < 0) else np.inf,
    }
    if len(pnl_array) > 0:
        var_95 = np.percentile(pnl_array, 5)
        cvar_95 = pnl_array[pnl_array <= var_95].mean() if np.any(pnl_array <= var_95) else var_95
        metrics['VaR_95'] = var_95
        metrics['CVaR_95'] = cvar_95
    return metrics

def plot_pnl_distribution(pnl_list):
    if len(pnl_list) < 5:
        return None
    df = pd.DataFrame(pnl_list, columns=['pnl'])
    fig = px.histogram(df, x='pnl', nbins=30, title='盈亏分布直方图', labels={'pnl':'净盈亏 (USDT)'})
    fig.add_vline(x=0, line_dash="dash", line_color="red")
    return fig

def plot_hourly_pnl(pnl_list_with_time):
    if len(pnl_list_with_time) < 5:
        return None
    df = pd.DataFrame(pnl_list_with_time, columns=['time', 'pnl'])
    df['hour_utc'] = df['time'].dt.hour
    hourly_stats = df.groupby('hour_utc')['pnl'].agg(['sum', 'count', 'mean']).reset_index()
    fig = px.bar(hourly_stats, x='hour_utc', y='sum', title='各 UTC 小时累计盈亏',
                 labels={'sum':'累计盈亏 (USDT)', 'hour_utc':'UTC 小时'})
    return fig

def monte_carlo_test(pnl_list, initial_equity, num_simulations=1000):
    if len(pnl_list) < 10:
        return None

    final_equities = []
    max_drawdowns = []
    max_consecutive_losses_list = []

    for _ in range(num_simulations):
        shuffled = random.sample(pnl_list, len(pnl_list))

        equity = initial_equity
        equity_curve = [equity]
        peak = equity
        max_dd = 0
        consecutive_losses = 0
        max_consecutive = 0

        for pnl in shuffled:
            equity += pnl
            equity_curve.append(equity)
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100
            if dd > max_dd:
                max_dd = dd

            if pnl < 0:
                consecutive_losses += 1
                if consecutive_losses > max_consecutive:
                    max_consecutive = consecutive_losses
            else:
                consecutive_losses = 0

        final_equities.append(equity)
        max_drawdowns.append(max_dd)
        max_consecutive_losses_list.append(max_consecutive)

    result = {
        '模拟次数': num_simulations,
        '最终权益均值': np.mean(final_equities),
        '最终权益中位数': np.median(final_equities),
        '最终权益标准差': np.std(final_equities),
        '最大回撤均值': np.mean(max_drawdowns),
        '最大回撤95分位': np.percentile(max_drawdowns, 95),
        '最大连续亏损均值': np.mean(max_consecutive_losses_list),
        '最大连续亏损95分位': np.percentile(max_consecutive_losses_list, 95),
        '破产概率%': sum(1 for e in final_equities if e < 0) / num_simulations * 100
    }
    return result

# ================================
# 10. 侧边栏（折叠次要信息）
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
        st.metric("OKX 资金费率", f"{funding_rate:.4f}%")
        st.session_state.latest_funding_rate = funding_rate / 100
    else:
        st.error(f"费率获取失败: {error_msg}")
        st.session_state.latest_funding_rate = 0.0

    with st.expander("📈 交易统计", expanded=False):
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
            total_pnl_amount = st.session_state.current_equity - st.session_state.initial_equity
            st.metric("总盈亏", f"{total_pnl_amount:.2f} USDT")

    with st.expander("📊 资金分析", expanded=False):
        if len(st.session_state.equity_history) >= 2:
            total_return = (st.session_state.current_equity - st.session_state.initial_equity) / st.session_state.initial_equity
            st.metric("总收益率", f"{total_return*100:.2f}%")

            times, equities = zip(*st.session_state.equity_history)
            equity_series = pd.Series(equities, index=times)
            rolling_max = equity_series.expanding().max()
            drawdown = (equity_series - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            st.metric("最大回撤", f"{max_drawdown*100:.2f}%")

            if len(st.session_state.daily_equity) >= 2:
                dates, eqs = zip(*st.session_state.daily_equity)
                daily_ret = pd.Series(eqs).pct_change().dropna()
                sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() != 0 else 0
                st.metric("夏普比率(年化)", f"{sharpe:.2f}")

            if stats['losses'] > 0:
                total_win = sum([p for p in st.session_state.pnl_list if p > 0])
                total_loss = -sum([p for p in st.session_state.pnl_list if p < 0])
                profit_factor = total_win / total_loss if total_loss != 0 else np.inf
                st.metric("盈亏比", f"{profit_factor:.2f}")

            trades = [log for log in st.session_state.signal_log if log.get('事件') == '平仓']
            if len(trades) >= 2:
                times = [datetime.strptime(log['时间'], "%Y-%m-%d %H:%M:%S") for log in trades]
                days = (max(times) - min(times)).days + 1
                trades_per_month = len(times) / days * 30 if days > 0 else 0
                st.metric("月均交易次数", f"{trades_per_month:.1f}")

    with st.expander("🎲 风险诊断", expanded=False):
        if len(st.session_state.pnl_list) >= 5:
            metrics = compute_pnl_distribution_metrics(st.session_state.pnl_list)
            cola, colb = st.columns(2)
            with cola:
                st.metric("VaR 95%", f"{metrics.get('VaR_95', 0):.2f}")
            with colb:
                st.metric("CVaR 95%", f"{metrics.get('CVaR_95', 0):.2f}")
            st.metric("平均盈亏", f"{metrics.get('平均盈亏', 0):.2f}")
            st.metric("盈亏中位数", f"{metrics.get('盈亏中位数', 0):.2f}")

            fig_dist = plot_pnl_distribution(st.session_state.pnl_list)
            if fig_dist:
                st.plotly_chart(fig_dist, use_container_width=True)

            trade_logs = [log for log in st.session_state.signal_log if log.get('事件') == '平仓']
            pnl_with_time = []
            for log in trade_logs:
                try:
                    t = datetime.strptime(log['时间'], "%Y-%m-%d %H:%M:%S")
                    pnl = float(log['净盈亏'])
                    pnl_with_time.append((t, pnl))
                except:
                    continue
            if len(pnl_with_time) >= 5:
                fig_hourly = plot_hourly_pnl(pnl_with_time)
                if fig_hourly:
                    st.plotly_chart(fig_hourly, use_container_width=True)

    with st.expander("⚙️ 滑点压力测试", expanded=False):
        new_entry_slip = st.slider("入场滑点 (%)", 0.0, 0.1, st.session_state.entry_slippage*100, 0.01) / 100
        new_exit_slip = st.slider("出场滑点 (%)", 0.0, 0.1, st.session_state.exit_slippage*100, 0.01) / 100
        if new_entry_slip != st.session_state.entry_slippage or new_exit_slip != st.session_state.exit_slippage:
            st.session_state.entry_slippage = new_entry_slip
            st.session_state.exit_slippage = new_exit_slip
            st.info("滑点已更新")

    with st.expander("🛡️ 风控设置", expanded=False):
        st.session_state.consecutive_loss_limit = st.number_input("连亏熔断阈值", min_value=1, max_value=10, value=st.session_state.consecutive_loss_limit)
        st.session_state.daily_loss_limit = st.number_input("日亏损限额 (USDT)", min_value=0, max_value=10000, value=st.session_state.daily_loss_limit, step=50)
        st.session_state.use_atr_stop = st.checkbox("启用 ATR 动态止损", value=st.session_state.use_atr_stop)
        st.session_state.use_ema_reverse = st.checkbox("启用 EMA 反转出场", value=st.session_state.use_ema_reverse)
        st.session_state.use_score_reverse = st.checkbox("启用评分反转出场", value=st.session_state.use_score_reverse)
        st.session_state.use_dynamic_position = st.checkbox("启用动态仓位缩放", value=st.session_state.use_dynamic_position)

    st.metric("当前连亏", stats['current_consecutive_losses'])
    st.metric("今日盈亏", f"{st.session_state.today_pnl:.2f} USDT")
    if st.session_state.system_halted:
        st.error("🚨 系统已熔断，请点击下方按钮重置")
    else:
        st.success("✅ 风控正常")

    if st.button("🔌 重置熔断"):
        st.session_state.system_halted = False
        st.session_state.last_price = 0
        st.session_state.active_signal = None
        st.session_state.last_signal_candle = None
        st.session_state.position = None
        st.session_state.price_changes = []
        st.session_state.stats['current_consecutive_losses'] = 0
        st.success("熔断已重置")

    if st.button("运行 Monte Carlo 模拟 (1000次)"):
        if len(st.session_state.pnl_list) < 10:
            st.warning("交易笔数不足10笔")
        else:
            with st.spinner("运行中..."):
                result = monte_carlo_test(st.session_state.pnl_list, st.session_state.initial_equity)
                if result:
                    st.json(result)

    st.markdown("---")
    st.subheader("📝 历史事件")
    if st.session_state.signal_log:
        log_list = list(st.session_state.signal_log)[::-1]
        log_df = pd.DataFrame(log_list)
        st.dataframe(log_df.head(20), width='stretch', height=350)
        if st.button("清除日志"):
            st.session_state.signal_log.clear()
            st.session_state.pnl_list.clear()
            st.rerun()
    else:
        st.info("等待交易...")

# ================================
# 11. 主循环（卡片化 UI）
# ================================
st.title("📈 ETH 100x 智能做单策略终端 v7.1 (最终优化版)")

try:
    ticker = exchange.fetch_ticker(SYMBOL)
    current_price = ticker['last']
    st.session_state.last_price = current_price

    df_5m, df_15m, df_1h = get_multi_timeframe_data()
    df_5m, df_15m, df_1h, latest_feat = compute_features(df_5m, df_15m, df_1h)
    mark_price = get_mark_price(df_5m)

    record_daily_equity_force()
    apply_funding_if_needed()
    check_liquidation(mark_price)

    entry_config = {
        'trend_weight': TREND_WEIGHT,
        'momentum_weight': MOMENTUM_WEIGHT,
        'model_weight': MODEL_WEIGHT,
        'conf_thres': FINAL_CONF_THRES,
        'min_atr_pct': MIN_ATR_PCT,
        'max_atr_pct': MAX_ATR_PCT,
        'volume_ratio_min': VOLUME_RATIO_MIN,
        'min_trend_strength': MIN_TREND_STRENGTH,
        'min_score_gap': MIN_SCORE_GAP,
    }
    entry_engine = EntryEngine(entry_config)
    exit_engine = ExitEngine(STOP_LOSS_PCT, TAKE_PROFIT_PCT)
    risk_engine = RiskEngine(st.session_state.consecutive_loss_limit, st.session_state.daily_loss_limit)

    if not st.session_state.system_halted:
        halted, reason = risk_engine.is_halted(st.session_state.stats['current_consecutive_losses'], st.session_state.today_pnl)
        if halted:
            st.session_state.system_halted = True
            st.error(f"🚨 风控触发: {reason}")

    current_5m_candle_time = df_5m.index[-1]
    new_candle = (st.session_state.last_5m_candle_time is None or
                  current_5m_candle_time > st.session_state.last_5m_candle_time)

    if st.session_state.position:
        c15_aligned = df_15m.asof(current_5m_candle_time)
        c1h_aligned = df_1h.asof(current_5m_candle_time)
        temp_15m = pd.DataFrame([c15_aligned], index=[current_5m_candle_time])
        temp_1h = pd.DataFrame([c1h_aligned], index=[current_5m_candle_time])
        t_long, t_short, _, _ = compute_trend_score(temp_15m, temp_1h)
        m_long, m_short = compute_momentum_score(df_5m)
        p_long, p_short = compute_model_prob(model, scaler, latest_feat, t_long, t_short)
        final_long = (t_long * TREND_WEIGHT + m_long * MOMENTUM_WEIGHT + p_long * MODEL_WEIGHT) / 100
        final_short = (t_short * TREND_WEIGHT + m_short * MOMENTUM_WEIGHT + p_short * MODEL_WEIGHT) / 100

        last_candle = df_5m.iloc[-1]
        exit_price, reason = exit_engine.check_exit(st.session_state.position, last_candle, current_price, df_5m, final_long, final_short)
        if reason:
            record_trade(exit_price, reason)

    direction = None
    final_score = 0
    filter_reasons = []
    signal_reason = ""
    scores_display = (0,0,0,0,0,0,0,0)
    t_long, t_short, m_long, m_short = 0, 0, 0, 0
    prob_long, prob_short = 50, 50
    final_long, final_short = 0, 0

    if new_candle and not st.session_state.system_halted:
        st.session_state.last_5m_candle_time = current_5m_candle_time

        direction, filter_reasons, scores, signal_reason = entry_engine.generate_signal(
            df_5m, df_15m, df_1h, latest_feat, model, scaler,
            last_signal_candle=st.session_state.last_signal_candle,
            current_candle_time=current_5m_candle_time
        )
        final_long, final_short, prob_long, prob_short, trend_long, trend_short, mom_long, mom_short = scores
        final_score = final_long if direction == 'LONG' else final_short
        t_long, t_short, m_long, m_short = trend_long, trend_short, mom_long, mom_short

        if direction:
            st.session_state.signal_log.append({
                "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "事件": "信号",
                "方向": direction,
                "置信度": f"{final_score:.1f}",
                "理由": signal_reason,
                "过滤": "无" if not filter_reasons else " | ".join(filter_reasons)
            })
        else:
            if filter_reasons:
                st.session_state.signal_log.append({
                    "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "事件": "信号",
                    "方向": "无",
                    "置信度": "-",
                    "理由": "过滤未通过",
                    "过滤": " | ".join(filter_reasons)
                })

        if direction and st.session_state.last_signal_candle != current_5m_candle_time:
            st.session_state.active_signal = direction
            st.session_state.last_signal_candle = current_5m_candle_time
        elif not direction and st.session_state.last_signal_candle != current_5m_candle_time:
            st.session_state.active_signal = None

        st.session_state.prediction_history.append({
            'time': current_5m_candle_time,
            'final_long': final_long,
            'final_short': final_short,
            'price': current_price
        })
        scores_display = scores
    else:
        # 非新K线时，重新计算当前评分用于显示
        c15_aligned = df_15m.asof(df_5m.index[-1])
        c1h_aligned = df_1h.asof(df_5m.index[-1])
        temp_15m = pd.DataFrame([c15_aligned], index=[df_5m.index[-1]])
        temp_1h = pd.DataFrame([c1h_aligned], index=[df_5m.index[-1]])
        t_long, t_short, _, _ = compute_trend_score(temp_15m, temp_1h)
        m_long, m_short = compute_momentum_score(df_5m)
        p_long, p_short = compute_model_prob(model, scaler, latest_feat, t_long, t_short)
        final_long = (t_long * TREND_WEIGHT + m_long * MOMENTUM_WEIGHT + p_long * MODEL_WEIGHT) / 100
        final_short = (t_short * TREND_WEIGHT + m_short * MOMENTUM_WEIGHT + p_short * MODEL_WEIGHT) / 100
        prob_long, prob_short = p_long, p_short

    # ========== 卡片布局开始 ==========

    # 第一行：价格、标记价、资金费率、杠杆滑块（四列）
    col1, col2, col3, col4 = st.columns([1,1,1,1.5])
    with col1:
        st.markdown(f"""
        <div class="card">
            <div class="card-label">ETH 实时价</div>
            <div class="card-value bull">${current_price:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="card">
            <div class="card-label">标记价格</div>
            <div class="card-value neutral">${mark_price:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        funding_color = "bull" if st.session_state.latest_funding_rate > 0 else "bear"
        st.markdown(f"""
        <div class="card">
            <div class="card-label">资金费率</div>
            <div class="card-value {funding_color}">{st.session_state.latest_funding_rate*100:.4f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="card leverage-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">杠杆倍数</div>', unsafe_allow_html=True)
        new_leverage = st.slider("##", min_value=1, max_value=100, value=st.session_state.leverage, step=1, label_visibility="collapsed")
        if new_leverage != st.session_state.leverage:
            st.session_state.leverage = new_leverage
        risk_color = "#ff3b5c" if new_leverage > 50 else "#00ff9d" if new_leverage > 20 else "#aaa"
        st.markdown(f'<div style="font-size: 1.8rem; font-weight:700; color:{risk_color};">{new_leverage}x</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # 第二行：趋势、动量、模型、信心（四列）
    col_t1, col_t2, col_t3, col_t4 = st.columns(4)

    with col_t1:
        trend_dir = "bull" if t_long > t_short else "bear" if t_short > t_long else "neutral"
        st.markdown(f"""
        <div class="card">
            <div class="card-label">趋势核 (多/空)</div>
            <div class="card-value {trend_dir}">{t_long:.0f}/{t_short:.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_t2:
        mom_dir = "bull" if m_long > m_short else "bear" if m_short > m_long else "neutral"
        st.markdown(f"""
        <div class="card">
            <div class="card-label">动量核 (多/空)</div>
            <div class="card-value {mom_dir}">{m_long:.0f}/{m_short:.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_t3:
        model_dir = "bull" if prob_long > prob_short else "bear"
        st.markdown(f"""
        <div class="card">
            <div class="card-label">AI 模型 (多/空)</div>
            <div class="card-value {model_dir}">{prob_long:.0f}%/{prob_short:.0f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col_t4:
        final_dir = "bull" if final_long > final_short else "bear" if final_short > final_long else "neutral"
        st.markdown(f"""
        <div class="card">
            <div class="card-label">最终信心</div>
            <div class="card-value {final_dir}">{max(final_long, final_short):.0f}%</div>
        </div>
        """, unsafe_allow_html=True)

    # 第三行：超大信号卡（横跨）
    if st.session_state.active_signal:
        direction = st.session_state.active_signal
        conf = final_score
        arrow = "▲" if direction == "LONG" else "▼"
        signal_class = "bull" if direction == "LONG" else "bear"
        signal_text = f"{direction} {conf:.0f}% {arrow}"
    else:
        signal_class = "neutral"
        signal_text = "无信号"

    st.markdown(f"""
    <div class="signal-card {signal_class}">
        <span class="signal-label">当前方向</span>
        <span class="signal-value {signal_class}">{signal_text}</span>
    </div>
    """, unsafe_allow_html=True)

    # 可选：显示过滤理由
    if filter_reasons:
        st.warning("⛔ 当前不满足信号条件: " + " | ".join(filter_reasons))
    else:
        if st.session_state.system_halted:
            st.error("⛔ 系统已熔断，暂停交易")
        else:
            st.success("✅ 所有基础过滤条件通过，等待高置信度信号...")

    # 开仓逻辑（新K线且有信号时）
    if new_candle and st.session_state.active_signal and st.session_state.position is None and not st.session_state.system_halted:
        side = st.session_state.active_signal
        st.success(f"🎯 **执行交易信号：{side}** (信心分 {final_score:.1f})")
        actual_entry, sl, tp, size = open_position(current_price, side, final_score)
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.write(f"**入场价:** {actual_entry:.2f}")
        sc2.write(f"**止损价:** {sl:.2f}")
        sc3.write(f"**止盈价:** {tp:.2f}")
        sc4.write(f"**仓位:** {size:.2f} 张")

    # 持仓信息展示
    if st.session_state.position:
        pos = st.session_state.position
        if pos['side'] == 'LONG':
            price_diff = current_price - pos['entry']
        else:
            price_diff = pos['entry'] - current_price
        gross_unrealized = price_diff * CONTRACT_SIZE * pos['size']
        exit_fee = current_price * CONTRACT_SIZE * pos['size'] * TAKER_FEE
        net_unrealized = gross_unrealized - exit_fee
        display_equity = st.session_state.current_equity + net_unrealized

        st.info(
            f"📌 当前持仓: {pos['side']} | "
            f"入场: {pos['entry']:.2f} | "
            f"止损: {pos['sl']:.2f} | "
            f"止盈: {pos['tp']:.2f} | "
            f"仓位: {pos['size']:.2f} 张 | "
            f"浮动盈亏: {net_unrealized:.2f} USDT (扣费)"
        )
        st.metric("预估权益", f"{display_equity:.2f} USDT")

    # 资金曲线图
    st.markdown("---")
    st.subheader("💰 资金曲线 (仅平仓后权益)")
    if len(st.session_state.equity_history) > 1:
        times, equities = zip(*st.session_state.equity_history)
        fig_equity = go.Figure()
        fig_equity.add_trace(go.Scatter(x=list(times), y=list(equities), mode='lines', name='权益', line=dict(color='#00ff9d')))
        fig_equity.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor='#0e1117', plot_bgcolor='#0e1117', font=dict(color='#fafafa'))
        st.plotly_chart(fig_equity, use_container_width=True)

    # K线图
    st.subheader("📈 K线图 (5分钟)")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df_5m.index,
        open=df_5m['open'],
        high=df_5m['high'],
        low=df_5m['low'],
        close=df_5m['close'],
        name='5m',
        increasing_line_color='#00ff9d',
        decreasing_line_color='#ff3b5c'
    ))
    fig.add_trace(go.Scatter(x=df_5m.index, y=df_5m['ema5'], line=dict(width=1, color='#ffaa00'), name='EMA5'))
    fig.add_trace(go.Scatter(x=df_5m.index, y=df_5m['ema20'], line=dict(width=1, color='#3399ff'), name='EMA20'))
    fig.update_layout(
        height=500,
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor='#0e1117',
        plot_bgcolor='#0e1117',
        font=dict(color='#fafafa')
    )
    st.plotly_chart(fig, use_container_width=True)

    # 最后更新时间
    st.markdown(f'<div class="update-time">最后更新: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>', unsafe_allow_html=True)

except Exception as e:
    st.error(f"系统异常: {e}")
    st.stop()
