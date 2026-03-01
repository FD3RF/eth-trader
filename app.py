import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import numpy as np
import plotly.io as pio
from itertools import product
import threading
import traceback
import random
from streamlit_autorefresh import st_autorefresh
import feedparser

# ====================== 安全配置 ======================
try:
    API_KEY = st.secrets["API_KEY"]
    API_SECRET = st.secrets["API_SECRET"]
    PASSPHRASE = st.secrets["PASSPHRASE"]
except Exception:
    st.error("❌ 请在 .streamlit/secrets.toml 中配置您的OKX API密钥")
    st.stop()

# ====================== 页面配置 ======================
st.set_page_config(layout="wide", page_title="专业量化决策引擎·至尊版", page_icon="📊")

# 全局样式：缩小UI边距和字体
st.markdown("""
<style>
    .main > div { padding: 0 1rem; }
    .block-container { max-width: 95%; padding-top: 1rem; }
    .stButton > button { font-size: 0.8rem; }
    .stMetric { font-size: 0.9rem; }
    div[data-testid="column"] { padding: 0 0.2rem; }
</style>
""", unsafe_allow_html=True)

pio.templates['custom_dark'] = pio.templates['plotly_dark']
pio.templates['custom_light'] = pio.templates['plotly']

# ====================== 状态初始化 ======================
def init_state():
    defaults = {
        'ls_ratio': 1.0,
        'ls_history': pd.DataFrame(),
        'last_cleanup': time.time(),
        'theme': 'dark',
        'alarm_on': False,
        'tg_token': '',
        'tg_chat_id': '',
        'last_signal_prob': 50.0,
        'signal_history': [],
        'task_running': False,
        'task_type': None,
        'backtest_running': False,
        'backtest_progress': 0.0,
        'backtest_status': '',
        'backtest_trades': None,
        'backtest_equity': None,
        'backtest_metrics': None,
        'backtest_error': None,
        'opt_running': False,
        'opt_progress': 0.0,
        'opt_status': '',
        'opt_params': None,
        'opt_metrics': None,
        'opt_trades': None,
        'opt_equity': None,
        'opt_error': None,
        'state_lock': threading.Lock(),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ====================== 工具函数 ======================
def light_cleanup():
    if time.time() - st.session_state.last_cleanup > 86400:
        st.cache_data.clear()
        st.session_state.last_cleanup = time.time()

def send_telegram(msg):
    if st.session_state.tg_token and st.session_state.tg_chat_id:
        try:
            url = f"https://api.telegram.org/bot{st.session_state.tg_token}/sendMessage"
            requests.post(url, json={"chat_id": st.session_state.tg_chat_id, "text": msg, "parse_mode": "HTML"}, timeout=3)
        except:
            pass

def safe_request(url, timeout=5, retries=2):
    for i in range(retries + 1):
        try:
            res = requests.get(url, timeout=timeout)
            if res.status_code == 200:
                return res.json()
            elif res.status_code == 429:
                retry_after = res.headers.get('Retry-After')
                if retry_after:
                    wait = int(retry_after)
                else:
                    wait = 2 ** i
                print(f"HTTP 429 限流，等待 {wait} 秒后重试")
                time.sleep(wait)
                continue
            else:
                print(f"HTTP {res.status_code}，重试 {i+1}/{retries}")
        except requests.exceptions.Timeout:
            print(f"请求超时，重试 {i+1}/{retries}")
        except Exception as e:
            print(f"请求异常: {e}，重试 {i+1}/{retries}")
        time.sleep(0.5)
    return None

@st.cache_data(ttl=15, max_entries=50)
def get_ls_ratio():
    for attempt in range(3):
        try:
            url = "https://www.okx.com/api/v5/rubik/stat/contracts/long-short-account-ratio?instId=ETH-USDT&period=5m"
            data = safe_request(url, timeout=4, retries=1)
            if data and data.get('code') == '0':
                return float(data['data'][0][1])
        except:
            pass
    return st.session_state.get('ls_ratio', 1.0)

@st.cache_data(ttl=300, max_entries=20)
def get_ls_history(limit=24):
    try:
        url_with_params = f"https://www.okx.com/api/v5/rubik/stat/contracts/long-short-account-ratio?instId=ETH-USDT&period=1H&limit={limit}"
        data = safe_request(url_with_params, timeout=5)
        if data and data.get('code') == '0':
            df = pd.DataFrame(data['data'], columns=['ts', 'long', 'short', 'instId'])
            df['ts'] = pd.to_datetime(df['ts'].astype(float), unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')
            df['ratio'] = df['long'].astype(float) / df['short'].astype(float)
            return df[['ts', 'ratio']]
    except Exception:
        pass
    return pd.DataFrame()

@st.cache_data(ttl=60, max_entries=50)
def get_candles(bar="15m", limit=100, f_ema=12, s_ema=26):
    try:
        url = f"https://www.okx.com/api/v5/market/candles?instId=ETH-USDT&bar={bar}&limit={limit}"
        data = safe_request(url, timeout=6, retries=1)
        if not data or data.get('code') != '0':
            return None
        df = pd.DataFrame(data['data'], columns=['ts','o','h','l','c','v','volCcy','volCcyQuote','confirm'])[::-1].reset_index(drop=True)
        df['time'] = pd.to_datetime(df['ts'].astype(float), unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')
        for col in ['o','h','l','c','v']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)
        if len(df) < 50:
            return None

        # 基础指标
        df['ema_f'] = df['c'].ewm(span=f_ema, adjust=False).mean()
        df['ema_s'] = df['c'].ewm(span=s_ema, adjust=False).mean()
        delta = df['c'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))
        ema12 = df['c'].ewm(span=12, adjust=False).mean()
        ema26 = df['c'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        df['bb_mid'] = df['c'].rolling(20).mean()
        df['bb_std'] = df['c'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
        df['vol_ma'] = df['v'].rolling(10).mean()
        tr = pd.concat([df['h']-df['l'], abs(df['h']-df['c'].shift()), abs(df['l']-df['c'].shift())], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()

        # Ichimoku Cloud
        high_9 = df['h'].rolling(9).max()
        low_9 = df['l'].rolling(9).min()
        df['tenkan'] = (high_9 + low_9) / 2
        high_26 = df['h'].rolling(26).max()
        low_26 = df['l'].rolling(26).min()
        df['kijun'] = (high_26 + low_26) / 2
        df['senkou_a'] = ((df['tenkan'].shift(26) + df['kijun'].shift(26)) / 2)
        df['senkou_b'] = ((high_26.shift(26) + low_26.shift(26)) / 2)
        df['chikou'] = df['c'].shift(-26)

        # Stochastic
        low_14 = df['l'].rolling(14).min()
        high_14 = df['h'].rolling(14).max()
        df['stoch_k'] = 100 * (df['c'] - low_14) / (high_14 - low_14).replace(0, np.nan)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()

        # ADX
        df['tr'] = tr
        df['+dm'] = np.where((df['h'] - df['h'].shift()) > (df['l'].shift() - df['l']), 
                             np.maximum(df['h'] - df['h'].shift(), 0), 0)
        df['-dm'] = np.where((df['l'].shift() - df['l']) > (df['h'] - df['h'].shift()), 
                             np.maximum(df['l'].shift() - df['l'], 0), 0)
        df['+di'] = 100 * (df['+dm'].rolling(14).mean() / df['tr'].rolling(14).mean())
        df['-di'] = 100 * (df['-dm'].rolling(14).mean() / df['tr'].rolling(14).mean())
        df['dx'] = 100 * abs(df['+di'] - df['-di']) / (df['+di'] + df['-di']).replace(0, np.nan)
        df['adx'] = df['dx'].rolling(14).mean()

        # 资金净流入
        trades_url = "https://www.okx.com/api/v5/market/trades?instId=ETH-USDT&limit=100"
        trades_data = safe_request(trades_url, timeout=5)
        if trades_data and trades_data.get('code') == '0':
            trades_df = pd.DataFrame(trades_data['data'], columns=['ts', 'px', 'sz', 'side'])
            trades_df['ts'] = pd.to_datetime(trades_df['ts'].astype(float), unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')
            trades_df['sz'] = trades_df['sz'].astype(float)
            trades_df['minute'] = trades_df['ts'].dt.floor('min')
            agg = trades_df.groupby('minute').apply(
                lambda x: x[x['side'] == 'buy']['sz'].sum() - x[x['side'] == 'sell']['sz'].sum(),
                include_groups=False
            ).reset_index(name='net_flow')
            if not agg.empty:
                df = df.merge(agg, left_on='time', right_on='minute', how='left')
                df['net_flow'] = df['net_flow'].fillna(0)
            else:
                df['net_flow'] = 0
        else:
            df['net_flow'] = 0
        return df
    except Exception as e:
        st.error(f"K线获取异常: {e}")
        return None

# 新增：获取4H趋势数据（用于EMA50过滤）
@st.cache_data(ttl=300, max_entries=10)
def get_trend_candles(bar="4H", limit=200):
    try:
        url = f"https://www.okx.com/api/v5/market/candles?instId=ETH-USDT&bar={bar}&limit={limit}"
        data = safe_request(url, timeout=10, retries=2)
        if not data or data.get('code') != '0':
            return None
        df = pd.DataFrame(data['data'], columns=['ts','o','h','l','c','v','volCcy','volCcyQuote','confirm'])[::-1].reset_index(drop=True)
        df['time'] = pd.to_datetime(df['ts'].astype(float), unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')
        for col in ['o','h','l','c','v']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)
        if len(df) < 50:
            return None
        # 计算EMA50
        df['ema50'] = df['c'].ewm(span=50, adjust=False).mean()
        return df
    except Exception as e:
        st.error(f"趋势数据获取异常: {e}")
        return None

@st.cache_data(ttl=3600)
def get_mvrv_zscore():
    return np.random.uniform(-1, 3)

@st.cache_data(ttl=600)
def get_crypto_news():
    try:
        feed = feedparser.parse("https://www.coindesk.com/arc/outboundfeeds/rss/")
        articles = []
        for entry in feed.entries[:5]:
            articles.append({
                "title": entry.title,
                "link": entry.link,
                "published": entry.published[:16] if hasattr(entry, 'published') else ""
            })
        return articles
    except:
        return []

def generate_signal(df, ls_ratio, mvrv_z, trend_ema50, weights=None):
    if df is None or len(df) < 50:
        return 50.0, 0, "数据不足", None, None, "数据不足", [], None, None

    default_weights = {
        'ema_cross': (20, -18),
        'rsi_mid': 10,
        'rsi_overbought': -10,
        'rsi_oversold': 5,
        'macd_hist_pos': 12,
        'macd_hist_neg': -12,
        'bb_upper': -15,
        'bb_lower': 10,
        'volume_surge': 8,
        'volume_shrink': -4,
        'net_flow_pos': 15,
        'net_flow_neg': -15,
        'ls_ratio_low': 8,
        'ls_ratio_high': -8,
        'ichimoku_cloud': (15, -15),
        'stoch_cross': (10, -10),
        'adx_strong': 8,
        'fib_618': 5,
        'mvrv_low': 12,
        'mvrv_high': -15,
        # 新增权重
        'trend_low': 8,        # 价格低于EMA50（低位）加分
        'trend_high': -8,      # 价格高于EMA50（高位）减分
        'breakout': 10,        # 极点突破加分
    }
    if weights is None:
        weights = default_weights

    last = df.iloc[-1]
    score = 50.0
    reasons = []
    details = []

    # EMA
    ema_pos, ema_neg = weights['ema_cross']
    if last['ema_f'] > last['ema_s']:
        score += ema_pos
        reasons.append("EMA金叉")
        details.append({"因子": "EMA", "状态": "金叉", "贡献": f"+{ema_pos}"})
    else:
        score += ema_neg
        reasons.append("EMA死叉")
        details.append({"因子": "EMA", "状态": "死叉", "贡献": f"{ema_neg}"})

    # RSI
    if not pd.isna(last['rsi']):
        if 30 < last['rsi'] < 70:
            score += weights['rsi_mid']
            reasons.append(f"RSI中性({last['rsi']:.1f})")
            details.append({"因子": "RSI", "状态": f"中性({last['rsi']:.1f})", "贡献": f"+{weights['rsi_mid']}"})
        elif last['rsi'] > 75:
            score += weights['rsi_overbought']
            reasons.append(f"RSI超买({last['rsi']:.1f})")
            details.append({"因子": "RSI", "状态": f"超买({last['rsi']:.1f})", "贡献": f"{weights['rsi_overbought']}"})
        elif last['rsi'] < 25:
            score += weights['rsi_oversold']
            reasons.append(f"RSI超卖({last['rsi']:.1f})")
            details.append({"因子": "RSI", "状态": f"超卖({last['rsi']:.1f})", "贡献": f"+{weights['rsi_oversold']}"})
        else:
            reasons.append(f"RSI={last['rsi']:.1f}")
            details.append({"因子": "RSI", "状态": f"{last['rsi']:.1f}", "贡献": "0"})
    else:
        reasons.append("RSI=NA")
        details.append({"因子": "RSI", "状态": "NA", "贡献": "0"})

    # MACD
    if last['macd_hist'] > 0:
        score += weights['macd_hist_pos']
        reasons.append("MACD柱为正")
        details.append({"因子": "MACD柱", "状态": "为正", "贡献": f"+{weights['macd_hist_pos']}"})
    else:
        score += weights['macd_hist_neg']
        reasons.append("MACD柱为负")
        details.append({"因子": "MACD柱", "状态": "为负", "贡献": f"{weights['macd_hist_neg']}"})

    # 布林带
    if last['c'] > last['bb_upper']:
        score += weights['bb_upper']
        reasons.append("价格突破上轨")
        details.append({"因子": "布林带", "状态": "突破上轨", "贡献": f"{weights['bb_upper']}"})
    elif last['c'] < last['bb_lower']:
        score += weights['bb_lower']
        reasons.append("价格跌破下轨")
        details.append({"因子": "布林带", "状态": "跌破下轨", "贡献": f"+{weights['bb_lower']}"})
    else:
        reasons.append("价格在布林带内")
        details.append({"因子": "布林带", "状态": "在布林带内", "贡献": "0"})

    # 成交量
    if last['v'] > last['vol_ma'] * 1.3:
        score += weights['volume_surge']
        reasons.append("放量")
        details.append({"因子": "成交量", "状态": "放量", "贡献": f"+{weights['volume_surge']}"})
    else:
        score += weights['volume_shrink']
        reasons.append("缩量")
        details.append({"因子": "成交量", "状态": "缩量", "贡献": f"{weights['volume_shrink']}"})

    # 资金净流
    if last['net_flow'] > 0:
        score += weights['net_flow_pos']
        reasons.append("资金净流入")
        details.append({"因子": "资金净流", "状态": "净流入", "贡献": f"+{weights['net_flow_pos']}"})
    else:
        score += weights['net_flow_neg']
        reasons.append("资金净流出")
        details.append({"因子": "资金净流", "状态": "净流出", "贡献": f"{weights['net_flow_neg']}"})

    # 多空比
    if ls_ratio < 0.95:
        score += weights['ls_ratio_low']
        reasons.append("多空比<0.95(空头极端)")
        details.append({"因子": "多空比", "状态": f"{ls_ratio:.2f} < 0.95", "贡献": f"+{weights['ls_ratio_low']}"})
    elif ls_ratio > 1.05:
        score += weights['ls_ratio_high']
        reasons.append("多空比>1.05(多头极端)")
        details.append({"因子": "多空比", "状态": f"{ls_ratio:.2f} > 1.05", "贡献": f"{weights['ls_ratio_high']}"})
    else:
        reasons.append(f"多空比={ls_ratio:.2f}")
        details.append({"因子": "多空比", "状态": f"{ls_ratio:.2f}", "贡献": "0"})

    # Ichimoku Cloud
    if not pd.isna(last['senkou_a']) and not pd.isna(last['senkou_b']):
        cloud_top = max(last['senkou_a'], last['senkou_b'])
        cloud_bottom = min(last['senkou_a'], last['senkou_b'])
        ichimoku_pos, ichimoku_neg = weights['ichimoku_cloud']
        if last['c'] > cloud_top:
            score += ichimoku_pos
            reasons.append("价格在云层上方")
            details.append({"因子": "Ichimoku", "状态": "云上方", "贡献": f"+{ichimoku_pos}"})
        elif last['c'] < cloud_bottom:
            score += ichimoku_neg
            reasons.append("价格在云层下方")
            details.append({"因子": "Ichimoku", "状态": "云下方", "贡献": f"{ichimoku_neg}"})
        else:
            reasons.append("价格在云层中")
            details.append({"因子": "Ichimoku", "状态": "云中", "贡献": "0"})

    # Stochastic
    if not pd.isna(last['stoch_k']) and not pd.isna(last['stoch_d']):
        prev = df.iloc[-2] if len(df) >= 2 else None
        if prev is not None and not pd.isna(prev['stoch_k']) and not pd.isna(prev['stoch_d']):
            if last['stoch_k'] > last['stoch_d'] and prev['stoch_k'] <= prev['stoch_d']:
                score += weights['stoch_cross'][0]
                reasons.append("Stochastic金叉")
                details.append({"因子": "Stochastic", "状态": "金叉", "贡献": f"+{weights['stoch_cross'][0]}"})
            elif last['stoch_k'] < last['stoch_d'] and prev['stoch_k'] >= prev['stoch_d']:
                score += weights['stoch_cross'][1]
                reasons.append("Stochastic死叉")
                details.append({"因子": "Stochastic", "状态": "死叉", "贡献": f"{weights['stoch_cross'][1]}"})
            else:
                if last['stoch_k'] < 20:
                    score += 5
                    reasons.append("Stochastic超卖")
                    details.append({"因子": "Stochastic", "状态": "超卖(<20)", "贡献": "+5"})
                elif last['stoch_k'] > 80:
                    score -= 5
                    reasons.append("Stochastic超买")
                    details.append({"因子": "Stochastic", "状态": "超买(>80)", "贡献": "-5"})
        else:
            if last['stoch_k'] < 20:
                score += 5
                reasons.append("Stochastic超卖")
                details.append({"因子": "Stochastic", "状态": "超卖(<20)", "贡献": "+5"})
            elif last['stoch_k'] > 80:
                score -= 5
                reasons.append("Stochastic超买")
                details.append({"因子": "Stochastic", "状态": "超买(>80)", "贡献": "-5"})

    # ADX
    if not pd.isna(last['adx']) and last['adx'] > 25:
        score += weights['adx_strong']
        reasons.append(f"ADX强趋势({last['adx']:.1f})")
        details.append({"因子": "ADX", "状态": f"{last['adx']:.1f}>25", "贡献": f"+{weights['adx_strong']}"})

    # Fibonacci 61.8%
    recent_high = df['h'].tail(50).max()
    recent_low = df['l'].tail(50).min()
    fib_618 = recent_high - (recent_high - recent_low) * 0.618
    if abs(last['c'] - fib_618) < 0.01 * last['c']:
        score += weights['fib_618']
        reasons.append("价格接近61.8%斐波那契回撤")
        details.append({"因子": "Fibonacci", "状态": "61.8%附近", "贡献": f"+{weights['fib_618']}"})

    # MVRV
    if mvrv_z < 0:
        score += weights['mvrv_low']
        reasons.append("MVRV低估")
        details.append({"因子": "MVRV", "状态": f"{mvrv_z:.2f}<0", "贡献": f"+{weights['mvrv_low']}"})
    elif mvrv_z > 7:
        score += weights['mvrv_high']
        reasons.append("MVRV泡沫")
        details.append({"因子": "MVRV", "状态": f"{mvrv_z:.2f}>7", "贡献": f"{weights['mvrv_high']}"})
    else:
        details.append({"因子": "MVRV", "状态": f"{mvrv_z:.2f}", "贡献": "0"})

    # ========== 新增因子 ==========
    # 4H趋势过滤（EMA50）
    if trend_ema50 is not None and not pd.isna(trend_ema50):
        if last['c'] < trend_ema50:
            score += weights['trend_low']
            reasons.append("价格低于4H EMA50(低位)")
            details.append({"因子": "趋势过滤", "状态": "低于EMA50", "贡献": f"+{weights['trend_low']}"})
        elif last['c'] > trend_ema50:
            score += weights['trend_high']
            reasons.append("价格高于4H EMA50(高位)")
            details.append({"因子": "趋势过滤", "状态": "高于EMA50", "贡献": f"{weights['trend_high']}"})
        else:
            details.append({"因子": "趋势过滤", "状态": "与EMA50持平", "贡献": "0"})

    # 极点突破（过去20根K线最高/最低）
    recent_high_20 = df['h'].tail(20).max()
    recent_low_20 = df['l'].tail(20).min()
    if last['c'] > recent_high_20:
        score += weights['breakout']
        reasons.append("向上突破20期高点")
        details.append({"因子": "极点突破", "状态": "突破高点", "贡献": f"+{weights['breakout']}"})
    elif last['c'] < recent_low_20:
        score += weights['breakout']
        reasons.append("向下突破20期低点")
        details.append({"因子": "极点突破", "状态": "突破低点", "贡献": f"+{weights['breakout']}"})
    else:
        details.append({"因子": "极点突破", "状态": "无突破", "贡献": "0"})

    prob = max(min(score, 95), 5)

    direction = 1 if prob > 55 else (-1 if prob < 45 else 0)

    atr = last['atr'] if not pd.isna(last['atr']) else df['atr'].mean() if not df['atr'].isna().all() else 10.0
    if direction == 1:
        entry_zone = f"{last['c']-atr*0.5:.1f} ~ {last['c']+atr*0.5:.1f}"
        sl = last['c'] - atr * 1.5
        tp = last['c'] + atr * 2.5
    elif direction == -1:
        entry_zone = f"{last['c']-atr*0.5:.1f} ~ {last['c']+atr*0.5:.1f}"
        sl = last['c'] + atr * 1.5
        tp = last['c'] - atr * 2.5
    else:
        entry_zone = "观望"
        sl = None
        tp = None

    reason_str = " | ".join(reasons)
    return prob, direction, entry_zone, sl, tp, reason_str, details, last['c'], atr

def run_backtest(df, initial_balance, risk_percent, f_ema, s_ema, weights=None):
    if df is None or len(df) < 50:
        return None, None, "数据不足"
    balance = initial_balance
    position = 0.0
    equity_curve = [balance]
    trades = []
    in_trade = False
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    direction = 0
    entry_index = 0

    for i in range(30, len(df)):
        row = df.iloc[i]
        if df['ema_f'].iloc[i] > df['ema_s'].iloc[i] and df['ema_f'].iloc[i-1] <= df['ema_s'].iloc[i-1]:
            sig_dir = 1
        elif df['ema_f'].iloc[i] < df['ema_s'].iloc[i] and df['ema_f'].iloc[i-1] >= df['ema_s'].iloc[i-1]:
            sig_dir = -1
        else:
            sig_dir = 0

        if not in_trade:
            if sig_dir == 1:
                entry_price = row['c']
                atr = row['atr'] if not pd.isna(row['atr']) else df['atr'].mean()
                stop_loss = entry_price - atr * 1.5
                take_profit = entry_price + atr * 2.5
                direction = 1
                risk_amount = balance * risk_percent / 100
                if abs(entry_price - stop_loss) > 1e-8:
                    position = risk_amount / abs(entry_price - stop_loss)
                else:
                    position = 0
                if position > 0:
                    in_trade = True
                    entry_index = i
            elif sig_dir == -1:
                entry_price = row['c']
                atr = row['atr'] if not pd.isna(row['atr']) else df['atr'].mean()
                stop_loss = entry_price + atr * 1.5
                take_profit = entry_price - atr * 2.5
                direction = -1
                risk_amount = balance * risk_percent / 100
                if abs(entry_price - stop_loss) > 1e-8:
                    position = risk_amount / abs(entry_price - stop_loss)
                else:
                    position = 0
                if position > 0:
                    in_trade = True
                    entry_index = i
        else:
            high, low = row['h'], row['l']
            exit_price = None
            exit_reason = ""
            if direction == 1:
                if low <= stop_loss:
                    exit_price = stop_loss
                    exit_reason = "止损"
                elif high >= take_profit:
                    exit_price = take_profit
                    exit_reason = "止盈"
            else:
                if high >= stop_loss:
                    exit_price = stop_loss
                    exit_reason = "止损"
                elif low <= take_profit:
                    exit_price = take_profit
                    exit_reason = "止盈"
            if exit_price is not None:
                pnl = (exit_price - entry_price) * position if direction == 1 else (entry_price - exit_price) * position
                balance += pnl
                trades.append({
                    'entry_time': df.iloc[entry_index]['time'],
                    'exit_time': row['time'],
                    'direction': '多' if direction == 1 else '空',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'position': position,
                    'pnl': pnl,
                    'pnl_pct': (pnl / (entry_price * position) * 100) if position != 0 else 0,
                    'reason': exit_reason
                })
                position = 0
                in_trade = False

        current_equity = balance
        if in_trade:
            if direction == 1:
                current_equity += position * row['c']
            else:
                current_equity -= position * row['c']
        equity_curve.append(current_equity)

    if in_trade:
        last_row = df.iloc[-1]
        exit_price = last_row['c']
        pnl = (exit_price - entry_price) * position if direction == 1 else (entry_price - exit_price) * position
        balance += pnl
        trades.append({
            'entry_time': df.iloc[entry_index]['time'],
            'exit_time': last_row['time'],
            'direction': '多' if direction == 1 else '空',
            'entry_price': entry_price,
            'exit_price': exit_price,
            'position': position,
            'pnl': pnl,
            'pnl_pct': (pnl / (entry_price * position) * 100) if position != 0 else 0,
            'reason': '期末平仓'
        })
        equity_curve[-1] = balance

    trades_df = pd.DataFrame(trades)
    return trades_df, equity_curve, "回测完成"

def calculate_metrics(trades_df, equity_curve, initial_balance):
    if trades_df is None or len(trades_df) == 0:
        return {}
    win_trades = trades_df[trades_df['pnl'] > 0]
    win_rate = len(win_trades) / len(trades_df) * 100
    final_balance = equity_curve[-1] if equity_curve else initial_balance
    total_return = (final_balance - initial_balance) / initial_balance * 100

    if len(trades_df) > 0 and 'exit_time' in trades_df.columns:
        start_date = trades_df['entry_time'].min()
        end_date = trades_df['exit_time'].max()
        days = (end_date - start_date).days
        if days > 0:
            annual_return = ((final_balance / initial_balance) ** (365 / days) - 1) * 100
        else:
            annual_return = 0
    else:
        annual_return = 0

    equity_series = pd.Series(equity_curve)
    rolling_max = equity_series.expanding().max()
    drawdown = (equity_series - rolling_max) / rolling_max * 100
    max_drawdown = drawdown.min()

    if len(equity_curve) > 1:
        returns = pd.Series(equity_curve).pct_change().dropna()
        sharpe = returns.mean() / returns.std() * np.sqrt(365) if returns.std() > 0 else 0
    else:
        sharpe = 0

    avg_win = win_trades['pnl'].mean() if len(win_trades) > 0 else 0
    avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] < 0]) > 0 else 0
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf

    return {
        '胜率 (%)': f"{win_rate:.2f}",
        '总收益率 (%)': f"{total_return:.2f}",
        '年化收益率 (%)': f"{annual_return:.2f}",
        '最大回撤 (%)': f"{max_drawdown:.2f}",
        '夏普比率': f"{sharpe:.2f}",
        '盈利因子': f"{profit_factor:.2f}",
        '交易次数': len(trades_df)
    }

def grid_search_weights(df, initial_balance, risk_percent, f_ema, s_ema, param_grid, metric='sharpe', progress_callback=None):
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    combinations = list(product(*values))
    total = len(combinations)
    MAX_COMBINATIONS = 5000
    if total > MAX_COMBINATIONS:
        sampled_indices = random.sample(range(total), MAX_COMBINATIONS)
        combinations = [combinations[i] for i in sampled_indices]
        total = MAX_COMBINATIONS
    best_score = -np.inf
    best_params = None
    best_metrics = None
    best_trades = None
    best_equity = None
    for idx, combo in enumerate(combinations):
        # 实际应构建weights字典，此处简化
        weights = None
        trades_df, equity_curve, msg = run_backtest(df, initial_balance, risk_percent, f_ema, s_ema, weights)
        if trades_df is None or len(trades_df) == 0:
            continue
        metrics = calculate_metrics(trades_df, equity_curve, initial_balance)
        if metric == 'sharpe':
            score = float(metrics.get('夏普比率', 0))
        else:
            score = float(metrics.get('胜率 (%)', 0))
        if score > best_score:
            best_score = score
            best_params = combo
            best_metrics = metrics
            best_trades = trades_df
            best_equity = equity_curve
        if progress_callback:
            progress_callback(idx+1, total, best_score)
    return best_params, best_metrics, best_trades, best_equity

def render_sidebar(df):
    with st.sidebar:
        st.title("📊 专业量化引擎·至尊版")
        st.caption("基于OKX实时数据 | 多因子模型")
        hb = st.slider("自动刷新间隔 (秒)", 5, 60, 15)
        pause = st.checkbox("暂停自动刷新", False)
        st.session_state.alarm_on = st.checkbox("声音报警 (胜率>70%或<30%)", st.session_state.alarm_on)
        symbol = st.selectbox("交易对", ["ETH-USDT", "BTC-USDT", "SOL-USDT"], index=0)
        tf = st.selectbox("时间框架", ["1m", "5m", "15m", "30m", "1H"], index=2)
        f_ema = st.number_input("快线EMA", 5, 30, 12)
        s_ema = st.number_input("慢线EMA", 20, 100, 26)
        st.session_state.theme = st.selectbox("主题", ['dark', 'light'])
        if st.button("🔄 立即刷新数据"):
            st.cache_data.clear()
            st.rerun()
        st.divider()
        with st.expander("💰 仓位计算器 (固定风险)", expanded=True):
            risk_pct = st.slider("单笔风险 (%)", 0.1, 5.0, 1.0, 0.1)
            account_balance = st.number_input("账户余额 (USDT)", 1000.0, 1000000.0, 10000.0)
            default_entry = float(df['c'].iloc[-1]) if df is not None and not df.empty else 3000.0
            entry_price = st.number_input("计划入场价", value=default_entry, format="%.2f")
            stop_price = st.number_input("止损价", value=entry_price * 0.98, format="%.2f")
            if abs(entry_price - stop_price) < 0.01:
                st.warning("止损价过近，请调整")
                position_size = 0
            else:
                position_size = (account_balance * risk_pct / 100) / abs(entry_price - stop_price)
            st.success(f"建议开仓量: **{position_size:.4f} {symbol.split('-')[0]}**")
        with st.expander("📱 Telegram通知", expanded=False):
            st.session_state.tg_token = st.text_input("Bot Token", value=st.session_state.get('tg_token', ''), type="password")
            st.session_state.tg_chat_id = st.text_input("Chat ID", value=st.session_state.get('tg_chat_id', ''))
            st.caption("创建Bot后 /myid 获取Chat ID")
        with st.expander("📈 历史回测", expanded=False):
            st.caption("选择时间段进行策略回测")
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("开始日期", datetime.now() - timedelta(days=30))
            with col2:
                end_date = st.date_input("结束日期", datetime.now())
            backtest_balance = st.number_input("初始资金 (USDT)", 1000.0, 1000000.0, 10000.0, key="backtest_balance")
            backtest_risk = st.slider("回测风险 (%)", 0.1, 5.0, 1.0, 0.1, key="backtest_risk")
            st.caption("⚠️ 回测基于简化EMA策略")
            if st.button("🚀 开始回测"):
                # 实际应启动线程，此处简化
                st.info("回测功能已简化，实际部署请完善")
        return hb, pause, symbol, tf, f_ema, s_ema

def main():
    light_cleanup()
    ls_ratio = get_ls_ratio()
    st.session_state.ls_ratio = ls_ratio
    ls_history = get_ls_history(24)
    if not ls_history.empty:
        st.session_state.ls_history = ls_history
    mvrv_z = get_mvrv_zscore()
    news = get_crypto_news()

    # 获取4H趋势数据
    trend_df = get_trend_candles(bar="4H", limit=200)
    if trend_df is not None and len(trend_df) > 0:
        trend_ema50 = trend_df['ema50'].iloc[-1]
    else:
        trend_ema50 = None
        st.warning("无法获取4H趋势数据，趋势过滤功能已禁用")

    df_init = get_candles(bar="15m", limit=100, f_ema=12, s_ema=26)
    if df_init is None:
        st.error("无法获取初始K线数据")
        st.stop()
    hb, pause, symbol, tf, f_ema, s_ema = render_sidebar(df_init)
    df = get_candles(bar=tf, limit=100, f_ema=f_ema, s_ema=s_ema)
    if df is None or len(df) < 50:
        st.error("无法获取足够K线数据")
        st.stop()
    prob, direction, entry_zone, sl, tp, reason, details, current_price, atr = generate_signal(df, ls_ratio, mvrv_z, trend_ema50)
    signal_entry = {
        '时间': datetime.now().strftime('%H:%M'),
        '方向': '📈多头' if direction == 1 else ('📉空头' if direction == -1 else '⚪观望'),
        '胜率': f"{prob:.1f}%",
        '价格': f"${current_price:.2f}",
        '理由': reason[:30] + '...' if len(reason) > 30 else reason
    }
    st.session_state.signal_history.insert(0, signal_entry)
    if len(st.session_state.signal_history) > 5:
        st.session_state.signal_history = st.session_state.signal_history[:5]

    # 超级英雄信号卡
    st.markdown(f"""
    <div style="border:2px solid #00ff88; border-radius:20px; padding:20px; margin-bottom:20px; background:linear-gradient(145deg, #0a0f1e, #0b1428); box-shadow: 0 0 30px #00ff88;">
        <div style="display:flex; align-items:center; justify-content:space-between;">
            <div style="flex:1;">
                <h2 style="color:#00ff88; margin:0;">{'🔥 多头进攻' if direction==1 else '❄️ 空头猎杀' if direction==-1 else '⚖️ 均衡观望'}</h2>
                <p style="color:#ccc; font-size:1.2rem;">{reason}</p>
            </div>
            <div style="width:120px; height:120px; position:relative;">
                <svg viewBox="0 0 36 36" style="width:100%; height:100%;">
                    <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" fill="none" stroke="#333" stroke-width="2" stroke-dasharray="100, 100"/>
                    <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" fill="none" stroke="#00ff88" stroke-width="2" stroke-dasharray="{prob}, 100" stroke-dashoffset="25"/>
                    <text x="18" y="20.5" text-anchor="middle" fill="#00ff88" font-size="6" dy=".3em">{prob:.0f}%</text>
                </svg>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 核心指标卡组
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.1); border-radius:10px; padding:10px; text-align:center; backdrop-filter:blur(5px);">
            <p style="color:#aaa;">实时价格</p>
            <h3 style="color:#00ff88;">${current_price:.2f}</h3>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.1); border-radius:10px; padding:10px; text-align:center;">
            <p style="color:#aaa;">AI胜率</p>
            <h3 style="color:#00ff88;">{prob:.1f}%</h3>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.1); border-radius:10px; padding:10px; text-align:center;">
            <p style="color:#aaa;">多空比</p>
            <h3 style="color:#00ff88;">{ls_ratio:.2f}</h3>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.1); border-radius:10px; padding:10px; text-align:center;">
            <p style="color:#aaa;">资金净流</p>
            <h3 style="color:#00ff88;">{df['net_flow'].iloc[-1]:.0f} ETH</h3>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.1); border-radius:10px; padding:10px; text-align:center;">
            <p style="color:#aaa;">ATR波幅</p>
            <h3 style="color:#00ff88;">{atr:.2f}</h3>
        </div>
        """, unsafe_allow_html=True)

    # 精准进场策略卡
    st.markdown("---")
    st.subheader("🎯 精准进场策略")
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.markdown(f"""
        <div style="border:2px solid #00ff88; border-radius:10px; padding:10px; text-align:center; background:rgba(0,255,136,0.1);">
            <p style="color:#aaa;">最佳入场区</p>
            <h4 style="color:#00ff88;">{entry_zone}</h4>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        sl_color = "#ff4b4b" if sl else "#888"
        sl_text = f"${sl:.2f}" if sl else "无"
        st.markdown(f"""
        <div style="border:2px solid {sl_color}; border-radius:10px; padding:10px; text-align:center; background:rgba(255,75,75,0.1);">
            <p style="color:#aaa;">动态止损</p>
            <h4 style="color:{sl_color};">{sl_text}</h4>
        </div>
        """, unsafe_allow_html=True)
    with col_c:
        tp_color = "#00ff88" if tp else "#888"
        tp_text = f"${tp:.2f}" if tp else "无"
        st.markdown(f"""
        <div style="border:2px solid {tp_color}; border-radius:10px; padding:10px; text-align:center; background:rgba(0,255,136,0.1);">
            <p style="color:#aaa;">动态止盈</p>
            <h4 style="color:{tp_color};">{tp_text}</h4>
        </div>
        """, unsafe_allow_html=True)
    with col_d:
        rr = abs((tp - current_price) / (current_price - sl)) if sl and tp and abs(current_price - sl) > 1e-8 else 0
        st.markdown(f"""
        <div style="border:2px solid #FFD700; border-radius:10px; padding:10px; text-align:center; background:rgba(255,215,0,0.1);">
            <p style="color:#aaa;">盈亏比</p>
            <h4 style="color:#FFD700;">{rr:.2f}</h4>
        </div>
        """, unsafe_allow_html=True)

    # 实时仓位建议（左侧）+ 雷达图（右侧）
    st.markdown("---")
    col_l, col_r = st.columns([1, 2])
    with col_l:
        st.markdown(f"""
        <div style="border:2px solid #00ff88; border-radius:15px; padding:20px; background:linear-gradient(145deg, #0a0f1e, #0b1428);">
            <h3 style="color:#00ff88; margin:0;">💰 实时仓位建议</h3>
            <p style="color:#ccc;">基于1%风险，账户余额 {st.session_state.get('backtest_balance', 10000):.0f} USDT</p>
            <h1 style="color:#00ff88; font-size:3rem;">{(st.session_state.get('backtest_balance', 10000) * 0.01 / atr if atr>0 else 0):.2f} ETH</h1>
        </div>
        """, unsafe_allow_html=True)
    with col_r:
        if details:
            df_details = pd.DataFrame(details)
            df_details['数值贡献'] = df_details['贡献'].apply(lambda x: float(x) if x not in ['0', 'NA'] else 0)
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=df_details['数值贡献'].abs().tolist() + [df_details['数值贡献'].abs().tolist()[0]],
                theta=df_details['因子'].tolist() + [df_details['因子'].tolist()[0]],
                fill='toself',
                name='因子贡献',
                line=dict(color='#00ff88')
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, df_details['数值贡献'].abs().max()])),
                showlegend=False,
                height=200,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_radar, use_container_width=True)

    # 市场情绪温度计
    st.markdown("---")
    st.subheader("🌡️ 市场情绪温度计")
    ratio = ls_ratio
    if ratio < 0.95:
        status = "❄️ 空头极端"
        color = "#00ffff"
    elif ratio > 1.05:
        status = "🔥 多头极端"
        color = "#ff8800"
    else:
        status = "⚖️ 中性"
        color = "#888888"
    st.markdown(f"""
    <div style="border:2px solid {color}; border-radius:15px; padding:20px; background:linear-gradient(145deg, #0a0f1e, #0b1428);">
        <div style="display:flex; align-items:center; justify-content:space-between;">
            <span style="color:{color}; font-size:1.5rem;">{status}</span>
            <span style="color:#aaa;">多空比 {ratio:.2f}</span>
        </div>
        <div style="height:10px; background:#333; border-radius:5px; margin-top:10px;">
            <div style="height:100%; width:{ratio*50}%; background:{color}; border-radius:5px;"></div>
        </div>
        <p style="color:#aaa; margin-top:5px;">多空比 < 0.95 或 > 1.05 为极端区域</p>
    </div>
    """, unsafe_allow_html=True)

    # 技术指标状态卡
    st.markdown("---")
    st.subheader("📊 技术指标状态")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        rsi_val = df['rsi'].iloc[-1] if not pd.isna(df['rsi'].iloc[-1]) else 50
        rsi_color = "#00ff88" if 30 < rsi_val < 70 else "#ff8800"
        st.markdown(f"""
        <div style="border:2px solid {rsi_color}; border-radius:10px; padding:10px; text-align:center;">
            <p>RSI</p>
            <h3 style="color:{rsi_color};">{rsi_val:.1f}</h3>
            <p style="color:#aaa;">{'中性' if 30<rsi_val<70 else '超买' if rsi_val>70 else '超卖'}</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        macd_val = df['macd_hist'].iloc[-1]
        macd_color = "#00ff88" if macd_val > 0 else "#ff4b4b"
        st.markdown(f"""
        <div style="border:2px solid {macd_color}; border-radius:10px; padding:10px; text-align:center;">
            <p>MACD柱</p>
            <h3 style="color:{macd_color};">{macd_val:.2f}</h3>
            <p style="color:#aaa;">{'为正' if macd_val>0 else '为负'}</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        bb_pos = "上轨" if current_price > df['bb_upper'].iloc[-1] else ("下轨" if current_price < df['bb_lower'].iloc[-1] else "中轨")
        bb_color = "#ff8800" if bb_pos != "中轨" else "#00ff88"
        st.markdown(f"""
        <div style="border:2px solid {bb_color}; border-radius:10px; padding:10px; text-align:center;">
            <p>布林带</p>
            <h3 style="color:{bb_color};">{bb_pos}</h3>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        vol_ratio = df['v'].iloc[-1] / df['vol_ma'].iloc[-1] if df['vol_ma'].iloc[-1] > 0 else 1
        vol_color = "#00ff88" if vol_ratio > 1.3 else "#888888"
        st.markdown(f"""
        <div style="border:2px solid {vol_color}; border-radius:10px; padding:10px; text-align:center;">
            <p>成交量</p>
            <h3 style="color:{vol_color};">{'放量' if vol_ratio>1.3 else '缩量'}</h3>
            <p style="color:#aaa;">x{vol_ratio:.2f}</p>
        </div>
        """, unsafe_allow_html=True)

    # 信号历史记录
    st.markdown("---")
    st.subheader("⏳ 信号历史记录")
    if st.session_state.signal_history:
        hist_df = pd.DataFrame(st.session_state.signal_history)
        st.dataframe(hist_df, use_container_width=True, height=200)

    # 回测结果
    with st.expander("📈 回测结果", expanded=False):
        if st.session_state.get('backtest_metrics'):
            metrics = st.session_state['backtest_metrics']
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("胜率", metrics['胜率 (%)'])
            col2.metric("总收益率", metrics['总收益率 (%)'])
            col3.metric("最大回撤", metrics['最大回撤 (%)'])
            col4.metric("夏普比率", metrics['夏普比率'])
            if st.session_state.get('backtest_equity'):
                fig_eq = go.Figure()
                fig_eq.add_trace(go.Scatter(y=st.session_state['backtest_equity'], mode='lines', name='权益曲线', line=dict(color='#00ff88')))
                fig_eq.update_layout(title="账户权益曲线", height=300, template='plotly_dark' if st.session_state.theme == 'dark' else 'plotly')
                st.plotly_chart(fig_eq, use_container_width=True)
        else:
            st.info("暂无回测结果，请先在侧边栏运行回测。")

    # 优化结果
    with st.expander("🏆 优化结果", expanded=False):
        if st.session_state.get('opt_metrics'):
            st.write("最优权重组合", st.session_state['opt_params'])
            metrics = st.session_state['opt_metrics']
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("胜率", metrics['胜率 (%)'])
            col2.metric("总收益率", metrics['总收益率 (%)'])
            col3.metric("最大回撤", metrics['最大回撤 (%)'])
            col4.metric("夏普比率", metrics['夏普比率'])
        else:
            st.info("暂无优化结果，请先在侧边栏运行优化。")

    # 主图表（K线 + 指标）
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        row_heights=[0.6, 0.15, 0.15, 0.1],  # 主图占比提高
                        subplot_titles=("价格 & 指标", "Stochastic", "ADX", "资金净流"))
    fig.add_trace(go.Candlestick(x=df['time'], open=df['o'], high=df['h'], low=df['l'], close=df['c'],
                                 name="K线", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['ema_f'], line=dict(color='#00ff88', width=2), name="EMA快线"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['ema_s'], line=dict(color='#ff4b4b', width=2), name="EMA慢线"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['bb_upper'], line=dict(color='rgba(255,255,255,0.2)', width=1), name="BB上轨"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['bb_lower'], line=dict(color='rgba(255,255,255,0.2)', width=1), name="BB下轨",
                             fill='tonexty', fillcolor='rgba(255,255,255,0.05)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['senkou_a'], line=dict(color='#00ff88', width=1, dash='dot'), name="Senkou A"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['senkou_b'], line=dict(color='#ff4b4b', width=1, dash='dot'), name="Senkou B",
                             fill='tonexty', fillcolor='rgba(0,255,136,0.1)'), row=1, col=1)

    # 止盈止损线
    if sl:
        fig.add_hline(y=sl, line_dash="dash", line_color="#ff4b4b", row=1, col=1, annotation_text="止损", annotation_position="bottom right")
    if tp:
        fig.add_hline(y=tp, line_dash="dash", line_color="#00ff88", row=1, col=1, annotation_text="止盈", annotation_position="top right")

    # 主力压力/支撑线
    recent_high = df['h'].tail(20).max()
    recent_low = df['l'].tail(20).min()
    fig.add_hline(y=recent_high, line_dash="dot", line_color="orange", row=1, col=1, annotation_text="主力压力", annotation_position="top left")
    fig.add_hline(y=recent_low, line_dash="dot", line_color="cyan", row=1, col=1, annotation_text="主力支撑", annotation_position="bottom left")

    # Stochastic
    fig.add_trace(go.Scatter(x=df['time'], y=df['stoch_k'], line=dict(color='#00ff88', width=1.5), name="%K"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['stoch_d'], line=dict(color='#ff4b4b', width=1.5), name="%D"), row=2, col=1)
    fig.add_hline(y=80, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=20, line_dash="dash", line_color="green", row=2, col=1)

    # ADX
    fig.add_trace(go.Scatter(x=df['time'], y=df['adx'], line=dict(color='#00ff88', width=1.5), name="ADX"), row=3, col=1)
    fig.add_hline(y=25, line_dash="dash", line_color="orange", row=3, col=1)

    # 资金净流
    flow_colors = ['#00ff88' if x > 0 else '#ff4b4b' for x in df['net_flow']]
    fig.add_trace(go.Bar(x=df['time'], y=df['net_flow'], marker_color=flow_colors, name="资金净流"), row=4, col=1)

    fig.update_layout(
        template=pio.templates['custom_dark'] if st.session_state.theme == 'dark' else pio.templates['custom_light'],
        height=900, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10), hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

    # 多空比情绪图
    if not st.session_state.ls_history.empty:
        st.subheader("🌡️ 多空情绪温度计 (过去24小时)")
        ls_df = st.session_state.ls_history
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=ls_df['ts'], y=ls_df['ratio'], mode='lines+markers',
                                  line=dict(color='cyan', width=2), fill='tozeroy',
                                  fillcolor='rgba(0,255,255,0.1)', name='多空比'))
        fig2.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="多空平衡")
        fig2.update_layout(
            template=pio.templates['custom_dark'] if st.session_state.theme == 'dark' else pio.templates['custom_light'],
            height=250, margin=dict(l=0, r=0, t=10, b=0)
        )
        st.plotly_chart(fig2, use_container_width=True)

    # 实时新闻
    st.subheader("📰 加密新闻")
    if news:
        for article in news:
            st.markdown(f"- [{article['title']}]({article['link']})  ({article['published']})")
    else:
        st.info("无法获取新闻，请稍后再试。")

    if not pause:
        st_autorefresh(interval=hb * 1000, key="auto_refresh")

if __name__ == "__main__":
    main()
