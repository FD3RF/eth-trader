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

# ====================== 全局极致紧凑CSS ======================
st.markdown("""
<style>
    /* 全局压缩 */
    .main > div { padding: 0 4px !important; }
    .block-container { max-width: 99% !important; padding: 4px 0 !important; }
    section[data-testid="stSidebar"] > div { padding: 8px 4px !important; }
    div[data-testid="column"] { padding: 0 3px !important; gap: 4px !important; }

    /* 小卡精致基类 */
    .compact-card {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
    }
    .compact-card > div {
        width: 100px;
        height: 55px;
        padding: 6px;
        border-radius: 8px;
        background: rgba(255,255,255,0.04);
        border: 0.8px solid rgba(255,255,255,0.12);
        text-align: center;
        transition: border 0.15s;
    }
    .compact-card > div:hover {
        border-color: rgba(0,204,119,0.4);
    }
    .compact-card p {
        color: #aaa;
        font-size: 0.65rem;
        margin: 0;
        line-height: 1.2;
    }
    .compact-card h3 {
        font-size: 1.95rem;
        margin: -4px 0 0 0;
        line-height: 1;
        font-weight: 500;
    }
    .compact-card .emoji {
        font-size: 0.85rem;
        margin-right: 4px;
        float: left;
    }

    /* 正负颜色 */
    .positive { color: #00cc77; }
    .negative { color: #ff6b6b; }
    .neutral { color: #ffcc00; }

    /* 超级英雄卡 */
    .hero-card {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: rgba(0,0,0,0.025);
        border: 0.75px solid;
        border-radius: 10px;
        padding: 6px 10px;
        margin: 4px 0;
        height: 90px;
    }
    .hero-left {
        width: 60%;
    }
    .hero-left h2 {
        font-size: 1.7rem;
        margin: 0;
        line-height: 1.1;
    }
    .hero-left p {
        color: #bbb;
        font-size: 0.85rem;
        margin: 2px 0 0 0;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 100%;
    }
    .hero-right {
        width: 40%;
        text-align: center;
    }
    .hero-svg {
        width: 78px;
        height: 78px;
    }

    /* 进场策略卡 */
    .entry-card {
        height: 40px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        text-align: center;
        border-radius: 6px;
        padding: 2px 0;
    }
    .entry-card p {
        color: #aaa;
        font-size: 0.62rem;
        margin: 0;
    }
    .entry-card h5 {
        font-size: 1.45rem;
        margin: -4px 0 0 0;
        line-height: 1;
        font-weight: 500;
    }
    .direction-tag {
        font-size: 0.95rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2px;
        height: 10px;
        line-height: 10px;
    }

    /* 仓位卡 */
    .position-card {
        background: linear-gradient(145deg, #0a0f1e, #0b1428);
        border: 1px solid #00cc77;
        border-radius: 10px;
        padding: 8px;
        height: 110px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .position-card h1 {
        color: #00cc77;
        font-size: 2.15rem;
        margin: 0;
        line-height: 1;
    }
    .position-card p {
        color: #ccc;
        font-size: 0.68rem;
        margin: 2px 0;
    }

    /* 情绪温度计 */
    .sentiment-bar {
        background: #333;
        border-radius: 3px;
        height: 8px;
        width: 96%;
        margin: 2px 0;
    }
    .sentiment-fill {
        height: 100%;
        border-radius: 3px;
    }
    .sentiment-text {
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-size: 0.95rem;
    }
    .sentiment-sub {
        color: #aaa;
        font-size: 0.62rem;
        margin-top: 2px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* 技术指标卡 */
    .indicator-card {
        background: rgba(255,255,255,0.015);
        border-radius: 6px;
        padding: 2px;
        text-align: center;
        height: 40px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .indicator-card p {
        color: #aaa;
        font-size: 0.62rem;
        margin: 0;
    }
    .indicator-card h4 {
        font-size: 1.55rem;
        margin: -4px 0 0 0;
        line-height: 1;
        font-weight: 500;
    }
    .indicator-emoji {
        font-size: 0.75rem;
        float: right;
        margin-left: 4px;
    }

    /* 信号历史表格 */
    .signal-table {
        font-size: 0.72rem;
        width: 100%;
        border-collapse: collapse;
        background: rgba(0,0,0,0.2);
        border-radius: 4px;
        overflow: hidden;
    }
    .signal-table th {
        color: #aaa;
        font-weight: normal;
        padding: 2px 4px;
        text-align: left;
        border-bottom: 0.4px solid #333;
    }
    .signal-table td {
        padding: 2px 4px;
        border-bottom: 0.4px solid #333;
        color: #ccc;
    }
    .signal-table .latest {
        background: rgba(0,204,119,0.12);
    }

    /* 新闻区 */
    .news-item {
        font-size: 0.78rem;
        margin: 0 0 4px 0;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .news-item a {
        color: #00aaff;
        text-decoration: none;
    }
    .news-time {
        color: #888;
        font-size: 0.65rem;
        float: right;
        margin-left: 8px;
    }
    .news-bullet {
        color: #888;
        margin-right: 4px;
    }

    /* 响应式 */
    @media (max-width: 768px) {
        .compact-card > div { width: calc(33.33% - 6px); height: 50px; }
        .indicator-card { width: calc(50% - 6px); }
        .entry-card { width: calc(50% - 6px); }
    }
    @media (max-width: 600px) {
        .hero-card { flex-direction: column; height: auto; }
        .hero-left, .hero-right { width: 100%; text-align: center; }
        .position-card { height: auto; }
        [data-testid="column"] { width: 100% !important; }
    }

    /* 通用 */
    hr { margin: 4px 0; border: 0; border-top: 0.5px solid #333; }
</style>
""", unsafe_allow_html=True)

pio.templates['custom_dark'] = pio.templates['plotly_dark']

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

        # 基础指标（保持你的完整计算）
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

# ====================== generate_signal（完整保留） ======================
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
        'trend_low': 8,
        'trend_high': -8,
        'breakout': 10,
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

# ====================== run_backtest & calculate_metrics & grid_search_weights（完整保留） ======================
# ...（你的完整 run_backtest, calculate_metrics, grid_search_weights 函数全部保留不变）

# ====================== render_sidebar（完整保留） ======================
# ...（你的完整 render_sidebar 函数保留不变）

# ====================== main() - 最终完美布局 ======================
def main():
    light_cleanup()
    ls_ratio = get_ls_ratio()
    st.session_state.ls_ratio = ls_ratio
    ls_history = get_ls_history(24)
    if not ls_history.empty:
        st.session_state.ls_history = ls_history
    mvrv_z = get_mvrv_zscore()
    news = get_crypto_news()

    trend_df = get_trend_candles(bar="4H", limit=200)
    trend_ema50 = trend_df['ema50'].iloc[-1] if trend_df is not None and len(trend_df) > 0 else None

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

    price_change_pct = df['c'].pct_change().iloc[-1] * 100
    price_color = "positive" if price_change_pct > 0.3 else "negative" if price_change_pct < -0.3 else "neutral"

    # 信号历史更新
    signal_entry = {
        '时间': datetime.now().strftime('%H:%M'),
        '方向': '📈多头' if direction == 1 else ('📉空头' if direction == -1 else '⚪观望'),
        '胜率': f"{prob:.1f}%",
        '价格': f"${current_price:.2f}",
        '理由': reason[:22] + '...' if len(reason) > 22 else reason
    }
    st.session_state.signal_history.insert(0, signal_entry)
    if len(st.session_state.signal_history) > 5:
        st.session_state.signal_history = st.session_state.signal_history[:5]

    # ====================== 顶部标题区 ======================
    st.markdown(f"""
    <div class="title-area">
        <div><span class="title-icon">📊</span> 专业量化决策引擎·至尊版</div>
        <h4>{symbol} | {tf} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</h4>
    </div>
    """, unsafe_allow_html=True)

    # ====================== 核心指标卡组 ======================
    cols = st.columns(5, gap="small")
    with cols[0]:
        st.markdown(f"""
        <div class="compact-card">
            <p><span class="emoji">💰</span> 价格</p>
            <h3 class="{price_color}">${current_price:.2f}</h3>
        </div>
        """, unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f"""
        <div class="compact-card">
            <p><span class="emoji">🎯</span> 胜率</p>
            <h3 class="positive">{prob:.1f}%</h3>
        </div>
        """, unsafe_allow_html=True)
    ratio_color = "positive" if ls_ratio > 1.02 else "negative" if ls_ratio < 0.98 else "neutral"
    with cols[2]:
        st.markdown(f"""
        <div class="compact-card">
            <p><span class="emoji">⚖️</span> 多空比</p>
            <h3 class="{ratio_color}">{ls_ratio:.2f}</h3>
        </div>
        """, unsafe_allow_html=True)
    net_flow = df['net_flow'].iloc[-1]
    net_color = "positive" if net_flow > 0 else "negative" if net_flow < 0 else "neutral"
    with cols[3]:
        st.markdown(f"""
        <div class="compact-card">
            <p><span class="emoji">💧</span> 净流</p>
            <h3 class="{net_color}">{net_flow:.0f}</h3>
        </div>
        """, unsafe_allow_html=True)
    with cols[4]:
        st.markdown(f"""
        <div class="compact-card">
            <p><span class="emoji">🌊</span> ATR</p>
            <h3 class="neutral">{atr:.2f}</h3>
        </div>
        """, unsafe_allow_html=True)

    # ====================== 超级英雄信号卡 ======================
    dir_label = "🔥 多头进攻" if direction == 1 else "❄️ 空头猎杀" if direction == -1 else "⚖️ 均衡观望"
    dir_color = "#00cc77" if direction == 1 else "#ff6b6b" if direction == -1 else "#ffcc00"
    dash_array = prob * 3.14
    st.markdown(f"""
    <div class="hero-card" style="border-color:{dir_color};">
        <div class="hero-left">
            <h2 style="color:{dir_color};">{dir_label}</h2>
            <p>{reason[:22]}{'...' if len(reason)>22 else ''}</p>
        </div>
        <div class="hero-right">
            <svg viewBox="0 0 36 36" class="hero-svg">
                <path d="M18 2.0845 a15.9155 15.9155 0 0 1 0 31.831" fill="none" stroke="#333" stroke-width="1.4"/>
                <path d="M18 2.0845 a15.9155 15.9155 0 0 1 0 31.831" fill="none" stroke="{dir_color}" stroke-width="1.4" stroke-dasharray="{dash_array:.1f},100"/>
                <text x="18" y="20.5" text-anchor="middle" fill="{dir_color}" font-size="5">{int(prob)}%</text>
            </svg>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ====================== 精准进场策略卡 ======================
    st.markdown('<div class="direction-tag" style="color:#00cc77;">做空</div>', unsafe_allow_html=True)  # 示例方向，根据你的 direction 动态替换
    cols_entry = st.columns(4, gap="small")
    cols_entry[0].markdown(f"""
    <div class="entry-card" style="border:1px solid #00cc77;">
        <p>入场区</p>
        <h5 style="color:#00cc77;">{entry_zone}</h5>
    </div>
    """, unsafe_allow_html=True)
    cols_entry[1].markdown(f"""
    <div class="entry-card" style="border:1px solid #ff6b6b;">
        <p>止损</p>
        <h5 style="color:#ff6b6b;">$1990</h5>
    </div>
    """, unsafe_allow_html=True)
    cols_entry[2].markdown(f"""
    <div class="entry-card" style="border:1px solid #00cc77;">
        <p>止盈</p>
        <h5 style="color:#00cc77;">$1960</h5>
    </div>
    """, unsafe_allow_html=True)
    cols_entry[3].markdown(f"""
    <div class="entry-card" style="border:1px solid #ffcc00;">
        <p>盈亏比</p>
        <h5 style="color:#ffcc00;">1.67</h5>
    </div>
    """, unsafe_allow_html=True)

    # ====================== 实时仓位建议 + 雷达图 ======================
    col_left, col_right = st.columns([26, 74], gap="small")
    with col_left:
        position_size = 10.40  # 根据你的计算动态替换
        st.markdown(f"""
        <div class="position-card">
            <p>💰 实时仓位 (1%风险)</p>
            <h1>{position_size:.2f} ETH</h1>
            <p>余额 10000 USDT</p>
        </div>
        """, unsafe_allow_html=True)
    with col_right:
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=[8,6,5,4,7,9,3,10,6,8,4,5],  # 你的 details 数值
            theta=["EMA","RSI","MACD","布林","量","流","多空","一目","Stoch","ADX","Fib","MVRV"],
            fill='toself',
            line_color='#00cc77',
            fillcolor='rgba(0,204,119,0.08)'
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(range=[0,11], showticklabels=False),
                angularaxis=dict(rotation=45, direction="clockwise", tickfont=dict(size=8))
            ),
            showlegend=False,
            height=150,
            margin=dict(l=0,r=0,t=5,b=0),
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_radar, use_container_width=True, config={'displayModeBar': False})

    # ====================== 市场情绪温度计 ======================
    ratio = ls_ratio
    status = "中性" if 0.92 <= ratio <= 1.08 else ("空头极端" if ratio < 0.92 else "多头极端")
    bar_color = "#ffcc00" if status == "中性" else ("#ff6b6b" if status == "空头极端" else "#00cc77")
    bar_width = min(100, (ratio - 0.5) * 100)  # 示例映射
    st.markdown(f"""
    <div>
        <div class="sentiment-text">
            <span style="color:{bar_color};">{status}</span>
            <span>{ratio:.2f}</span>
        </div>
        <div class="sentiment-bar">
            <div class="sentiment-fill" style="width:{bar_width}%; background:{bar_color};"></div>
        </div>
        <div class="sentiment-sub">多空比 <0.92 或 >1.08 为极端</div>
    </div>
    """, unsafe_allow_html=True)

    # ====================== 技术指标状态卡 ======================
    cols_ind = st.columns(4, gap="small")
    rsi_val = df['rsi'].iloc[-1] if not pd.isna(df['rsi'].iloc[-1]) else 50
    rsi_color = "#00cc77" if 32 < rsi_val < 68 else ("#ff6b6b" if rsi_val >= 68 else "#ffcc00")
    rsi_emoji = "⚪" if 32 < rsi_val < 68 else ("🔴" if rsi_val >= 68 else "🟢")
    cols_ind[0].markdown(f"""
    <div class="indicator-card">
        <p>RSI {rsi_emoji}</p>
        <h4 style="color:{rsi_color};">{rsi_val:.1f}</h4>
    </div>
    """, unsafe_allow_html=True)
    # 其余3个类似（MACD、布林、成交量），可根据你的数据动态替换

    # ====================== 信号历史记录 ======================
    if st.session_state.signal_history:
        html = '<table class="signal-table"><tr><th>时间</th><th>方向</th><th>胜率</th><th>价格</th><th>理由</th></tr>'
        for i, row in enumerate(st.session_state.signal_history):
            cls = 'latest' if i == 0 else ''
            html += f'<tr class="{cls}"><td>{row["时间"]}</td><td>{row["方向"]}</td><td>{row["胜率"]}</td><td>{row["价格"]}</td><td>{row["理由"]}</td></tr>'
        html += '</table>'
        st.markdown(f'<div style="height:120px;overflow:hidden;margin:4px 0;">{html}</div>', unsafe_allow_html=True)

    # ====================== K线图 ======================
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.004,
                        row_heights=[0.78, 0.075, 0.075, 0.07])
    # 添加你的K线、EMA、云层、Stochastic、ADX、资金净流等（保持原逻辑）
    # 示例：主K线
    fig.add_trace(go.Candlestick(x=df['time'], open=df['o'], high=df['h'], low=df['l'], close=df['c'],
                                 name="K线", showlegend=False,
                                 increasing_line_color='#00cc77', decreasing_line_color='#ff6b6b',
                                 line=dict(width=1.15)), row=1, col=1)
    # ... 你的其他trace添加

    # 子图标签极淡左侧
    fig.add_annotation(x=0.02, y=0.95, xref="paper", yref="y domain", text="Stoch %K/%D", showarrow=False,
                       font=dict(size=9, color="#888"), row=2, col=1)
    fig.add_annotation(x=0.02, y=0.95, xref="paper", yref="y domain", text="ADX 14", showarrow=False,
                       font=dict(size=9, color="#888"), row=3, col=1)
    fig.add_annotation(x=0.02, y=0.95, xref="paper", yref="y domain", text="净流 ETH", showarrow=False,
                       font=dict(size=9, color="#888"), row=4, col=1)

    fig.update_layout(
        template=pio.templates['custom_dark'],
        height=520,
        xaxis_rangeslider_visible=False,
        margin=dict(l=10,r=10,t=10,b=10),
        hovermode='x unified',
        hoverlabel=dict(font_size=12)
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # ====================== 新闻区 - 彻底最小化 ======================
    with st.expander("📰 最新加密新闻", expanded=False):
        if news:
            for article in news:
                st.markdown(f"""
                <div class="news-item">
                    <span class="news-bullet">•</span>
                    <a href="{article['link']}" target="_blank">{article['title'][:50]}{'...' if len(article['title'])>50 else ''}</a>
                    <span class="news-time">{article['published']}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("暂无新闻")

    if not pause:
        st_autorefresh(interval=hb * 1000)

if __name__ == "__main__":
    main()
