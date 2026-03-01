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
import math
import random

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
        # 全局任务锁标志
        'task_running': False,
        'task_type': None,
        # 回测状态
        'backtest_running': False,
        'backtest_progress': 0.0,
        'backtest_status': '',
        'backtest_trades': None,
        'backtest_equity': None,
        'backtest_metrics': None,
        'backtest_error': None,
        # 优化状态
        'opt_running': False,
        'opt_progress': 0.0,
        'opt_status': '',
        'opt_params': None,
        'opt_metrics': None,
        'opt_trades': None,
        'opt_equity': None,
        'opt_error': None,
        # 用于线程安全的锁
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

# ====================== 增强版安全请求（支持429退避）======================
def safe_request(url, timeout=5, retries=2):
    """统一的安全请求函数，带超时、重试和429退避，无UI调用（线程安全）"""
    for i in range(retries + 1):
        try:
            res = requests.get(url, timeout=timeout)
            if res.status_code == 200:
                return res.json()
            elif res.status_code == 429:
                # 限流：尝试读取 Retry-After 头
                retry_after = res.headers.get('Retry-After')
                if retry_after:
                    wait = int(retry_after)
                else:
                    wait = 2 ** i  # 指数退避
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

# ====================== 数据获取 ======================
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
        if len(df) < 30:
            return None

        # 技术指标
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

        # 资金净流入
        trades_url = "https://www.okx.com/api/v5/market/trades?instId=ETH-USDT&limit=100"
        trades_data = safe_request(trades_url, timeout=5)
        if trades_data and trades_data.get('code') == '0':
            trades_df = pd.DataFrame(trades_data['data'], columns=['ts', 'px', 'sz', 'side'])
            trades_df['ts'] = pd.to_datetime(trades_df['ts'].astype(float), unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')
            trades_df['sz'] = trades_df['sz'].astype(float)
            trades_df['minute'] = trades_df['ts'].dt.floor('min')
            agg = trades_df.groupby('minute').apply(
                lambda x: x[x['side'] == 'buy']['sz'].sum() - x[x['side'] == 'sell']['sz'].sum()
            ).reset_index(name='net_flow')
            if not agg.empty:
                df = df.merge(agg, left_on='time', right_on='minute', how='left')
                df['net_flow'].fillna(0, inplace=True)
            else:
                df['net_flow'] = 0
        else:
            df['net_flow'] = 0
        return df
    except Exception as e:
        st.error(f"K线获取异常: {e}")
        return None

@st.cache_resource(ttl=3600, max_entries=5)
def fetch_historical_candles(bar, start_date, end_date, f_ema, s_ema):
    """获取历史K线，使用资源缓存，返回DataFrame"""
    all_dfs = []
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
    current_end = end_ts
    max_requests = 200
    request_count = 0

    while current_end > start_ts and request_count < max_requests:
        try:
            url = f"https://www.okx.com/api/v5/market/history-candles?instId=ETH-USDT&bar={bar}&limit=100&after={current_end}"
            data = safe_request(url, timeout=10, retries=1)
            if not data or data.get('code') != '0':
                break
            seg_data = data.get('data', [])
            if not seg_data:
                break
            df_seg = pd.DataFrame(seg_data, columns=['ts','o','h','l','c','v','volCcy','volCcyQuote','confirm'])
            df_seg = df_seg[::-1]
            df_seg['time'] = pd.to_datetime(df_seg['ts'].astype(float), unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')
            for col in ['o','h','l','c','v']:
                df_seg[col] = pd.to_numeric(df_seg[col], errors='coerce')
            df_seg.dropna(inplace=True)
            all_dfs.append(df_seg)
            current_end = int(df_seg['ts'].iloc[0]) - 1
            request_count += 1
            if current_end <= start_ts:
                break
        except Exception as e:
            print(f"历史数据请求异常: {e}")
            break

    if not all_dfs:
        return None

    df = pd.concat(all_dfs, ignore_index=True)
    df = df.sort_values('time').reset_index(drop=True)
    df = df[df['time'] >= pd.Timestamp(start_date).tz_localize('Asia/Shanghai')].reset_index(drop=True)
    if len(df) < 50:
        return None

    # 计算指标
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
    df['net_flow'] = 0  # 历史数据无法获取净流入
    return df

# ====================== 信号生成 ======================
def generate_signal(df, ls_ratio, weights=None):
    if df is None or len(df) < 30:
        return 50.0, 0, "数据不足", None, None, "数据不足"

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
        'ls_ratio_high': -8
    }
    if weights is None:
        weights = default_weights

    last = df.iloc[-1]
    score = 50.0
    reasons = []

    ema_pos, ema_neg = weights['ema_cross']
    if last['ema_f'] > last['ema_s']:
        score += ema_pos
        reasons.append("EMA金叉")
    else:
        score += ema_neg
        reasons.append("EMA死叉")

    if not pd.isna(last['rsi']):
        if 30 < last['rsi'] < 70:
            score += weights['rsi_mid']
            reasons.append(f"RSI中性({last['rsi']:.1f})")
        elif last['rsi'] > 75:
            score += weights['rsi_overbought']
            reasons.append(f"RSI超买({last['rsi']:.1f})")
        elif last['rsi'] < 25:
            score += weights['rsi_oversold']
            reasons.append(f"RSI超卖({last['rsi']:.1f})")
        else:
            reasons.append(f"RSI={last['rsi']:.1f}")
    else:
        reasons.append("RSI=NA")

    if last['macd_hist'] > 0:
        score += weights['macd_hist_pos']
        reasons.append("MACD柱为正")
    else:
        score += weights['macd_hist_neg']
        reasons.append("MACD柱为负")

    if last['c'] > last['bb_upper']:
        score += weights['bb_upper']
        reasons.append("价格突破上轨")
    elif last['c'] < last['bb_lower']:
        score += weights['bb_lower']
        reasons.append("价格跌破下轨")
    else:
        reasons.append("价格在布林带内")

    if last['v'] > last['vol_ma'] * 1.3:
        score += weights['volume_surge']
        reasons.append("放量")
    else:
        score += weights['volume_shrink']
        reasons.append("缩量")

    if last['net_flow'] > 0:
        score += weights['net_flow_pos']
        reasons.append("资金净流入")
    else:
        score += weights['net_flow_neg']
        reasons.append("资金净流出")

    if ls_ratio < 0.95:
        score += weights['ls_ratio_low']
        reasons.append("多空比<0.95(空头极端)")
    elif ls_ratio > 1.05:
        score += weights['ls_ratio_high']
        reasons.append("多空比>1.05(多头极端)")
    else:
        reasons.append(f"多空比={ls_ratio:.2f}")

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
        sl = tp = None

    reason_str = " | ".join(reasons)
    return prob, direction, entry_zone, sl, tp, reason_str

# ====================== 回测核心 ======================
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
        prob, sig_dir, _, _, _, _ = generate_signal(df.iloc[:i+1], 1.0, weights=weights)

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

# ====================== 增强版网格搜索（带组合数限制和稀疏更新）======================
def grid_search_weights(df, initial_balance, risk_percent, f_ema, s_ema, param_grid, metric='sharpe', progress_callback=None):
    keys = ['ema_cross_pos', 'ema_cross_neg', 'rsi_mid', 'rsi_overbought', 'rsi_oversold',
            'macd_hist_pos', 'macd_hist_neg', 'bb_upper', 'bb_lower',
            'volume_surge', 'volume_shrink', 'net_flow_pos', 'net_flow_neg',
            'ls_ratio_low', 'ls_ratio_high']
    values = [param_grid[k] for k in keys]
    combinations = list(product(*values))
    total = len(combinations)

    # 限制组合数，超过5000则随机采样5000组
    MAX_COMBINATIONS = 5000
    if total > MAX_COMBINATIONS:
        print(f"组合数 {total} 超过上限 {MAX_COMBINATIONS}，随机采样 {MAX_COMBINATIONS} 组")
        sampled_indices = random.sample(range(total), MAX_COMBINATIONS)
        combinations = [combinations[i] for i in sampled_indices]
        total = MAX_COMBINATIONS

    best_score = -np.inf
    best_params = None
    best_metrics = None
    best_trades = None
    best_equity = None

    # 进度更新步长（每完成1%更新一次）
    progress_step = max(1, total // 100)

    for idx, combo in enumerate(combinations):
        weights = {
            'ema_cross': (combo[0], combo[1]),
            'rsi_mid': combo[2],
            'rsi_overbought': combo[3],
            'rsi_oversold': combo[4],
            'macd_hist_pos': combo[5],
            'macd_hist_neg': combo[6],
            'bb_upper': combo[7],
            'bb_lower': combo[8],
            'volume_surge': combo[9],
            'volume_shrink': combo[10],
            'net_flow_pos': combo[11],
            'net_flow_neg': combo[12],
            'ls_ratio_low': combo[13],
            'ls_ratio_high': combo[14]
        }
        trades_df, equity_curve, msg = run_backtest(df, initial_balance, risk_percent, f_ema, s_ema, weights)
        if trades_df is None or len(trades_df) == 0:
            continue
        metrics = calculate_metrics(trades_df, equity_curve, initial_balance)
        if metric == 'sharpe':
            score = float(metrics.get('夏普比率', 0))
        elif metric == 'return':
            score = float(metrics.get('总收益率 (%)', 0))
        elif metric == 'win_rate':
            score = float(metrics.get('胜率 (%)', 0))
        elif metric == 'profit_factor':
            score = float(metrics.get('盈利因子', 0))
        else:
            score = float(metrics.get('夏普比率', 0))

        if score > best_score:
            best_score = score
            best_params = combo
            best_metrics = metrics
            best_trades = trades_df
            best_equity = equity_curve

        # 每 progress_step 组或最后一组时更新进度
        if progress_callback and (idx % progress_step == 0 or idx == total - 1):
            progress_callback(idx+1, total, best_score)

    return best_params, best_metrics, best_trades, best_equity

# ====================== 侧边栏UI ======================
def render_sidebar(df):
    with st.sidebar:
        st.title("📊 专业量化引擎·至尊版")
        st.caption("基于OKX实时数据 | 多因子模型")

        # 刷新控制
        hb = st.slider("自动刷新间隔 (秒)", 5, 60, 15)
        pause = st.checkbox("暂停自动刷新", False)
        st.session_state.alarm_on = st.checkbox("声音报警 (胜率>70%或<30%)", st.session_state.alarm_on)

        # 交易参数
        symbol = st.selectbox("交易对", ["ETH-USDT", "BTC-USDT", "SOL-USDT"], index=0)
        tf = st.selectbox("时间框架", ["1m", "5m", "15m", "30m", "1H"], index=2)
        f_ema = st.number_input("快线EMA", 5, 30, 12)
        s_ema = st.number_input("慢线EMA", 20, 100, 26)

        # 主题
        st.session_state.theme = st.selectbox("主题", ['dark', 'light'])

        if st.button("🔄 立即刷新数据"):
            st.cache_data.clear()
            st.rerun()

        st.divider()

        # 仓位计算器
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

        # Telegram配置
        with st.expander("📱 Telegram通知", expanded=False):
            st.session_state.tg_token = st.text_input("Bot Token", value=st.session_state.get('tg_token', ''), type="password")
            st.session_state.tg_chat_id = st.text_input("Chat ID", value=st.session_state.get('tg_chat_id', ''))
            st.caption("创建Bot后 /myid 获取Chat ID")

        # ====================== 回测区域 ======================
        with st.expander("📈 历史回测", expanded=False):
            st.caption("选择时间段进行策略回测")
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("开始日期", datetime.now() - timedelta(days=30))
            with col2:
                end_date = st.date_input("结束日期", datetime.now())
            backtest_balance = st.number_input("初始资金 (USDT)", 1000.0, 1000000.0, 10000.0, key="backtest_balance")
            backtest_risk = st.slider("回测风险 (%)", 0.1, 5.0, 1.0, 0.1, key="backtest_risk")

            # 回测说明：净流入因子在回测中被忽略
            st.caption("⚠️ 回测未包含资金净流入因子（历史数据不可得）")

            # 显示错误信息（如果有）
            if st.session_state.get('backtest_error'):
                st.error(f"回测出错: {st.session_state['backtest_error']}")

            if st.session_state.get('backtest_running', False):
                st.info(f"回测进行中... {st.session_state.get('backtest_status', '')}")
                st.progress(st.session_state.get('backtest_progress', 0.0))
            else:
                if st.button("🚀 开始回测"):
                    # 检查是否有其他任务正在运行
                    if st.session_state.get('task_running', False):
                        st.warning(f"另一个任务正在运行 ({st.session_state['task_type']})，请稍后再试")
                    else:
                        # 打包状态更新为字典，一次性写入
                        with st.session_state['state_lock']:
                            new_state = {
                                'task_running': True,
                                'task_type': '回测',
                                'backtest_running': True,
                                'backtest_progress': 0.0,
                                'backtest_status': "正在获取历史数据...",
                                'backtest_error': None
                            }
                            for k, v in new_state.items():
                                st.session_state[k] = v

                        def run_backtest_thread():
                            try:
                                df_hist = fetch_historical_candles(tf, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), f_ema, s_ema)
                                if df_hist is not None:
                                    with st.session_state['state_lock']:
                                        st.session_state['backtest_status'] = "正在回测..."
                                    trades_df, equity_curve, msg = run_backtest(df_hist, backtest_balance, backtest_risk, f_ema, s_ema)
                                    if trades_df is not None:
                                        metrics = calculate_metrics(trades_df, equity_curve, backtest_balance)
                                        with st.session_state['state_lock']:
                                            # 一次性更新多个相关状态
                                            updates = {
                                                'backtest_trades': trades_df,
                                                'backtest_equity': equity_curve,
                                                'backtest_metrics': metrics,
                                                'backtest_status': "完成"
                                            }
                                            for k, v in updates.items():
                                                st.session_state[k] = v
                                    else:
                                        with st.session_state['state_lock']:
                                            st.session_state['backtest_status'] = f"回测失败: {msg}"
                                else:
                                    with st.session_state['state_lock']:
                                        st.session_state['backtest_status'] = "历史数据获取失败"
                            except Exception as e:
                                with st.session_state['state_lock']:
                                    st.session_state['backtest_error'] = traceback.format_exc()
                                    st.session_state['backtest_status'] = f"错误: {e}"
                            finally:
                                with st.session_state['state_lock']:
                                    st.session_state['backtest_running'] = False
                                    st.session_state['task_running'] = False
                                    st.session_state['task_type'] = None
                                    st.session_state['backtest_progress'] = 1.0

                        threading.Thread(target=run_backtest_thread).start()
                        st.rerun()

        # ====================== 因子优化区域 ======================
        with st.expander("⚙️ 因子权重优化 (网格搜索)", expanded=False):
            st.caption("设置各因子权重的搜索范围，点击开始优化（可能耗时较长）")
            col1, col2 = st.columns(2)
            with col1:
                ema_pos_range = st.text_input("EMA金叉加分", "15,20,25")
                ema_neg_range = st.text_input("EMA死叉减分", "-20,-18,-15")
                rsi_mid_range = st.text_input("RSI中性加分", "5,10,15")
                rsi_overbought_range = st.text_input("RSI超买减分", "-15,-10,-5")
                rsi_oversold_range = st.text_input("RSI超卖加分", "0,5,10")
                macd_pos_range = st.text_input("MACD柱正加分", "8,12,15")
                macd_neg_range = st.text_input("MACD柱负减分", "-15,-12,-8")
            with col2:
                bb_upper_range = st.text_input("突破上轨减分", "-20,-15,-10")
                bb_lower_range = st.text_input("跌破下轨加分", "5,10,15")
                vol_surge_range = st.text_input("放量加分", "5,8,12")
                vol_shrink_range = st.text_input("缩量减分", "-6,-4,-2")
                flow_pos_range = st.text_input("净流入加分", "10,15,20")
                flow_neg_range = st.text_input("净流出减分", "-20,-15,-10")
                ls_low_range = st.text_input("多空比<0.95加分", "5,8,12")
                ls_high_range = st.text_input("多空比>1.05减分", "-12,-8,-5")

            metric_opt = st.selectbox("优化目标", ["夏普比率", "总收益率", "胜率", "盈利因子"])
            opt_balance = st.number_input("优化初始资金 (USDT)", 1000.0, 1000000.0, 10000.0, key="opt_balance")
            opt_risk = st.slider("优化风险 (%)", 0.1, 5.0, 1.0, 0.1, key="opt_risk")

            def parse_range(s):
                items = [x.strip() for x in s.split(',') if x.strip()]
                return [float(x) for x in items]

            # 优化说明：净流入因子在回测中被忽略
            st.caption("⚠️ 优化未包含资金净流入因子（历史数据不可得）")

            # 显示错误信息（如果有）
            if st.session_state.get('opt_error'):
                st.error(f"优化出错: {st.session_state['opt_error']}")

            if st.session_state.get('opt_running', False):
                st.info(f"优化进行中... {st.session_state.get('opt_status', '')}")
                st.progress(st.session_state.get('opt_progress', 0.0))
            else:
                if st.button("🚀 开始优化"):
                    # 检查是否有其他任务正在运行
                    if st.session_state.get('task_running', False):
                        st.warning(f"另一个任务正在运行 ({st.session_state['task_type']})，请稍后再试")
                    else:
                        param_grid = {
                            'ema_cross_pos': parse_range(ema_pos_range),
                            'ema_cross_neg': parse_range(ema_neg_range),
                            'rsi_mid': parse_range(rsi_mid_range),
                            'rsi_overbought': parse_range(rsi_overbought_range),
                            'rsi_oversold': parse_range(rsi_oversold_range),
                            'macd_hist_pos': parse_range(macd_pos_range),
                            'macd_hist_neg': parse_range(macd_neg_range),
                            'bb_upper': parse_range(bb_upper_range),
                            'bb_lower': parse_range(bb_lower_range),
                            'volume_surge': parse_range(vol_surge_range),
                            'volume_shrink': parse_range(vol_shrink_range),
                            'net_flow_pos': parse_range(flow_pos_range),
                            'net_flow_neg': parse_range(flow_neg_range),
                            'ls_ratio_low': parse_range(ls_low_range),
                            'ls_ratio_high': parse_range(ls_high_range)
                        }
                        for k, v in param_grid.items():
                            if not v:
                                st.error(f"{k} 输入不能为空")
                                st.stop()
                        total_combinations = np.prod([len(v) for v in param_grid.values()])
                        if total_combinations > 5000:
                            st.warning(f"组合数过多 ({total_combinations})，将随机采样 5000 组进行优化。")

                        with st.session_state['state_lock']:
                            new_state = {
                                'task_running': True,
                                'task_type': '优化',
                                'opt_running': True,
                                'opt_progress': 0.0,
                                'opt_status': "正在获取历史数据...",
                                'opt_error': None
                            }
                            for k, v in new_state.items():
                                st.session_state[k] = v

                        def run_opt_thread():
                            try:
                                df_hist = fetch_historical_candles(
                                    tf,
                                    (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
                                    datetime.now().strftime("%Y-%m-%d"),
                                    f_ema, s_ema
                                )
                                if df_hist is None:
                                    with st.session_state['state_lock']:
                                        st.session_state['opt_status'] = "历史数据获取失败"
                                    return
                                with st.session_state['state_lock']:
                                    st.session_state['opt_status'] = "正在网格搜索..."

                                def progress_callback(current, total, best):
                                    with st.session_state['state_lock']:
                                        st.session_state['opt_progress'] = current / total
                                        st.session_state['opt_status'] = f"已测试 {current}/{total} 组，当前最优 {best:.2f}"

                                best_params, best_metrics, best_trades, best_equity = grid_search_weights(
                                    df_hist, opt_balance, opt_risk, f_ema, s_ema, param_grid,
                                    metric='sharpe' if metric_opt == '夏普比率' else
                                           'return' if metric_opt == '总收益率' else
                                           'win_rate' if metric_opt == '胜率' else 'profit_factor',
                                    progress_callback=progress_callback
                                )
                                if best_params:
                                    with st.session_state['state_lock']:
                                        updates = {
                                            'opt_params': best_params,
                                            'opt_metrics': best_metrics,
                                            'opt_trades': best_trades,
                                            'opt_equity': best_equity,
                                            'opt_status': "完成"
                                        }
                                        for k, v in updates.items():
                                            st.session_state[k] = v
                                else:
                                    with st.session_state['state_lock']:
                                        st.session_state['opt_status'] = "优化失败，未找到有效参数组合"
                            except Exception as e:
                                with st.session_state['state_lock']:
                                    st.session_state['opt_error'] = traceback.format_exc()
                                    st.session_state['opt_status'] = f"错误: {e}"
                            finally:
                                with st.session_state['state_lock']:
                                    st.session_state['opt_running'] = False
                                    st.session_state['task_running'] = False
                                    st.session_state['task_type'] = None
                                    st.session_state['opt_progress'] = 1.0

                        threading.Thread(target=run_opt_thread).start()
                        st.rerun()

        return hb, pause, symbol, tf, f_ema, s_ema

# ====================== 主界面 ======================
def main():
    light_cleanup()

    # 获取多空比数据
    ls_ratio = get_ls_ratio()
    st.session_state.ls_ratio = ls_ratio
    ls_history = get_ls_history(24)
    if not ls_history.empty:
        st.session_state.ls_history = ls_history

    # 初始数据（用于侧边栏默认价格）
    df_init = get_candles(bar="15m", limit=100, f_ema=12, s_ema=26)
    if df_init is None:
        st.error("无法获取初始K线数据")
        st.stop()

    # 侧边栏
    hb, pause, symbol, tf, f_ema, s_ema = render_sidebar(df_init)

    # 获取最新K线
    df = get_candles(bar=tf, limit=100, f_ema=f_ema, s_ema=s_ema)
    if df is None or len(df) < 30:
        st.error("无法获取足够K线数据")
        st.stop()

    # 生成实时信号
    prob, direction, entry_zone, sl, tp, reason = generate_signal(df, ls_ratio)

    # Telegram推送
    if (prob > 70 or prob < 30) and abs(prob - st.session_state.last_signal_prob) > 5:
        if prob > 70:
            emoji = "🚀"
            side = "多头"
        else:
            emoji = "⚠️"
            side = "空头"
        rr = abs((tp - df['c'].iloc[-1]) / (df['c'].iloc[-1] - sl)) if sl and tp and abs(df['c'].iloc[-1] - sl) > 1e-8 else 0
        msg = f"""{emoji} {side}信号！
交易对: {symbol} | {tf}
胜率: {prob:.1f}%
价格: ${df['c'].iloc[-1]:.2f}
入场区: {entry_zone}
止损: ${sl:.2f}
止盈: ${tp:.2f}
盈亏比: 1:{rr:.2f}
理由: {reason}"""
        send_telegram(msg)
        st.session_state.last_signal_prob = prob

    # 声音报警
    if st.session_state.alarm_on and (prob > 70 or prob < 30):
        st.audio("https://www.soundjay.com/misc/sounds/bell-ringing-05.mp3", format="audio/mp3")

    # 标题
    st.markdown(f"""
        <h1 style='text-align: center; color: #00ff88; font-family: "Courier New", monospace;'>
            专业量化决策引擎·至尊版
        </h1>
        <h4 style='text-align: center; color: #aaa; margin-top: -10px;'>
            {symbol} | {tf} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </h4>
    """, unsafe_allow_html=True)

    # 核心指标卡片
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("实时价格", f"${df['c'].iloc[-1]:.2f}", f"{df['c'].pct_change().iloc[-1]*100:.2f}%")
    col2.metric("AI胜率", f"{prob:.1f}%")
    col3.metric("多空比", f"{ls_ratio:.2f}")
    col4.metric("资金净流", f"{df['net_flow'].iloc[-1]:.0f} ETH")
    col5.metric("ATR波幅", f"{df['atr'].iloc[-1]:.2f}")

    st.divider()

    # 策略卡片
    if direction == 1:
        box_color = "#00ff88"
        action = "🔥 多头策略"
    elif direction == -1:
        box_color = "#ff4b4b"
        action = "❄️ 空头策略"
    else:
        box_color = "#FFD700"
        action = "⚖️ 观望"
    rr_value = abs((tp - df['c'].iloc[-1]) / (df['c'].iloc[-1] - sl)) if sl and tp and abs(df['c'].iloc[-1] - sl) > 1e-8 else 0
    st.markdown(f"""
    <div style="border:2px solid {box_color}; border-radius:15px; padding:20px; margin-bottom:20px; background:rgba(0,0,0,0.3);">
        <h2 style="color:{box_color}; margin:0;">{action}</h2>
        <p style="color:#ccc;">胜率 <b style="color:{box_color};">{prob:.1f}%</b> | 信号理由: {reason}</p>
        <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:15px; margin-top:15px;">
            <div><span style="color:#aaa;">🎯 最佳入场区</span><br><b style="color:#00ff88;">{entry_zone}</b></div>
            <div><span style="color:#aaa;">🛡️ 动态止损</span><br><b style="color:#ff4b4b;">{f'${sl:.2f}' if sl else '无'}</b></div>
            <div><span style="color:#aaa;">💰 动态止盈</span><br><b style="color:#00ff88;">{f'${tp:.2f}' if tp else '无'}</b></div>
        </div>
        <div style="margin-top:15px;">
            <span style="color:#aaa;">📊 盈亏比: </span><b>{rr_value:.2f}</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 主图表
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        row_heights=[0.6, 0.2, 0.2],
                        subplot_titles=("价格 & 指标", "MACD", "资金净流"))
    fig.add_trace(go.Candlestick(x=df['time'], open=df['o'], high=df['h'], low=df['l'], close=df['c'],
                                 name="K线", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['ema_f'], line=dict(color='#00ff88', width=2), name="EMA快线"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['ema_s'], line=dict(color='#ff4b4b', width=2), name="EMA慢线"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['bb_upper'], line=dict(color='rgba(255,255,255,0.2)', width=1), name="BB上轨"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['bb_lower'], line=dict(color='rgba(255,255,255,0.2)', width=1), name="BB下轨",
                             fill='tonexty', fillcolor='rgba(255,255,255,0.05)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['macd'], line=dict(color='#00ff88', width=1.5), name="MACD"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['macd_signal'], line=dict(color='#ff4b4b', width=1.5), name="信号线"), row=2, col=1)
    colors_hist = ['#00ff88' if val > 0 else '#ff4b4b' for val in df['macd_hist']]
    fig.add_trace(go.Bar(x=df['time'], y=df['macd_hist'], marker_color=colors_hist, name="MACD柱"), row=2, col=1)
    flow_colors = ['#00ff88' if x > 0 else '#ff4b4b' for x in df['net_flow']]
    fig.add_trace(go.Bar(x=df['time'], y=df['net_flow'], marker_color=flow_colors, name="资金净流"), row=3, col=1)
    fig.update_layout(
        template=pio.templates['custom_dark'] if st.session_state.theme == 'dark' else pio.templates['custom_light'],
        height=750, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10), hovermode='x unified'
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

    # 显示回测结果
    if st.session_state.get('backtest_metrics') and not st.session_state.get('backtest_running', False):
        with st.expander("📊 回测结果", expanded=False):
            metrics = st.session_state['backtest_metrics']
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("胜率", metrics['胜率 (%)'])
            col2.metric("总收益率", metrics['总收益率 (%)'])
            col3.metric("最大回撤", metrics['最大回撤 (%)'])
            col4.metric("夏普比率", metrics['夏普比率'])
            col1.metric("年化收益率", metrics['年化收益率 (%)'])
            col2.metric("盈利因子", metrics['盈利因子'])
            col3.metric("交易次数", metrics['交易次数'])
            if st.session_state.get('backtest_equity'):
                fig_eq = go.Figure()
                fig_eq.add_trace(go.Scatter(y=st.session_state['backtest_equity'], mode='lines', name='权益曲线', line=dict(color='#00ff88')))
                fig_eq.update_layout(title="账户权益曲线", height=300, template='plotly_dark' if st.session_state.theme == 'dark' else 'plotly')
                st.plotly_chart(fig_eq, use_container_width=True)
            if st.session_state.get('backtest_trades') is not None:
                st.subheader("交易记录")
                display_df = st.session_state['backtest_trades'][['entry_time', 'exit_time', 'direction', 'entry_price', 'exit_price', 'pnl', 'pnl_pct', 'reason']].copy()
                display_df['entry_time'] = display_df['entry_time'].dt.strftime('%m-%d %H:%M')
                display_df['exit_time'] = display_df['exit_time'].dt.strftime('%m-%d %H:%M')
                display_df['pnl'] = display_df['pnl'].round(2)
                display_df['pnl_pct'] = display_df['pnl_pct'].round(2)
                st.dataframe(display_df, use_container_width=True)

    # 显示优化结果
    if st.session_state.get('opt_metrics') and not st.session_state.get('opt_running', False):
        with st.expander("🏆 最优权重组合", expanded=False):
            st.subheader("最优权重")
            param_names = ['EMA金叉', 'EMA死叉', 'RSI中性', 'RSI超买', 'RSI超卖',
                           'MACD柱正', 'MACD柱负', '突破上轨', '跌破下轨',
                           '放量', '缩量', '净流入', '净流出', '多空比低', '多空比高']
            param_df = pd.DataFrame({'因子': param_names, '权重': st.session_state['opt_params']})
            st.dataframe(param_df, use_container_width=True)
            st.subheader("优化指标")
            metrics = st.session_state['opt_metrics']
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("胜率", metrics['胜率 (%)'])
            col2.metric("总收益率", metrics['总收益率 (%)'])
            col3.metric("最大回撤", metrics['最大回撤 (%)'])
            col4.metric("夏普比率", metrics['夏普比率'])
            if st.session_state.get('opt_equity'):
                fig_eq = go.Figure()
                fig_eq.add_trace(go.Scatter(y=st.session_state['opt_equity'], mode='lines', name='最优权益', line=dict(color='#ffaa00')))
                fig_eq.update_layout(title="最优参数权益曲线", height=300, template='plotly_dark' if st.session_state.theme == 'dark' else 'plotly')
                st.plotly_chart(fig_eq, use_container_width=True)

    # 前端自动刷新
    if not pause:
        st.markdown(f'<meta http-equiv="refresh" content="{hb}">', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
