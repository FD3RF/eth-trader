import streamlit as st
import pandas as pd
import requests
from datetime import datetime
from collections import deque
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.signal import argrelextrema
import time

# 尝试导入自动刷新组件，若未安装则优雅降级
try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None

# 尝试导入 yfinance（备用数据源），若未安装则禁用备用功能
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

# ==================== v15.3 配置 ====================
SYMBOL = "ETH"
CURRENCY = "USD"
INTERVAL = "5"
LIMIT = 400
INTERVAL_MINUTES = 5

# 评分权重（可调常量）
SCORE_BASE = 45
SCORE_SLOPE_MAX = 50
SCORE_RSI_STRONG = 50
SCORE_RSI_WEAK = 25
SCORE_DIV_CLASSIC = 35
SCORE_DIV_HIDDEN = 50
SCORE_VOL_CONFIRM = 25
SCORE_VOL_PENALTY = -20
BLEED_PENALTY_FACTOR = 0.7

# ==================== Session State ====================
if 'initialized' not in st.session_state:
    st.session_state.update({
        # 核心策略参数
        'fast': 8, 'slow': 21, 'rsi_period': 14,
        'score_thresh': 80,
        'bleed_threshold': 76,
        'volume_confirm': True,
        'enable_sound': True,
        'auto_refresh': False,
        # 风控参数
        'use_atr_sl': True,
        'sl_m': 1.8, 'tp1_m': 1.2, 'tp2_m': 2.5,
        'trailing_sl_pct': 0.5,
        'partial_tp_pct': 0.5,
        'risk_per_trade': 1.0,
        'account_balance': 10000.0,
        # 数据与历史
        'history': deque(maxlen=200),
        'candles': deque(maxlen=1500),
        'last_signal_time': None,
        'api_fail': 0,
        'last_error': "",
        'initialized': True
    })

# ==================== 数据获取（仅 CryptoCompare + Yahoo Finance 备用）====================
@st.cache_data(ttl=12, show_spinner=False)
def fetch_data(interval='5m', limit=LIMIT):
    """
    从 CryptoCompare 获取 K 线数据，失败时尝试 Yahoo Finance（若可用）。
    支持 interval: '5m', '15m', '1h'
    """
    # 将 interval 转换为 CryptoCompare 的 aggregate 参数（分钟数）
    interval_map = {
        '5m': 5,
        '15m': 15,
        '1h': 60
    }
    aggregate = interval_map.get(interval, 5)
    
    # CryptoCompare 历史分钟数据端点
    url = f"https://min-api.cryptocompare.com/data/v2/histominute?fsym=ETH&tsym=USD&limit={limit}&aggregate={aggregate}"
    
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        if data.get('Response') == 'Error':
            raise Exception(data.get('Message', 'Unknown error'))
        
        klines = data['Data']['Data']
        df = pd.DataFrame(klines)
        # 选择所需列并重命名
        df = df[['time', 'open', 'high', 'low', 'close', 'volumefrom']]
        df.columns = ['ts', 'open', 'high', 'low', 'close', 'volume']
        df['ts'] = pd.to_datetime(df['ts'], unit='s')  # CryptoCompare 时间戳为秒
        df.set_index('ts', inplace=True)
        df = df.astype(float)
        
        st.session_state.last_error = ""  # 清除错误
        return df
        
    except Exception as e:
        st.session_state.api_fail += 1
        st.session_state.last_error = f"CryptoCompare {interval} 失败: {str(e)[:100]}"
        
        # 尝试 Yahoo Finance 备用（需安装 yfinance）
        if YFINANCE_AVAILABLE:
            try:
                # Yahoo Finance 使用不同的 interval 表示
                yf_interval_map = {
                    '5m': '5m',
                    '15m': '15m',
                    '1h': '60m'
                }
                yf_interval = yf_interval_map.get(interval, '5m')
                
                # 获取足够的数据（5天应能满足 limit 需求）
                ticker = yf.Ticker("ETH-USD")
                df_yf = ticker.history(period="5d", interval=yf_interval)
                
                if df_yf.empty:
                    raise Exception("Yahoo Finance 返回空数据")
                
                # 重命名列以匹配标准格式
                df_yf = df_yf[['Open', 'High', 'Low', 'Close', 'Volume']]
                df_yf.columns = ['open', 'high', 'low', 'close', 'volume']
                df_yf.index = pd.to_datetime(df_yf.index)
                
                # 只保留最新的 limit 根（Yahoo 返回数据可能多于 limit）
                if len(df_yf) > limit:
                    df_yf = df_yf.tail(limit)
                
                st.session_state.last_error = ""  # 成功则清除错误
                return df_yf
                
            except Exception as e2:
                st.session_state.last_error += f" | Yahoo Finance 失败: {str(e2)[:100]}"
        else:
            st.session_state.last_error += " | Yahoo Finance 未安装"
        
        return pd.DataFrame()

# ==================== 指标计算 ====================
def get_indicators(df):
    if df.empty:
        return df
    df = df.copy()
    df['ema_fast'] = df['close'].ewm(span=st.session_state.fast, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=st.session_state.slow, adjust=False).mean()

    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/st.session_state.rsi_period).mean()
    loss = -delta.where(delta < 0, 0).ewm(alpha=1/st.session_state.rsi_period).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-9)))

    tr = pd.concat([df['high']-df['low'],
                    (df['high']-df['close'].shift()).abs(),
                    (df['low']-df['close'].shift()).abs()], axis=1).max(1)
    df['atr'] = tr.ewm(alpha=1/14).mean()

    df['macd_line'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
    df['macd_signal'] = df['macd_line'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd_line'] - df['macd_signal']

    df['vol_ema20'] = df['volume'].ewm(span=20, adjust=False).mean()
    df['vol_ema50'] = df['volume'].ewm(span=50, adjust=False).mean()

    return df

# ==================== 阴跌评分 ====================
def calc_bleed_score(df, lookback=60):
    data = df.tail(lookback)
    if len(data) < 15:
        return 0

    atr_ratio = data['atr'].iloc[-1] / (data['atr'].mean() + 1e-9)
    s_atr = max(0, (1 - atr_ratio)) * 50

    down_mask = data['close'] < data['open']
    down_pct = down_mask.mean() * 25
    consec = down_mask.rolling(8).sum().max()
    s_consec = min(30, consec * 5)

    vol_ratio_short = data['volume'].iloc[-1] / (data['vol_ema20'].iloc[-1] + 1e-9)
    vol_ratio_long = data['volume'].iloc[-1] / (data['vol_ema50'].iloc[-1] + 1e-9)
    s_vol = max(0, (1 - min(vol_ratio_short, vol_ratio_long))) * 40

    lower_highs = (data['high'].diff() < 0).mean() * 20
    lower_lows = (data['low'].diff() < 0).mean() * 15

    total = s_atr + down_pct + s_consec + s_vol + lower_highs + lower_lows
    return min(100, total)

# ==================== MACD 背离 ====================
def detect_divergence(df, lookback=120, order=7):
    data = df.tail(lookback)
    if len(data) < 20:
        return None, None

    p_low = argrelextrema(data['close'].values, np.less, order=order)[0]
    m_low = argrelextrema(data['macd_line'].values, np.greater, order=order)[0]
    p_high = argrelextrema(data['close'].values, np.greater, order=order)[0]
    m_high = argrelextrema(data['macd_line'].values, np.less, order=order)[0]

    bull_classic = bull_hidden = None
    if len(p_low) >= 2 and len(m_low) >= 2:
        i1, i2 = p_low[-2], p_low[-1]
        if i2 - i1 >= 12 and data['close'].iloc[i2] < data['close'].iloc[i1] * 0.998:
            if data['macd_line'].iloc[i2] > data['macd_line'].iloc[i1]:
                bull_classic = {'type': '底背离 (经典)', 'time2': data.index[i2], 'strength': '中'}

        if data['close'].iloc[i2] < data['close'].iloc[i1] and data['macd_line'].iloc[i2] > data['macd_line'].iloc[i1]:
            bull_hidden = {'type': '底隐背离 (强势)', 'time2': data.index[i2], 'strength': '强'}

    bear_classic = bear_hidden = None
    if len(p_high) >= 2 and len(m_high) >= 2:
        i1, i2 = p_high[-2], p_high[-1]
        if i2 - i1 >= 12 and data['close'].iloc[i2] > data['close'].iloc[i1] * 1.002:
            if data['macd_line'].iloc[i2] < data['macd_line'].iloc[i1]:
                bear_classic = {'type': '顶背离 (经典)', 'time2': data.index[i2], 'strength': '中'}

        if data['close'].iloc[i2] > data['close'].iloc[i1] and data['macd_line'].iloc[i2] < data['macd_line'].iloc[i1]:
            bear_hidden = {'type': '顶隐背离 (强势)', 'time2': data.index[i2], 'strength': '强'}

    return (bull_classic or bull_hidden), (bear_classic or bear_hidden)

# ==================== 多周期共振 ====================
def multi_timeframe_resonance(df_5m, df_15m, df_1h, side):
    if df_15m.empty or df_1h.empty:
        return False
    if side == 'BUY':
        res_15 = df_15m['ema_fast'].iloc[-1] > df_15m['ema_slow'].iloc[-1] and df_15m['close'].iloc[-1] > df_15m['ema_fast'].iloc[-1]
        res_1h = df_1h['ema_fast'].iloc[-1] > df_1h['ema_slow'].iloc[-1] and df_1h['close'].iloc[-1] > df_1h['ema_fast'].iloc[-1]
    else:
        res_15 = df_15m['ema_fast'].iloc[-1] < df_15m['ema_slow'].iloc[-1] and df_15m['close'].iloc[-1] < df_15m['ema_fast'].iloc[-1]
        res_1h = df_1h['ema_fast'].iloc[-1] < df_1h['ema_slow'].iloc[-1] and df_1h['close'].iloc[-1] < df_1h['ema_fast'].iloc[-1]
    return res_15 and res_1h

# ==================== 综合信心评分 ====================
def get_brain_score(side, df, bleed_score, bull_div, bear_div):
    if len(df) < 10:
        return SCORE_BASE

    last = df.iloc[-1]
    score = SCORE_BASE

    slope = (last['ema_fast'] - df['ema_fast'].iloc[-6]) / last['close'] * 1000
    score += min(SCORE_SLOPE_MAX, max(0, abs(slope) * 90))

    r = last['rsi']
    if side == 'BUY':
        score += SCORE_RSI_STRONG if 50 < r < 75 else SCORE_RSI_WEAK if 40 < r < 85 else 0
    else:
        score += SCORE_RSI_STRONG if 25 < r < 50 else SCORE_RSI_WEAK if 15 < r < 60 else 0

    if bull_div:
        score += SCORE_DIV_HIDDEN if "隐" in bull_div['type'] else SCORE_DIV_CLASSIC
    if bear_div:
        score += SCORE_DIV_HIDDEN if "隐" in bear_div['type'] else SCORE_DIV_CLASSIC

    if st.session_state.volume_confirm:
        vol_confirm = (df['volume'].iloc[-1] > df['vol_ema20'].iloc[-1]) if side == 'BUY' else (df['volume'].iloc[-1] < df['vol_ema20'].iloc[-1])
        score += SCORE_VOL_CONFIRM if vol_confirm else SCORE_VOL_PENALTY

    score -= bleed_score * BLEED_PENALTY_FACTOR
    return max(0, min(150, score))

# ==================== 仓位建议 ====================
def suggest_position_size(confidence, bleed_score):
    if bleed_score > st.session_state.bleed_threshold or confidence < 50:
        return 0.0

    base_risk = st.session_state.risk_per_trade / 100
    confidence_factor = confidence / 100
    adjusted_risk = base_risk * confidence_factor * (1 - bleed_score / 200)

    return round(adjusted_risk * st.session_state.account_balance / 100, 2)

# ==================== 止损止盈 ====================
def sltp(price, side, atr):
    if st.session_state.use_atr_sl and atr > 0:
        risk = atr * st.session_state.sl_m
        if side == 'BUY':
            return price - risk, price + risk * st.session_state.tp1_m, price + risk * st.session_state.tp2_m
        else:
            return price + risk, price - risk * st.session_state.tp1_m, price - risk * st.session_state.tp2_m
    else:
        # 固定比例
        if side == 'BUY':
            return price * 0.995, price * 1.01, price * 1.02
        else:
            return price * 1.005, price * 0.99, price * 0.98

# ==================== 追踪止损 & 分批止盈 ====================
def update_position(rec, current_price):
    if rec['result'] != 'pending':
        return rec

    if rec['side'] == 'BUY':
        if current_price > rec.get('highest_price', rec['entry']):
            rec['highest_price'] = current_price
            new_sl = rec['highest_price'] * (1 - st.session_state.trailing_sl_pct / 100)
            rec['trailing_sl'] = max(rec.get('trailing_sl', rec['sl']), new_sl)

        if current_price >= rec['tp1'] and not rec.get('partial_closed', False):
            rec['partial_closed'] = True
            rec['remaining_position'] = 1 - st.session_state.partial_tp_pct

        if current_price <= rec.get('trailing_sl', rec['sl']):
            rec['result'] = '止损出局'
            rec['exit_price'] = current_price
            rec['exit_time'] = datetime.now().strftime('%Y-%m-%d %H:%M')
    else:
        if current_price < rec.get('lowest_price', rec['entry']):
            rec['lowest_price'] = current_price
            new_sl = rec['lowest_price'] * (1 + st.session_state.trailing_sl_pct / 100)
            rec['trailing_sl'] = min(rec.get('trailing_sl', rec['sl']), new_sl)

        if current_price <= rec['tp1'] and not rec.get('partial_closed', False):
            rec['partial_closed'] = True
            rec['remaining_position'] = 1 - st.session_state.partial_tp_pct

        if current_price >= rec.get('trailing_sl', rec['sl']):
            rec['result'] = '止损出局'
            rec['exit_price'] = current_price
            rec['exit_time'] = datetime.now().strftime('%Y-%m-%d %H:%M')

    return rec

# ==================== 计算统计指标 ====================
def calculate_stats(hist_df, current_price):
    if hist_df.empty:
        return 0.0, 0.0, "无持仓"

    hist_df['actual_pnl_%'] = np.where(
        hist_df['result'] != 'pending',
        np.where(
            hist_df['side'] == 'BUY',
            (hist_df['exit_price'] - hist_df['entry']) / hist_df['entry'] * 100,
            (hist_df['entry'] - hist_df['exit_price']) / hist_df['entry'] * 100
        ),
        np.nan
    )

    hist_df['floating_pnl_%'] = np.where(
        hist_df['side'] == 'BUY',
        (current_price - hist_df['entry']) / hist_df['entry'] * 100,
        (hist_df['entry'] - current_price) / hist_df['entry'] * 100
    )

    max_pnl_closed = hist_df.loc[hist_df['result'] != 'pending', 'actual_pnl_%'].max()
    max_pnl_floating = hist_df.loc[hist_df['result'] == 'pending', 'floating_pnl_%'].max()
    max_pnl = max(
        max_pnl_closed if pd.notna(max_pnl_closed) else -np.inf,
        max_pnl_floating if pd.notna(max_pnl_floating) else -np.inf
    )
    max_pnl = max_pnl if pd.notna(max_pnl) else 0.0

    min_pnl_closed = hist_df.loc[hist_df['result'] != 'pending', 'actual_pnl_%'].min()
    min_pnl_floating = hist_df.loc[hist_df['result'] == 'pending', 'floating_pnl_%'].min()
    max_dd = min(
        min_pnl_closed if pd.notna(min_pnl_closed) else np.inf,
        min_pnl_floating if pd.notna(min_pnl_floating) else np.inf
    )
    max_dd = max_dd if pd.notna(max_dd) else 0.0

    now = datetime.now()
    durations = []
    for _, row in hist_df.iterrows():
        entry_time = pd.to_datetime(row['time'])
        if row['result'] != 'pending' and pd.notna(row.get('exit_time')):
            exit_time = pd.to_datetime(row['exit_time'])
            durations.append((exit_time - entry_time).total_seconds() / 60)
        else:
            durations.append((now - entry_time).total_seconds() / 60)
    avg_hold = f"{np.mean(durations):.0f} 分钟" if durations else "无持仓"

    return max_pnl, max_dd, avg_hold

# ==================== UI ====================
st.set_page_config(page_title="ETH V15.3 Ultimate Deep Insight (No Binance)", layout="wide")

st.markdown("""
<style>
    .stApp { background: #0e1117; color: #e6edf3; }
    .metric-container { background: #161b22; border-radius: 12px; padding: 16px; border-left: 5px solid #58a6ff; margin: 8px 0; }
    .signal-active { background: linear-gradient(135deg, #1f2a3a, #0f1a2a); border: 2px solid #39d353; border-radius: 12px; padding: 20px; margin: 12px 0; }
    .signal-warning { border-color: #f85149; }
    .alert-box { background: #2d1a1a; border: 2px solid #f85149; border-radius: 10px; padding: 15px; margin: 10px 0; color: #ffcccc; font-weight: bold; }
    .position-suggest { background: #1a2a1a; border: 2px solid #39d353; border-radius: 10px; padding: 15px; margin: 10px 0; }
    .stats-card { background: #1e293b; border-radius: 12px; padding: 16px; margin: 12px 0; text-align: center; }
</style>
""", unsafe_allow_html=True)

st.title("ETH V15.3 Ultimate Deep Insight – 多周期共振 + 动态风控 (无币安版)")

# 侧边栏
with st.sidebar:
    st.header("⚙️ 策略参数")
    st.session_state.fast = st.number_input("EMA快线", 3, 30, 8)
    st.session_state.slow = st.number_input("EMA慢线", 10, 100, 21)
    st.session_state.score_thresh = st.slider("信号触发阈值", 50, 100, 80)
    st.session_state.bleed_threshold = st.slider("阴跌禁入阈值", 40, 100, 76)
    st.session_state.volume_confirm = st.checkbox("启用成交量确认过滤", value=True)
    st.session_state.enable_sound = st.checkbox("启用信号声音提示", value=True)
    st.session_state.auto_refresh = st.checkbox("自动刷新页面", value=False)
    st.divider()

    st.header("📊 风控参数")
    st.session_state.use_atr_sl = st.checkbox("使用ATR动态止损", value=True)
    if st.session_state.use_atr_sl:
        st.session_state.sl_m = st.slider("ATR止损倍数", 1.0, 5.0, 1.8)
        st.session_state.tp1_m = st.slider("TP1倍数", 0.5, 3.0, 1.2, step=0.1)
        st.session_state.tp2_m = st.slider("TP2倍数", 1.0, 5.0, 2.5, step=0.1)
    else:
        st.info("将使用固定比例止损：止损0.5%，TP1 1.0%，TP2 2.0%")
    st.session_state.trailing_sl_pct = st.slider("追踪止损%", 0.1, 1.0, 0.5, step=0.1)
    st.session_state.partial_tp_pct = st.slider("TP1分批平仓%", 0.3, 0.8, 0.5, step=0.1)
    st.session_state.risk_per_trade = st.slider("单笔风险 %", 0.5, 5.0, 1.0, step=0.1)
    st.session_state.account_balance = st.number_input("账户余额 (USD)", 1000.0, 1000000.0, 10000.0, step=100.0)

    st.divider()
    if st.button("🗑 清空历史信号"):
        st.session_state.history.clear()
        st.rerun()
    if st.button("🔄 强制刷新数据"):
        st.cache_data.clear()
        st.rerun()
    if st.session_state.last_error:
        with st.expander("⚠️ API 错误日志"):
            st.write(st.session_state.last_error)

# 自动刷新处理
if st.session_state.auto_refresh:
    if st_autorefresh is None:
        st.warning("请安装 streamlit-autorefresh 以使用自动刷新功能：`pip install streamlit-autorefresh`")
    else:
        st_autorefresh(interval=st.session_state.get('refresh', 20) * 1000, key="auto_refresh")

# 主逻辑
raw_data_5m = fetch_data('5m')
raw_data_15m = fetch_data('15m', 200)
raw_data_1h = fetch_data('1h', 200)

if len(raw_data_5m) > 100:
    df_5m = get_indicators(raw_data_5m)
    df_15m = get_indicators(raw_data_15m) if not raw_data_15m.empty else pd.DataFrame()
    df_1h = get_indicators(raw_data_1h) if not raw_data_1h.empty else pd.DataFrame()

    if df_15m.empty:
        st.warning("⚠️ 15分钟数据获取失败，共振检查可能不完整。")
    if df_1h.empty:
        st.warning("⚠️ 1小时数据获取失败，共振检查可能不完整。")

    bleed_score = calc_bleed_score(df_5m)
    bull_div, bear_div = detect_divergence(df_5m)

    last = df_5m.iloc[-1]
    is_bull = last['ema_fast'] > last['ema_slow'] and last['close'] > last['ema_fast']
    is_bear = last['ema_fast'] < last['ema_slow'] and last['close'] < last['ema_fast']

    side = "BUY" if is_bull else "SELL" if is_bear else None

    resonance_ok = multi_timeframe_resonance(df_5m, df_15m, df_1h, side) if side and not df_15m.empty and not df_1h.empty else False

    confidence = get_brain_score(side, df_5m, bleed_score, bull_div, bear_div) if side else 0

    can_trade = confidence >= st.session_state.score_thresh and bleed_score <= st.session_state.bleed_threshold and resonance_ok

    position_size = suggest_position_size(confidence, bleed_score) if can_trade and side else 0.0

    # 仪表盘
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("综合信心分", f"{confidence:.1f}", delta="可入场" if can_trade else "观望", delta_color="normal" if can_trade else "inverse")
    c2.metric("阴跌风险", f"{bleed_score:.1f}", delta="高危" if bleed_score > 76 else "安全", delta_color="inverse")
    c3.metric("背离预警", (bull_div['type'] if bull_div else "") or (bear_div['type'] if bear_div else "无"), delta_color="inverse" if bull_div or bear_div else "normal")
    c4.metric("多周期共振", "✅ 通过" if resonance_ok else "❌ 未通过", delta_color="normal" if resonance_ok else "inverse")
    c5.metric("当前价格", f"{last['close']:.2f}")

    if position_size > 0:
        st.markdown(f"""
        <div class="position-suggest">
            <strong>建议仓位（{side}）</strong><br>
            风险金额：${position_size:.2f} (占账户 {st.session_state.risk_per_trade}% × 信心因子)<br>
            建议数量：{position_size / last['close']:.4f} ETH
        </div>
        """, unsafe_allow_html=True)

    # 信号触发
    if can_trade and side and st.session_state.last_signal_time != df_5m.index[-1]:
        sl, tp1, tp2 = sltp(last['close'], side, last['atr'])
        st.session_state.history.appendleft({
            'time': df_5m.index[-1].strftime('%Y-%m-%d %H:%M'),
            'side': side,
            'entry': round(last['close'], 2),
            'sl': round(sl, 2),
            'tp1': round(tp1, 2),
            'tp2': round(tp2, 2),
            'confidence': round(confidence, 1),
            'bleed': round(bleed_score, 1),
            'div': bull_div['type'] if bull_div else bear_div['type'] if bear_div else "",
            'resonance': resonance_ok,
            'suggested_size': round(position_size / last['close'], 4),
            'result': 'pending',
            'partial_closed': False,
            'trailing_sl': sl,
            'highest_price': last['close'] if side == 'BUY' else None,
            'lowest_price': last['close'] if side == 'SELL' else None,
            'exit_price': None,
            'exit_time': None
        })
        st.session_state.last_signal_time = df_5m.index[-1]
        st.toast(f"🚀 {side} 顶级信号触发！信心 {confidence:.1f}", icon="🔥")
        if st.session_state.enable_sound:
            st.components.v1.html("""
            <audio autoplay>
                <source src="https://www.soundjay.com/buttons/beep-07.mp3" type="audio/mpeg">
            </audio>
            """, height=0)

    # 更新追踪止损 & 分批止盈
    for rec in st.session_state.history:
        if rec['result'] == 'pending':
            current_price = df_5m['close'].iloc[-1]
            rec = update_position(rec, current_price)

    # 信号统计汇总
    if st.session_state.history:
        hist_df_raw = pd.DataFrame(list(st.session_state.history))
        max_pnl, max_dd, avg_hold = calculate_stats(hist_df_raw, df_5m['close'].iloc[-1])

        cols = st.columns(3)
        cols[0].metric("最大 PNL", f"{max_pnl:+.2f}%", delta_color="normal" if max_pnl >= 0 else "inverse")
        cols[1].metric("最大回撤", f"{max_dd:+.2f}%", delta_color="inverse")
        cols[2].metric("平均持仓时长", avg_hold)

    # 图表
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.55, 0.15, 0.3])

    fig.add_trace(go.Candlestick(x=df_5m.index, open=df_5m['open'], high=df_5m['high'], low=df_5m['low'], close=df_5m['close'], name="ETH"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_5m.index, y=df_5m['ema_fast'], line=dict(color='#ffd700'), name="Fast"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_5m.index, y=df_5m['ema_slow'], line=dict(color='#4da9ff'), name="Slow"), row=1, col=1)

    vol_colors = ['#26de81' if c >= o else '#fc5c65' for c, o in zip(df_5m['close'], df_5m['open'])]
    fig.add_trace(go.Bar(x=df_5m.index, y=df_5m['volume'], marker_color=vol_colors), row=2, col=1)

    fig.add_trace(go.Bar(x=df_5m.index, y=df_5m['macd_hist'], marker_color='orange'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_5m.index, y=df_5m['macd_line'], line=dict(color='#00ff9d')), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_5m.index, y=df_5m['macd_signal'], line=dict(color='#ff4d88', dash='dot')), row=3, col=1)

    if bull_div:
        fig.add_annotation(x=bull_div['time2'], y=df_5m['low'].loc[bull_div['time2']]*0.992,
                           text=bull_div['type'], showarrow=True, arrowhead=2, ax=50, ay=-60,
                           font=dict(color="#39d353", size=14), row=1, col=1)
    if bear_div:
        fig.add_annotation(x=bear_div['time2'], y=df_5m['high'].loc[bear_div['time2']]*1.008,
                           text=bear_div['type'], showarrow=True, arrowhead=2, ax=-50, ay=60,
                           font=dict(color="#f85149", size=14), row=1, col=1)

    fig.update_layout(height=940, template="plotly_dark", showlegend=False, xaxis_rangeslider_visible=False, margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig, use_container_width=True)

    # 信号记录表格
    if st.session_state.history:
        st.subheader("📜 信号记录（含追踪止损 & 分批状态）")
        hist_df = pd.DataFrame(list(st.session_state.history))
        current_price = df_5m['close'].iloc[-1]

        # 计算盈亏显示列
        hist_df['盈亏%'] = np.where(
            hist_df['result'] != 'pending',
            np.where(
                hist_df['side'] == 'BUY',
                (hist_df['exit_price'] - hist_df['entry']) / hist_df['entry'] * 100,
                (hist_df['entry'] - hist_df['exit_price']) / hist_df['entry'] * 100
            ),
            np.where(
                hist_df['side'] == 'BUY',
                (current_price - hist_df['entry']) / hist_df['entry'] * 100,
                (hist_df['entry'] - current_price) / hist_df['entry'] * 100
            )
        ).round(2)
        hist_df['盈亏%'] = hist_df['盈亏%'].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "待定")
        hist_df['部分止盈'] = hist_df['partial_closed'].map({True: '✅', False: '❌'})

        display_cols = ['time', 'side', 'entry', 'sl', 'tp1', 'tp2', '盈亏%', 'result', '部分止盈']
        hist_display = hist_df[display_cols].copy()

        def style_result(val):
            if val == 'pending':
                return 'background-color: #ffd70020; color: #ffd700'
            elif val == '止损出局':
                return 'background-color: #f8514920; color: #f85149'
            else:
                return 'background-color: #39d35320; color: #39d353'

        styled = hist_display.style.applymap(style_result, subset=['result'])
        styled = styled.applymap(lambda x: 'color: #39d353' if x == 'BUY' else 'color: #f85149', subset=['side'])

        st.dataframe(styled)

        csv = hist_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 导出全部记录", csv, "eth_v15_signals.csv", "text/csv")

else:
    st.info("⌛ 正在深度同步多周期行情数据... (需要至少100根K线)")

st.caption("ETH V15.3 Ultimate Deep Insight | 多周期共振 + 追踪止损 + 分批止盈 + 阴跌最强防御 + 精确统计 | 无币安 | 2026")
