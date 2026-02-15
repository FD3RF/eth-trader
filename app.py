# -*- coding: utf-8 -*-
"""
🚀 合約智能監控中心 · 終極資金曲線版（風險因子驅動）
市場環境層 | 信號層 | 風險因子層 | 資金管理層 | 生存保護層
多幣種卡片｜資金曲線｜簡易回測｜交易日誌｜風險預警
"""

import streamlit as st
import pandas as pd
import numpy as np
import ta
import ccxt
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
from streamlit_autorefresh import st_autorefresh
import warnings
import joblib
import os
from collections import Counter

warnings.filterwarnings('ignore')

# ==================== 全域配置 ====================
SYMBOLS = ["ETH/USDT", "BTC/USDT", "SOL/USDT"]
R_BASE = 0.01                       # 基礎風險 1%
MAX_LEVERAGE = 20.0                 # 最大槓桿
STOP_ATR = 1.5                      # 止損倍數
TAKE_ATR = 3.0                      # 止盈倍數
CONSECUTIVE_LOSS_LIMIT = 3          # 連虧煞車閾值
CONSECUTIVE_STOP_HOURS = 24         # 連虧暫停小時數
MAX_DRAWDOWN = 20.0                  # 最大回撤警戒線
DAILY_LOSS_LIMIT = 300.0             # 日虧損限額
MIN_ATR_PCT = 0.8                    # 最小波動率要求（低於此值風險減半，但不禁止交易）

# ==================== 免費數據獲取器（支援多幣種）====================
class FreeDataFetcherV5:
    """支援多幣種的免費數據獲取器"""
    def __init__(self, symbols=None):
        if symbols is None:
            symbols = SYMBOLS
        self.symbols = symbols
        self.periods = ['15m', '1h', '4h', '1d']
        self.limit = 500
        self.timeout = 10
        self.exchange = ccxt.mexc({'enableRateLimit': True, 'timeout': 30000})
        self.fng_url = "https://api.alternative.me/fng/"
        self.chain_netflow = 5234
        self.chain_whale = 128

    def fetch_kline(self, symbol, timeframe):
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=self.limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            return df, "MEXC"
        except Exception as e:
            st.warning(f"{symbol} {timeframe} 獲取失敗: {e}")
            return None, None

    def fetch_fear_greed(self):
        try:
            resp = requests.get(self.fng_url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                return int(data['data'][0]['value'])
        except:
            pass
        return 50

    def fetch_all(self):
        all_data = {}
        fear_greed = self.fetch_fear_greed()
        for symbol in self.symbols:
            data_dict = {}
            price_sources = []
            for period in self.periods:
                df, src = self.fetch_kline(symbol, period)
                if df is not None:
                    data_dict[period] = self._add_indicators(df)
                    price_sources.append(src)
            if data_dict:
                all_data[symbol] = {
                    "data_dict": data_dict,
                    "current_price": data_dict['15m']['close'].iloc[-1] if '15m' in data_dict else None,
                    "source": price_sources[0] if price_sources else "MEXC",
                    "fear_greed": fear_greed,
                    "chain_netflow": self.chain_netflow,
                    "chain_whale": self.chain_whale,
                }
        return all_data

    def _add_indicators(self, df):
        df = df.copy()
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma60'] = df['close'].rolling(60).mean()
        df['ema20'] = df['close'].ewm(span=20).mean()
        df['ema50'] = df['close'].ewm(span=50).mean()
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
        df['atr_pct'] = df['atr'] / df['close'] * 100.0
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
        df['adx'] = adx.adx()
        return df


# ==================== 市場環境層 ====================
def evaluate_market(df_dict):
    """判斷市場狀態：趨勢/震盪/禁止交易（僅供參考，不強制禁止）"""
    if '15m' not in df_dict:
        return "未知", 0.0, 0.0
    df = df_dict['15m']
    last = df.iloc[-1]

    ema20 = last['ema20']
    ema50 = last['ema50']
    adx = last['adx']
    atr_pct = last['atr_pct']

    # 異常波動檢測（僅警告，不禁止）
    body = abs(last['close'] - last['open'])
    if body > 3 * last['atr']:
        return "異常波動", atr_pct, adx

    if ema20 > ema50 and adx > 20:
        return "趨勢", atr_pct, adx
    elif adx < 25:
        return "震盪", atr_pct, adx
    else:
        return "不明確", atr_pct, adx


# ==================== 五層共振評分（用於風險因子）====================
def five_layer_score(df_dict, fear_greed, chain_netflow, chain_whale):
    """
    五層共振評分，每層20分，總分0-100
    返回：(方向, 總分, 各層分數) 方向保留供參考，不影響信號
    """
    if not df_dict or '15m' not in df_dict or '1h' not in df_dict or '4h' not in df_dict or '1d' not in df_dict:
        return 0, 0, {}

    df_15m = df_dict['15m']
    df_1h = df_dict['1h']
    df_4h = df_dict['4h']
    df_1d = df_dict['1d']

    last_15m = df_15m.iloc[-1]
    last_1h = df_1h.iloc[-1]
    last_4h = df_4h.iloc[-1]
    last_1d = df_1d.iloc[-1]

    # 1. 趨勢層 (20分)
    trend_score = 0
    trend_dir = 0
    adx = last_15m['adx']
    if adx > 25:
        trend_score = 20
        trend_dir = 1 if last_15m['ema20'] > last_15m['ema50'] else -1
    elif adx > 20:
        trend_score = 10
        trend_dir = 1 if last_15m['ema20'] > last_15m['ema50'] else -1

    # 2. 多週期層 (20分)
    multi_score = 0
    multi_dir = 0
    dir_15m = 1 if last_15m['ema20'] > last_15m['ema50'] else -1
    dir_1h = 1 if last_1h['ema20'] > last_1h['ema50'] else -1
    dir_4h = 1 if last_4h['ema20'] > last_4h['ema50'] else -1
    dir_1d = 1 if last_1d['ema20'] > last_1d['ema50'] else -1

    if dir_15m == dir_1h == dir_4h == dir_1d:
        multi_score = 20
        multi_dir = dir_15m
    elif dir_15m == dir_1h == dir_4h:
        multi_score = 15
        multi_dir = dir_15m
    elif dir_15m == dir_1h:
        multi_score = 10
        multi_dir = dir_15m

    # 3. 資金面層 (20分) - 模擬
    fund_score = 0
    fund_dir = 0

    # 4. 鏈上/情緒層 (20分)
    chain_score = 0
    chain_dir = 0
    if chain_netflow > 5000 and chain_whale > 100:
        chain_score = 20
        chain_dir = 1
    elif fear_greed < 30:
        chain_score = 15
        chain_dir = 1
    elif fear_greed > 70:
        chain_score = 15
        chain_dir = -1

    # 5. 動量層 (20分)
    momentum_score = 0
    momentum_dir = 0
    rsi = last_15m['rsi']
    macd_diff = last_15m['macd'] - last_15m['macd_signal']
    if rsi > 55 and macd_diff > 0:
        momentum_score = 20
        momentum_dir = 1
    elif rsi < 45 and macd_diff < 0:
        momentum_score = 20
        momentum_dir = -1
    elif rsi > 50:
        momentum_score = 10
        momentum_dir = 1
    elif rsi < 50:
        momentum_score = 10
        momentum_dir = -1

    total_score = trend_score + multi_score + fund_score + chain_score + momentum_score

    # 最終方向（僅供參考）
    dirs = [trend_dir, multi_dir, fund_dir, chain_dir, momentum_dir]
    dirs = [d for d in dirs if d != 0]
    if len(dirs) >= 3:
        count = Counter(dirs)
        final_dir = count.most_common(1)[0][0]
    else:
        final_dir = 0

    layer_scores = {
        "趨勢": trend_score,
        "多週期": multi_score,
        "資金面": fund_score,
        "鏈上": chain_score,
        "動量": momentum_score
    }
    return final_dir, total_score, layer_scores


# ==================== 入場信號層（獨立於五層評分）====================
def generate_entry_signal(df_dict, market_mode):
    """根據市場模式生成入場信號，不依賴五層評分"""
    if '15m' not in df_dict:
        return 0
    df = df_dict['15m']
    last = df.iloc[-1]

    if market_mode == "趨勢":
        ema20 = last['ema20']
        ema50 = last['ema50']
        # 趨勢多：EMA20 > EMA50 且 價格回踩EMA20 且 RSI未過熱
        if (ema20 > ema50 and 
            last['close'] >= ema20 * 0.99 and 
            last['rsi'] < 70 and last['rsi'] > 40):
            return 1
        # 趨勢空：EMA20 < EMA50 且 價格反彈至EMA20 且 RSI未超賣
        elif (ema20 < ema50 and 
              last['close'] <= ema20 * 1.01 and 
              last['rsi'] > 30 and last['rsi'] < 60):
            return -1
        else:
            return 0
    elif market_mode == "震盪":
        bb_upper = last['bb_high']
        bb_lower = last['bb_low']
        # 下軌買
        if last['close'] <= bb_lower * 1.01 and last['rsi'] < 30:
            return 1
        # 上軌賣
        elif last['close'] >= bb_upper * 0.99 and last['rsi'] > 70:
            return -1
        else:
            return 0
    else:
        # 其他狀態（異常波動、不明確）不開倉
        return 0


# ==================== 風險控制層 ====================
def calculate_stops(entry_price, side, atr_value):
    stop_distance = STOP_ATR * atr_value
    take_distance = TAKE_ATR * atr_value
    if side == 1:
        stop = entry_price - stop_distance
        take = entry_price + take_distance
    else:
        stop = entry_price + stop_distance
        take = entry_price - take_distance
    return stop, take, take_distance/stop_distance


# ==================== 風險因子計算 ====================
def calculate_risk_factors(five_total, atr_pct, drawdown, consecutive_losses):
    """
    計算最終風險係數 R_final = R_base * F_score * F_vol * F_dd * F_loss
    限制在 [0.001, 0.02] 之間
    """
    # 1. 質量因子 F_score = five_total / 100
    F_score = five_total / 100.0
    F_score = max(0.1, min(1.0, F_score))  # 限制範圍

    # 2. 波動因子 F_vol 分檔
    if atr_pct < 0.8:
        F_vol = 0.5
    elif atr_pct <= 2.5:
        F_vol = 1.0
    else:  # >2.5%
        F_vol = 0.7

    # 3. 回撤因子 F_dd
    if drawdown < 10:
        F_dd = 1.0
    elif drawdown <= 20:
        F_dd = 0.5
    else:
        F_dd = 0.3

    # 4. 連虧因子 F_loss
    if consecutive_losses < 3:
        F_loss = 1.0
    elif consecutive_losses <= 4:
        F_loss = 0.5
    else:
        F_loss = 0.2

    # 計算最終風險
    R_final = R_BASE * F_score * F_vol * F_dd * F_loss
    # 限制範圍
    R_final = max(0.001, min(0.02, R_final))
    return R_final, F_score, F_vol, F_dd, F_loss


# ==================== 資金管理層 ====================
def calculate_position_size(balance, entry_price, stop_price, R_final, max_leverage=MAX_LEVERAGE):
    """根據最終風險比例計算倉位"""
    risk_amount = balance * R_final
    stop_distance = abs(entry_price - stop_price)
    if stop_distance == 0:
        return 0.0
    position_value = risk_amount / stop_distance * entry_price
    max_position = balance * max_leverage
    position_value = min(position_value, max_position)
    quantity = position_value / entry_price
    return round(quantity, 3)


# ==================== 生存保護層（連虧、回撤、日虧損）====================
class SurvivalProtection:
    """生存保護：記錄連續虧損、回撤、日虧損，並提供因子計算所需數據"""
    def __init__(self):
        self.consecutive_losses = 0
        self.peak_balance = 10000.0
        self.mode_switch_time = None
        self.trading_paused_until = None
        self.daily_loss_triggered = False
        self.last_mode = None
        self.daily_pnl = 0.0

    def update(self, trade_result, current_balance, current_mode, last_kline_time, daily_pnl):
        if trade_result < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        if current_balance > self.peak_balance:
            self.peak_balance = current_balance
        drawdown = (self.peak_balance - current_balance) / self.peak_balance * 100.0

        if self.last_mode is not None and current_mode != self.last_mode:
            self.mode_switch_time = last_kline_time
        self.last_mode = current_mode

        if daily_pnl < -DAILY_LOSS_LIMIT:
            self.daily_loss_triggered = True

        # 檢查是否暫停交易（僅日虧損超限時）
        paused = False
        if self.daily_loss_triggered:
            paused = True

        return paused, drawdown

    def can_trade(self, current_time):
        if self.daily_loss_triggered:
            return False
        return True


# ==================== 強平價格計算 ====================
def calculate_liquidation_price(entry_price, side, leverage):
    if side == "多單":
        return entry_price * (1 - 1.0/leverage)
    else:
        return entry_price * (1 + 1.0/leverage)


# ==================== 簡易回測（適配新邏輯）====================
def run_backtest(df_dict, market_func, signal_func, five_func, initial_balance=10000.0, lookback_days=30):
    """簡易回測（忽略風險因子，僅用信號方向測試）"""
    df = df_dict['15m'].copy()
    lookback = lookback_days * 96
    df = df.iloc[-lookback:] if len(df) > lookback else df

    balance = initial_balance
    peak = balance
    trades = 0
    wins = 0
    losses = 0
    total_profit = 0.0
    total_loss = 0.0
    max_drawdown = 0.0

    position = None
    entry_price = 0.0

    for i in range(len(df)):
        row = df.iloc[i]
        temp_dict = {'15m': df.iloc[:i+1], '1h': None, '4h': None, '1d': None}
        market_mode, _, _ = market_func(temp_dict)
        # 信號不依賴五層評分，但五層評分仍需計算（此處用默認值）
        signal = signal_func(temp_dict, market_mode)

        if market_mode in ["異常波動", "不明確"]:
            continue

        if position is None:
            if signal == 1:
                position = 'long'
                entry_price = row['close']
            elif signal == -1:
                position = 'short'
                entry_price = row['close']
        else:
            if (position == 'long' and signal <= 0) or (position == 'short' and signal >= 0):
                exit_price = row['close']
                if position == 'long':
                    pnl = (exit_price - entry_price) / entry_price * 100.0
                else:
                    pnl = (entry_price - exit_price) / entry_price * 100.0
                trades += 1
                if pnl > 0:
                    wins += 1
                    total_profit += pnl
                else:
                    losses += 1
                    total_loss += abs(pnl)
                balance *= (1.0 + pnl/100.0)
                if balance > peak:
                    peak = balance
                else:
                    dd = (peak - balance) / peak * 100.0
                    if dd > max_drawdown:
                        max_drawdown = dd
                position = None
        if balance > peak:
            peak = balance

    win_rate = wins / trades if trades > 0 else 0.0
    total_return = (balance - initial_balance) / initial_balance * 100.0
    profit_factor = total_profit / total_loss if total_loss > 0 else 0.0
    return {
        '勝率': f"{win_rate*100:.1f}%",
        '總收益': f"{total_return:.1f}%",
        '最大回撤': f"{max_drawdown:.1f}%",
        '盈虧比': f"{profit_factor:.2f}",
        '交易次數': trades
    }


# ==================== 初始化session_state ====================
def init_session_state():
    if 'account_balance' not in st.session_state:
        st.session_state.account_balance = 10000.0
    if 'daily_pnl' not in st.session_state:
        st.session_state.daily_pnl = 0.0
    if 'peak_balance' not in st.session_state:
        st.session_state.peak_balance = 10000.0
    if 'last_date' not in st.session_state:
        st.session_state.last_date = datetime.now().date()
    if 'balance_history' not in st.session_state:
        st.session_state.balance_history = []
    if 'trade_log' not in st.session_state:
        st.session_state.trade_log = []
    if 'auto_enabled' not in st.session_state:
        st.session_state.auto_enabled = False
    if 'auto_position' not in st.session_state:
        st.session_state.auto_position = None
    if 'signal_history' not in st.session_state:
        st.session_state.signal_history = []
    if 'protection' not in st.session_state:
        st.session_state.protection = SurvivalProtection()


def update_risk_stats(current_price, sim_entry, sim_side, sim_quantity, sim_leverage):
    today = datetime.now().date()
    if today != st.session_state.last_date:
        st.session_state.daily_pnl = 0.0
        st.session_state.last_date = today
        st.session_state.protection.daily_loss_triggered = False

    if sim_entry > 0 and current_price:
        if sim_side == "多單":
            pnl = (current_price - sim_entry) * sim_quantity * sim_leverage
        else:
            pnl = (sim_entry - current_price) * sim_quantity * sim_leverage
        st.session_state.daily_pnl = pnl
        st.session_state.protection.daily_pnl = pnl

    current_balance = st.session_state.account_balance + st.session_state.daily_pnl
    if current_balance > st.session_state.peak_balance:
        st.session_state.peak_balance = current_balance
    drawdown = (st.session_state.peak_balance - current_balance) / st.session_state.peak_balance * 100.0
    return drawdown


# ==================== 主界面 ====================
st.set_page_config(page_title="合約智能監控·終極資金曲線版", layout="wide")
st.markdown("""
<style>
.stApp { background-color: #0B0E14; color: white; }
.ai-box { background: #1A1D27; border-radius: 10px; padding: 20px; border-left: 6px solid #00F5A0; }
.metric { background: #232734; padding: 15px; border-radius: 8px; }
.signal-buy { color: #00F5A0; font-weight: bold; }
.signal-sell { color: #FF5555; font-weight: bold; }
.profit { color: #00F5A0; }
.loss { color: #FF5555; }
.warning { color: #FFA500; }
.danger { color: #FF0000; font-weight: bold; }
.info-box { background: #1A2A3A; border-left: 6px solid #00F5A0; padding: 10px; border-radius: 5px; margin-bottom: 10px; }
.trade-plan { background: #232734; padding: 15px; border-radius: 8px; margin-top: 10px; border-left: 6px solid #FFAA00; }
.dashboard { background: #1A1D27; padding: 15px; border-radius: 8px; border-left: 6px solid #00F5A0; margin-bottom: 10px; }
.card { background: #1A1D27; border-radius: 5px; padding: 10px; text-align: center; cursor: pointer; }
.card:hover { background: #2A2D37; }
</style>
""", unsafe_allow_html=True)

st.title("🧠 合約智能監控中心 · 終極資金曲線版")
st.caption("市場環境｜獨立信號｜五層風險因子｜波動分級｜回撤保護｜連虧降級")

init_session_state()
ai_model = None

# 側邊欄
with st.sidebar:
    st.header("⚙️ 控制面板")
    selected_symbol = st.selectbox("主交易對", SYMBOLS, index=0, key="selected_symbol")
    main_period = st.selectbox("主圖週期", ["15m", "1h", "4h", "1d"], index=0)
    auto_refresh = st.checkbox("開啟自動刷新", value=True)
    refresh_interval = st.number_input("刷新間隔(秒)", min_value=5, max_value=60, value=10, step=1, disabled=not auto_refresh)
    if auto_refresh:
        st_autorefresh(interval=refresh_interval * 1000, key="auto_refresh")

    st.markdown("---")
    st.subheader("📈 模擬合約")
    sim_entry = st.number_input("開倉價", value=0.0, format="%.2f", step=0.01)
    sim_side = st.selectbox("方向", ["多單", "空單"])
    sim_leverage = st.slider("槓桿倍數", min_value=1.0, max_value=100.0, value=10.0, step=1.0)
    sim_quantity = st.number_input("數量", value=0.01, format="%.4f", step=0.001)

    st.markdown("---")
    st.subheader("💰 風控設置")
    account_balance = st.number_input("初始資金 (USDT)", value=st.session_state.account_balance, step=1000.0, format="%.2f")
    daily_loss_limit = st.number_input("日虧損限額 (USDT)", value=DAILY_LOSS_LIMIT, step=50.0, format="%.2f")
    risk_per_trade_display = st.slider("基礎單筆風險 (%)", min_value=0.5, max_value=3.0, value=R_BASE*100, step=0.5) / 100.0
    st.session_state.account_balance = account_balance

    st.markdown("---")
    st.subheader("📊 簡易回測")
    backtest_days = st.slider("回測天數", min_value=7, max_value=90, value=30, step=1)
    if st.button("運行回測"):
        with st.spinner("回測中..."):
            fetcher = FreeDataFetcherV5(symbols=[selected_symbol])
            backtest_data = fetcher.fetch_all()
            if backtest_data and selected_symbol in backtest_data:
                bt_result = run_backtest(
                    backtest_data[selected_symbol]["data_dict"],
                    evaluate_market,
                    generate_entry_signal,
                    five_layer_score,  # 回測中五層評分未被使用，僅佔位
                    initial_balance=account_balance,
                    lookback_days=backtest_days
                )
                st.success("回測完成")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("勝率", bt_result['勝率'])
                    st.metric("總收益", bt_result['總收益'])
                with col2:
                    st.metric("最大回撤", bt_result['最大回撤'])
                    st.metric("盈虧比", bt_result['盈虧比'])
                with col3:
                    st.metric("交易次數", bt_result['交易次數'])

# 獲取數據
with st.spinner("獲取全市場數據..."):
    fetcher = FreeDataFetcherV5(symbols=SYMBOLS)
    all_data = fetcher.fetch_all()

# 多幣種卡片
st.markdown("### 🔥 多幣種即時信號")
cols = st.columns(len(SYMBOLS))
for i, sym in enumerate(SYMBOLS):
    if sym in all_data:
        df_dict = all_data[sym]["data_dict"]
        mode, _, _ = evaluate_market(df_dict)
        signal = generate_entry_signal(df_dict, mode)
        dir_icon = {1: "🟢 多", -1: "🔴 空", 0: "⚪ 觀"}[signal]
        with cols[i]:
            if st.button(f"{sym}\n{dir_icon}\n{mode}", key=f"card_{sym}"):
                st.session_state.selected_symbol = sym
                st.rerun()

# 當前選中的幣種
if selected_symbol not in all_data:
    selected_symbol = SYMBOLS[0]
data = all_data[selected_symbol]
data_dict = data["data_dict"]
current_price = data["current_price"]
fear_greed = data["fear_greed"]
source_display = data["source"]
chain_netflow = data["chain_netflow"]
chain_whale = data["chain_whale"]

# 五層共振評分
five_dir, five_total, layer_scores = five_layer_score(data_dict, fear_greed, chain_netflow, chain_whale)
st.session_state.five_total = five_total

# 市場環境評估
market_mode, atr_pct, adx = evaluate_market(data_dict)

# 入場信號（獨立）
entry_signal = generate_entry_signal(data_dict, market_mode)

# ATR值
atr_value = data_dict['15m']['atr'].iloc[-1] if '15m' in data_dict else 0.0

# 計算風險因子和最終風險比例
drawdown = update_risk_stats(current_price, sim_entry, sim_side, sim_quantity, sim_leverage)
consecutive_losses = st.session_state.protection.consecutive_losses
R_final, F_score, F_vol, F_dd, F_loss = calculate_risk_factors(five_total, atr_pct, drawdown, consecutive_losses)

# 交易計劃（僅在有信號時）
stop_loss = take_profit = risk_reward = None
position_size = 0.0
if entry_signal != 0 and atr_value > 0:
    stop_loss, take_profit, risk_reward = calculate_stops(current_price, entry_signal, atr_value)
    # 用 R_final 計算倉位
    position_size = calculate_position_size(
        st.session_state.account_balance,
        current_price,
        stop_loss,
        R_final
    )

# 生存保護層檢查
protection = st.session_state.protection
now = datetime.now()
paused, drawdown_protect = protection.update(0.0, st.session_state.account_balance + st.session_state.daily_pnl,
                                             market_mode, now, st.session_state.daily_pnl)  # trade_result佔位
can_trade = protection.can_trade(now)

# 顯示狀態
st.markdown(f"""
<div class="info-box">
    ✅ 價格源：{source_display} | 恐懼貪婪：{fear_greed} | 市場狀態：{market_mode} | 五層總分：{five_total}
    <br>⚠️ 鏈上數據為模擬值 | { '🔴 交易暫停中' if not can_trade else '' }
</div>
""", unsafe_allow_html=True)

if not can_trade:
    reason = []
    if protection.daily_loss_triggered:
        reason.append("日虧損超限")
    st.error(f"🚨 交易暫停: {', '.join(reason)}")

# 主布局
col_left, col_right = st.columns([2.2, 1.3])

with col_left:
    st.markdown(f"<h5>市場狀態: <span style='color:green;'>{market_mode}</span> | ADX: {adx:.1f} | ATR%: {atr_pct:.2f}% | 五層總分: {five_total}</h5>", unsafe_allow_html=True)

    # 五層熱力圖
    st.subheader("🔥 五層權重（風險因子）")
    cols = st.columns(5)
    layer_names = list(layer_scores.keys())
    layer_values = list(layer_scores.values())
    colors = ['#00F5A0', '#00F5A0', '#FFAA00', '#FF5555', '#FFAA00']
    for i, col in enumerate(cols):
        with col:
            val = layer_values[i]
            bg_color = colors[i] if val > 10 else '#555'
            st.markdown(f"""
            <div style="background:{bg_color}22; border-left:4px solid {bg_color}; padding:10px; border-radius:5px; text-align:center;">
                <h4>{layer_names[i]}</h4>
                <h2>{val}</h2>
            </div>
            """, unsafe_allow_html=True)

    # K線圖
    st.subheader(f"📊 {selected_symbol} K線 ({main_period})")
    if main_period in data_dict:
        df = data_dict[main_period].tail(100).copy()
        df['日期'] = df['timestamp']
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           row_heights=[0.7, 0.3],
                           subplot_titles=(f"{selected_symbol} {main_period}", "RSI"))
        fig.add_trace(go.Candlestick(x=df['日期'], open=df['open'], high=df['high'],
                                     low=df['low'], close=df['close'], name="K線"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['日期'], y=df['ema20'], name="EMA20", line=dict(color="orange")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['日期'], y=df['ema50'], name="EMA50", line=dict(color="blue")), row=1, col=1)
        if entry_signal != 0:
            last_date = df['日期'].iloc[-1]
            last_price = df['close'].iloc[-1]
            arrow_text = "▲ 多" if entry_signal == 1 else "▼ 空"
            arrow_color = "green" if entry_signal == 1 else "red"
            fig.add_annotation(x=last_date, y=last_price * (1.02 if entry_signal==1 else 0.98),
                               text=arrow_text, showarrow=True, arrowhead=2, arrowcolor=arrow_color)
        fig.add_trace(go.Scatter(x=df['日期'], y=df['rsi'], name="RSI", line=dict(color="purple")), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
        fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=600)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("K線數據不可用")

with col_right:
    st.subheader("🧠 即時決策")
    dir_map = {1: "🔴 做多", -1: "🔵 做空", 0: "⚪ 觀望"}
    st.markdown(f'<div class="ai-box">{dir_map[entry_signal]}<br>五層總分: {five_total}/100</div>', unsafe_allow_html=True)

    # 顯示風險因子
    st.markdown(f"""
    <div style="background:#1A1D27; padding:15px; border-radius:8px; margin:10px 0;">
        <h4>⚖️ 風險因子</h4>
        <p>基礎風險: {R_BASE*100:.1f}%</p>
        <p>質量因子 (F_score): {F_score:.2f}</p>
        <p>波動因子 (F_vol): {F_vol:.2f}</p>
        <p>回撤因子 (F_dd): {F_dd:.2f}</p>
        <p>連虧因子 (F_loss): {F_loss:.2f}</p>
        <p><strong>最終風險: {R_final*100:.3f}%</strong></p>
    </div>
    """, unsafe_allow_html=True)

    if entry_signal != 0 and stop_loss and take_profit:
        st.markdown(f"""
        <div class="trade-plan">
            <h4>📋 交易計劃</h4>
            <p>入場價: <span style="color:#00F5A0">${current_price:.2f}</span></p>
            <p>止損價: <span style="color:#FF5555">${stop_loss:.2f}</span> (虧損 {abs(current_price-stop_loss)/current_price*100:.2f}%)</p>
            <p>止盈價: <span style="color:#00F5A0">${take_profit:.2f}</span> (盈虧比 {risk_reward:.2f})</p>
            <p>建議倉位: {position_size} {selected_symbol.split('/')[0]}</p>
        </div>
        """, unsafe_allow_html=True)

    st.metric("當前價格", f"${current_price:.2f}" if current_price else "N/A")

    # 風險儀表盤
    with st.container():
        st.markdown('<div class="dashboard">', unsafe_allow_html=True)
        st.markdown("#### 📊 風險儀表盤")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.metric("帳戶餘額", f"${st.session_state.account_balance:.2f}")
            st.metric("日盈虧", f"${st.session_state.daily_pnl:.2f}", delta_color="inverse")
        with col_r2:
            st.metric("當前回撤", f"{drawdown:.2f}%")
            st.metric("日虧損剩餘", f"${daily_loss_limit + st.session_state.daily_pnl:.2f}")
            st.metric("連續虧損", consecutive_losses)
        if st.session_state.balance_history:
            st.line_chart(st.session_state.balance_history)
        st.markdown('</div>', unsafe_allow_html=True)

    # 資金面快照
    with st.expander("💰 資金面快照", expanded=True):
        st.write("資金費率: **暫缺（模擬）**")
        st.write("OI變化: **暫缺（模擬）**")
        st.write("多空比: **暫缺（模擬）**")

    with st.expander("🔗 鏈上&情緒", expanded=False):
        st.write(f"交易所淨流入: **{chain_netflow:+.0f} {selected_symbol.split('/')[0]}** (模擬)")
        st.write(f"大額轉帳: **{chain_whale}** 筆 (模擬)")
        st.write(f"恐懼貪婪指數: **{fear_greed}**")

    # 模擬持倉
    if sim_entry > 0 and current_price:
        if sim_side == "多單":
            pnl = (current_price - sim_entry) * sim_quantity * sim_leverage
            pnl_pct = (current_price - sim_entry) / sim_entry * sim_leverage * 100.0
            liq_price = calculate_liquidation_price(sim_entry, "多單", sim_leverage)
        else:
            pnl = (sim_entry - current_price) * sim_quantity * sim_leverage
            pnl_pct = (sim_entry - current_price) / sim_entry * sim_leverage * 100.0
            liq_price = calculate_liquidation_price(sim_entry, "空單", sim_leverage)
        color_class = "profit" if pnl >= 0 else "loss"
        distance = abs(current_price - liq_price) / current_price * 100.0
        st.markdown(f"""
        <div class="metric">
            <h4>模擬持倉</h4>
            <p>{sim_side} | {sim_leverage:.1f}x</p>
            <p>開倉: ${sim_entry:.2f}</p>
            <p class="{color_class}">盈虧: ${pnl:.2f} ({pnl_pct:.2f}%)</p>
            <p>強平價: <span class="warning">${liq_price:.2f}</span> (距 {distance:.1f}%)</p>
        </div>
        """, unsafe_allow_html=True)
        if distance < 5:
            st.warning("⚠️ 接近強平線！")
    else:
        st.info("輸入開倉價查看模擬")

    # 策略自動測試
    st.markdown("---")
    st.subheader("🧪 策略自動測試")
    auto_enabled = st.checkbox("啟用自動跟隨信號（模擬）", value=st.session_state.auto_enabled)
    st.session_state.auto_enabled = auto_enabled

    if auto_enabled and can_trade and entry_signal != 0:
        if st.session_state.auto_position is None:
            st.session_state.auto_position = {
                'side': 'long' if entry_signal == 1 else 'short',
                'entry': current_price,
                'time': datetime.now(),
                'leverage': MAX_LEVERAGE,
                'stop': stop_loss,
                'take': take_profit,
                'size': position_size
            }
            st.success(f"✅ 自動開{st.session_state.auto_position['side']}倉 @ {current_price:.2f}")
        else:
            pos = st.session_state.auto_position
            if (pos['side'] == 'long' and (current_price <= pos['stop'] or current_price >= pos['take'])) or \
               (pos['side'] == 'short' and (current_price >= pos['stop'] or current_price <= pos['take'])) or \
               (entry_signal == -1 and pos['side'] == 'long') or \
               (entry_signal == 1 and pos['side'] == 'short'):
                if pos['side'] == 'long':
                    pnl = (current_price - pos['entry']) * pos['leverage']
                else:
                    pnl = (pos['entry'] - current_price) * pos['leverage']
                pnl_pct = pnl / pos['entry'] * 100.0
                protection.update(pnl, st.session_state.account_balance + st.session_state.daily_pnl,
                                  market_mode, now, st.session_state.daily_pnl)
                st.session_state.trade_log.append({
                    '開倉時間': pos['time'].strftime('%H:%M'),
                    '方向': pos['side'],
                    '開倉價': f"{pos['entry']:.2f}",
                    '平倉時間': datetime.now().strftime('%H:%M'),
                    '平倉價': f"{current_price:.2f}",
                    '盈虧': f"{pnl:.2f}",
                    '盈虧%': f"{pnl_pct:.1f}%"
                })
                st.session_state.balance_history.append(st.session_state.account_balance + st.session_state.daily_pnl)
                st.info(f"📉 平倉 {pos['side']}，盈虧: ${pnl:.2f}")
                st.session_state.auto_position = None

    if st.session_state.auto_position:
        pos = st.session_state.auto_position
        pnl = (current_price - pos['entry']) * (1.0 if pos['side']=='long' else -1.0) * pos['leverage']
        pnl_pct = (current_price - pos['entry']) / pos['entry'] * pos['leverage'] * 100.0 * (1.0 if pos['side']=='long' else -1.0)
        liq_price = calculate_liquidation_price(pos['entry'], "多單" if pos['side']=='long' else "空單", pos['leverage'])
        distance = abs(current_price - liq_price) / current_price * 100.0
        color_class = "profit" if pnl >= 0 else "loss"
        st.markdown(f"""
        <div class="metric">
            <h4>自動模擬持倉</h4>
            <p>方向: {'多' if pos['side']=='long' else '空'} | 槓桿: {pos['leverage']:.1f}x</p>
            <p>開倉: ${pos['entry']:.2f} ({pos['time'].strftime('%H:%M')})</p>
            <p class="{color_class}">盈虧: ${pnl:.2f} ({pnl_pct:.2f}%)</p>
            <p>強平價: <span class="warning">${liq_price:.2f}</span> (距 {distance:.1f}%)</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("手動平倉", key="auto_close"):
            if pos['side'] == 'long':
                pnl = (current_price - pos['entry']) * pos['leverage']
            else:
                pnl = (pos['entry'] - current_price) * pos['leverage']
            pnl_pct = pnl / pos['entry'] * 100.0
            protection.update(pnl, st.session_state.account_balance + st.session_state.daily_pnl,
                              market_mode, now, st.session_state.daily_pnl)
            st.session_state.trade_log.append({
                '開倉時間': pos['time'].strftime('%H:%M'),
                '方向': pos['side'],
                '開倉價': f"{pos['entry']:.2f}",
                '平倉時間': datetime.now().strftime('%H:%M'),
                '平倉價': f"{current_price:.2f}",
                '盈虧': f"{pnl:.2f}",
                '盈虧%': f"{pnl_pct:.1f}%"
            })
            st.session_state.balance_history.append(st.session_state.account_balance + st.session_state.daily_pnl)
            st.success(f"平倉，盈虧: ${pnl:.2f}")
            st.session_state.auto_position = None
            st.rerun()
    else:
        if auto_enabled:
            if can_trade:
                st.info("等待信號開倉")
            else:
                st.warning("交易暫停中")

    # 交易日誌
    with st.expander("📋 交易日誌"):
        if st.session_state.trade_log:
            st.dataframe(pd.DataFrame(st.session_state.trade_log), use_container_width=True)
        else:
            st.info("暫無交易記錄")

    # 歷史信號
    if entry_signal != 0:
        current_dir = "多" if entry_signal == 1 else "空"
        if not st.session_state.signal_history or st.session_state.signal_history[-1]['方向'] != current_dir:
            st.session_state.signal_history.append({
                '時間': datetime.now().strftime("%H:%M"),
                '方向': current_dir,
                '市場': market_mode,
                '五層總分': five_total
            })
            st.session_state.signal_history = st.session_state.signal_history[-20:]

    with st.expander("📋 歷史信號記錄"):
        if st.session_state.signal_history:
            st.dataframe(pd.DataFrame(st.session_state.signal_history), use_container_width=True)
        else:
            st.info("暫無歷史信號")
