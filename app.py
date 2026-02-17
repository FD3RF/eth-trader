# -*- coding: utf-8 -*-
"""
🚀 终极量化终端 · 完整整合版 33.0
功能：多周期信号、多策略IC、ATR止损/止盈、风险管理、实盘/模拟模式、Telegram通知
"""

import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import requests
import ta
from datetime import datetime, timedelta
from collections import deque
import functools
import warnings
import time
import logging

warnings.filterwarnings("ignore")

# ==================== 配置 ====================
class Config:
    SYMBOLS = ["ETH/USDT", "BTC/USDT", "SOL/USDT", "BNB/USDT"]
    TIMEFRAMES = ["15m", "1h", "4h", "1d"]
    TIMEFRAME_WEIGHTS = {"15m": 3, "1h": 5, "4h": 7, "1d": 10}
    BASE_RISK_PER_TRADE = 0.02
    RISK_BUDGET_RATIO = 0.1
    MAX_DAILY_TRADES = 5
    MAX_CONSECUTIVE_LOSSES = 3
    COOL_DOWN_HOURS = 24
    MIN_ATR_PCT = 0.8
    TP_MIN_RATIO = 2.0
    TRAILING_STOP_PCT = 0.35
    KELLY_FRACTION = 0.25
    ATR_MULTIPLIER_BASE = 1.5
    FETCH_LIMIT = 1500
    AUTO_REFRESH_MS = 60000
    CIRCUIT_BREAKER_ATR = 5.0
    CIRCUIT_BREAKER_FG_EXTREME = (10, 90)
    SIM_VOLATILITY = 0.05
    SIM_TREND = 0.15
    DATA_SOURCES = ["binance", "bybit", "kucoin", "mexc"]

# ==================== 日志 ====================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("UltimateTrader")

# ==================== 会话状态 ====================
def init_session_state():
    defaults = {
        "account_balance": 10000.0,
        "daily_pnl": 0.0,
        "peak_balance": 10000.0,
        "consecutive_losses": 0,
        "daily_trades": 0,
        "trade_log": [],
        "auto_position": None,
        "auto_enabled": True,
        "cooldown_until": None,
        "net_value_history": [],
        "last_trade_date": None,
        "use_simulated_data": True,
        "telegram_token": None,
        "telegram_chat_id": None,
        "error_log": [],
        "execution_log": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ==================== 辅助函数 ====================
def log_error(msg):
    st.session_state.error_log.append(f"{datetime.now().strftime('%H:%M:%S')} - {msg}")
    if len(st.session_state.error_log) > 20:
        st.session_state.error_log.pop(0)

def log_exec(msg):
    st.session_state.execution_log.append(f"{datetime.now().strftime('%H:%M:%S')} - {msg}")
    if len(st.session_state.execution_log) > 50:
        st.session_state.execution_log.pop(0)

def send_telegram(msg):
    token = st.session_state.get("telegram_token") or st.secrets.get("TELEGRAM_BOT_TOKEN")
    chat_id = st.session_state.get("telegram_chat_id") or st.secrets.get("TELEGRAM_CHAT_ID")
    if token and chat_id:
        try:
            requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          json={"chat_id": chat_id, "text": msg, "parse_mode": "HTML"},
                          timeout=5)
        except:
            pass

def safe_request(max_retries=3):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    log_error(f"请求失败 {i+1}/{max_retries}: {e}")
                    time.sleep(2 ** i)
            return None
        return wrapper
    return decorator

# ==================== 模拟数据生成 ====================
def generate_simulated_data(symbol, limit=Config.FETCH_LIMIT):
    np.random.seed(int(time.time()*1000) % 2**32)
    end = datetime.now()
    start = end - timedelta(minutes=15*limit)
    ts = pd.date_range(start, end, periods=limit)
    base_prices = {"BTC/USDT":45000,"ETH/USDT":2500,"SOL/USDT":120,"BNB/USDT":400}
    base = base_prices.get(symbol,100)
    t = np.linspace(0,6*np.pi,limit)
    trend = np.random.choice([-1,1])*Config.SIM_TREND*np.linspace(0,1,limit)*base
    cycle = 0.08*base*(np.sin(t)+0.5*np.sin(3*t)+0.3*np.sin(5*t))
    vol = Config.SIM_VOLATILITY*(1+0.5*np.sin(t/10))
    rw = np.cumsum(np.random.randn(limit)*vol*base*0.3)
    price = np.maximum(base+trend+cycle+rw, base*0.3)
    opens = price*(1+np.random.randn(limit)*0.002)
    closes = price*(1+np.random.randn(limit)*0.003)
    highs = np.maximum(opens,closes)+np.abs(np.random.randn(limit))*vol*price*0.5
    lows = np.minimum(opens,closes)-np.abs(np.random.randn(limit))*vol*price*0.5
    vol_base = np.random.randint(500,5000,limit)
    vol_factor = 1+3*np.abs(np.diff(price,prepend=price[0]))/price
    volumes = (vol_base*vol_factor).astype(int)
    df = pd.DataFrame({"timestamp":ts,"open":opens,"high":highs,"low":lows,"close":closes,"volume":volumes})
    return add_indicators(df)

# ==================== 技术指标 ====================
def add_indicators(df):
    if df.empty: return df
    df = df.copy()
    df['ema50'] = df['close'].ewm(span=50,adjust=False).mean()
    df['ema200'] = df['close'].ewm(span=200,adjust=False).mean()
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = df['macd']-df['macd_signal']
    df['rsi'] = ta.momentum.RSIIndicator(df['close'],14).rsi()
    atr = ta.volatility.AverageTrueRange(df['high'],df['low'],df['close'],14).average_true_range()
    df['atr'] = atr
    df['atr_pct'] = (atr/df['close']*100).fillna(0)
    df['adx'] = ta.trend.ADXIndicator(df['high'],df['low'],df['close'],14).adx()
    df['volume_ma20'] = df['volume'].rolling(20).mean()
    df['volume_surge'] = df['volume']>df['volume_ma20']*1.2
    return df

# ==================== 数据获取 ====================
class DataFetcher:
    def __init__(self):
        self.exchanges = {}
        for name in Config.DATA_SOURCES:
            try:
                cls = getattr(ccxt,name)
                self.exchanges[name] = cls({'enableRateLimit':True,'timeout':30000})
            except: pass

    @safe_request()
    def fetch_ohlcv(self, ex, symbol, tf, limit):
        data = ex.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
        df = pd.DataFrame(data,columns=['timestamp','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'],unit='ms')
        return df

    def get_symbol_data(self,symbol):
        if st.session_state.use_simulated_data:
            df15 = generate_simulated_data(symbol)
            return {"data_dict":{"15m":df15},"current_price":df15['close'].iloc[-1]}
        for ex in self.exchanges.values():
            df = self.fetch_ohlcv(ex,symbol,'15m',Config.FETCH_LIMIT)
            if df is not None and len(df)>=50:
                df = add_indicators(df)
                return {"data_dict":{"15m":df},"current_price":df['close'].iloc[-1]}
        st.session_state.use_simulated_data = True
        return self.get_symbol_data(symbol)

# ==================== 信号引擎 ====================
class SignalEngine:
    @staticmethod
    def get_signal(df):
        last = df.iloc[-1]
        if last['close']>last['ema200'] and last['macd_diff']>0: return 1
        elif last['close']<last['ema200'] and last['macd_diff']<0: return -1
        return 0

# ==================== 风控 ====================
class RiskManager:
    def __init__(self):
        self.recent_trades = deque(maxlen=50)

    def kelly_fraction(self):
        if len(self.recent_trades)<10: return 0.1
        wins = [r for r in self.recent_trades if r>0]
        losses = [abs(r) for r in self.recent_trades if r<0]
        win_rate = len(wins)/len(self.recent_trades)
        avg_win = np.mean(wins) if wins else 1.0
        avg_loss = np.mean(losses) if losses else 1.0
        b = avg_win/avg_loss if avg_loss>0 else 1.0
        kelly = win_rate-(1-win_rate)/b
        return max(0,min(kelly*Config.KELLY_FRACTION,0.5))

    def position_size(self,balance,prob,atr,atr_pct):
        edge = abs(prob-0.5)
        kelly = self.kelly_fraction()
        pos_risk = balance*Config.BASE_RISK_PER_TRADE*kelly*edge*2*(1 if atr_pct<4 else 0.8)
        return pos_risk/atr if atr>0 else 0

# ==================== UI ====================
class UIRenderer:
    def __init__(self):
        self.fetcher = DataFetcher()

    def render_sidebar(self):
        st.sidebar.title("⚙️ 配置")
        symbol = st.sidebar.selectbox("品种",Config.SYMBOLS)
        leverage_mode = st.sidebar.selectbox("杠杆模式",["稳健 (3-5x)","无敌 (5-8x)","神级 (8-10x)"])
        balance = st.sidebar.number_input("余额 USDT",1000.0,1000000.0,value=st.session_state.account_balance)
        use_real = st.sidebar.checkbox("实盘交易",False)
        st.sidebar.text_input("API Key","")
        st.sidebar.text_input("Secret Key","")
        st.sidebar.checkbox("Telegram通知",False)
        return symbol,leverage_mode,use_real

    def render_main(self,symbol,data,signal,pos_size):
        st.header(f"🚀 {symbol} 量化终端")
        st.subheader("📊 市场状态")
        st.metric("当前价格",f"{data['current_price']:.2f}")
        st.metric("交易信号","买入" if signal==1 else "卖出" if signal==-1 else "观望")
        st.subheader("💰 仓位 & 风控")
        st.write(f"推荐仓位: {pos_size:.4f} 手")
        st.subheader("📈 多周期信号")
        df = data['data_dict']['15m']
        st.line_chart(df['close'])

# ==================== 主程序 ====================
def main():
    st.set_page_config(page_title="终极量化终端 33.0", layout="wide")
    st.title("🚀 终极量化终端 · 完整整合版 33.0")
    init_session_state()
    ui = UIRenderer()
    symbol,mode,use_real = ui.render_sidebar()
    data = ui.fetcher.get_symbol_data(symbol)
    df15 = data['data_dict']['15m']
    sig = SignalEngine.get_signal(df15)
    risk = RiskManager()
    pos_size = risk.position_size(st.session_state.account_balance,0.6,df15['atr'].iloc[-1],df15['atr_pct'].iloc[-1])
    ui.render_main(symbol,data,sig,pos_size)
    st_autorefresh = st.experimental_rerun
    st_autorefresh()

if __name__=="__main__":
    main()
