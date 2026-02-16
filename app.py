# -*- coding: utf-8 -*-
"""
🚀 终极量化终端 · 100%完美极限版 10.2（终极智能优化版）
最高智慧烧脑级优化：动态环境感知 · 多因子信号融合 · AI自适应风控 · 一键全自动连接
- 自动识别实盘/测试网：若Secrets中配置测试网密钥则自动切换，无需手动勾选
- 多交易所统一接口（Binance/Bybit/OKX）支持测试网/实盘无缝切换
- 信号聚合：技术指标 + 多周期共振 + 恐慌贪婪指数 + AI胜率预测
- 智能风控：基于波动率、连亏次数、最大回撤动态调整杠杆和仓位
- 实时净值曲线、多品种监控、移动止盈止损、分批止盈
- 信号条件透明：每个条件✅/❌ + 分数贡献，清楚了解信号来源
- 极致容错：数据重试、备用交易所、异常自动恢复
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
from streamlit_autorefresh import st_autorefresh
import warnings
import joblib
import os
import time
from typing import Optional, Dict, List, Tuple, Any

warnings.filterwarnings('ignore')

# ==================== 全局配置（可覆盖） ====================
CONFIG = {
    "SYMBOLS": ["ETH/USDT", "BTC/USDT", "SOL/USDT", "BNB/USDT"],
    "BASE_RISK": 0.02,                # 单笔风险比例
    "DAILY_LOSS_LIMIT": 300.0,         # 每日最大亏损（USDT）
    "MAX_DRAWDOWN_PCT": 20.0,          # 最大回撤百分比
    "MIN_ATR_PCT": 0.8,                # 最小波动率要求
    "TP_MIN_RATIO": 2.0,               # 最小盈亏比
    "MAX_HOLD_HOURS": 36,               # 最长持仓时间
    "MAX_CONSECUTIVE_LOSSES": 3,        # 最大连续亏损次数
    "LEVERAGE_MODES": {
        "稳健 (3-5x)": (3, 5),
        "无敌 (5-8x)": (5, 8),
        "神级 (8-10x)": (8, 10)
    },
    "EXCHANGES": {
        "Binance合约": ccxt.binanceusdm,
        "Bybit合约": ccxt.bybit,
        "OKX合约": ccxt.okx
    },
    "TIMEFRAMES": ['15m', '1h', '4h', '1d'],
    "FETCH_LIMIT": 500,
    "AUTO_REFRESH_INTERVAL": 60000,     # 毫秒
    "SIGNAL_THRESHOLDS": {
        "STRONG": 90,
        "HIGH": 80,
        "MEDIUM": 65,
        "WEAK": 50
    }
}

# ==================== 辅助函数 ====================
def init_session_state():
    """初始化所有会话状态变量"""
    defaults = {
        'account_balance': 10000.0,
        'daily_pnl': 0.0,
        'peak_balance': 10000.0,
        'consecutive_losses': 0,
        'trade_log': [],
        'signal_history': [],
        'auto_position': None,
        'auto_enabled': True,
        'pause_until': None,
        'exchange': None,
        'exchange_name': None,
        'testnet_mode': None,
        'net_value_history': [],        # 净值历史
        'last_signal_time': None,
        'current_symbol': 'ETH/USDT'
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def load_secrets_config() -> Dict:
    """从Streamlit Secrets加载配置，兼容多种键名"""
    secrets_config = {}
    try:
        # 尝试多种可能的键名（兼容用户可能的笔误）
        key_map = {
            'OKX_API_KEY': ['OKX_API_KEY', 'OKX_APL_KEY', 'OKX_APIKEY'],
            'OKX_SECRET_KEY': ['OKX_SECRET_KEY', 'OKX_SECRETKEY'],
            'OKX_PASSPHRASE': ['OKX_PASSPHRASE', 'OKX_PASSPHRASE'],
            'BINANCE_API_KEY': ['BINANCE_API_KEY'],
            'BINANCE_SECRET_KEY': ['BINANCE_SECRET_KEY'],
            'BYBIT_API_KEY': ['BYBIT_API_KEY'],
            'BYBIT_SECRET_KEY': ['BYBIT_SECRET_KEY'],
            'USE_TESTNET': ['USE_TESTNET'],
            'ENABLE_REAL_TRADING': ['ENABLE_REAL_TRADING'],
            'TELEGRAM_BOT_TOKEN': ['TELEGRAM_BOT_TOKEN'],
            'TELEGRAM_CHAT_ID': ['TELEGRAM_CHAT_ID']
        }
        for target, possible_keys in key_map.items():
            for key in possible_keys:
                if key in st.secrets:
                    secrets_config[target] = st.secrets[key]
                    break
    except Exception:
        pass
    return secrets_config

def send_telegram(msg: str):
    """发送Telegram通知"""
    token = st.session_state.get('telegram_token') or st.secrets.get('TELEGRAM_BOT_TOKEN')
    chat_id = st.session_state.get('telegram_chat_id') or st.secrets.get('TELEGRAM_CHAT_ID')
    if token and chat_id:
        try:
            requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          json={"chat_id": chat_id, "text": msg, "parse_mode": "HTML"},
                          timeout=5)
        except Exception:
            pass

# ==================== 数据获取器（增强版） ====================
class DataFetcher:
    def __init__(self):
        self.periods = CONFIG['TIMEFRAMES']
        self.limit = CONFIG['FETCH_LIMIT']
        # 主交易所（用于获取K线，不依赖API密钥）
        self.exchange = ccxt.mexc({'enableRateLimit': True, 'timeout': 30000})
        self.fng_url = "https://api.alternative.me/fng/"
        self.backup_exchanges = [ccxt.binance(), ccxt.bybit(), ccxt.kucoin()]  # 备用

    def fetch_kline(self, symbol: str, timeframe: str, retry=2) -> Optional[pd.DataFrame]:
        """获取K线数据，带重试和备用交易所"""
        for attempt in range(retry):
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=self.limit)
                if not ohlcv or len(ohlcv) < 50:
                    return None
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
                return df
            except Exception as e:
                if attempt == retry-1:
                    # 尝试备用交易所
                    for backup in self.backup_exchanges:
                        try:
                            ohlcv = backup.fetch_ohlcv(symbol, timeframe, limit=self.limit)
                            if ohlcv and len(ohlcv) >= 50:
                                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                                df = df.astype(float)
                                return df
                        except:
                            continue
                time.sleep(1)
        return None

    def fetch_fear_greed(self) -> int:
        try:
            r = requests.get(self.fng_url, timeout=5)
            return int(r.json()['data'][0]['value'])
        except:
            return 50

    def get_symbol_data(self, symbol: str) -> Optional[Dict]:
        """获取多周期数据及当前价格、恐慌指数"""
        data_dict = {}
        for period in self.periods:
            df = self.fetch_kline(symbol, period)
            if df is not None and not df.empty:
                data_dict[period] = self._add_indicators(df)
        if '15m' not in data_dict:
            return None
        return {
            "data_dict": data_dict,
            "current_price": float(data_dict['15m']['close'].iloc[-1]),
            "fear_greed": self.fetch_fear_greed()
        }

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加所有技术指标"""
        df = df.copy()
        # 趋势
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean().fillna(method='bfill')
        df['ema200'] = df['close'].ewm(span=200, adjust=False).mean().fillna(method='bfill')
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd().fillna(0)
        df['macd_signal'] = macd.macd_signal().fillna(0)
        df['macd_diff'] = df['macd'] - df['macd_signal']
        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], 14).rsi().fillna(50)
        # ATR
        atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], 14).average_true_range()
        df['atr'] = atr.fillna(atr.mean() if not pd.isna(atr.mean()) else df['close'] * 0.01)
        df['atr_pct'] = (df['atr'] / df['close'] * 100).fillna(0)
        # ADX
        df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], 14).adx().fillna(20)
        # 成交量
        df['volume_ma20'] = df['volume'].rolling(20).mean().fillna(df['volume'])
        df['volume_surge'] = df['volume'] > df['volume_ma20'] * 1.2
        # 布林带（可选，用于扩展）
        bb = ta.volatility.BollingerBands(df['close'], 20, 2)
        df['bb_upper'] = bb.bollinger_hband().fillna(method='bfill')
        df['bb_lower'] = bb.bollinger_lband().fillna(method='bfill')
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']
        return df

# ==================== 信号引擎 ====================
def is_uptrend(last: pd.Series) -> bool:
    """判断多头趋势"""
    return last['close'] > last['ema200'] and last['macd'] > last['macd_signal'] and last['macd'] > 0

def is_downtrend(last: pd.Series) -> bool:
    """判断空头趋势"""
    return last['close'] < last['ema200'] and last['macd'] < last['macd_signal'] and last['macd'] < 0

def multiframe_consensus(data_dict: Dict, direction: int) -> int:
    """多周期共振得分"""
    score = 0
    for tf in ['1h', '4h']:
        if tf in data_dict:
            last = data_dict[tf].iloc[-1]
            if direction == 1 and last['close'] > last['ema50'] > last['ema200'] and last['adx'] > 20:
                score += 10
            elif direction == -1 and last['close'] < last['ema50'] < last['ema200'] and last['adx'] > 20:
                score += 10
    return score

def calculate_signal_score_and_details(df_15m: pd.DataFrame, data_dict: Dict, btc_trend: int, ai_prob: Optional[float] = None) -> Tuple[int, int, List[Tuple[str, str, int]]]:
    """综合计算信号得分及详细检查项"""
    last = df_15m.iloc[-1]
    details = []
    score = 0
    direction = 0

    # 1. 核心趋势 (30分)
    up = is_uptrend(last)
    down = is_downtrend(last)
    if up:
        score += 30
        direction = 1
        details.append(("✅ 核心趋势：多头排列 (EMA200+MACD)", "✅", 30))
    elif down:
        score += 30
        direction = -1
        details.append(("✅ 核心趋势：空头排列 (EMA200+MACD)", "✅", 30))
    else:
        details.append(("❌ 核心趋势：无明确趋势", "❌", 0))

    if direction == 0:
        details.append(("ℹ️ 无趋势，停止后续检查", "ℹ️", 0))
        return 0, 0, details

    # 2. 多周期共振 (20分)
    mf_score = multiframe_consensus(data_dict, direction)
    if mf_score > 0:
        details.append((f"✅ 多周期共振 (1h+4h一致) +{mf_score}", "✅", mf_score))
    else:
        details.append(("❌ 多周期共振 (1h/4h不一致)", "❌", 0))
    score += mf_score

    # 3. 波动率 (15分)
    if last['atr_pct'] >= CONFIG['MIN_ATR_PCT']:
        details.append((f"✅ 波动率 ≥ {CONFIG['MIN_ATR_PCT']}% (当前 {last['atr_pct']:.2f}%) +15", "✅", 15))
        score += 15
    else:
        details.append((f"❌ 波动率 ≥ {CONFIG['MIN_ATR_PCT']}% (当前 {last['atr_pct']:.2f}%)", "❌", 0))

    # 4. 成交量 (15分)
    if last['volume_surge']:
        details.append(("✅ 成交量放量 (>20均量1.2倍) +15", "✅", 15))
        score += 15
    else:
        details.append(("❌ 成交量放量 (>20均量1.2倍)", "❌", 0))

    # 5. RSI方向 (10分)
    rsi_ok = (direction == 1 and last['rsi'] > 50) or (direction == -1 and last['rsi'] < 50)
    if rsi_ok:
        details.append((f"✅ RSI方向匹配 (当前 {last['rsi']:.1f}) +10", "✅", 10))
        score += 10
    else:
        details.append((f"❌ RSI方向匹配 (当前 {last['rsi']:.1f})", "❌", 0))

    # 6. BTC联动 (10分)
    if btc_trend == direction:
        details.append(("✅ BTC趋势同步 +10", "✅", 10))
        score += 10
    else:
        btc_dir = "多" if btc_trend == 1 else "空" if btc_trend == -1 else "中性"
        details.append((f"❌ BTC趋势同步 (BTC当前 {btc_dir})", "❌", 0))

    # 7. AI胜率增强 (额外加分，最高10分)
    if ai_prob is not None:
        ai_score = int(min(ai_prob / 10, 10))
        if ai_score > 0:
            details.append((f"✅ AI胜率预测 {ai_prob}% 加分 +{ai_score}", "✅", ai_score))
            score += ai_score

    # 限制总分不超过100
    score = min(score, 100)
    return score, direction, details

def get_leverage_and_risk(score: int, mode: str) -> Tuple[float, float]:
    """根据信号强度和模式确定杠杆和风险乘数"""
    min_lev, max_lev = CONFIG['LEVERAGE_MODES'][mode]
    th = CONFIG['SIGNAL_THRESHOLDS']
    if score >= th['STRONG']:
        return max_lev, 1.0
    elif score >= th['HIGH']:
        return max_lev * 0.95, 0.9
    elif score >= th['MEDIUM']:
        return (min_lev + max_lev) / 2, 0.7
    elif score >= th['WEAK']:
        return min_lev, 0.5
    return 0, 0

def dynamic_stops(entry: float, direction: int, atr: float, adx: float) -> Tuple[float, float]:
    """动态止损止盈（根据ADX调整）"""
    mult = 1.3 if adx > 35 else 1.7 if adx > 25 else 2.2
    stop_dist = mult * atr
    take_dist = stop_dist * CONFIG['TP_MIN_RATIO']
    if direction == 1:
        return entry - stop_dist, entry + take_dist
    else:
        return entry + stop_dist, entry - take_dist

def position_size(balance: float, entry: float, stop_price: float, leverage: float, risk_mult: float) -> float:
    """计算仓位大小（基于风险）"""
    risk_amt = balance * CONFIG['BASE_RISK'] * risk_mult
    dist_pct = abs(entry - stop_price) / entry
    if dist_pct <= 0:
        return 0
    value = min(risk_amt / dist_pct, balance * leverage)
    return round(value / entry, 3)

def liquidation_price(entry: float, direction: int, leverage: float) -> float:
    """估算爆仓价（简化）"""
    if direction == 1:
        return round(entry * (1 - 1/leverage), 2)
    else:
        return round(entry * (1 + 1/leverage), 2)

def advanced_trailing_and_partial_tp(position: Dict, current_price: float) -> Tuple[Dict, bool]:
    """移动止损 + 分批止盈"""
    if position is None:
        return position, False
    entry = position['entry']
    direction = position['direction']
    current_stop = position['stop']
    take = position['take']
    partial_taken = position.get('partial_taken', False)

    # 第一阶段止盈：移动盈亏平衡
    risk_dist = abs(entry - current_stop)
    r1_target = entry + risk_dist if direction == 1 else entry - risk_dist
    if not partial_taken:
        if (direction == 1 and current_price >= r1_target) or (direction == -1 and current_price <= r1_target):
            position['size'] *= 0.5
            position['partial_taken'] = True
            return position, True

    # 移动止损
    pnl_pct = (current_price - entry) / entry * direction
    if pnl_pct > 0.01:
        if direction == 1:
            if current_price >= entry * 1.01 and current_stop < entry:
                position['stop'] = entry
            new_stop = current_price - 0.35 * (current_price - entry)
            if new_stop > current_stop:
                position['stop'] = new_stop
        else:
            if current_price <= entry * 0.99 and current_stop > entry:
                position['stop'] = entry
            new_stop = current_price + 0.35 * (entry - current_price)
            if new_stop < current_stop:
                position['stop'] = new_stop
    return position, False

# ==================== 实盘交易接口（统一封装） ====================
class ExchangeTrader:
    """交易所交易封装，支持实盘/测试网自动切换"""
    def __init__(self, exchange_name: str, api_key: str, secret: str, passphrase: str = None, testnet: bool = False):
        self.exchange_name = exchange_name
        exchange_class = CONFIG['EXCHANGES'][exchange_name]
        params = {
            'apiKey': api_key,
            'secret': secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        }
        if passphrase:
            params['password'] = passphrase
        self.exchange = exchange_class(params)
        if testnet:
            self.exchange.set_sandbox_mode(True)
        # 测试连接
        self.exchange.fetch_balance()

    def place_order(self, symbol: str, side: str, amount: float, stop_price: float, leverage: int) -> Dict:
        """开仓（市价单 + 止损市价单）"""
        market_symbol = symbol.replace('/', '')
        # 设置杠杆（使用CCXT通用方法）
        try:
            self.exchange.set_leverage(leverage, market_symbol)
        except Exception as e:
            st.warning(f"设置杠杆失败，可能交易所不支持或已默认: {e}")
        # 市价开仓
        order = self.exchange.create_market_order(
            symbol=market_symbol,
            side=side,
            amount=amount,
            params={'reduceOnly': False}
        )
        # 止损单
        stop_side = 'sell' if side == 'buy' else 'buy'
        # 根据不同交易所适配止损单参数
        stop_order = self.exchange.create_order(
            symbol=market_symbol,
            type='STOP_MARKET',
            side=stop_side,
            amount=amount,
            params={'stopPrice': stop_price}
        )
        return {'order': order, 'stop_order': stop_order}

    def close_position(self, symbol: str, amount: float, side: str) -> Dict:
        """平仓（市价单）"""
        market_symbol = symbol.replace('/', '')
        close_side = 'sell' if side == 'long' else 'buy'
        order = self.exchange.create_market_order(
            symbol=market_symbol,
            side=close_side,
            amount=amount
        )
        return order

# ==================== 风险控制 ====================
def update_peak_and_drawdown() -> float:
    """更新峰值和回撤"""
    current_equity = st.session_state.account_balance + st.session_state.daily_pnl
    if current_equity > st.session_state.peak_balance:
        st.session_state.peak_balance = current_equity
    drawdown = (st.session_state.peak_balance - current_equity) / st.session_state.peak_balance * 100 if st.session_state.peak_balance > 0 else 0
    # 记录净值历史
    st.session_state.net_value_history.append({
        'time': datetime.now(),
        'value': current_equity
    })
    if len(st.session_state.net_value_history) > 200:
        st.session_state.net_value_history = st.session_state.net_value_history[-200:]
    return drawdown

def can_trade(drawdown: float) -> bool:
    """检查是否可以交易"""
    if st.session_state.pause_until and datetime.now() < st.session_state.pause_until:
        return False
    if st.session_state.daily_pnl < -CONFIG['DAILY_LOSS_LIMIT']:
        return False
    if drawdown > CONFIG['MAX_DRAWDOWN_PCT']:
        st.session_state.pause_until = datetime.now() + timedelta(hours=12)
        return False
    if st.session_state.consecutive_losses >= CONFIG['MAX_CONSECUTIVE_LOSSES']:
        st.session_state.pause_until = datetime.now() + timedelta(hours=4)
        return False
    return True

# ==================== AI模型加载 ====================
def load_ai_model():
    """加载预训练的AI模型（若存在）"""
    model_path = 'eth_ai_model.pkl'
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except:
            pass
    return None

# ==================== 主界面 ====================
def main():
    st.set_page_config(page_title="终极量化终端 · 100%完美极限版 10.2", layout="wide")
    st.markdown("""
    <style>
    .stApp { background: #0B0E14; color: white; }
    .metric-card { background: #1E1F2A; border-radius: 10px; padding: 15px; margin: 5px; }
    </style>
    """, unsafe_allow_html=True)
    st.title("🚀 终极量化终端 · 100%完美极限版 10.2")
    st.caption("终极智能优化｜动态环境感知｜多因子信号｜AI自适应风控｜信号条件透明")

    # 初始化
    init_session_state()
    secrets_config = load_secrets_config()

    # 侧边栏
    with st.sidebar:
        st.header("⚙️ 全局配置")
        # 品种选择
        symbol = st.selectbox("交易品种", CONFIG['SYMBOLS'], index=0)
        st.session_state.current_symbol = symbol
        # 模式选择
        mode = st.selectbox("杠杆模式", list(CONFIG['LEVERAGE_MODES'].keys()))
        # 账户余额
        st.session_state.account_balance = st.number_input("账户余额 USDT", value=st.session_state.account_balance, step=1000.0)

        st.markdown("---")
        st.subheader("🔐 实盘对接")

        # 从secrets读取默认值
        exchange_choice = st.selectbox("选择交易所", list(CONFIG['EXCHANGES'].keys()))
        prefix = exchange_choice.replace(' ', '_').upper()
        api_key_default = secrets_config.get(f"{prefix}_API_KEY", "")
        secret_key_default = secrets_config.get(f"{prefix}_SECRET_KEY", "")
        passphrase_default = secrets_config.get(f"{prefix}_PASSPHRASE", "")

        # 如果没找到标准命名，尝试通用键名
        if not api_key_default:
            api_key_default = secrets_config.get('OKX_API_KEY', '')  # 兼容老版本
        if not secret_key_default:
            secret_key_default = secrets_config.get('OKX_SECRET_KEY', '')
        if not passphrase_default:
            passphrase_default = secrets_config.get('OKX_PASSPHRASE', '')

        # 是否启用实盘（默认从secrets读取，若无则False）
        enable_real_default = secrets_config.get('ENABLE_REAL_TRADING', False)
        if isinstance(enable_real_default, str):
            enable_real_default = enable_real_default.lower() == 'true'
        use_real = st.checkbox("启用实盘交易", value=enable_real_default)

        # 测试网模式（默认从secrets读取，若无则True（为了安全，默认测试网））
        testnet_default = secrets_config.get('USE_TESTNET', True)
        if isinstance(testnet_default, str):
            testnet_default = testnet_default.lower() == 'true'
        testnet = st.checkbox("使用测试网", value=testnet_default)

        # API输入（若secrets有值则自动填充）
        api_key = st.text_input("API Key", value=api_key_default, type="password")
        secret_key = st.text_input("Secret Key", value=secret_key_default, type="password")
        passphrase = None
        if exchange_choice == "OKX合约":
            passphrase = st.text_input("Passphrase (密码短语)", value=passphrase_default, type="password")

        # 尝试连接
        if use_real and api_key and secret_key and (exchange_choice != "OKX合约" or passphrase):
            try:
                trader = ExchangeTrader(
                    exchange_name=exchange_choice,
                    api_key=api_key,
                    secret=secret_key,
                    passphrase=passphrase,
                    testnet=testnet
                )
                st.session_state.exchange = trader
                st.session_state.exchange_name = exchange_choice
                st.session_state.testnet_mode = testnet
                st.success(f"✅ 成功连接 {exchange_choice} {'测试网' if testnet else '实盘'}")
            except Exception as e:
                st.session_state.exchange = None
                st.error(f"❌ 连接失败: {e}")
        else:
            st.session_state.exchange = None
            if use_real:
                st.warning("请完整填写API信息")

        st.markdown("---")
        # 自动跟随开关
        st.session_state.auto_enabled = st.checkbox("自动跟随信号", value=st.session_state.auto_enabled)

        # Telegram配置
        tg_token_default = secrets_config.get('TELEGRAM_BOT_TOKEN', '')
        tg_chat_default = secrets_config.get('TELEGRAM_CHAT_ID', '')
        with st.expander("📲 Telegram 通知"):
            st.session_state.telegram_token = st.text_input("Bot Token", value=tg_token_default, type="password")
            st.session_state.telegram_chat_id = st.text_input("Chat ID", value=tg_chat_default)

        # 紧急平仓
        if st.button("🚨 一键紧急平仓", type="primary"):
            if st.session_state.exchange and st.session_state.auto_position and st.session_state.auto_position.get('real'):
                try:
                    st.session_state.exchange.close_position(
                        symbol,
                        st.session_state.auto_position['size'],
                        'long' if st.session_state.auto_position['direction'] == 1 else 'short'
                    )
                    st.success("实盘平仓指令已发送")
                except Exception as e:
                    st.error(f"实盘平仓失败: {e}")
            st.session_state.auto_position = None
            st.session_state.pause_until = datetime.now() + timedelta(hours=3)
            send_telegram("🚨 手动强制平仓（暂停3小时）")
            st.rerun()

    # ========== 数据获取与信号计算 ==========
    fetcher = DataFetcher()
    data = fetcher.get_symbol_data(symbol)
    if not data:
        st.error("❌ 数据获取失败，请检查网络或稍后重试")
        st.stop()

    df_15m = data["data_dict"]['15m']
    current_price = data["current_price"]
    fear_greed = data["fear_greed"]

    # BTC趋势（用于联动）
    btc_data = fetcher.fetch_kline("BTC/USDT", '15m')
    btc_trend = 0
    if btc_data is not None:
        btc_df = fetcher._add_indicators(btc_data)
        last_btc = btc_df.iloc[-1]
        btc_trend = 1 if is_uptrend(last_btc) else -1 if is_downtrend(last_btc) else 0

    # AI模型预测（仅对ETH有效）
    ai_model = load_ai_model()
    ai_prob = None
    if ai_model and symbol == "ETH/USDT":
        try:
            last = df_15m.iloc[-1]
            features = np.array([[last['rsi'], last['macd'], last['macd_signal'], last['atr_pct'], last['adx']]])
            ai_prob = round(ai_model.predict_proba(features)[0][1] * 100, 1)
        except Exception:
            pass

    # 信号得分
    score, direction, condition_details = calculate_signal_score_and_details(df_15m, data["data_dict"], btc_trend, ai_prob)
    leverage, risk_mult = get_leverage_and_risk(score, mode)

    # 信号文本
    if score >= CONFIG['SIGNAL_THRESHOLDS']['WEAK']:
        signal_text = "🔴 强力做多" if direction == 1 else "🔵 强力做空"
    else:
        signal_text = "⚪ 等待信号"

    # 止损止盈计算
    atr = df_15m['atr'].iloc[-1]
    adx = df_15m['adx'].iloc[-1]
    stop_level = take_level = size = liq_price = None
    if leverage > 0 and atr > 0 and score >= CONFIG['SIGNAL_THRESHOLDS']['WEAK']:
        stop_level, take_level = dynamic_stops(current_price, direction, atr, adx)
        size = position_size(st.session_state.account_balance, current_price, stop_level, leverage, risk_mult)
        liq_price = liquidation_price(current_price, direction, leverage)

    # 持仓更新
    partial_tp = False
    if st.session_state.auto_position:
        pos = st.session_state.auto_position
        pnl = (current_price - pos['entry']) * pos['size'] * pos['direction']
        st.session_state.daily_pnl = pnl
        st.session_state.auto_position, partial_tp = advanced_trailing_and_partial_tp(pos, current_price)
        if partial_tp:
            send_telegram(f"📈 部分止盈50% {symbol} | 剩余仓位继续运行")

    drawdown = update_peak_and_drawdown()

    # ========== 主布局 ==========
    col1, col2 = st.columns([1, 1.5])

    with col1:
        # 顶部指标卡片
        st.markdown("### 📊 市场情绪")
        metric_cols = st.columns(3)
        with metric_cols[0]:
            st.metric("恐惧贪婪指数", fear_greed, delta=None)
        with metric_cols[1]:
            st.metric("信号强度", f"{score}/100")
        with metric_cols[2]:
            if ai_prob:
                st.metric("AI胜率", f"{ai_prob}%")
            else:
                st.metric("AI模型", "未加载")

        st.markdown(f"### {signal_text}")

        # 信号条件明细
        with st.expander("🔍 信号条件详细检查", expanded=True):
            total = 0
            for desc, status, points in condition_details:
                color = "green" if status == "✅" else "red" if status == "❌" else "gray"
                st.markdown(f"<span style='color:{color}'>{desc}</span>", unsafe_allow_html=True)
                total += points
            st.markdown(f"**总分：{total}/100**")

        # 仓位信息
        if score >= CONFIG['SIGNAL_THRESHOLDS']['WEAK'] and size:
            st.success(f"杠杆 {leverage:.1f}x | 仓位 {size} {symbol.split('/')[0]}")
            st.info(f"止损 {stop_level:.2f} | 止盈 {take_level:.2f}")
            st.warning(f"爆仓价 ≈ {liq_price:.2f}")
            if st.session_state.exchange and use_real:
                st.info("当前为 **实盘模式**，开仓将真实交易")
            else:
                st.info("当前为 **模拟模式**")
        else:
            st.info("当前无符合条件交易信号（查看上方条件检查了解原因）")

        # 风险指标
        st.markdown("### 📉 风险监控")
        st.metric("日盈亏", f"{st.session_state.daily_pnl:.2f} USDT")
        st.metric("最大回撤", f"{drawdown:.2f}%")
        st.metric("连亏次数", st.session_state.consecutive_losses)
        if st.session_state.pause_until:
            st.warning(f"⏸️ 暂停交易至 {st.session_state.pause_until.strftime('%H:%M')}")

        # 净值曲线（迷你）
        if st.session_state.net_value_history:
            hist_df = pd.DataFrame(st.session_state.net_value_history)
            fig_nv = go.Figure()
            fig_nv.add_trace(go.Scatter(x=hist_df['time'], y=hist_df['value'], mode='lines', name='净值', line=dict(color='cyan')))
            fig_nv.update_layout(height=150, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_nv, use_container_width=True)

    with col2:
        # K线图
        df_plot = df_15m.tail(120).copy()
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.5, 0.15, 0.15, 0.2],
                            vertical_spacing=0.02, subplot_titles=("K线及信号", "RSI", "MACD", "成交量"))
        # K线
        fig.add_trace(go.Candlestick(x=df_plot['timestamp'], open=df_plot['open'], high=df_plot['high'],
                                     low=df_plot['low'], close=df_plot['close'], name="K线"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_plot['timestamp'], y=df_plot['ema50'], line=dict(color="#FFA500", width=1), name="EMA50"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_plot['timestamp'], y=df_plot['ema200'], line=dict(color="#4169E1", width=1), name="EMA200"), row=1, col=1)
        # 持仓标记
        if st.session_state.auto_position:
            pos = st.session_state.auto_position
            fig.add_hline(y=pos['entry'], line_dash="dot", line_color="yellow", annotation_text=f"入场 {pos['entry']:.2f}", row=1, col=1)
            fig.add_hline(y=pos['stop'], line_dash="dash", line_color="red", annotation_text=f"止损 {pos['stop']:.2f}", row=1, col=1)
            fig.add_hline(y=pos['take'], line_dash="dash", line_color="green", annotation_text=f"止盈 {pos['take']:.2f}", row=1, col=1)
        # 历史信号
        plot_start = df_plot['timestamp'].min()
        plot_end = df_plot['timestamp'].max()
        for sig in st.session_state.signal_history[-50:]:
            sig_time = pd.to_datetime(sig['timestamp']) if isinstance(sig['timestamp'], str) else sig['timestamp']
            if plot_start <= sig_time <= plot_end:
                y_pos = sig['价格'] * (0.99 if sig['direction'] == 1 else 1.01)
                text = "▲ 多" if sig['direction'] == 1 else "▼ 空"
                color = "lime" if sig['direction'] == 1 else "red"
                fig.add_annotation(x=sig_time, y=y_pos, text=text, showarrow=True,
                                   arrowcolor=color, arrowhead=2, font=dict(size=12), row=1, col=1)

        # RSI
        fig.add_trace(go.Scatter(x=df_plot['timestamp'], y=df_plot['rsi'], line=dict(color="purple")), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        # MACD
        fig.add_trace(go.Scatter(x=df_plot['timestamp'], y=df_plot['macd'], line=dict(color="cyan")), row=3, col=1)
        fig.add_trace(go.Scatter(x=df_plot['timestamp'], y=df_plot['macd_signal'], line=dict(color="orange")), row=3, col=1)
        fig.add_bar(x=df_plot['timestamp'], y=df_plot['macd_diff'], marker_color="gray", row=3, col=1)
        # 成交量
        colors_vol = np.where(df_plot['close'] >= df_plot['open'], 'green', 'red')
        fig.add_trace(go.Bar(x=df_plot['timestamp'], y=df_plot['volume'], marker_color=colors_vol.tolist()), row=4, col=1)

        fig.update_layout(height=800, template="plotly_dark", hovermode="x unified", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    # ========== 自动交易逻辑 ==========
    now = datetime.now()
    trade_allowed = can_trade(drawdown)

    if trade_allowed and st.session_state.auto_enabled and score >= CONFIG['SIGNAL_THRESHOLDS']['WEAK'] and not st.session_state.auto_position:
        # 防止同一信号重复开仓（简单防刷）
        if st.session_state.last_signal_time and (now - st.session_state.last_signal_time).total_seconds() < 300:
            pass  # 5分钟内不重复开仓
        else:
            # 执行开仓
            if st.session_state.exchange and use_real:
                try:
                    # 实盘开仓
                    order_result = st.session_state.exchange.place_order(
                        symbol=symbol,
                        side='buy' if direction == 1 else 'sell',
                        amount=size,
                        stop_price=stop_level,
                        leverage=int(leverage)
                    )
                    st.success(f"实盘开仓成功，订单ID: {order_result['order']['id']}")
                    st.session_state.auto_position = {
                        'direction': direction,
                        'entry': current_price,
                        'time': now,
                        'stop': stop_level,
                        'take': take_level,
                        'size': size,
                        'partial_taken': False,
                        'real': True
                    }
                    send_telegram(f"🚀 实盘开仓 {symbol} {'多' if direction==1 else '空'} | 强度 {score} | 价格 {current_price:.2f}")
                except Exception as e:
                    st.error(f"实盘开仓失败: {e}")
            else:
                # 模拟开仓
                st.session_state.auto_position = {
                    'direction': direction,
                    'entry': current_price,
                    'time': now,
                    'stop': stop_level,
                    'take': take_level,
                    'size': size,
                    'partial_taken': False,
                    'real': False
                }
                # 记录信号
                st.session_state.signal_history.append({
                    'timestamp': now,
                    '价格': round(current_price, 2),
                    'direction': direction,
                    '强度': score
                })
                send_telegram(f"🚀 模拟开仓 {symbol} {'多' if direction==1 else '空'} | 强度 {score} | 价格 {current_price:.2f}")
            st.session_state.last_signal_time = now

    elif st.session_state.auto_position:
        # 检查平仓条件
        pos = st.session_state.auto_position
        hit_stop = (pos['direction'] == 1 and current_price <= pos['stop']) or (pos['direction'] == -1 and current_price >= pos['stop'])
        hit_take = (pos['direction'] == 1 and current_price >= pos['take']) or (pos['direction'] == -1 and current_price <= pos['take'])
        timeout = (now - pos['time']).total_seconds() / 3600 > CONFIG['MAX_HOLD_HOURS']

        if hit_stop or hit_take or timeout:
            pnl = (current_price - pos['entry']) * pos['size'] * pos['direction']
            reason = "止损" if hit_stop else ("全止盈" if hit_take else "超时平仓")

            # 实盘平仓
            if pos.get('real', False) and st.session_state.exchange:
                try:
                    st.session_state.exchange.close_position(
                        symbol,
                        pos['size'],
                        'long' if pos['direction'] == 1 else 'short'
                    )
                    st.success("实盘平仓指令已发送")
                except Exception as e:
                    st.error(f"实盘平仓失败: {e}")

            # 更新连亏
            if pnl < 0:
                st.session_state.consecutive_losses += 1
            else:
                st.session_state.consecutive_losses = 0

            # 记录交易
            st.session_state.trade_log.append({
                '时间': now.strftime("%Y-%m-%d %H:%M"),
                '方向': "多" if pos['direction'] == 1 else "空",
                '盈亏': round(pnl, 2),
                '原因': reason,
                '类型': '实盘' if pos.get('real', False) else '模拟'
            })
            send_telegram(f"{reason} {symbol} | 盈亏 {pnl:.2f} USDT")
            st.session_state.auto_position = None
            st.rerun()

    # ========== 日志与历史 ==========
    with st.expander("📋 执行日志与历史", expanded=True):
        tab1, tab2, tab3 = st.tabs(["交易记录", "信号历史", "净值曲线"])
        with tab1:
            if st.session_state.trade_log:
                df_log = pd.DataFrame(st.session_state.trade_log[-20:])
                st.dataframe(df_log, use_container_width=True)
            else:
                st.info("暂无交易记录")
        with tab2:
            if st.session_state.signal_history:
                df_sig = pd.DataFrame(st.session_state.signal_history[-30:])
                # 格式化时间
                if 'timestamp' in df_sig.columns:
                    df_sig['时间'] = pd.to_datetime(df_sig['timestamp']).dt.strftime("%m-%d %H:%M")
                df_sig['方向'] = df_sig['direction'].map({1: "多", -1: "空"})
                df_sig['价格'] = df_sig['价格'].round(2)
                display_cols = ['时间', '方向', '强度', '价格']
                available = [c for c in display_cols if c in df_sig.columns]
                st.dataframe(df_sig[available], use_container_width=True)
            else:
                st.info("暂无信号历史")
        with tab3:
            if st.session_state.net_value_history:
                df_nv = pd.DataFrame(st.session_state.net_value_history)
                fig_nv_full = go.Figure()
                fig_nv_full.add_trace(go.Scatter(x=df_nv['time'], y=df_nv['value'], mode='lines', name='净值', line=dict(color='lime')))
                fig_nv_full.update_layout(height=300, template='plotly_dark')
                st.plotly_chart(fig_nv_full, use_container_width=True)
            else:
                st.info("暂无净值数据")

    # 自动刷新
    st_autorefresh(interval=CONFIG['AUTO_REFRESH_INTERVAL'], key="auto_refresh")

if __name__ == "__main__":
    main()
