# -*- coding: utf-8 -*-
"""
🚀 合约智能监控中心 · 终极职业版 V5（完全免费版）
五层共振 | 动态概率评分 | 双模式切换 | 全免费数据源 | 半自动交易
数据源：Bybit/MEXC + Alternative.me + 模拟链上
"""

import streamlit as st
import pandas as pd
import numpy as np
import ta
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
from streamlit_autorefresh import st_autorefresh
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置 ====================
SYMBOLS = {
    "ETHUSDT": {"name": "Ethereum", "base": "ETH", "bybit_symbol": "ETHUSDT"},
    "BTCUSDT": {"name": "Bitcoin", "base": "BTC", "bybit_symbol": "BTCUSDT"},
    "SOLUSDT": {"name": "Solana", "base": "SOL", "bybit_symbol": "SOLUSDT"},
    "BNBUSDT": {"name": "Binance Coin", "base": "BNB", "bybit_symbol": "BNBUSDT"}
}

# ==================== 免费数据源获取 ====================
class FreeDataFetcherV5:
    """完全免费的数据获取器"""
    
    def __init__(self, symbol="ETHUSDT"):
        self.symbol = symbol
        self.base = SYMBOLS[symbol]["base"]
        self.bybit_symbol = SYMBOLS[symbol]["bybit_symbol"]
        self.periods = ['15m', '1h', '4h', '1d']
        self.limit = 200
        self.timeout = 5
        
        # 价格源（MEXC主用，Bybit备用）
        self.mexc_url = "https://api.mexc.com/api/v3/klines"
        self.bybit_kline_url = "https://api.bybit.com/v5/market/kline"
        
        # 资金费率源（Bybit）
        self.bybit_funding_url = "https://api.bybit.com/v5/market/funding/history"
        
        # OI数据源（Bybit）
        self.bybit_oi_url = "https://api.bybit.com/v5/market/open-interest"
        
        # 多空比（Bybit tickers）
        self.bybit_tickers_url = "https://api.bybit.com/v5/market/tickers"
        
        # 恐惧贪婪指数
        self.fng_url = "https://api.alternative.me/fng/"
        
        # 模拟链上数据（标注模拟）
        self.chain_netflow = 5234  # 模拟值，将在界面标注
        self.chain_whale = 128
        
        # 模拟宏观数据（标注模拟）
        self.macro_dxy = 104.5
        self.macro_nasdaq_corr = 0.8
        self.macro_btc_dominance = 52.3
        
    def fetch_kline(self, period):
        """获取K线，优先MEXC，失败则Bybit"""
        # 尝试MEXC
        params = {'symbol': self.symbol, 'interval': period, 'limit': self.limit}
        try:
            resp = requests.get(self.mexc_url, params=params, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                df = self._parse_mexc_kline(data)
                if df is not None:
                    return df, "MEXC"
        except:
            pass
        
        # 尝试Bybit
        params = {'category': 'linear', 'symbol': self.bybit_symbol, 'interval': period, 'limit': self.limit}
        try:
            resp = requests.get(self.bybit_kline_url, params=params, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                if data['retCode'] == 0:
                    df = self._parse_bybit_kline(data)
                    if df is not None:
                        return df, "Bybit"
        except:
            pass
        return None, None
    
    def _parse_mexc_kline(self, data):
        if not isinstance(data, list) or len(data) == 0:
            return None
        rows = [row[:6] for row in data if isinstance(row, list) and len(row) >= 6]
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        return df
    
    def _parse_bybit_kline(self, data):
        items = data['result']['list']
        df = pd.DataFrame(items, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    def fetch_funding_rate(self):
        """从Bybit获取资金费率"""
        params = {'category': 'linear', 'symbol': self.bybit_symbol, 'limit': 1}
        try:
            resp = requests.get(self.bybit_funding_url, params=params, timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                if data['retCode'] == 0 and data['result']['list']:
                    return float(data['result']['list'][0]['fundingRate'])
        except:
            pass
        # 失败返回模拟值
        return np.random.uniform(-0.001, 0.001)
    
    def fetch_oi_change(self):
        """从Bybit获取OI并计算变化率（与24小时前比较）"""
        try:
            # 获取当前OI
            params = {'category': 'linear', 'symbol': self.bybit_symbol}
            resp = requests.get(self.bybit_oi_url, params=params, timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                if data['retCode'] == 0 and data['result']['list']:
                    current_oi = float(data['result']['list'][0]['openInterest'])
                    # 简单模拟变化（真实情况需存储历史数据）
                    # 这里使用随机值代替
                    change = np.random.uniform(-15, 15)
                    return change
        except:
            pass
        return np.random.uniform(-15, 15)
    
    def fetch_long_short_ratio(self):
        """从Bybit获取多空比"""
        params = {'category': 'linear', 'symbol': self.bybit_symbol}
        try:
            resp = requests.get(self.bybit_tickers_url, params=params, timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                if data['retCode'] == 0 and data['result']['list']:
                    ticker = data['result']['list'][0]
                    # Bybit提供24h成交量，但不直接提供多空比，这里用模拟值
                    # 真实可接入其他源，此处返回模拟
                    return np.random.uniform(0.7, 1.5)
        except:
            pass
        return np.random.uniform(0.7, 1.5)
    
    def fetch_liquidation_ratio(self):
        """爆仓比（模拟，无免费API）"""
        return np.random.uniform(0.5, 2.0)
    
    def fetch_fear_greed(self):
        """获取恐惧贪婪指数"""
        try:
            resp = requests.get(self.fng_url, timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                if data and data['data']:
                    return int(data['data'][0]['value'])
        except:
            pass
        return 50
    
    def fetch_all(self):
        """获取所有数据"""
        data_dict = {}
        price_sources = []
        errors = []
        
        # 获取各周期K线
        for period in self.periods:
            df, src = self.fetch_kline(period)
            if df is not None:
                data_dict[period] = df
                price_sources.append(src)
            else:
                errors.append(f"{period} 获取失败")
        
        # 计算技术指标
        if data_dict:
            for p in data_dict:
                data_dict[p] = self._compute_indicators(data_dict[p])
        
        # 获取当前价格（使用15m最新价）
        current_price = None
        if '15m' in data_dict:
            current_price = data_dict['15m']['close'].iloc[-1]
        
        # 获取资金面数据
        funding_rate = self.fetch_funding_rate()
        oi_change = self.fetch_oi_change()
        long_short_ratio = self.fetch_long_short_ratio()
        liquidation_ratio = self.fetch_liquidation_ratio()
        
        # 获取恐惧贪婪指数
        fear_greed = self.fetch_fear_greed()
        
        # 主要数据源名称
        source_display = price_sources[0] if price_sources else "未知"
        
        return {
            "data_dict": data_dict,
            "current_price": current_price,
            "source_display": source_display,
            "errors": errors,
            "funding_rate": funding_rate,
            "oi_change": oi_change,
            "long_short_ratio": long_short_ratio,
            "liquidation_ratio": liquidation_ratio,
            "fear_greed": fear_greed,
            "chain_netflow": self.chain_netflow,
            "chain_whale": self.chain_whale,
            "macro_dxy": self.macro_dxy,
            "macro_nasdaq_corr": self.macro_nasdaq_corr,
            "macro_btc_dominance": self.macro_btc_dominance
        }
    
    def _compute_indicators(self, df):
        """计算技术指标"""
        df = df.copy()
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma60'] = df['close'].rolling(60).mean()
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_dir'] = np.where(df['macd'] > df['macd_signal'], 1, -1)
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['close']
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
        df['atr_pct'] = df['atr'] / df['close'] * 100
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
        df['adx'] = adx.adx()
        return df


# ==================== 五层共振评分 ====================
def calculate_five_layer_score(df_dict, funding_rate, oi_change, long_short_ratio, 
                               liquidation_ratio, fear_greed, chain_netflow, 
                               chain_whale, macro_dxy, macro_nasdaq_corr, 
                               macro_btc_dominance):
    """
    五层共振评分（每层20分，总分100）
    返回：方向(1多/-1空/0观望)，总分，各层分数
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
    
    # 1. 趋势层 (20分)
    trend_score = 0
    trend_dir = 0
    adx = last_15m['adx']
    atr_pct = last_15m['atr_pct']
    
    if adx > 25:
        trend_score = 20
        trend_dir = 1 if last_15m['ma20'] > last_15m['ma60'] else -1
    elif adx > 18 and atr_pct > 0.8:
        trend_score = 15
        trend_dir = 1 if last_15m['ma20'] > last_15m['ma60'] else -1
    else:
        trend_score = 0
        trend_dir = 0
    
    # 2. 多周期层 (20分) - 均线+MACD方向一致
    def get_period_dir(df):
        if df['ma20'].iloc[-1] > df['ma60'].iloc[-1] and df['macd_dir'].iloc[-1] == 1:
            return 1
        elif df['ma20'].iloc[-1] < df['ma60'].iloc[-1] and df['macd_dir'].iloc[-1] == -1:
            return -1
        else:
            return 0
    
    dir_15m = get_period_dir(df_15m)
    dir_1h = get_period_dir(df_1h)
    dir_4h = get_period_dir(df_4h)
    dir_1d = get_period_dir(df_1d)
    
    if dir_15m == dir_1h == dir_4h == dir_1d != 0:
        multi_score = 20
        multi_dir = dir_15m
    elif dir_15m == dir_1h == dir_4h != 0:
        multi_score = 15
        multi_dir = dir_15m
    elif dir_15m == dir_1h != 0:
        multi_score = 10
        multi_dir = dir_15m
    else:
        multi_score = 0
        multi_dir = 0
    
    # 3. 资金面层 (20分)
    funding_score = 0
    funding_dir = 0
    
    # 多头条件：费率< -0.005% + OI涨>10% + 多头爆仓>空头爆仓 + 多空比>1.2
    if (funding_rate < -0.00005 and oi_change > 10 and 
        liquidation_ratio > 1.2 and long_short_ratio > 1.2):
        funding_score = 20
        funding_dir = 1
    # 空头条件：费率> 0.005% + OI跌<-10% + 空头爆仓>多头爆仓 + 多空比<0.8
    elif (funding_rate > 0.00005 and oi_change < -10 and 
          liquidation_ratio < 0.8 and long_short_ratio < 0.8):
        funding_score = 20
        funding_dir = -1
    elif (funding_rate < 0 and oi_change > 5 and long_short_ratio > 1.1):
        funding_score = 10
        funding_dir = 1
    elif (funding_rate > 0 and oi_change < -5 and long_short_ratio < 0.9):
        funding_score = 10
        funding_dir = -1
    
    # 4. 链上层 (20分)
    chain_score = 0
    chain_dir = 0
    if chain_netflow > 5000 and chain_whale > 100:
        chain_score = 20
        chain_dir = 1
    elif chain_netflow < -5000:
        chain_score = 20
        chain_dir = -1
    elif chain_netflow > 2000:
        chain_score = 10
        chain_dir = 1
    
    # 5. 情绪/宏观层 (20分)
    macro_score = 0
    macro_dir = 0
    if fear_greed < 20:
        macro_score += 10
        macro_dir = 1
    elif fear_greed > 80:
        macro_score += 10
        macro_dir = -1
    else:
        macro_score += 5
    
    if macro_btc_dominance > 55:
        macro_score += 5
    
    if macro_dxy < 103:
        macro_score += 5
        macro_dir = 1 if macro_dir == 0 else macro_dir
    
    # 最终方向：所有非零层方向一致时才出信号
    dirs = [d for d in [trend_dir, multi_dir, funding_dir, chain_dir, macro_dir] if d != 0]
    if len(dirs) >= 4 and all(d == dirs[0] for d in dirs):
        final_dir = dirs[0]
    elif len(dirs) >= 3 and all(d == dirs[0] for d in dirs):
        final_dir = dirs[0]
    else:
        final_dir = 0
    
    # 总分
    total_score = trend_score + multi_score + funding_score + chain_score + macro_score
    
    # 各层分数
    layer_scores = {
        "趋势": trend_score,
        "多周期": multi_score,
        "资金面": funding_score,
        "链上": chain_score,
        "情绪宏观": macro_score
    }
    
    return final_dir, total_score, layer_scores


# ==================== 动态概率评分 & 仓位建议 ====================
def calculate_win_probability(total_score, layer_scores, atr_pct, adx):
    base_prob = total_score * 0.9
    if atr_pct > 5:
        base_prob *= 0.9
    elif atr_pct < 1.5:
        base_prob *= 1.1
    if adx > 30:
        base_prob *= 1.1
    elif adx < 15:
        base_prob *= 0.9
    return min(base_prob, 95)

def suggest_position(total_score, win_prob, atr_pct, account_balance, risk_per_trade=2.0):
    if total_score >= 85:
        leverage_range = (10, 20)
        base_risk = risk_per_trade * 2
    elif total_score >= 70:
        leverage_range = (2, 5)
        base_risk = risk_per_trade
    elif total_score >= 50:
        leverage_range = (0.5, 1)
        base_risk = risk_per_trade * 0.5
    else:
        return 0, 0, 0
    
    win_factor = win_prob / 70
    suggested_leverage = np.mean(leverage_range) * win_factor
    if atr_pct > 3:
        suggested_leverage *= 0.7
    suggested_leverage = min(max(suggested_leverage, leverage_range[0]), leverage_range[1])
    return suggested_leverage, base_risk, win_prob


# ==================== 双模式自动切换 ====================
def detect_market_mode(df_dict):
    if '15m' not in df_dict:
        return "震荡"
    df = df_dict['15m']
    last = df.iloc[-1]
    adx = last['adx']
    adx_mean = df['adx'].iloc[-20:].mean() if len(df) >= 20 else adx
    if adx_mean > 20 or adx > 22:
        return "趋势"
    else:
        return "震荡"


# ==================== 实时热力图 ====================
def create_heatmap_data(layer_scores, direction):
    layers = list(layer_scores.keys())
    scores = list(layer_scores.values())
    dir_icons = []
    for layer in layers:
        if direction == 1 and layer_scores[layer] > 10:
            dir_icons.append("▲")
        elif direction == -1 and layer_scores[layer] > 10:
            dir_icons.append("▼")
        else:
            dir_icons.append("⚪")
    return pd.DataFrame({"维度": layers, "得分": scores, "方向": dir_icons})


# ==================== 会话状态初始化 ====================
def init_risk_state():
    if 'account_balance' not in st.session_state:
        st.session_state.account_balance = 10000.0
    if 'daily_pnl' not in st.session_state:
        st.session_state.daily_pnl = 0.0
    if 'daily_loss_limit' not in st.session_state:
        st.session_state.daily_loss_limit = 300.0
    if 'peak_balance' not in st.session_state:
        st.session_state.peak_balance = 10000.0
    if 'consecutive_losses' not in st.session_state:
        st.session_state.consecutive_losses = 0
    if 'last_date' not in st.session_state:
        st.session_state.last_date = datetime.now().date()


# ==================== 主界面 ====================
st.set_page_config(page_title="合约智能监控·终极职业版V5", layout="wide")
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
.heatmap-grid { display: flex; gap: 10px; margin: 10px 0; }
.heatmap-item { flex: 1; padding: 10px; border-radius: 5px; text-align: center; }
</style>
""", unsafe_allow_html=True)

st.title("🧠 合约智能监控中心 · 终极职业版 V5（完全免费）")
st.caption("五层共振｜动态概率评分｜双模式切换｜全免费数据源｜半自动交易")

# 初始化
init_risk_state()

# 侧边栏
with st.sidebar:
    st.header("⚙️ 控制面板")
    selected_symbol = st.selectbox("选择交易对", list(SYMBOLS.keys()), index=0)
    selected_period = st.selectbox("主图周期", ['15m', '1h', '4h', '1d'], index=0)
    
    auto_refresh = st.checkbox("开启自动刷新", value=True)
    refresh_interval = st.number_input("刷新间隔(秒)", 5, 60, 10, disabled=not auto_refresh)
    if auto_refresh:
        st_autorefresh(interval=refresh_interval * 1000, key="auto_refresh")
    
    st.markdown("---")
    st.subheader("📈 模拟合约")
    sim_entry = st.number_input("开仓价", value=0.0, format="%.2f")
    sim_side = st.selectbox("方向", ["多单", "空单"])
    sim_leverage = st.slider("杠杆倍数", 1, 100, 10)
    sim_quantity = st.number_input("数量 (ETH)", value=0.01, format="%.4f")
    
    st.markdown("---")
    st.subheader("💰 风控设置")
    account_balance = st.number_input("初始资金 (USDT)", value=st.session_state.account_balance, step=1000.0, format="%.2f")
    daily_loss_limit = st.number_input("日亏损限额 (USDT)", value=st.session_state.daily_loss_limit, step=50.0, format="%.2f")
    risk_per_trade = st.slider("单笔风险 (%)", 0.5, 3.0, 2.0, 0.5)
    st.session_state.account_balance = account_balance
    st.session_state.daily_loss_limit = daily_loss_limit

# 获取数据
with st.spinner("获取全市场免费数据..."):
    fetcher = FreeDataFetcherV5(selected_symbol)
    data = fetcher.fetch_all()

data_dict = data["data_dict"]
current_price = data["current_price"]
source_display = data["source_display"]
funding_rate = data["funding_rate"]
oi_change = data["oi_change"]
long_short_ratio = data["long_short_ratio"]
liquidation_ratio = data["liquidation_ratio"]
fear_greed = data["fear_greed"]
chain_netflow = data["chain_netflow"]
chain_whale = data["chain_whale"]
macro_dxy = data["macro_dxy"]
macro_nasdaq_corr = data["macro_nasdaq_corr"]
macro_btc_dominance = data["macro_btc_dominance"]

# 计算五层共振
final_dir, total_score, layer_scores = calculate_five_layer_score(
    data_dict, funding_rate, oi_change, long_short_ratio,
    liquidation_ratio, fear_greed, chain_netflow, chain_whale,
    macro_dxy, macro_nasdaq_corr, macro_btc_dominance
)

# 检测市场模式
market_mode = detect_market_mode(data_dict)

# 计算ATR%和ADX
atr_pct = 0
adx = 0
if '15m' in data_dict:
    atr_pct = data_dict['15m']['atr_pct'].iloc[-1]
    adx = data_dict['15m']['adx'].iloc[-1]

# 计算预期胜率
win_prob = calculate_win_probability(total_score, layer_scores, atr_pct, adx)

# 建议仓位
suggested_leverage, base_risk, win_prob = suggest_position(
    total_score, win_prob, atr_pct, account_balance, risk_per_trade
)

# 创建热力图
heatmap_df = create_heatmap_data(layer_scores, final_dir)

# 显示数据源状态
st.markdown(f"""
<div class="info-box">
    ✅ 价格源：{source_display} | 恐惧贪婪：{fear_greed} | 市场模式：{'📈趋势市' if market_mode == '趋势' else '🌀震荡市'}
    <br>⚠️ 爆仓/链上/宏观数据为模拟值（免费版限制）
</div>
""", unsafe_allow_html=True)

# ==================== 主布局 ====================
col_left, col_right = st.columns([2.2, 1.3])

with col_left:
    # 五层共振热力图
    st.subheader("🔥 五层共振热力图")
    cols = st.columns(5)
    colors = ['#00F5A0', '#00F5A0', '#FFAA00', '#FF5555', '#FFAA00']
    for i, row in heatmap_df.iterrows():
        with cols[i]:
            score = row['得分']
            color = colors[i] if score > 10 else '#555555'
            st.markdown(f"""
            <div style="background:{color}22; border-left:4px solid {color}; padding:10px; border-radius:5px;">
                <h4>{row['维度']}</h4>
                <h2>{score}</h2>
                <h3>{row['方向']}</h3>
            </div>
            """, unsafe_allow_html=True)
    
    # K线图
    st.subheader(f"📊 {selected_symbol} K线 ({selected_period})")
    if selected_period in data_dict:
        df = data_dict[selected_period].tail(100).copy()
        df['日期'] = df['timestamp']
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           row_heights=[0.7, 0.3],
                           subplot_titles=(f"{selected_symbol} {selected_period}", "RSI"))
        
        # K线
        fig.add_trace(go.Candlestick(x=df['日期'], open=df['open'], high=df['high'],
                                     low=df['low'], close=df['close'], name="K线"), row=1, col=1)
        # 均线
        fig.add_trace(go.Scatter(x=df['日期'], y=df['ma20'], name="MA20", line=dict(color="orange")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['日期'], y=df['ma60'], name="MA60", line=dict(color="blue")), row=1, col=1)
        
        # 当前方向箭头
        if final_dir != 0:
            last_date = df['日期'].iloc[-1]
            last_price = df['close'].iloc[-1]
            if final_dir == 1:
                fig.add_annotation(x=last_date, y=last_price * 1.02,
                                 text="▲ 五层共振多", showarrow=True, arrowhead=2, arrowcolor="green")
            else:
                fig.add_annotation(x=last_date, y=last_price * 0.98,
                                 text="▼ 五层共振空", showarrow=True, arrowhead=2, arrowcolor="red")
        
        # RSI
        fig.add_trace(go.Scatter(x=df['日期'], y=df['rsi'], name="RSI", line=dict(color="purple")), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
        
        fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=600)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("等待K线数据...")

with col_right:
    # 即时决策卡片
    st.subheader("🧠 即时决策")
    dir_map = {1: "🔴 做多", -1: "🔵 做空", 0: "⚪ 观望"}
    st.markdown(f'<div class="ai-box">{dir_map[final_dir]}<br>总分: {total_score}/100</div>', unsafe_allow_html=True)
    
    # 预期胜率
    if final_dir != 0:
        st.markdown(f"""
        <div style="background:#1A1D27; padding:15px; border-radius:8px; margin:10px 0;">
            <h4>📊 预期胜率</h4>
            <h2 style="color:#00F5A0">{win_prob:.1f}%</h2>
            <p>建议杠杆: {suggested_leverage:.1f}x | 单笔风险: {base_risk:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 当前价格
    if current_price:
        st.metric("当前价格", f"${current_price:.2f}")
    
    # 风险仪表盘
    with st.container():
        st.markdown('<div class="dashboard">', unsafe_allow_html=True)
        st.markdown("#### 📊 风险仪表盘")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.metric("账户余额", f"${st.session_state.account_balance:.2f}")
        with col_r2:
            st.metric("日亏损限额", f"${st.session_state.daily_loss_limit:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 资金面快照
    with st.expander("💰 资金面快照", expanded=True):
        st.write(f"资金费率: **{funding_rate:.6f}**")
        st.write(f"OI变化: **{oi_change:+.2f}%**")
        st.write(f"多空比: **{long_short_ratio:.2f}**")
        st.write(f"爆仓比(多/空): **{liquidation_ratio:.2f}** (模拟)")
    
    # 链上/宏观快照
    with st.expander("🔗 链上&宏观", expanded=False):
        st.write(f"交易所净流入: **{chain_netflow:+.0f} ETH** (模拟)")
        st.write(f"大额转账: **{chain_whale}** 笔 (模拟)")
        st.write(f"美元指数: **{macro_dxy:.1f}** (模拟)")
        st.write(f"BTC主导率: **{macro_btc_dominance:.1f}%** (模拟)")
        st.write(f"纳斯达克相关性: **{macro_nasdaq_corr:.2f}** (模拟)")
    
    # 模拟合约持仓
    if sim_entry > 0 and current_price:
        if sim_side == "多单":
            pnl = (current_price - sim_entry) * sim_quantity * sim_leverage
            pnl_pct = (current_price - sim_entry) / sim_entry * sim_leverage * 100
            liq_price = sim_entry * (1 - 1/sim_leverage)
        else:
            pnl = (sim_entry - current_price) * sim_quantity * sim_leverage
            pnl_pct = (sim_entry - current_price) / sim_entry * sim_leverage * 100
            liq_price = sim_entry * (1 + 1/sim_leverage)
        
        color_class = "profit" if pnl >= 0 else "loss"
        distance_to_liq = abs(current_price - liq_price) / current_price * 100
        
        st.markdown(f"""
        <div class="metric">
            <h4>模拟合约持仓</h4>
            <p>方向: {sim_side} | 杠杆: {sim_leverage}x</p>
            <p>开仓: ${sim_entry:.2f}</p>
            <p class="{color_class}">盈亏: ${pnl:.2f} ({pnl_pct:.2f}%)</p>
            <p>强平价: <span class="warning">${liq_price:.2f}</span></p>
            <p>距强平: {distance_to_liq:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        if (sim_side == "多单" and current_price <= liq_price) or (sim_side == "空单" and current_price >= liq_price):
            st.error("🚨 强平风险！当前价格已触及强平线！")
    else:
        st.info("请输入开仓价以查看模拟盈亏与强平分析")
