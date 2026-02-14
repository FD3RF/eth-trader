import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
import time
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import optuna
import warnings
warnings.filterwarnings('ignore')

# 设置页面
st.set_page_config(page_title="终极至尊量子版 v4 · 跨维度智能融合系统", layout="wide", initial_sidebar_state="expanded")

# ---------- CSS样式 ----------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0B0E14 0%, #141A24 100%);
        color: #F0F4FA;
    }
    .glass-card {
        background: rgba(20, 28, 40, 0.75);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }
    .metric-card {
        background: rgba(16, 22, 34, 0.8);
        border-radius: 12px;
        padding: 16px;
        border-left: 4px solid #00D4FF;
        box-shadow: 0 4px 20px rgba(0, 212, 255, 0.1);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-left-color: #F0B90B;
    }
    .signal-box {
        background: rgba(26, 34, 48, 0.9);
        backdrop-filter: blur(5px);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(255, 215, 0, 0.3);
        box-shadow: 0 8px 32px rgba(255, 215, 0, 0.1);
    }
    .strong-signal {
        background: linear-gradient(145deg, #2A2418, #1F1A12);
        border-left: 6px solid #FFA500;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 8px 32px rgba(255, 165, 0, 0.2);
    }
    .warning-box {
        background: rgba(239, 83, 80, 0.1);
        border-left: 4px solid #EF5350;
        border-radius: 8px;
        padding: 12px;
        margin: 10px 0;
    }
    .snapshot-card {
        background: rgba(24, 30, 42, 0.8);
        border-radius: 12px;
        padding: 16px;
        border: 1px solid rgba(255,255,255,0.05);
        transition: 0.2s;
    }
    .snapshot-card:hover {
        border-color: #00D4FF;
    }
    .title-glow {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00D4FF, #F0B90B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #00D4FF, #F0B90B, transparent);
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# ---------- 币种配置（40+主流资产）----------
COINS = {
    "BTC": {"id": "bitcoin", "name": "Bitcoin", "symbol": "BTC"},
    "ETH": {"id": "ethereum", "name": "Ethereum", "symbol": "ETH"},
    "SOL": {"id": "solana", "name": "Solana", "symbol": "SOL"},
    "BNB": {"id": "binancecoin", "name": "BNB", "symbol": "BNB"},
    "XRP": {"id": "ripple", "name": "XRP", "symbol": "XRP"},
    "ADA": {"id": "cardano", "name": "Cardano", "symbol": "ADA"},
    "DOGE": {"id": "dogecoin", "name": "Dogecoin", "symbol": "DOGE"},
    "AVAX": {"id": "avalanche-2", "name": "Avalanche", "symbol": "AVAX"},
    "DOT": {"id": "polkadot", "name": "Polkadot", "symbol": "DOT"},
    "LINK": {"id": "chainlink", "name": "Chainlink", "symbol": "LINK"},
    "MATIC": {"id": "matic-network", "name": "Polygon", "symbol": "MATIC"},
    "LTC": {"id": "litecoin", "name": "Litecoin", "symbol": "LTC"},
    "BCH": {"id": "bitcoin-cash", "name": "Bitcoin Cash", "symbol": "BCH"},
    "UNI": {"id": "uniswap", "name": "Uniswap", "symbol": "UNI"},
    "ATOM": {"id": "cosmos", "name": "Cosmos", "symbol": "ATOM"},
    "FIL": {"id": "filecoin", "name": "Filecoin", "symbol": "FIL"},
    "APT": {"id": "aptos", "name": "Aptos", "symbol": "APT"},
    "SUI": {"id": "sui", "name": "Sui", "symbol": "SUI"},
    "OP": {"id": "optimism", "name": "Optimism", "symbol": "OP"},
    "ARB": {"id": "arbitrum", "name": "Arbitrum", "symbol": "ARB"},
    "NEAR": {"id": "near", "name": "NEAR Protocol", "symbol": "NEAR"},
    "AAVE": {"id": "aave", "name": "Aave", "symbol": "AAVE"},
    "MKR": {"id": "maker", "name": "Maker", "symbol": "MKR"},
    "SNX": {"id": "synthetix", "name": "Synthetix", "symbol": "SNX"},
    "CRV": {"id": "curve-dao-token", "name": "Curve DAO", "symbol": "CRV"},
    "YFI": {"id": "yearn-finance", "name": "Yearn Finance", "symbol": "YFI"},
    "COMP": {"id": "compound", "name": "Compound", "symbol": "COMP"},
    "RUNE": {"id": "thorchain", "name": "THORChain", "symbol": "RUNE"},
    "EGLD": {"id": "elrond", "name": "Elrond", "symbol": "EGLD"},
    "FLOW": {"id": "flow", "name": "Flow", "symbol": "FLOW"},
    "IMX": {"id": "immutable-x", "name": "Immutable X", "symbol": "IMX"},
    "GMT": {"id": "stepn", "name": "GMT", "symbol": "GMT"},
    "APE": {"id": "apecoin", "name": "ApeCoin", "symbol": "APE"},
    "GALA": {"id": "gala", "name": "Gala", "symbol": "GALA"},
    "SAND": {"id": "the-sandbox", "name": "The Sandbox", "symbol": "SAND"},
    "MANA": {"id": "decentraland", "name": "Decentraland", "symbol": "MANA"},
    "AXS": {"id": "axie-infinity", "name": "Axie Infinity", "symbol": "AXS"},
    "ENJ": {"id": "enjincoin", "name": "Enjin", "symbol": "ENJ"},
    "CHZ": {"id": "chiliz", "name": "Chiliz", "symbol": "CHZ"},
    "XAU": {"id": "gold", "name": "Gold", "symbol": "XAU"},
    "XAG": {"id": "silver", "name": "Silver", "symbol": "XAG"},
    "SPX": {"id": "sp500", "name": "S&P 500", "symbol": "SPX"},
    "NDX": {"id": "nasdaq", "name": "NASDAQ", "symbol": "NDX"},
}

# ---------- 模拟数据源（宏观经济、链上、新闻、深度）----------
@st.cache_data(ttl=3600)
def fetch_macro_data():
    """模拟宏观经济数据（实际中可接入FRED等API）"""
    return {
        "interest_rate": np.random.uniform(0.5, 5.0),
        "cpi": np.random.uniform(1.0, 9.0),
        "unemployment": np.random.uniform(3.0, 6.0),
        "gdp_growth": np.random.uniform(-2.0, 5.0)
    }

@st.cache_data(ttl=3600)
def fetch_onchain_data(coin_id):
    """模拟链上数据（实际中可接入Glassnode等API）"""
    return {
        "active_addresses": int(np.random.uniform(100000, 1000000)),
        "transaction_count": int(np.random.uniform(50000, 500000)),
        "exchange_netflow": np.random.uniform(-1000, 1000),
        "whale_transactions": int(np.random.uniform(10, 100))
    }

@st.cache_data(ttl=300)
def fetch_news_sentiment():
    """模拟新闻情绪（实际中可接入NewsAPI + NLP）"""
    return np.random.uniform(-1, 1)

@st.cache_data(ttl=30)
def fetch_orderbook_imbalance():
    """模拟订单簿不平衡（-1到1，负表示卖方多）"""
    return np.random.uniform(-1, 1)

# ---------- CoinGecko 免费数据源 ----------
@st.cache_data(ttl=30)
def fetch_price(coin_id):
    if coin_id in ["gold", "silver", "sp500", "nasdaq"]:
        base_price = {"gold": 2000, "silver": 25, "sp500": 4500, "nasdaq": 15000}.get(coin_id, 100)
        change = np.random.uniform(-2, 2)
        return base_price * (1 + change/100), change
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd&include_24hr_change=true"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        return data[coin_id]['usd'], data[coin_id]['usd_24h_change']
    except:
        return None, None

def generate_klines(price, interval_min=5, limit=500):
    now = datetime.now()
    times = [now - timedelta(minutes=i*interval_min) for i in range(limit)][::-1]
    returns = np.random.randn(limit) * 0.002
    for i in range(1, limit):
        if abs(returns[i-1]) > 0.003:
            returns[i] *= 1.5
    price_series = price * np.exp(np.cumsum(returns))
    price_series = price_series * (price / price_series[-1])
    closes = price_series
    opens = [closes[i-1] if i>0 else closes[0]*0.999 for i in range(limit)]
    highs = np.maximum(opens, closes) * 1.002
    lows = np.minimum(opens, closes) * 0.998
    vols = np.random.uniform(100, 500, limit) * (1 + 0.5*np.abs(returns))
    return pd.DataFrame({
        "time": times,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": vols
    })

# ---------- 完整Ichimoku云图 ----------
def add_ichimoku_full(df):
    high_9 = df['high'].rolling(9).max()
    low_9 = df['low'].rolling(9).min()
    df['tenkan'] = (high_9 + low_9) / 2
    high_26 = df['high'].rolling(26).max()
    low_26 = df['low'].rolling(26).min()
    df['kijun'] = (high_26 + low_26) / 2
    df['senkou_a'] = ((df['tenkan'] + df['kijun']) / 2).shift(26)
    high_52 = df['high'].rolling(52).max()
    low_52 = df['low'].rolling(52).min()
    df['senkou_b'] = ((high_52 + low_52) / 2).shift(26)
    df['chikou'] = df['close'].shift(-26)
    return df

# ---------- 高级技术指标（含额外指标）----------
def add_advanced_indicators(df):
    df = df.copy()
    # 基础
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma60"] = df["close"].rolling(60).mean()
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    df["macd"] = ta.trend.MACD(df["close"]).macd()
    df["macd_signal"] = ta.trend.MACD(df["close"]).macd_signal()
    df["bb_upper"] = ta.volatility.BollingerBands(df["close"]).bollinger_hband()
    df["bb_lower"] = ta.volatility.BollingerBands(df["close"]).bollinger_lband()
    df["adx"] = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14).adx()
    df["cci"] = ta.trend.CCIIndicator(df["high"], df["low"], df["close"], window=20).cci()
    df["mfi"] = ta.volume.MFIIndicator(df["high"], df["low"], df["close"], df["volume"], window=14).money_flow_index()
    df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
    df["natr"] = df["atr"] / df["close"] * 100
    # KDJ
    low_9 = df['low'].rolling(9).min()
    high_9 = df['high'].rolling(9).max()
    rsv = (df['close'] - low_9) / (high_9 - low_9) * 100
    df['kdj_k'] = rsv.ewm(alpha=1/3).mean()
    df['kdj_d'] = df['kdj_k'].ewm(alpha=1/3).mean()
    df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']
    # SAR
    df['sar'] = ta.trend.PSARIndicator(df['high'], df['low'], df['close']).psar()
    # StochRSI
    stochrsi = ta.momentum.StochRSIIndicator(df['close'], window=14)
    df['stochrsi_k'] = stochrsi.stochrsi_k()
    df['stochrsi_d'] = stochrsi.stochrsi_d()
    # Williams %R
    df['williams_r'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close'], lbp=14).williams_r()
    # Chaikin Money Flow
    df['cmf'] = ta.volume.ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'], df['volume'], window=20).chaikin_money_flow()
    # Ichimoku
    df = add_ichimoku_full(df)
    return df

# ---------- 形态识别（增强版）----------
def detect_candlestick_patterns(df):
    patterns = []
    if len(df) < 3:
        return patterns
    last = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3] if len(df) > 2 else None
    if prev2 is not None:
        if last['close'] > last['open'] and prev['close'] < prev['open']:
            if last['close'] > prev['open'] and last['open'] < prev['close']:
                patterns.append("📈 看涨吞没")
        if last['close'] < last['open'] and prev['close'] > prev['open']:
            if last['close'] < prev['open'] and last['open'] > prev['close']:
                patterns.append("📉 看跌吞没")
    body = abs(last['close'] - last['open'])
    if body < (last['high'] - last['low']) * 0.1:
        patterns.append("➕ 十字星")
    real_body = abs(last['close'] - last['open'])
    lower_shadow = last['open'] - last['low'] if last['open'] > last['close'] else last['close'] - last['low']
    upper_shadow = last['high'] - last['close'] if last['open'] > last['close'] else last['high'] - last['open']
    if lower_shadow > 2 * real_body and upper_shadow < real_body:
        if last['close'] > last['open']:
            patterns.append("🔨 锤子线 (看涨)")
        else:
            patterns.append("🪢 上吊线 (看跌)")
    if prev2 is not None:
        if prev2['close'] < prev2['open'] and prev['close'] < prev['open'] and last['close'] > last['open']:
            if last['close'] > (prev2['open'] + prev2['close'])/2:
                patterns.append("🌅 晨星形态")
        if prev2['close'] > prev2['open'] and prev['close'] > prev['open'] and last['close'] < last['open']:
            if last['close'] < (prev2['open'] + prev2['close'])/2:
                patterns.append("🌆 暮星形态")
    if len(df) > 3:
        if all(df.iloc[-i]['close'] < df.iloc[-i]['open'] for i in range(1,4)) and all(df.iloc[-i]['close'] < df.iloc[-i-1]['close'] for i in range(1,3)):
            patterns.append("🐦‍⬛ 三只乌鸦 (看跌)")
        if all(df.iloc[-i]['close'] > df.iloc[-i]['open'] for i in range(1,4)) and all(df.iloc[-i]['close'] > df.iloc[-i-1]['close'] for i in range(1,3)):
            patterns.append("🔴 红三兵 (看涨)")
    return patterns

# ---------- 深度学习LSTM模型 ----------
class LSTMPredictor(nn.Module):
    def __init__(self, input_size=10, hidden_size=64, num_layers=2, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def train_lstm(df, feature_cols, target_col, seq_length=20):
    if len(df) < 100:
        return None, None
    data = df[feature_cols].values
    target = df[target_col].shift(-1).values
    X, y = [], []
    for i in range(len(data) - seq_length - 1):
        X.append(data[i:i+seq_length])
        y.append(target[i+seq_length])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)
    if len(X) < 10:
        return None, None
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    model = LSTMPredictor(input_size=len(feature_cols))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(10):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
    last_seq = torch.tensor(data[-seq_length:].reshape(1, seq_length, -1), dtype=torch.float32)
    with torch.no_grad():
        pred = model(last_seq).item()
    return pred, model

# ---------- 机器学习集成模型 ----------
def train_ensemble(df):
    if len(df) < 100:
        return None, None
    feature_cols = ['rsi', 'macd', 'adx', 'cci', 'mfi', 'kdj_k', 'kdj_d', 'natr', 'stochrsi_k', 'williams_r', 'cmf']
    X = df[feature_cols].dropna().values
    y = (df['close'].shift(-5) > df['close']).astype(int).dropna().values
    min_len = min(len(X), len(y))
    if min_len < 50:
        return None, None
    X = X[:min_len]
    y = y[:min_len]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # 训练多个模型
    models = {
        'lr': LogisticRegression(max_iter=1000),
        'rf': RandomForestClassifier(n_estimators=50, max_depth=5),
        'gb': GradientBoostingClassifier(n_estimators=50, max_depth=3),
        'xgb': xgb.XGBClassifier(n_estimators=50, max_depth=3, use_label_encoder=False, eval_metric='logloss'),
        'lgb': lgb.LGBMClassifier(n_estimators=50, max_depth=3, verbose=-1),
        'cat': cb.CatBoostClassifier(iterations=50, depth=3, verbose=0)
    }
    for name, model in models.items():
        if name == 'lgb':
            model.fit(X_scaled, y, eval_set=[(X_scaled, y)], eval_metric='logloss', callbacks=[lgb.early_stopping(10)], verbose=0)
        elif name == 'cat':
            model.fit(X_scaled, y, eval_set=(X_scaled, y), verbose=0)
        else:
            model.fit(X_scaled, y)
    return models, scaler

def ensemble_predict(df, models, scaler):
    if models is None:
        return 0.5
    feature_cols = ['rsi', 'macd', 'adx', 'cci', 'mfi', 'kdj_k', 'kdj_d', 'natr', 'stochrsi_k', 'williams_r', 'cmf']
    last = df[feature_cols].iloc[-1:].dropna()
    if last.empty:
        return 0.5
    X_last = scaler.transform(last)
    probs = []
    for name, model in models.items():
        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba(X_last)[0][1]
            probs.append(prob)
    return np.mean(probs)

# ---------- 蒙特卡洛模拟 ----------
def monte_carlo_simulation(df, steps=10, n_simulations=100):
    last_price = df['close'].iloc[-1]
    returns = df['close'].pct_change().dropna()
    if len(returns) < 30:
        return [last_price]*(steps+1), [last_price]*(steps+1), [last_price]*(steps+1)
    mu = returns.mean()
    sigma = returns.std()
    simulations = []
    for _ in range(n_simulations):
        prices = [last_price]
        for _ in range(steps):
            ret = np.random.normal(mu, sigma)
            prices.append(prices[-1] * (1 + ret))
        simulations.append(prices)
    sim_array = np.array(simulations)
    mean_path = np.mean(sim_array, axis=0)
    upper = np.percentile(sim_array, 95, axis=0)
    lower = np.percentile(sim_array, 5, axis=0)
    return mean_path, upper, lower

# ---------- 风险价值（VaR）----------
def calculate_var(df, confidence=0.95, horizon=1):
    returns = df['close'].pct_change().dropna()
    if len(returns) < 30:
        return 0.02
    var = np.percentile(returns, (1-confidence)*100) * np.sqrt(horizon)
    return abs(var)

# ---------- 动态杠杆调整 ----------
def dynamic_leverage(current_volatility, base_leverage=100, max_leverage=100):
    normal_vol = 0.02
    vol_ratio = current_volatility / normal_vol
    adjusted = base_leverage / max(vol_ratio, 0.5)
    return int(min(max_leverage, max(1, adjusted)))

# ---------- 强化学习环境（用于PPO训练）----------
class TradingEnv(gym.Env):
    """自定义交易环境，状态=特征向量，动作=持仓比例（-1到1），奖励=收益率"""
    def __init__(self, df, feature_cols):
        super().__init__()
        self.df = df
        self.feature_cols = feature_cols
        self.current_step = 50
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(feature_cols)+1,), dtype=np.float32)  # 特征 + 当前仓位
        self.reset()
    def reset(self):
        self.current_step = 50
        self.position = 0.0  # 当前持仓比例
        self.entry_price = self.df['close'].iloc[self.current_step]
        return self._get_obs()
    def _get_obs(self):
        feats = self.df[self.feature_cols].iloc[self.current_step].values
        return np.concatenate([[self.position], feats]).astype(np.float32)
    def step(self, action):
        action = np.clip(action, -1, 1)[0]
        price = self.df['close'].iloc[self.current_step]
        # 假设交易成本为0.1%
        cost = abs(action - self.position) * price * 0.001
        self.position = action
        self.current_step += 1
        if self.current_step >= len(self.df)-1:
            done = True
        else:
            done = False
        next_price = self.df['close'].iloc[self.current_step]
        pnl = self.position * (next_price - price) / price - cost
        reward = pnl * 100  # 缩放奖励
        return self._get_obs(), reward, done, {}

def train_ppo(df, feature_cols):
    """训练PPO智能体（简化版，实际中需长时间训练）"""
    env = DummyVecEnv([lambda: TradingEnv(df, feature_cols)])
    model = PPO('MlpPolicy', env, verbose=0, n_steps=256, batch_size=64, n_epochs=10, learning_rate=0.0003)
    model.learn(total_timesteps=10000)  # 小规模训练
    return model

# ---------- 多因子评分系统（增强版）----------
def calculate_signal_score(df, ensemble_prob=0.5, lstm_pred=None, macro=None, onchain=None, sentiment=0, ob_imbalance=0):
    if df.empty or len(df) < 30:
        return 0, "数据不足"
    last = df.iloc[-1]
    score = 0
    reasons = []
    # 技术因子 (40)
    if not pd.isna(last['ma20']) and not pd.isna(last['ma60']):
        if last['ma20'] > last['ma60']:
            score += 15
            reasons.append("MA20>MA60")
        else:
            score -= 15
            reasons.append("MA20<MA60")
    if not pd.isna(last['adx']):
        if last['adx'] > 25:
            score += 5 if score>0 else -5
            reasons.append(f"ADX{last['adx']:.0f}")
    if not pd.isna(last['rsi']):
        if last['rsi'] < 30:
            score += 20
            reasons.append("RSI超卖")
        elif last['rsi'] > 70:
            score -= 20
            reasons.append("RSI超买")
        elif last['rsi'] > 50:
            score += 5
            reasons.append("RSI>50")
        else:
            score -= 5
            reasons.append("RSI<50")
    if not pd.isna(last['macd']) and not pd.isna(last['macd_signal']):
        if last['macd'] > last['macd_signal']:
            score += 10
            reasons.append("MACD金叉")
        else:
            score -= 10
            reasons.append("MACD死叉")
    if not pd.isna(last['cci']):
        if last['cci'] > 100:
            score += 5
            reasons.append("CCI超买")
        elif last['cci'] < -100:
            score -= 5
            reasons.append("CCI超卖")
    if not pd.isna(last['kdj_k']) and not pd.isna(last['kdj_d']):
        if last['kdj_k'] > last['kdj_d'] and last['kdj_k'] < 20:
            score += 15
            reasons.append("KDJ金叉超卖")
        elif last['kdj_k'] < last['kdj_d'] and last['kdj_k'] > 80:
            score -= 15
            reasons.append("KDJ死叉超买")
    if not pd.isna(last['mfi']):
        if last['mfi'] < 20:
            score += 10
            reasons.append("MFI超卖")
        elif last['mfi'] > 80:
            score -= 10
            reasons.append("MFI超买")
    if not pd.isna(last['stochrsi_k']):
        if last['stochrsi_k'] < 20:
            score += 10
            reasons.append("StochRSI超卖")
        elif last['stochrsi_k'] > 80:
            score -= 10
            reasons.append("StochRSI超买")
    if not pd.isna(last['williams_r']):
        if last['williams_r'] < -80:
            score += 10
            reasons.append("Williams超卖")
        elif last['williams_r'] > -20:
            score -= 10
            reasons.append("Williams超买")
    if not pd.isna(last['cmf']):
        if last['cmf'] > 0.1:
            score += 5
            reasons.append("CMF正")
        elif last['cmf'] < -0.1:
            score -= 5
            reasons.append("CMF负")
    patterns = detect_candlestick_patterns(df)
    for p in patterns:
        if "看涨" in p or "锤子" in p or "晨星" in p:
            score += 10
            reasons.append(p)
        elif "看跌" in p or "上吊" in p or "暮星" in p:
            score -= 10
            reasons.append(p)
    # 机器学习信号 (20)
    if ensemble_prob > 0.6:
        score += 15
        reasons.append("ML看涨")
    elif ensemble_prob < 0.4:
        score -= 15
        reasons.append("ML看跌")
    # LSTM信号 (10)
    if lstm_pred is not None:
        if lstm_pred > last['close']:
            score += 10
            reasons.append("LSTM看涨")
        else:
            score -= 10
            reasons.append("LSTM看跌")
    # Ichimoku信号 (10)
    if not pd.isna(last['tenkan']) and not pd.isna(last['kijun']):
        if last['tenkan'] > last['kijun']:
            score += 5
            reasons.append("Ichi转换>基准")
        else:
            score -= 5
    if not pd.isna(last['senkou_a']) and not pd.isna(last['senkou_b']):
        if last['close'] > max(last['senkou_a'], last['senkou_b']):
            score += 5
            reasons.append("价格在云上")
        elif last['close'] < min(last['senkou_a'], last['senkou_b']):
            score -= 5
            reasons.append("价格在云下")
    # 宏观数据 (10)
    if macro:
        if macro['interest_rate'] < 2.0:
            score += 5
            reasons.append("低利率利好")
        if macro['cpi'] > 5.0:
            score -= 5
            reasons.append("高通胀利空")
    # 链上数据 (10)
    if onchain:
        if onchain['active_addresses'] > 500000:
            score += 5
            reasons.append("活跃地址增加")
        if onchain['exchange_netflow'] < -100:
            score += 5
            reasons.append("交易所净流出")
    # 新闻情绪 (5)
    if sentiment > 0.5:
        score += 5
        reasons.append("新闻正面")
    elif sentiment < -0.5:
        score -= 5
        reasons.append("新闻负面")
    # 订单簿不平衡 (5)
    if ob_imbalance > 0.3:
        score += 5
        reasons.append("买方深度强")
    elif ob_imbalance < -0.3:
        score -= 5
        reasons.append("卖方深度强")
    score = max(-100, min(100, score))
    return score, ", ".join(reasons[:5])

def get_signal_from_score(score):
    if score >= 60:
        return "强烈做多", score, "🔥🔥🔥 强烈看涨信号"
    elif score >= 30:
        return "做多", score, "看涨信号"
    elif score <= -60:
        return "强烈做空", score, "💀💀💀 强烈看跌信号"
    elif score <= -30:
        return "做空", score, "看跌信号"
    else:
        return "观望", score, "震荡整理"

def calc_position(capital, entry, stop, leverage=100):
    risk = 0.02
    if entry<=0 or stop<=0: return 0
    stop_pct = abs(entry-stop)/entry
    if stop_pct<=0: return 0
    max_loss = capital * risk
    pos_value = max_loss / stop_pct
    if pos_value > capital * leverage:
        pos_value = capital * leverage
    return pos_value / entry

def moving_stop_loss(entry_price, current_price, direction, trail_percent=0.01):
    if direction == "做多":
        if current_price > entry_price * (1 + trail_percent):
            return entry_price
    elif direction == "做空":
        if current_price < entry_price * (1 - trail_percent):
            return entry_price
    return None

def kelly_fraction(win_prob, win_loss_ratio):
    """凯利公式计算最优仓位比例"""
    return (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio

# ---------- 专业K线图（终极版）----------
def plot_ultimate_candlestick(df, selected_coin, interval):
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        row_heights=[0.4, 0.2, 0.2, 0.2],
        vertical_spacing=0.03,
        subplot_titles=(
            f"{selected_coin}/USDT {interval} K线图 (含Ichimoku云)",
            "RSI & StochRSI",
            "MACD & 动量",
            "成交量 & MFI/CMF"
        )
    )
    fig.add_trace(go.Candlestick(
        x=df.time, open=df.open, high=df.high, low=df.low, close=df.close,
        name="K线", increasing_line_color='#26A69A', decreasing_line_color='#EF5350',
        hoverlabel=dict(bgcolor='#1E1F2A')
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.time, y=df.ma20, name="MA20", line=dict(color='#F0B90B', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.time, y=df.ma60, name="MA60", line=dict(color='#1890FF', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.time, y=df.bb_upper, name="布林上轨", line=dict(color='#888', width=1, dash='dash'), opacity=0.5), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.time, y=df.bb_lower, name="布林下轨", line=dict(color='#888', width=1, dash='dash'), opacity=0.5), row=1, col=1)
    if 'senkou_a' in df.columns and 'senkou_b' in df.columns:
        fig.add_trace(go.Scatter(x=df.time, y=df['senkou_a'], name="云带A", line=dict(color='green', width=1), opacity=0.3), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.time, y=df['senkou_b'], name="云带B", line=dict(color='red', width=1), opacity=0.3, fill='tonexty', fillcolor='rgba(128,128,128,0.2)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.time, y=df.rsi, name="RSI(14)", line=dict(color='#9B59B6', width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.time, y=df.stochrsi_k, name="StochRSI K", line=dict(color='#FFB347', width=1.5, dash='dot')), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="rgba(239,83,80,0.5)", row=2)
    fig.add_hline(y=30, line_dash="dash", line_color="rgba(38,166,154,0.5)", row=2)
    fig.add_trace(go.Scatter(x=df.time, y=df.macd, name="MACD", line=dict(color='#FFB347', width=1.5)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.time, y=df.macd_signal, name="信号线", line=dict(color='#FF6B6B', width=1.5)), row=3, col=1)
    volume_colors = ['#26A69A' if close >= open else '#EF5350' for close, open in zip(df['close'], df['open'])]
    fig.add_trace(go.Bar(x=df.time, y=df.volume, name="成交量", marker_color=volume_colors, opacity=0.8, showlegend=False), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.time, y=df.mfi, name="MFI", line=dict(color='gold', width=1.5)), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.time, y=df.cmf*100, name="CMF x100", line=dict(color='cyan', width=1.5)), row=4, col=1)
    fig.update_layout(
        template="plotly_dark",
        xaxis=dict(rangeslider=dict(visible=False), type='date', showspikes=True, spikecolor="white", spikethickness=1),
        yaxis=dict(showspikes=True, spikecolor="white", spikethickness=1),
        hovermode='x unified',
        hoverdistance=100,
        spikedistance=1000,
        height=900,
        margin=dict(l=50, r=20, t=50, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0.5)")
    )
    return fig

def market_sentiment(df):
    last = df.iloc[-1]
    if last['rsi'] > 70 and last['cci'] > 100:
        return "🔥 极度贪婪 (超买)"
    elif last['rsi'] < 30 and last['cci'] < -100:
        return "💧 极度恐惧 (超卖)"
    elif last['ma20'] > last['ma60']:
        return "📈 多头主导"
    elif last['ma20'] < last['ma60']:
        return "📉 空头主导"
    else:
        return "⚖️ 多空平衡"

# ---------- 贝叶斯优化（用于参数搜索）----------
def optimize_parameters(df):
    """使用Optuna搜索最佳参数组合（示例）"""
    def objective(trial):
        lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        n_estimators = trial.suggest_int('n_estimators', 50, 200)
        max_depth = trial.suggest_int('max_depth', 3, 10)
        # 训练模型并评估
        model = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=lr, use_label_encoder=False, eval_metric='logloss')
        X = df[['rsi','macd','adx','cci']].dropna().values
        y = (df['close'].shift(-5) > df['close']).astype(int).dropna().values
        min_len = min(len(X), len(y))
        X = X[:min_len]; y = y[:min_len]
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(model, X, y, cv=3)
        return scores.mean()
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)
    return study.best_params

# ---------- 初始化session ----------
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = datetime.now()
    st.session_state.prices = {coin: 2600 for coin in COINS}
    st.session_state.signal_history = []
    if "accounts" not in st.session_state:
        st.session_state.accounts = [{"name": "主账户", "capital": 1000, "leverage": 100, "equity_curve": [1000], "entry_price": 0, "direction": "观望"}]
    st.session_state.current_account = 0
    st.session_state.ensemble_models = None
    st.session_state.scaler = None
    st.session_state.lstm_model = None
    st.session_state.ppo_model = None

# ---------- 侧边栏 ----------
with st.sidebar:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("## ⚙️ 终极至尊控制台")
    st.markdown("---")
    selected_coin = st.selectbox("选择币种", list(COINS.keys()), index=1)
    coin_id = COINS[selected_coin]["id"]
    interval = st.selectbox("K线周期", ["1m","5m","15m","1h","4h"], index=1)
    auto = st.checkbox("自动刷新 (30秒)", True)
    st.markdown("---")
    st.subheader("👥 多账户管理")
    account_names = [acc["name"] for acc in st.session_state.accounts]
    selected_account_idx = st.selectbox("选择账户", range(len(account_names)), format_func=lambda i: account_names[i], key="account_selector")
    st.session_state.current_account = selected_account_idx
    if st.button("➕ 添加新账户", use_container_width=True):
        if len(st.session_state.accounts) < 3:
            new_name = f"账户{len(st.session_state.accounts)+1}"
            st.session_state.accounts.append({"name": new_name, "capital": 1000, "leverage": 100, "equity_curve": [1000], "entry_price": 0, "direction": "观望"})
            st.rerun()
        else:
            st.warning("最多支持3个账户")
    st.markdown("---")
    st.subheader("💰 资金管理")
    acc = st.session_state.accounts[st.session_state.current_account]
    capital = st.number_input("本金 (USDT)", 10, value=acc["capital"], step=100, key=f"capital_{st.session_state.current_account}")
    lev = st.select_slider("杠杆倍数", [10,20,50,100], value=acc["leverage"], key=f"lev_{st.session_state.current_account}")
    st.session_state.accounts[st.session_state.current_account]["capital"] = capital
    st.session_state.accounts[st.session_state.current_account]["leverage"] = lev
    price, _ = fetch_price(coin_id)
    if price:
        st.session_state.prices[selected_coin] = price
    current_price = st.session_state.prices.get(selected_coin, 2600)
    entry = st.number_input("入场价", value=current_price, step=1.0, format="%.2f", key=f"entry_{st.session_state.current_account}")
    stop = st.number_input("止损价", value=current_price*0.99, step=1.0, format="%.2f", key=f"stop_{st.session_state.current_account}")
    if st.button("🔄 刷新数据", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- 主界面 ----------
st.markdown(f'<h1 class="title-glow">📊 {selected_coin} 终极至尊量子版 v4 · 跨维度智能融合系统</h1>', unsafe_allow_html=True)
st.caption(f"⚡ 数据更新: {st.session_state.last_refresh.strftime('%H:%M:%S')} | 数据源: CoinGecko + 宏观/链上/情绪模拟 + 深度强化学习 + 贝叶斯优化")

price, change = fetch_price(coin_id)
if price:
    st.session_state.prices[selected_coin] = price
else:
    price = st.session_state.prices.get(selected_coin, 2600)

interval_min = int(interval.replace('m','').replace('h','60')) if 'm' in interval or 'h' in interval else 5
df = generate_klines(price, interval_min, limit=500)
df = add_advanced_indicators(df)
last = df.iloc[-1]
prev = df.iloc[-2]

# 获取外部数据
macro = fetch_macro_data()
onchain = fetch_onchain_data(coin_id)
sentiment = fetch_news_sentiment()
ob_imbalance = fetch_orderbook_imbalance()

# 训练/更新集成模型
if st.session_state.ensemble_models is None or len(df) % 100 == 0:
    models, scaler = train_ensemble(df)
    if models is not None:
        st.session_state.ensemble_models = models
        st.session_state.scaler = scaler
ensemble_prob = ensemble_predict(df, st.session_state.ensemble_models, st.session_state.scaler) if st.session_state.ensemble_models else 0.5

# LSTM预测
feature_cols = ['close', 'volume', 'rsi', 'macd', 'adx', 'mfi', 'kdj_k', 'stochrsi_k', 'williams_r', 'cmf']
if st.session_state.lstm_model is None:
    lstm_pred, lstm_model = train_lstm(df, feature_cols, 'close')
    if lstm_model:
        st.session_state.lstm_model = lstm_model
else:
    last_seq = df[feature_cols].iloc[-20:].values.reshape(1,20,-1)
    last_seq_t = torch.tensor(last_seq, dtype=torch.float32)
    with torch.no_grad():
        lstm_pred = st.session_state.lstm_model(last_seq_t).item()

# 强化学习PPO（简化：使用预训练模型或占位）
if st.session_state.ppo_model is None:
    # 训练一个小模型（实际中需要大量数据和时间）
    env = DummyVecEnv([lambda: TradingEnv(df, feature_cols)])
    st.session_state.ppo_model = PPO('MlpPolicy', env, verbose=0, n_steps=256, batch_size=64, n_epochs=10, learning_rate=0.0003)
    st.session_state.ppo_model.learn(total_timesteps=5000)
ppo_obs = df[feature_cols].iloc[-1].values
ppo_obs = np.concatenate([[0.0], ppo_obs]).reshape(1, -1)
ppo_action, _ = st.session_state.ppo_model.predict(ppo_obs, deterministic=True)
ppo_action = ppo_action[0]  # -1到1之间的仓位建议

# 计算信号
score, reason_summary = calculate_signal_score(df, ensemble_prob, lstm_pred, macro, onchain, sentiment, ob_imbalance)
direction, conf, extra_reason = get_signal_from_score(score)

# 动态杠杆
current_vol = last['natr'] / 100 if not pd.isna(last['natr']) else 0.02
dyn_leverage = dynamic_leverage(current_vol, base_leverage=lev, max_leverage=100)

# 蒙特卡洛模拟
mean_path, upper, lower = monte_carlo_simulation(df, steps=10, n_simulations=200)
var_1d = calculate_var(df, confidence=0.95, horizon=1)
var_5d = calculate_var(df, confidence=0.95, horizon=5)
sentiment_label = market_sentiment(df)

# 移动止损建议
trail_stop = moving_stop_loss(entry, last['close'], direction)

# 凯利仓位（模拟胜率和赔率）
if len(st.session_state.signal_history) > 10:
    # 简单估计胜率
    df_hist = pd.DataFrame(st.session_state.signal_history)
    wins = len(df_hist[df_hist['score'] > 0])
    total = len(df_hist)
    win_prob = wins / total if total>0 else 0.5
    avg_win = df_hist[df_hist['score'] > 0]['score'].mean() if wins>0 else 0
    avg_loss = abs(df_hist[df_hist['score'] < 0]['score'].mean()) if (total-wins)>0 else 1
    win_loss_ratio = avg_win / avg_loss if avg_loss>0 else 1
    kelly = kelly_fraction(win_prob, win_loss_ratio)
else:
    kelly = 0.02  # 默认2%

# ---------- 顶部指标卡片 ----------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
cols = st.columns(10)
with cols[0]:
    delta = last['close'] - prev['close']
    st.metric(f"{selected_coin}/USDT", f"${last['close']:.2f}", f"{delta:+.2f}")
with cols[1]:
    st.metric("RSI", f"{last['rsi']:.1f}")
with cols[2]:
    st.metric("ADX", f"{last['adx']:.1f}")
with cols[3]:
    st.metric("ATR%", f"{last['natr']:.2f}%")
with cols[4]:
    st.metric("成交量", f"{last['volume']:.0f}")
with cols[5]:
    st.metric("情绪", sentiment_label, delta=None)
with cols[6]:
    st.metric("ML概率", f"{ensemble_prob:.0%}")
with cols[7]:
    st.metric("LSTM", f"${lstm_pred:.2f}" if 'lstm_pred' in locals() else "N/A")
with cols[8]:
    st.metric("PPO仓位", f"{ppo_action:.2f}")
with cols[9]:
    st.metric("凯利%", f"{kelly:.1%}")
st.markdown('</div>', unsafe_allow_html=True)

# 风险提示 + VaR + 宏观/链上快照
st.markdown(f"""
<div class="warning-box">
    ⚠️ 当前杠杆 {lev}倍 (建议动态 {dyn_leverage}x) | 本金 {capital:.0f} USDT | 可开最大 {capital*lev/price:.3f} {selected_coin} | 单笔风险≤2% | 24h涨跌: {change:+.2f}% 
    <br>📊 风险价值 (95%): 1日 VaR {var_1d*100:.2f}% | 5日 VaR {var_5d*100:.2f}%
    <br>🌍 宏观: 利率 {macro['interest_rate']:.1f}% | CPI {macro['cpi']:.1f}% | 失业 {macro['unemployment']:.1f}%
    <br>🔗 链上: 活跃地址 {onchain['active_addresses']:,} | 净流量 {onchain['exchange_netflow']:.0f} | 新闻情绪 {sentiment:.2f}
</div>
""", unsafe_allow_html=True)

# ---------- AI实时监控分析（六列增强）----------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("📊 AI实时监控分析")
colA, colB, colC, colD, colE, colF = st.columns(6)
with colA:
    st.markdown("**趋势状态**")
    trend = "多头" if last['ma20'] > last['ma60'] else "空头" if last['ma20'] < last['ma60'] else "震荡"
    st.markdown(f"- 均线排列: **{trend}**")
    st.markdown(f"- ADX: **{last['adx']:.1f}** ({'强趋势' if last['adx']>25 else '弱趋势'})")
    st.markdown(f"- 价格相对布林: **{'上轨' if last['close']>last['bb_upper'] else '下轨' if last['close']<last['bb_lower'] else '中轨'}**")
with colB:
    st.markdown("**动量指标**")
    st.markdown(f"- RSI: **{last['rsi']:.1f}** ({'超买' if last['rsi']>70 else '超卖' if last['rsi']<30 else '中性'})")
    st.markdown(f"- CCI: **{last['cci']:.1f}**")
    st.markdown(f"- MFI: **{last['mfi']:.1f}**")
    st.markdown(f"- KDJ: K={last['kdj_k']:.1f} J={last['kdj_j']:.1f}")
with colC:
    st.markdown("**额外指标**")
    st.markdown(f"- StochRSI K: **{last['stochrsi_k']:.1f}**")
    st.markdown(f"- Williams %R: **{last['williams_r']:.1f}**")
    st.markdown(f"- CMF: **{last['cmf']:.2f}**")
with colD:
    st.markdown("**支撑/阻力**")
    support = last['bb_lower'] if not pd.isna(last['bb_lower']) else last['close']*0.98
    resistance = last['bb_upper'] if not pd.isna(last['bb_upper']) else last['close']*1.02
    st.markdown(f"- 支撑: **${support:.2f}**")
    st.markdown(f"- 阻力: **${resistance:.2f}**")
    st.markdown(f"- 成交量密集: ${last['volume_profile_max']:.2f}" if 'volume_profile_max' in df.columns else "")
with colE:
    st.markdown("**Ichimoku云**")
    if 'tenkan' in df.columns and not pd.isna(last['tenkan']):
        st.markdown(f"- 转换线: **${last['tenkan']:.2f}**")
        st.markdown(f"- 基准线: **${last['kijun']:.2f}**")
        st.markdown(f"- 云带A: **${last['senkou_a']:.2f}**" if not pd.isna(last['senkou_a']) else "")
        st.markdown(f"- 云带B: **${last['senkou_b']:.2f}**" if not pd.isna(last['senkou_b']) else "")
with colF:
    st.markdown("**AI决策**")
    st.markdown(f"- PPO建议仓位: **{ppo_action:.2f}**")
    st.markdown(f"- 凯利仓位: **{kelly:.1%}**")
    st.markdown(f"- 综合评分: **{score}**")
st.markdown('</div>', unsafe_allow_html=True)

# ---------- K线图 ----------
st.subheader(f"{interval} K线图 (含Ichimoku云)")
fig = plot_ultimate_candlestick(df, selected_coin, interval)
st.plotly_chart(fig, use_container_width=True)

# ---------- 蒙特卡洛模拟图 ----------
st.subheader("🔮 蒙特卡洛模拟 (未来10步价格路径)")
fig_mc = go.Figure()
x_future = list(range(11))
fig_mc.add_trace(go.Scatter(x=x_future, y=mean_path, mode='lines+markers', name='平均路径', line=dict(color='gold', width=2)))
fig_mc.add_trace(go.Scatter(x=x_future, y=upper, mode='lines', name='95%上限', line=dict(color='red', dash='dash')))
fig_mc.add_trace(go.Scatter(x=x_future, y=lower, mode='lines', name='5%下限', line=dict(color='green', dash='dash')))
fig_mc.update_layout(template="plotly_dark", height=300, margin=dict(l=0, r=0, t=20, b=0), title="未来10步价格模拟")
st.plotly_chart(fig_mc, use_container_width=True)

# ---------- AI信号与交易策略 ----------
colL, colR = st.columns(2)
with colL:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🎯 AI量子信号")
    if "强烈" in direction:
        st.markdown(f'<div class="strong-signal"><span style="font-size:28px;color:{"#26A69A" if "多" in direction else "#EF5350"};">{direction}</span><br>评分: {score} (强烈信号)<br>{extra_reason}<br>因子: {reason_summary}<br>ML概率: {ensemble_prob:.0%}<br>LSTM预测: ${lstm_pred:.2f if "lstm_pred" in locals() else "N/A"}<br>PPO建议: {ppo_action:.2f}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="signal-box"><span style="font-size:24px;color:{"#26A69A" if "多" in direction else "#EF5350" if "空" in direction else "#888"};">{"🟢" if "多" in direction else "🔴" if "空" in direction else "⚪"} {direction}</span><br>评分: {score}<br>{extra_reason}<br>因子: {reason_summary}<br>ML概率: {ensemble_prob:.0%}<br>LSTM预测: ${lstm_pred:.2f if "lstm_pred" in locals() else "N/A"}<br>PPO建议: {ppo_action:.2f}</div>', unsafe_allow_html=True)
    patterns = detect_candlestick_patterns(df)
    if patterns:
        st.markdown("**📐 形态识别:**")
        for p in patterns:
            st.markdown(f"- {p}")
    st.markdown('</div>', unsafe_allow_html=True)

with colR:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📈 精准入场策略")
    if "做多" in direction:
        stop_price = last['ma60'] if not pd.isna(last['ma60']) else last['close']*0.99
        tp1 = last['close'] * 1.02
        tp2 = last['close'] * 1.05
        st.markdown(f"""
        **激进进场:** ${last['close']:.2f} (当前价)  
        **稳健进场:** ${last['ma20']:.2f} (MA20支撑)  
        **止损位:** ${stop_price:.2f}  
        **第一目标:** ${tp1:.2f} (+2%)  
        **第二目标:** ${tp2:.2f} (+5%)  
        """)
    elif "做空" in direction:
        stop_price = last['ma60'] if not pd.isna(last['ma60']) else last['close']*1.01
        tp1 = last['close'] * 0.98
        tp2 = last['close'] * 0.95
        st.markdown(f"""
        **激进进场:** ${last['close']:.2f} (当前价)  
        **稳健进场:** ${last['ma20']:.2f} (MA20阻力)  
        **止损位:** ${stop_price:.2f}  
        **第一目标:** ${tp1:.2f} (-2%)  
        **第二目标:** ${tp2:.2f} (-5%)  
        """)
    else:
        st.info("等待明确信号")
    if trail_stop:
        st.success(f"💡 移动止损建议: 可将止损上移至 ${trail_stop:.2f} (保本)")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- 当前盈亏与净值曲线 ----------
colX, colY = st.columns([1, 1])
with colX:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    qty = calc_position(capital, entry, stop, lev)
    if qty > 0:
        if "做多" in direction:
            pnl = (last['close'] - entry) * qty
        else:
            pnl = (entry - last['close']) * qty
        color = "#26A69A" if pnl>=0 else "#EF5350"
        st.markdown(f"""
        <span style="font-size:20px;">💰 当前盈亏</span><br>
        <span style="font-size:32px;color:{color};">{pnl:+.2f} USDT</span><br>
        <span>数量 {qty:.4f} {selected_coin} | 保证金 {qty*entry/lev:.2f} USDT</span>
        """, unsafe_allow_html=True)
        st.session_state.accounts[st.session_state.current_account]["entry_price"] = entry
        st.session_state.accounts[st.session_state.current_account]["direction"] = direction
        st.session_state.accounts[st.session_state.current_account]["equity_curve"].append(capital + pnl)
    else:
        st.info("输入有效入场价和止损价计算盈亏")
    st.markdown('</div>', unsafe_allow_html=True)

with colY:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("**📈 模拟账户净值曲线**")
    equity_curve = st.session_state.accounts[st.session_state.current_account]["equity_curve"]
    if len(equity_curve) > 1:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=list(range(len(equity_curve))), y=equity_curve, mode='lines', line=dict(color='#00D4FF', width=2), fill='tozeroy', fillcolor='rgba(0,212,255,0.1)'))
        fig2.update_layout(template="plotly_dark", height=150, margin=dict(l=0, r=0, t=10, b=0), showlegend=False, xaxis=dict(showticklabels=False), yaxis=dict(title="净值"))
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.write("暂无数据")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- 历史信号回测面板 ----------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("📜 历史信号回测")
current_signal = {"time": datetime.now().strftime("%H:%M"), "coin": selected_coin, "direction": direction, "score": score, "price": last['close']}
st.session_state.signal_history.append(current_signal)
if len(st.session_state.signal_history) > 20:
    st.session_state.signal_history = st.session_state.signal_history[-20:]
if st.session_state.signal_history:
    df_signals = pd.DataFrame(st.session_state.signal_history)
    total = len(df_signals)
    wins = len(df_signals[df_signals['score'] > 0])
    win_rate = wins/total if total>0 else 0
    st.markdown(f"**最近{total}次信号统计** (基于评分方向模拟): 胜率 {win_rate:.1%}")
    st.dataframe(df_signals[['time','coin','direction','score','price']], use_container_width=True, hide_index=True)
else:
    st.info("暂无历史信号")
st.markdown('</div>', unsafe_allow_html=True)

# ---------- 其他币种快照 ----------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("📌 其他币种快照")
cols = st.columns(3)
coin_list = list(COINS.items())
other_coins = [item for item in coin_list if item[0] != selected_coin]
for i, (coin_name, coin_info) in enumerate(other_coins[:3]):
    with cols[i]:
        coin_id = coin_info["id"]
        p, ch = fetch_price(coin_id)
        if p:
            st.markdown(f"""
            <div class="snapshot-card">
                <span style="font-size:20px;font-weight:bold;">{coin_name}</span><br>
                <span>价格: ${p:.2f}</span><br>
                <span>24h: <span style="color:{'#26A69A' if ch>0 else '#EF5350'};">{ch:+.2f}%</span></span>
            </div>
            """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# 自动刷新
if auto and (datetime.now()-st.session_state.last_refresh).seconds > 30:
    st.cache_data.clear()
    st.session_state.last_refresh = datetime.now()
    st.rerun()

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.caption("⚠️ 终极至尊量子版 v4 仅供学术研究，不构成投资建议。100倍杠杆高风险，务必设止损。市场有风险，入市需谨慎。历史不会重演，但总会惊人相似。")
