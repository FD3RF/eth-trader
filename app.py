# -*- coding: utf-8 -*-
"""
🚀 合约智能监控中心 · 终极神级版（多币种+AI交易计划）
五层共振 + AI决策 + 动态止损止盈 + 历史信号
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

warnings.filterwarnings('ignore')

# ==================== 配置 ====================
SYMBOLS = ["ETH/USDT", "BTC/USDT", "SOL/USDT"]  # 支持的交易对

# ==================== 免费数据获取器（支持多币种）====================
class FreeDataFetcherV5:
    """支持多币种的免费数据获取器"""
    
    def __init__(self, symbols=None):
        if symbols is None:
            symbols = SYMBOLS
        self.symbols = symbols
        self.periods = ['15m', '1h', '4h', '1d']
        self.limit = 500
        self.timeout = 10
        
        # MEXC交易所实例
        self.exchange = ccxt.mexc({
            'enableRateLimit': True,
            'timeout': 30000,
        })
        
        # 恐惧贪婪指数
        self.fng_url = "https://api.alternative.me/fng/"
        
        # 模拟链上数据（标注模拟）
        self.chain_netflow = 5234
        self.chain_whale = 128
    
    def fetch_kline(self, symbol, timeframe):
        """获取单个币种K线"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=self.limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            return df, "MEXC"
        except Exception as e:
            st.warning(f"{symbol} {timeframe} 获取失败: {e}")
            return None, None
    
    def fetch_fear_greed(self):
        """获取恐惧贪婪指数"""
        try:
            resp = requests.get(self.fng_url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                return int(data['data'][0]['value'])
        except:
            pass
        return 50
    
    def fetch_all(self):
        """获取所有币种所有周期的数据"""
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
        """添加技术指标"""
        df = df.copy()
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma60'] = df['close'].rolling(60).mean()
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
        df['atr_pct'] = df['atr'] / df['close'] * 100
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
        df['adx'] = adx.adx()
        return df


# ==================== 五层共振评分 ====================
def five_layer_score(df_dict, fear_greed, chain_netflow, chain_whale):
    """
    计算五层共振总分和方向
    返回：(方向: 1多/-1空/0观望, 总分, 各层分数)
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

    # 1. 趋势层 (30分)
    trend_score = 0
    trend_dir = 0
    adx = last_15m['adx']
    atr_pct = last_15m['atr_pct']

    if adx > 25 or (adx > 18 and atr_pct > 0.8):
        trend_score = 30
        trend_dir = 1 if last_15m['ma20'] > last_15m['ma60'] else -1

    # 2. 多周期共振 (25分)
    multi_score = 0
    multi_dir = 0
    # 检查均线排列
    if all(df['close'].iloc[-1] > df['ma60'].iloc[-1] for df in [df_15m, df_1h, df_4h, df_1d]):
        multi_score = 25
        multi_dir = 1
    elif all(df['close'].iloc[-1] < df['ma60'].iloc[-1] for df in [df_15m, df_1h, df_4h, df_1d]):
        multi_score = 25
        multi_dir = -1
    elif all(df['close'].iloc[-1] > df['ma20'].iloc[-1] for df in [df_15m, df_1h, df_4h]):
        multi_score = 15
        multi_dir = 1

    # 3. 资金面层（无真实数据，暂用模拟）
    fund_score = 0
    fund_dir = 0

    # 4. 链上/情绪层 (15分)
    chain_score = 0
    chain_dir = 0
    if chain_netflow > 5000 and chain_whale > 100:
        chain_score = 15
        chain_dir = 1
    elif fear_greed < 30:
        chain_score = 10
        chain_dir = 1
    elif fear_greed > 70:
        chain_score = 10
        chain_dir = -1

    # 5. 动量层 (10分)
    momentum_score = 0
    momentum_dir = 0
    if last_15m['rsi'] > 55 and last_15m['macd'] > last_15m['macd_signal']:
        momentum_score = 10
        momentum_dir = 1
    elif last_15m['rsi'] < 45 and last_15m['macd'] < last_15m['macd_signal']:
        momentum_score = 10
        momentum_dir = -1

    # 最终方向：至少三层一致
    dirs = [d for d in [trend_dir, multi_dir, fund_dir, chain_dir, momentum_dir] if d != 0]
    if len(dirs) >= 3 and all(d == dirs[0] for d in dirs):
        final_dir = dirs[0]
    else:
        final_dir = 0

    total_score = trend_score + multi_score + fund_score + chain_score + momentum_score
    layer_scores = {
        "趋势": trend_score,
        "多周期": multi_score,
        "资金面": fund_score,
        "链上情绪": chain_score,
        "动量": momentum_score
    }
    return final_dir, total_score, layer_scores


# ==================== AI预测模块 ====================
def load_ai_model():
    """加载预训练的XGBoost模型"""
    model_path = 'eth_ai_model.pkl'
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            return model
        except Exception as e:
            st.warning(f"AI模型加载失败: {e}")
            return None
    else:
        return None

def ai_predict(model, features):
    """使用模型预测上涨概率，features应为长度为7的列表"""
    if model is None:
        return np.random.randint(40, 60)
    try:
        prob = model.predict_proba([features])[0][1] * 100
        return prob
    except Exception as e:
        st.error(f"AI预测出错: {e}")
        return 50


# ==================== 交易计划生成 ====================
def generate_trade_plan(direction, current_price, atr_value, ai_prob):
    """
    根据方向、价格、ATR、AI胜率生成止损止盈价
    止损 = 当前价 ± 1.5 * ATR
    止盈 = 当前价 ∓ 3 * ATR (风险回报比1:2)
    返回 (止损价, 止盈价, 盈亏比)
    """
    if direction == 0 or atr_value == 0 or current_price == 0:
        return None, None, None
    stop_distance = 1.5 * atr_value
    take_distance = 3.0 * atr_value  # 1:2 盈亏比
    if direction == 1:  # 做多
        stop_loss = current_price - stop_distance
        take_profit = current_price + take_distance
    else:  # 做空
        stop_loss = current_price + stop_distance
        take_profit = current_price - take_distance
    risk_reward = take_distance / stop_distance  # 盈亏比
    return stop_loss, take_profit, risk_reward


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
        leverage_range = (5, 10)
        base_risk = risk_per_trade
    elif total_score >= 70:
        leverage_range = (2, 5)
        base_risk = risk_per_trade * 0.8
    elif total_score >= 50:
        leverage_range = (1, 2)
        base_risk = risk_per_trade * 0.5
    else:
        return 0, 0, 0

    if atr_pct > 3:
        leverage_range = (leverage_range[0]*0.7, leverage_range[1]*0.7)
    suggested_leverage = np.mean(leverage_range)
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


# ==================== 风险状态管理 ====================
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

def update_risk_stats(current_price, sim_entry, sim_side, sim_quantity, sim_leverage):
    today = datetime.now().date()
    if today != st.session_state.last_date:
        st.session_state.daily_pnl = 0.0
        st.session_state.last_date = today
    if sim_entry > 0 and current_price:
        if sim_side == "多单":
            pnl = (current_price - sim_entry) * sim_quantity * sim_leverage
        else:
            pnl = (sim_entry - current_price) * sim_quantity * sim_leverage
        st.session_state.daily_pnl = pnl
    current_balance = st.session_state.account_balance + st.session_state.daily_pnl
    if current_balance > st.session_state.peak_balance:
        st.session_state.peak_balance = current_balance
    drawdown = (st.session_state.peak_balance - current_balance) / st.session_state.peak_balance * 100
    return drawdown


# ==================== 强平价格计算 ====================
def calculate_liquidation_price(entry_price, side, leverage):
    if side == "多单":
        return entry_price * (1 - 1/leverage)
    else:
        return entry_price * (1 + 1/leverage)


# ==================== 主界面 ====================
st.set_page_config(page_title="合约智能监控·终极神级版+交易计划", layout="wide")
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
</style>
""", unsafe_allow_html=True)

st.title("🧠 合约智能监控中心 · 终极神级版（AI交易计划）")
st.caption("五层共振 + AI决策 + 动态止损止盈 + 历史信号")

# 初始化
init_risk_state()
ai_model = load_ai_model()

# 侧边栏
with st.sidebar:
    st.header("⚙️ 控制面板")
    
    # 币种选择
    selected_symbol = st.selectbox("主交易对", SYMBOLS, index=0, key="selected_symbol")
    
    main_period = st.selectbox("主图周期", ["15m", "1h", "4h", "1d"], index=0)
    
    auto_refresh = st.checkbox("开启自动刷新", value=True)
    refresh_interval = st.number_input("刷新间隔(秒)", 5, 60, 10, disabled=not auto_refresh)
    if auto_refresh:
        st_autorefresh(interval=refresh_interval * 1000, key="auto_refresh")
    
    st.markdown("---")
    st.subheader("📈 模拟合约")
    sim_entry = st.number_input("开仓价", value=0.0, format="%.2f")
    sim_side = st.selectbox("方向", ["多单", "空单"])
    sim_leverage = st.slider("杠杆倍数", 1, 100, 10)
    sim_quantity = st.number_input("数量", value=0.01, format="%.4f")
    
    st.markdown("---")
    st.subheader("💰 风控设置")
    account_balance = st.number_input("初始资金 (USDT)", value=st.session_state.account_balance, step=1000.0, format="%.2f")
    daily_loss_limit = st.number_input("日亏损限额 (USDT)", value=st.session_state.daily_loss_limit, step=50.0, format="%.2f")
    risk_per_trade = st.slider("单笔风险 (%)", 0.5, 3.0, 2.0, 0.5)
    st.session_state.account_balance = account_balance
    st.session_state.daily_loss_limit = daily_loss_limit
    
    # ========== 信号阈值设置 ==========
    st.markdown("---")
    st.subheader("🎛️ 信号阈值")
    long_threshold = st.slider("做多信号阈值 (总分)", 50, 95, 80, key="long_threshold")
    short_threshold = st.slider("做空信号阈值 (总分)", 5, 50, 20, key="short_threshold")

# 获取数据
with st.spinner("获取全市场数据..."):
    fetcher = FreeDataFetcherV5(symbols=SYMBOLS)
    all_data = fetcher.fetch_all()

# 计算所有币种的五层共振分数
all_scores = {}
for sym, data in all_data.items():
    data_dict = data["data_dict"]
    fear_greed = data["fear_greed"]
    chain_netflow = data["chain_netflow"]
    chain_whale = data["chain_whale"]
    final_dir, total_score, layer_scores = five_layer_score(data_dict, fear_greed, chain_netflow, chain_whale)
    all_scores[sym] = total_score
st.session_state.all_scores = all_scores

# 当前选中的币种数据
if selected_symbol not in all_data:
    selected_symbol = SYMBOLS[0]
data = all_data[selected_symbol]
data_dict = data["data_dict"]
current_price = data["current_price"]
fear_greed = data["fear_greed"]
source_display = data["source"]
chain_netflow = data["chain_netflow"]
chain_whale = data["chain_whale"]

# 计算当前币种的五层共振
final_dir, total_score, layer_scores = five_layer_score(data_dict, fear_greed, chain_netflow, chain_whale)
st.session_state.total_score = total_score   # 用于下单按钮

# 检测市场模式
market_mode = detect_market_mode(data_dict)

# 计算ATR%和ADX
atr_pct = 0
adx = 0
atr_value = 0  # ATR绝对值
if '15m' in data_dict:
    df_15m = data_dict['15m']
    atr_series = df_15m['atr']
    if not atr_series.empty:
        atr_value = atr_series.iloc[-1]
    atr_pct = df_15m['atr_pct'].iloc[-1]
    adx = df_15m['adx'].iloc[-1]

# 计算预期胜率（基于五层）
win_prob = calculate_win_probability(total_score, layer_scores, atr_pct, adx)

# AI预测
ai_prob = 50
if ai_model and '15m' in data_dict:
    try:
        last = data_dict['15m'].iloc[-1]
        features = [
            last['rsi'],
            last['ma20'],
            last['ma60'],
            last['macd'],
            last['macd_signal'],
            last['atr_pct'],
            last['adx']
        ]
        ai_prob = ai_predict(ai_model, features)
    except Exception as e:
        st.error(f"AI特征提取失败: {e}")
        ai_prob = 50

# 综合信号方向
if final_dir != 0 and ai_prob > 60:
    signal_dir = final_dir
    combined_win = (win_prob * 0.6 + ai_prob * 0.4)
elif final_dir != 0 and ai_prob > 50:
    signal_dir = final_dir
    combined_win = win_prob * 0.7 + ai_prob * 0.3
else:
    signal_dir = 0
    combined_win = 0

# 生成交易计划
stop_loss, take_profit, risk_reward = generate_trade_plan(signal_dir, current_price, atr_value, ai_prob)

# 仓位建议
suggested_leverage, base_risk, _ = suggest_position(total_score, combined_win, atr_pct, account_balance, risk_per_trade)

# 更新风控
drawdown = update_risk_stats(current_price, sim_entry, sim_side, sim_quantity, sim_leverage)

# 创建热力图
heatmap_df = create_heatmap_data(layer_scores, final_dir)

# ========== 显示数据源状态 ==========
if source_display != "无":
    st.markdown(f"""
    <div class="info-box">
        ✅ 价格源：{source_display} | 恐惧贪婪：{fear_greed} | AI模型：{'已加载' if ai_model else '未加载(使用模拟)'}
        <br>⚠️ 链上数据为模拟值（可替换为Dune免费API）
    </div>
    """, unsafe_allow_html=True)

# ========== 最佳品种提示 ==========
if all_scores:
    best_symbol = max(all_scores, key=all_scores.get)
    best_score = all_scores[best_symbol]
    st.info(f"🔥 当前最佳机会：**{best_symbol}**（总分 {best_score}）")

# 主布局
col_left, col_right = st.columns([2.2, 1.3])

with col_left:
    # 市场状态
    if data_dict:
        state_color = "green" if market_mode == "趋势" else "orange"
        st.markdown(f"<h5>市场状态: <span style='color:{state_color};'>{market_mode}</span></h5>", unsafe_allow_html=True)

    # 五层共振热力图
    st.subheader("🔥 五层共振热力图")
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

    # K线图
    st.subheader(f"📊 {selected_symbol} K线 ({main_period})")
    if main_period in data_dict:
        df = data_dict[main_period].tail(100).copy()
        df['日期'] = df['timestamp']
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           row_heights=[0.7, 0.3],
                           subplot_titles=(f"{selected_symbol} {main_period}", "RSI"))
        # K线
        fig.add_trace(go.Candlestick(x=df['日期'], open=df['open'], high=df['high'],
                                     low=df['low'], close=df['close'], name="K线"), row=1, col=1)
        # 均线
        fig.add_trace(go.Scatter(x=df['日期'], y=df['ma20'], name="MA20", line=dict(color="orange")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['日期'], y=df['ma60'], name="MA60", line=dict(color="blue")), row=1, col=1)
        # 方向箭头
        if signal_dir != 0:
            last_date = df['日期'].iloc[-1]
            last_price = df['close'].iloc[-1]
            arrow_text = "▲ 多" if signal_dir == 1 else "▼ 空"
            arrow_color = "green" if signal_dir == 1 else "red"
            fig.add_annotation(x=last_date, y=last_price * (1.02 if signal_dir==1 else 0.98),
                               text=arrow_text, showarrow=True, arrowhead=2, arrowcolor=arrow_color)
        # RSI
        fig.add_trace(go.Scatter(x=df['日期'], y=df['rsi'], name="RSI", line=dict(color="purple")), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
        fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=600)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("K线数据不可用")

with col_right:
    st.subheader("🧠 即时决策")
    dir_map = {1: "🔴 做多", -1: "🔵 做空", 0: "⚪ 观望"}
    st.markdown(f'<div class="ai-box">{dir_map[signal_dir]}<br>五层总分: {total_score}/100</div>', unsafe_allow_html=True)

    if signal_dir != 0:
        st.markdown(f"""
        <div style="background:#1A1D27; padding:15px; border-radius:8px; margin:10px 0;">
            <h4>🤖 AI预测胜率</h4>
            <h2 style="color:#00F5A0">{ai_prob:.1f}%</h2>
            <p>建议杠杆: {suggested_leverage:.1f}x | 风险: {base_risk:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 显示交易计划（止损止盈）
        if stop_loss and take_profit:
            st.markdown(f"""
            <div class="trade-plan">
                <h4>📋 AI交易计划</h4>
                <p>入场价: <span style="color:#00F5A0">${current_price:.2f}</span></p>
                <p>止损价: <span style="color:#FF5555">${stop_loss:.2f}</span> (亏损 {abs(current_price-stop_loss)/current_price*100:.2f}%)</p>
                <p>止盈价: <span style="color:#00F5A0">${take_profit:.2f}</span> (盈亏比 {risk_reward:.2f})</p>
                <p>ATR(14): {atr_value:.2f}</p>
            </div>
            """, unsafe_allow_html=True)

    st.metric("当前价格", f"${current_price:.2f}" if current_price else "N/A")

    # ========== 风险仪表盘 ==========
    with st.container():
        st.markdown('<div class="dashboard">', unsafe_allow_html=True)
        st.markdown("#### 📊 风险仪表盘")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.metric("账户余额", f"${st.session_state.account_balance:.2f}")
            st.metric("日盈亏", f"${st.session_state.daily_pnl:.2f}", delta_color="inverse")
        with col_r2:
            st.metric("当前回撤", f"{drawdown:.2f}%")
            st.metric("日亏损剩余", f"${st.session_state.daily_loss_limit + st.session_state.daily_pnl:.2f}")
        
        # 大号显示建议杠杆
        if suggested_leverage > 0:
            st.markdown(f"<h3 style='color:#00F5A0; text-align:center;'>建议杠杆：{suggested_leverage:.1f}x</h3>", unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

    # ========== 资金面快照 ==========
    with st.expander("💰 资金面快照", expanded=True):
        st.write("资金费率: **暂缺（模拟）**")
        st.write("OI变化: **暂缺（模拟）**")
        st.write("多空比: **暂缺（模拟）**")

    # ========== 链上&情绪 ==========
    with st.expander("🔗 链上&情绪", expanded=False):
        st.write(f"交易所净流入: **{chain_netflow:+.0f} {selected_symbol.split('/')[0]}** (模拟)")
        st.write(f"大额转账: **{chain_whale}** 笔 (模拟)")
        st.write(f"恐惧贪婪指数: **{fear_greed}**")

    # ========== 模拟合约持仓 ==========
    if sim_entry > 0 and current_price:
        if sim_side == "多单":
            pnl = (current_price - sim_entry) * sim_quantity * sim_leverage
            pnl_pct = (current_price - sim_entry) / sim_entry * sim_leverage * 100
            liq_price = calculate_liquidation_price(sim_entry, "多单", sim_leverage)
        else:
            pnl = (sim_entry - current_price) * sim_quantity * sim_leverage
            pnl_pct = (sim_entry - current_price) / sim_entry * sim_leverage * 100
            liq_price = calculate_liquidation_price(sim_entry, "空单", sim_leverage)
        color_class = "profit" if pnl >= 0 else "loss"
        distance = abs(current_price - liq_price) / current_price * 100
        st.markdown(f"""
        <div class="metric">
            <h4>模拟持仓</h4>
            <p>{sim_side} | {sim_leverage}x</p>
            <p>开仓: ${sim_entry:.2f}</p>
            <p class="{color_class}">盈亏: ${pnl:.2f} ({pnl_pct:.2f}%)</p>
            <p>强平价: <span class="warning">${liq_price:.2f}</span> (距 {distance:.1f}%)</p>
        </div>
        """, unsafe_allow_html=True)
        if distance < 5:
            st.warning("⚠️ 接近强平线！")
    else:
        st.info("输入开仓价查看模拟")

    # ========== 一键复制交易计划 ==========
    if st.button("📋 复制当前交易计划"):
        plan_text = f"""
        交易对：{selected_symbol}
        方向：{'多' if signal_dir==1 else '空' if signal_dir==-1 else '观望'}
        当前价格：${current_price:.2f}
        五层总分：{total_score}
        AI预测胜率：{ai_prob:.1f}%
        建议杠杆：{suggested_leverage:.1f}x
        """
        if stop_loss and take_profit:
            plan_text += f"\n止损价：${stop_loss:.2f}\n止盈价：${take_profit:.2f}\n盈亏比：{risk_reward:.2f}"
        st.code(plan_text)
        st.info("请手动复制以上计划")

    # ========== 历史信号记录 ==========
    if 'signal_history' not in st.session_state:
        st.session_state.signal_history = []

    # 检测新信号（与上次记录的信号不同）
    if total_score >= st.session_state.long_threshold or total_score <= st.session_state.short_threshold:
        current_dir = "多" if total_score >= st.session_state.long_threshold else "空" if total_score <= st.session_state.short_threshold else "观望"
        if not st.session_state.signal_history or st.session_state.signal_history[-1]['方向'] != current_dir:
            st.session_state.signal_history.append({
                '时间': datetime.now().strftime("%H:%M"),
                '方向': current_dir,
                '总分': total_score
            })
            st.session_state.signal_history = st.session_state.signal_history[-20:]

    with st.expander("📋 历史信号记录"):
        if st.session_state.signal_history:
            st.dataframe(pd.DataFrame(st.session_state.signal_history), use_container_width=True)
        else:
            st.info("暂无历史信号")
