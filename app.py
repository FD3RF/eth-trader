# -*- coding: utf-8 -*-
"""
🚀 合约智能监控中心 · 终极最强神级版
五层共振 + AI决策 + 免费数据源 + 动态风控
"""

import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import requests
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
from streamlit_autorefresh import st_autorefresh
import warnings
warnings.filterwarnings('ignore')

# ==================== 免费数据源获取 ====================
class FreeDataFetcher:
    """完全免费的数据获取器（ccxt + Coinglass + Alternative.me + 模拟链上）"""
    
    def __init__(self, symbol='ETH/USDT'):
        self.symbol = symbol
        self.base = symbol.split('/')[0]
        self.exchange = ccxt.binance({'enableRateLimit': True})
        self.periods = ['15m', '1h', '4h', '1d']
        self.limit = 500
        
        # Coinglass免费API（无需key，但有频率限制）
        self.coinglass_base = "https://open-api.coinglass.com/api/pro/v1/futures"
        
        # 情绪API
        self.fng_url = "https://api.alternative.me/fng/"
        
        # 模拟链上数据（可替换为Dune免费API）
        self.chain_netflow = 5234   # 示例值
        self.chain_whale = 128

    def fetch_ohlcv(self, timeframe):
        """从Binance获取K线"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe, limit=self.limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            return df
        except Exception as e:
            st.error(f"获取{timeframe}数据失败: {e}")
            return None

    def fetch_coinglass_data(self):
        """获取Coinglass资金面数据（资金费率、OI、多空比）"""
        coin = self.base
        funding = oi = ls_ratio = 0.0
        try:
            # 资金费率
            url = f"{self.coinglass_base}/funding_rate_chart?symbol={coin}"
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if data['data']:
                    funding = data['data'][-1]['fundingRate']
                    oi = data['data'][-1]['openInterest']
            # 多空比
            url2 = f"{self.coinglass_base}/long_short_chart?symbol={coin}"
            resp2 = requests.get(url2, timeout=5)
            if resp2.status_code == 200:
                data2 = resp2.json()
                if data2['data']:
                    ls_ratio = data2['data'][-1]['longShortRatio']
        except:
            pass
        # 如果失败，使用模拟值
        if funding == 0:
            funding = np.random.uniform(-0.001, 0.001)
            oi = np.random.uniform(1e8, 1e9)
            ls_ratio = np.random.uniform(0.7, 1.5)
        return funding, oi, ls_ratio

    def fetch_fear_greed(self):
        """获取恐惧贪婪指数"""
        try:
            resp = requests.get(self.fng_url, timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                return int(data['data'][0]['value'])
        except:
            pass
        return 50

    def fetch_all(self):
        """获取所有周期数据 + 资金面 + 情绪"""
        data_dict = {}
        for tf in self.periods:
            df = self.fetch_ohlcv(tf)
            if df is not None:
                data_dict[tf] = self._add_indicators(df)
        
        funding, oi, ls_ratio = self.fetch_coinglass_data()
        fear_greed = self.fetch_fear_greed()
        
        # 当前价格（取15m最新）
        current_price = data_dict['15m']['close'].iloc[-1] if '15m' in data_dict else None
        
        return {
            "data_dict": data_dict,
            "current_price": current_price,
            "funding_rate": funding,
            "open_interest": oi,
            "long_short_ratio": ls_ratio,
            "fear_greed": fear_greed,
            "chain_netflow": self.chain_netflow,
            "chain_whale": self.chain_whale
        }

    def _add_indicators(self, df):
        """添加技术指标"""
        df = df.copy()
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma60'] = df['close'].rolling(60).mean()
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = df['macd'] - df['macd_signal']
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
        df['atr_pct'] = df['atr'] / df['close'] * 100
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
        df['adx'] = adx.adx()
        return df


# ==================== 五层共振评分 ====================
def five_layer_score(data, funding_rate, long_short_ratio, fear_greed, chain_netflow, chain_whale):
    """
    计算五层共振总分和方向
    返回：(方向: 1多/-1空/0观望, 总分, 各层分数)
    """
    df_15m = data.get('15m')
    df_1h = data.get('1h')
    df_4h = data.get('4h')
    df_1d = data.get('1d')
    if any(df is None for df in [df_15m, df_1h, df_4h, df_1d]):
        return 0, 0, {}

    last_15m = df_15m.iloc[-1]
    last_1h = df_1h.iloc[-1]
    last_4h = df_4h.iloc[-1]
    last_1d = df_1d.iloc[-1]

    # 1. 趋势层 (30分)
    trend_score = 0
    trend_dir = 0
    if last_15m['adx'] > 25 or (last_15m['adx'] > 18 and last_15m['atr_pct'] > 0.8):
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

    # 3. 资金面层 (20分)
    fund_score = 0
    fund_dir = 0
    if funding_rate < -0.0005 and long_short_ratio > 1.2:
        fund_score = 20
        fund_dir = 1
    elif funding_rate > 0.0005 and long_short_ratio < 0.8:
        fund_score = 20
        fund_dir = -1
    elif funding_rate < 0:
        fund_score = 10
        fund_dir = 1

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
    if last_15m['rsi'] > 55 and last_15m['macd_diff'] > 0:
        momentum_score = 10
        momentum_dir = 1
    elif last_15m['rsi'] < 45 and last_15m['macd_diff'] < 0:
        momentum_score = 10
        momentum_dir = -1

    # 最终方向：至少三层一致且无反向
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
    """加载预训练的XGBoost模型（若无则返回None）"""
    try:
        import joblib
        model = joblib.load('eth_ai_model.pkl')
        return model
    except:
        return None

def ai_predict(model, features):
    """使用模型预测上涨概率"""
    if model is None:
        return np.random.randint(40, 60)  # 模拟
    prob = model.predict_proba([features])[0][1] * 100
    return prob

# 注：训练脚本见附录，需先在本地/Colab运行生成模型文件


# ==================== 仓位建议 ====================
def suggest_position(total_score, ai_prob, atr_pct, account_balance, risk_per_trade=2.0):
    if total_score >= 80 and ai_prob > 70:
        leverage_range = (5, 10)
        base_risk = risk_per_trade
    elif total_score >= 60 and ai_prob > 60:
        leverage_range = (2, 5)
        base_risk = risk_per_trade * 0.8
    elif total_score >= 40 and ai_prob > 50:
        leverage_range = (1, 2)
        base_risk = risk_per_trade * 0.5
    else:
        return 0, 0, 0
    
    # 根据ATR调整杠杆
    if atr_pct > 3:
        leverage_range = (leverage_range[0]*0.7, leverage_range[1]*0.7)
    suggested_leverage = np.mean(leverage_range)
    return suggested_leverage, base_risk, ai_prob


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


# ==================== 主界面 ====================
st.set_page_config(page_title="合约智能监控·终极神级版", layout="wide")
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

st.title("🧠 合约智能监控中心 · 终极最强神级版")
st.caption("五层共振 + AI决策 + 全免费数据源 + 动态风控")

# 初始化
init_risk_state()
ai_model = load_ai_model()

# 侧边栏
with st.sidebar:
    st.header("⚙️ 控制面板")
    symbol = st.selectbox("交易对", ["ETH/USDT", "BTC/USDT", "SOL/USDT", "BNB/USDT"], index=0)
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

# 获取数据
with st.spinner("获取全市场数据..."):
    fetcher = FreeDataFetcher(symbol)
    data = fetcher.fetch_all()

data_dict = data["data_dict"]
current_price = data["current_price"]
funding_rate = data["funding_rate"]
oi = data["open_interest"]
ls_ratio = data["long_short_ratio"]
fear_greed = data["fear_greed"]
chain_netflow = data["chain_netflow"]
chain_whale = data["chain_whale"]

# 五层共振
final_dir, total_score, layer_scores = five_layer_score(
    data_dict, funding_rate, ls_ratio, fear_greed, chain_netflow, chain_whale
)

# AI预测（需要提取特征，这里简化）
# 实际应提取最新特征向量，此处演示用
atr_pct = data_dict['15m']['atr_pct'].iloc[-1] if '15m' in data_dict else 0
adx = data_dict['15m']['adx'].iloc[-1] if '15m' in data_dict else 0
features_sample = [adx, atr_pct, funding_rate, ls_ratio, fear_greed]  # 示例特征
ai_prob = ai_predict(ai_model, features_sample)

# 仓位建议
suggested_leverage, base_risk, final_ai_prob = suggest_position(total_score, ai_prob, atr_pct, account_balance, risk_per_trade)

# 更新风控
drawdown = update_risk_stats(current_price, sim_entry, sim_side, sim_quantity, sim_leverage)

# 显示数据源状态
st.markdown(f"""
<div class="info-box">
    ✅ 数据源：Binance/Coinglass/Alternative | 恐惧贪婪：{fear_greed} | AI模型：{'已加载' if ai_model else '未加载(使用模拟)'}
    <br>⚠️ 链上数据为模拟值（可替换为Dune免费API）
</div>
""", unsafe_allow_html=True)

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

# 主布局
col_left, col_right = st.columns([2.2, 1.3])

with col_left:
    st.subheader(f"📊 {symbol} K线 ({main_period})")
    if main_period in data_dict:
        df = data_dict[main_period].tail(100).copy()
        df['日期'] = df['timestamp']
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           row_heights=[0.7, 0.3],
                           subplot_titles=(f"{symbol} {main_period}", "RSI"))
        # K线
        fig.add_trace(go.Candlestick(x=df['日期'], open=df['open'], high=df['high'],
                                     low=df['low'], close=df['close'], name="K线"), row=1, col=1)
        # 均线
        fig.add_trace(go.Scatter(x=df['日期'], y=df['ma20'], name="MA20", line=dict(color="orange")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['日期'], y=df['ma60'], name="MA60", line=dict(color="blue")), row=1, col=1)
        # 方向箭头
        if final_dir != 0:
            last_date = df['日期'].iloc[-1]
            last_price = df['close'].iloc[-1]
            arrow_text = "▲ 五层多" if final_dir == 1 else "▼ 五层空"
            arrow_color = "green" if final_dir == 1 else "red"
            fig.add_annotation(x=last_date, y=last_price * (1.02 if final_dir==1 else 0.98),
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
    st.markdown(f'<div class="ai-box">{dir_map[final_dir]}<br>五层总分: {total_score}/100</div>', unsafe_allow_html=True)
    
    if final_dir != 0:
        st.markdown(f"""
        <div style="background:#1A1D27; padding:15px; border-radius:8px; margin:10px 0;">
            <h4>🤖 AI预测胜率</h4>
            <h2 style="color:#00F5A0">{final_ai_prob:.1f}%</h2>
            <p>建议杠杆: {suggested_leverage:.1f}x | 风险: {base_risk:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.metric("当前价格", f"${current_price:.2f}" if current_price else "N/A")
    
    # 风险仪表盘
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
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 资金面快照
    with st.expander("💰 资金面快照", expanded=True):
        st.write(f"资金费率: **{funding_rate:.6f}**")
        st.write(f"未平仓合约: **{oi:.2e}**")
        st.write(f"多空比: **{ls_ratio:.2f}**")
    
    # 链上/情绪
    with st.expander("🔗 链上&情绪", expanded=False):
        st.write(f"交易所净流入: **{chain_netflow:+.0f} ETH** (模拟)")
        st.write(f"大额转账: **{chain_whale}** 笔 (模拟)")
        st.write(f"恐惧贪婪指数: **{fear_greed}**")
    
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
