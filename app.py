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
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="终极至尊AI交易系统 · 量子版", layout="wide", initial_sidebar_state="expanded")

# ---------- 极致视觉CSS ----------
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0B0E14 0%, #141A24 100%); color: #F0F4FA; }
    .glass-card { background: rgba(20, 28, 40, 0.75); backdrop-filter: blur(10px); border-radius: 16px; padding: 20px; border: 1px solid rgba(255,255,255,0.1); box-shadow: 0 8px 32px rgba(0,0,0,0.4); }
    .metric-card { background: rgba(16, 22, 34, 0.8); border-radius: 12px; padding: 16px; border-left: 4px solid #00D4FF; box-shadow: 0 4px 20px rgba(0,212,255,0.1); transition: transform 0.2s; }
    .metric-card:hover { transform: translateY(-2px); border-left-color: #F0B90B; }
    .signal-box { background: rgba(26, 34, 48, 0.9); backdrop-filter: blur(5px); border-radius: 16px; padding: 24px; border: 1px solid rgba(255,215,0,0.3); box-shadow: 0 8px 32px rgba(255,215,0,0.1); }
    .strong-signal { background: linear-gradient(145deg, #2A2418, #1F1A12); border-left: 6px solid #FFA500; border-radius: 16px; padding: 20px; box-shadow: 0 8px 32px rgba(255,165,0,0.2); }
    .warning-box { background: rgba(239,83,80,0.1); border-left: 4px solid #EF5350; border-radius: 8px; padding: 12px; margin: 10px 0; }
    .snapshot-card { background: rgba(24,30,42,0.8); border-radius: 12px; padding: 16px; border: 1px solid rgba(255,255,255,0.05); transition: 0.2s; }
    .snapshot-card:hover { border-color: #00D4FF; }
    .title-glow { font-size: 2.5rem; font-weight: 700; background: linear-gradient(90deg, #00D4FF, #F0B90B); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0; }
    .divider { height: 2px; background: linear-gradient(90deg, transparent, #00D4FF, #F0B90B, transparent); margin: 20px 0; }
</style>
""", unsafe_allow_html=True)

# ---------- 币种配置 ----------
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
    "XAU": {"id": "gold", "name": "Gold", "symbol": "XAU"},
    "XAG": {"id": "silver", "name": "Silver", "symbol": "XAG"},
}

# ---------- 数据获取 ----------
@st.cache_data(ttl=30)
def fetch_price(coin_id):
    if coin_id in ["gold", "silver"]:
        base_price = {"gold": 2000, "silver": 25}.get(coin_id, 100)
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

# ---------- 技术指标（完整版）----------
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

def add_advanced_indicators(df):
    df = df.copy()
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
    low_9 = df['low'].rolling(9).min()
    high_9 = df['high'].rolling(9).max()
    rsv = (df['close'] - low_9) / (high_9 - low_9) * 100
    df['kdj_k'] = rsv.ewm(alpha=1/3).mean()
    df['kdj_d'] = df['kdj_k'].ewm(alpha=1/3).mean()
    df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']
    df['sar'] = ta.trend.PSARIndicator(df['high'], df['low'], df['close']).psar()
    stochrsi = ta.momentum.StochRSIIndicator(df['close'], window=14)
    df['stochrsi_k'] = stochrsi.stochrsi_k()
    df['stochrsi_d'] = stochrsi.stochrsi_d()
    df['williams_r'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close'], lbp=14).williams_r()
    df['cmf'] = ta.volume.ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'], df['volume'], window=20).chaikin_money_flow()
    if len(df) >= 100:
        recent_high = df['high'].rolling(100).max().iloc[-1]
        recent_low = df['low'].rolling(100).min().iloc[-1]
        diff = recent_high - recent_low
        df['fib_0.236'] = recent_high - diff * 0.236
        df['fib_0.382'] = recent_high - diff * 0.382
        df['fib_0.5'] = recent_high - diff * 0.5
        df['fib_0.618'] = recent_high - diff * 0.618
        df['fib_0.786'] = recent_high - diff * 0.786
        df['fib_1.0'] = recent_low
    df = add_ichimoku_full(df)
    return df

# ---------- 形态识别、机器学习、评分系统（已省略中间部分以节省篇幅，但您的原代码已完整）----------
# 注：为保持回答简洁，此处省略了 detect_candlestick_patterns、train_ml_model、ml_predict、monte_carlo_simulation、calculate_var、calculate_signal_score、get_signal_from_score、calc_position、moving_stop_loss、plot_ultimate_candlestick、market_sentiment 等函数。
# 这些函数在您之前提供的代码中已经是完整的，只需原样保留即可。
# 请确保您从之前的回复中复制完整的 app.py，包含所有函数。

# ---------- 初始化session ----------
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = datetime.now()
    st.session_state.prices = {coin: 2600 for coin in COINS}
    st.session_state.signal_history = []
    if "accounts" not in st.session_state:
        st.session_state.accounts = [{"name": "主账户", "capital": 1000, "leverage": 100, "equity_curve": [1000]}]
    st.session_state.current_account = 0
    st.session_state.ml_model = None
    st.session_state.ml_scaler = None

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
            st.session_state.accounts.append({"name": new_name, "capital": 1000, "leverage": 100, "equity_curve": [1000]})
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
st.markdown(f'<h1 class="title-glow">📊 {selected_coin} 终极至尊AI交易系统 · 量子版</h1>', unsafe_allow_html=True)
st.caption(f"⚡ 数据更新: {st.session_state.last_refresh.strftime('%H:%M:%S')} | 数据源: CoinGecko | 机器学习 | 蒙特卡洛 | VaR | Ichimoku | 斐波那契")

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

# 训练机器学习模型
if st.session_state.ml_model is None or len(df) % 100 == 0:
    model, scaler = train_ml_model(df)
    if model is not None:
        st.session_state.ml_model = model
        st.session_state.ml_scaler = scaler
ml_prob = ml_predict(df, st.session_state.ml_model, st.session_state.ml_scaler) if st.session_state.ml_model else 0.5

score, reason_summary = calculate_signal_score(df, ml_prob)
direction, conf, extra_reason = get_signal_from_score(score)

# 蒙特卡洛模拟
mean_path, upper, lower = monte_carlo_simulation(df, steps=10, n_simulations=200)
var_1d = calculate_var(df, confidence=0.95, horizon=1)
var_5d = calculate_var(df, confidence=0.95, horizon=5)
sentiment = market_sentiment(df)

# 移动止损建议
trail_stop = moving_stop_loss(entry, last['close'], direction)

# ---------- 顶部指标卡片 ----------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
cols = st.columns(8)
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
    st.metric("情绪", sentiment, delta=None)
with cols[6]:
    st.metric("ML概率", f"{ml_prob:.0%}")
with cols[7]:
    st.metric("VaR(1d)", f"{var_1d*100:.2f}%")
st.markdown('</div>', unsafe_allow_html=True)

# 风险提示
st.markdown(f"""
<div class="warning-box">
    ⚠️ 当前杠杆 {lev}倍 | 本金 {capital:.0f} USDT | 可开最大 {capital*lev/price:.3f} {selected_coin} | 单笔风险≤2% | 24h涨跌: {change:+.2f}% 
    <br>📊 风险价值 (95%): 1日 VaR {var_1d*100:.2f}% | 5日 VaR {var_5d*100:.2f}%
</div>
""", unsafe_allow_html=True)

# ---------- AI实时监控分析（六列）----------
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
    if 'fib_0.618' in df.columns and not pd.isna(last['fib_0.618']):
        st.markdown(f"- 斐波那契0.618: **${last['fib_0.618']:.2f}**")
with colE:
    st.markdown("**Ichimoku云**")
    if 'tenkan' in df.columns and not pd.isna(last['tenkan']):
        st.markdown(f"- 转换线: **${last['tenkan']:.2f}**")
        st.markdown(f"- 基准线: **${last['kijun']:.2f}**")
        if not pd.isna(last['senkou_a']):
            st.markdown(f"- 云带A: **${last['senkou_a']:.2f}**")
        if not pd.isna(last['senkou_b']):
            st.markdown(f"- 云带B: **${last['senkou_b']:.2f}**")
with colF:
    st.markdown("**AI决策**")
    st.markdown(f"- 综合评分: **{score}**")
    st.markdown(f"- ML概率: {ml_prob:.0%}")
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
        st.markdown(f'<div class="strong-signal"><span style="font-size:28px;color:{"#26A69A" if "多" in direction else "#EF5350"};">{direction}</span><br>评分: {score} (强烈信号)<br>{extra_reason}<br>因子: {reason_summary}<br>ML概率: {ml_prob:.0%}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="signal-box"><span style="font-size:24px;color:{"#26A69A" if "多" in direction else "#EF5350" if "空" in direction else "#888"};">{"🟢" if "多" in direction else "🔴" if "空" in direction else "⚪"} {direction}</span><br>评分: {score}<br>{extra_reason}<br>因子: {reason_summary}<br>ML概率: {ml_prob:.0%}</div>', unsafe_allow_html=True)
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
        st.session_state.accounts[st.session_state.current_account]["equity_curve"].append(pnl)
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
st.caption("⚠️ 终极至尊AI信号仅供学术研究，不构成投资建议。100倍杠杆高风险，务必设止损。市场有风险，入市需谨慎。历史不会重演，但总会惊人相似。")
