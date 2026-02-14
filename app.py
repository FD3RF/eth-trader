import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
import time
from datetime import datetime, timedelta
import random

st.set_page_config(page_title="至尊AI交易终端 · 多币种", layout="wide", initial_sidebar_state="expanded")

# ---------- 极致视觉CSS ----------
st.markdown("""
<style>
    /* 全局背景渐变 */
    .stApp {
        background: linear-gradient(135deg, #0B0E14 0%, #141A24 100%);
        color: #F0F4FA;
    }
    
    /* 玻璃态卡片 */
    .glass-card {
        background: rgba(20, 28, 40, 0.75);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }
    
    /* 指标卡片 */
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
    
    /* 信号框 */
    .signal-box {
        background: rgba(26, 34, 48, 0.9);
        backdrop-filter: blur(5px);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(255, 215, 0, 0.3);
        box-shadow: 0 8px 32px rgba(255, 215, 0, 0.1);
    }
    
    /* 强烈信号高亮 */
    .strong-signal {
        background: linear-gradient(145deg, #2A2418, #1F1A12);
        border-left: 6px solid #FFA500;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 8px 32px rgba(255, 165, 0, 0.2);
    }
    
    /* 警告框 */
    .warning-box {
        background: rgba(239, 83, 80, 0.1);
        border-left: 4px solid #EF5350;
        border-radius: 8px;
        padding: 12px;
        margin: 10px 0;
    }
    
    /* 快照卡片 */
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
    
    /* 自定义标题 */
    .title-glow {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00D4FF, #F0B90B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    
    /* 分割线 */
    .divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #00D4FF, #F0B90B, transparent);
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# ---------- 币种配置 ----------
COINS = {
    "ETH": {"id": "ethereum", "name": "Ethereum", "symbol": "ETH"},
    "BTC": {"id": "bitcoin", "name": "Bitcoin", "symbol": "BTC"},
    "SOL": {"id": "solana", "name": "Solana", "symbol": "SOL"}
}

# ---------- CoinGecko 免费数据源 ----------
@st.cache_data(ttl=30)
def fetch_price(coin_id):
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd&include_24hr_change=true"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        return data[coin_id]['usd'], data[coin_id]['usd_24h_change']
    except:
        return None, None

def generate_klines(price, interval_min=5, limit=200):
    """基于当前价格生成模拟K线"""
    now = datetime.now()
    times = [now - timedelta(minutes=i*interval_min) for i in range(limit)][::-1]
    returns = np.random.randn(limit) * 0.002
    price_series = price * np.exp(np.cumsum(returns))
    price_series = price_series * (price / price_series[-1])
    
    closes = price_series
    opens = [closes[i-1] if i>0 else closes[0]*0.999 for i in range(limit)]
    highs = np.maximum(opens, closes) * 1.002
    lows = np.minimum(opens, closes) * 0.998
    vols = np.random.uniform(100, 500, limit)
    
    return pd.DataFrame({
        "time": times,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": vols
    })

def add_advanced_indicators(df):
    """添加高级技术指标"""
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
    return df

def detect_candlestick_patterns(df):
    """识别K线形态（与之前相同）"""
    patterns = []
    if len(df) < 3:
        return patterns
    last = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3] if len(df) > 2 else None
    
    # 吞没形态
    if prev2 is not None:
        if last['close'] > last['open'] and prev['close'] < prev['open']:
            if last['close'] > prev['open'] and last['open'] < prev['close']:
                patterns.append("📈 看涨吞没")
        if last['close'] < last['open'] and prev['close'] > prev['open']:
            if last['close'] < prev['open'] and last['open'] > prev['close']:
                patterns.append("📉 看跌吞没")
    
    # 十字星
    body = abs(last['close'] - last['open'])
    if body < (last['high'] - last['low']) * 0.1:
        patterns.append("➕ 十字星")
    
    # 锤子线/上吊线
    real_body = abs(last['close'] - last['open'])
    lower_shadow = last['open'] - last['low'] if last['open'] > last['close'] else last['close'] - last['low']
    upper_shadow = last['high'] - last['close'] if last['open'] > last['close'] else last['high'] - last['open']
    if lower_shadow > 2 * real_body and upper_shadow < real_body:
        if last['close'] > last['open']:
            patterns.append("🔨 锤子线 (看涨)")
        else:
            patterns.append("🪢 上吊线 (看跌)")
    
    # 晨星/暮星
    if prev2 is not None:
        if prev2['close'] < prev2['open'] and prev['close'] < prev['open'] and last['close'] > last['open']:
            if last['close'] > (prev2['open'] + prev2['close'])/2:
                patterns.append("🌅 晨星形态")
        if prev2['close'] > prev2['open'] and prev['close'] > prev['open'] and last['close'] < last['open']:
            if last['close'] < (prev2['open'] + prev2['close'])/2:
                patterns.append("🌆 暮星形态")
    
    return patterns

def calculate_signal_score(df):
    """多因子评分系统"""
    if df.empty or len(df) < 30:
        return 0, "数据不足"
    last = df.iloc[-1]
    score = 0
    reasons = []
    
    # 趋势因子 (30)
    if not pd.isna(last['ma20']) and not pd.isna(last['ma60']):
        if last['ma20'] > last['ma60']:
            score += 20
            reasons.append("MA20>MA60")
        else:
            score -= 20
            reasons.append("MA20<MA60")
    if not pd.isna(last['adx']):
        if last['adx'] > 25:
            score += 10 if score>0 else -10
            reasons.append(f"ADX{last['adx']:.0f}")
    
    # 动量因子 (40)
    if not pd.isna(last['rsi']):
        if last['rsi'] < 30:
            score += 30
            reasons.append("RSI超卖")
        elif last['rsi'] > 70:
            score -= 30
            reasons.append("RSI超买")
        elif last['rsi'] > 50:
            score += 10
            reasons.append("RSI>50")
        else:
            score -= 10
            reasons.append("RSI<50")
    
    if not pd.isna(last['macd']) and not pd.isna(last['macd_signal']):
        if last['macd'] > last['macd_signal']:
            score += 15
            reasons.append("MACD金叉")
        else:
            score -= 15
            reasons.append("MACD死叉")
    
    if not pd.isna(last['cci']):
        if last['cci'] > 100:
            score += 10
            reasons.append("CCI超买")
        elif last['cci'] < -100:
            score -= 10
            reasons.append("CCI超卖")
    
    # 成交量因子 (20)
    if not pd.isna(last['mfi']):
        if last['mfi'] < 20:
            score += 15
            reasons.append("MFI超卖")
        elif last['mfi'] > 80:
            score -= 15
            reasons.append("MFI超买")
    
    # 形态因子 (10)
    patterns = detect_candlestick_patterns(df)
    for p in patterns:
        if "看涨" in p or "锤子" in p or "晨星" in p:
            score += 10
            reasons.append(p)
        elif "看跌" in p or "上吊" in p or "暮星" in p:
            score -= 10
            reasons.append(p)
    
    score = max(-100, min(100, score))
    return score, ", ".join(reasons[:3])

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

def plot_professional_candlestick(df, selected_coin, interval):
    """币安风格专业K线图（增强版）"""
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        row_heights=[0.5, 0.25, 0.25],
        vertical_spacing=0.03,
        subplot_titles=(
            f"{selected_coin}/USDT {interval} K线图",
            "RSI (14) & MACD",
            "成交量"
        )
    )
    
    # 主图K线
    fig.add_trace(go.Candlestick(
        x=df.time,
        open=df.open,
        high=df.high,
        low=df.low,
        close=df.close,
        name="K线",
        increasing_line_color='#26A69A',
        decreasing_line_color='#EF5350',
        showlegend=True,
        hoverlabel=dict(bgcolor='#1E1F2A', font_size=12)
    ), row=1, col=1)
    
    # 均线
    fig.add_trace(go.Scatter(x=df.time, y=df.ma20, name="MA20", line=dict(color='#F0B90B', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.time, y=df.ma60, name="MA60", line=dict(color='#1890FF', width=1.5)), row=1, col=1)
    
    # 布林带
    fig.add_trace(go.Scatter(x=df.time, y=df.bb_upper, name="布林上轨", line=dict(color='#888888', width=1, dash='dash'), opacity=0.5), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.time, y=df.bb_lower, name="布林下轨", line=dict(color='#888888', width=1, dash='dash'), opacity=0.5), row=1, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=df.time, y=df.rsi, name="RSI(14)", line=dict(color='#9B59B6', width=2)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="rgba(239, 83, 80, 0.5)", row=2)
    fig.add_hline(y=30, line_dash="dash", line_color="rgba(38, 166, 154, 0.5)", row=2)
    fig.add_hline(y=50, line_dash="dot", line_color="#888888", row=2, opacity=0.3)
    
    # MACD (叠加在RSI图上，使用次坐标轴)
    fig.add_trace(go.Scatter(x=df.time, y=df.macd, name="MACD", line=dict(color='#FFB347', width=1.5)), row=2, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=df.time, y=df.macd_signal, name="信号线", line=dict(color='#FF6B6B', width=1.5)), row=2, col=1, secondary_y=False)
    
    # 成交量（按涨跌着色）
    volume_colors = ['#26A69A' if close >= open else '#EF5350' 
                     for close, open in zip(df['close'], df['open'])]
    fig.add_trace(go.Bar(
        x=df.time, y=df.volume, 
        name="成交量",
        marker_color=volume_colors,
        marker_line_width=0,
        opacity=0.8,
        showlegend=False
    ), row=3, col=1)
    
    # 十字光标与布局
    fig.update_layout(
        template="plotly_dark",
        xaxis=dict(rangeslider=dict(visible=False), type='date', showspikes=True, spikecolor="white", spikethickness=1, spikemode="across"),
        yaxis=dict(showspikes=True, spikecolor="white", spikethickness=1, spikemode="across"),
        hovermode='x unified',
        hoverdistance=100,
        spikedistance=1000,
        height=700,
        margin=dict(l=50, r=20, t=50, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0.5)", font=dict(size=11))
    )
    
    fig.update_yaxes(title_text="价格 (USDT)", row=1, col=1, tickformat=".2f")
    fig.update_yaxes(title_text="RSI/MACD", row=2, col=1)
    fig.update_yaxes(title_text="成交量", row=3, col=1)
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    
    return fig

# 模拟市场情绪（基于最新指标）
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

# ---------- 初始化session ----------
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = datetime.now()
    st.session_state.prices = {coin: 2600 for coin in COINS}
    st.session_state.trade_history = []  # 存储模拟交易记录，用于统计
    st.session_state.equity_curve = [1000]  # 模拟账户净值曲线

# ---------- 侧边栏 ----------
with st.sidebar:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("## ⚙️ 至尊控制台")
    st.markdown("---")
    
    selected_coin = st.selectbox("选择币种", list(COINS.keys()), index=0)
    coin_id = COINS[selected_coin]["id"]
    
    interval = st.selectbox("K线周期", ["1m","5m","15m","1h","4h"], index=1)
    auto = st.checkbox("自动刷新 (30秒)", True)
    
    st.markdown("---")
    st.subheader("💰 资金管理")
    capital = st.number_input("本金 (USDT)", 10, value=1000, step=100)
    lev = st.select_slider("杠杆倍数", [10,20,50,100], value=100)
    
    price, _ = fetch_price(coin_id)
    if price:
        st.session_state.prices[selected_coin] = price
    current_price = st.session_state.prices.get(selected_coin, 2600)
    
    entry = st.number_input("入场价", value=current_price, step=1.0, format="%.2f")
    stop = st.number_input("止损价", value=current_price*0.99, step=1.0, format="%.2f")
    
    if st.button("🔄 刷新数据", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- 主界面 ----------
st.markdown(f'<h1 class="title-glow">📊 {selected_coin} 至尊AI交易终端</h1>', unsafe_allow_html=True)
st.caption(f"⚡ 数据更新: {st.session_state.last_refresh.strftime('%H:%M:%S')} | 数据源: CoinGecko | 信号仅供参考")

# 获取实时价格
price, change = fetch_price(coin_id)
if price:
    st.session_state.prices[selected_coin] = price
else:
    price = st.session_state.prices.get(selected_coin, 2600)

# 生成K线数据
interval_min = int(interval.replace('m','').replace('h','60')) if 'm' in interval or 'h' in interval else 5
df = generate_klines(price, interval_min)
df = add_advanced_indicators(df)
last = df.iloc[-1]
prev = df.iloc[-2]

# 计算信号
score, reason_summary = calculate_signal_score(df)
direction, conf, extra_reason = get_signal_from_score(score)

# 市场情绪
sentiment = market_sentiment(df)

# ---------- 顶部指标卡片（增强版）----------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
cols = st.columns(6)
with cols[0]:
    delta = last['close'] - prev['close']
    st.metric(f"{selected_coin}/USDT", f"${last['close']:.2f}", f"{delta:+.2f}")
with cols[1]:
    st.metric("RSI(14)", f"{last['rsi']:.1f}")
with cols[2]:
    st.metric("ADX", f"{last['adx']:.1f}")
with cols[3]:
    st.metric("ATR%", f"{last['natr']:.2f}%")
with cols[4]:
    st.metric("成交量", f"{last['volume']:.0f}")
with cols[5]:
    st.metric("情绪", sentiment, delta=None)
st.markdown('</div>', unsafe_allow_html=True)

# 风险提示
st.markdown(f"""
<div class="warning-box">
    ⚠️ 当前杠杆 {lev}倍 | 本金 {capital:.0f} USDT | 可开最大 {capital*lev/price:.3f} {selected_coin} | 单笔风险≤2% | 24h涨跌: {change:+.2f}% 
</div>
""", unsafe_allow_html=True)

# ---------- AI实时监控分析（三列玻璃卡片）----------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("📊 AI实时监控分析")
colA, colB, colC = st.columns(3)
with colA:
    st.markdown("**趋势状态**")
    trend = "多头" if last['ma20'] > last['ma60'] else "空头" if last['ma20'] < last['ma60'] else "震荡"
    st.markdown(f"- 均线排列: **{trend}**")
    st.markdown(f"- ADX趋势强度: **{'强趋势' if last['adx']>25 else '弱趋势/震荡'}**")
    st.markdown(f"- 价格相对布林: **{'上轨附近' if last['close']>last['bb_upper'] else '下轨附近' if last['close']<last['bb_lower'] else '中轨'}**")
with colB:
    st.markdown("**动量指标**")
    st.markdown(f"- RSI: **{last['rsi']:.1f}** ({'超买' if last['rsi']>70 else '超卖' if last['rsi']<30 else '中性'})")
    st.markdown(f"- CCI: **{last['cci']:.1f}**")
    st.markdown(f"- MFI: **{last['mfi']:.1f}**")
with colC:
    st.markdown("**支撑/阻力**")
    support = last['bb_lower'] if not pd.isna(last['bb_lower']) else last['close']*0.98
    resistance = last['bb_upper'] if not pd.isna(last['bb_upper']) else last['close']*1.02
    st.markdown(f"- 支撑: **${support:.2f}**")
    st.markdown(f"- 阻力: **${resistance:.2f}**")
    st.markdown(f"- 24h涨跌: **{change:+.2f}%**" if change else "-")
st.markdown('</div>', unsafe_allow_html=True)

# ---------- K线图 ----------
st.subheader(f"{interval} K线图")
fig = plot_professional_candlestick(df, selected_coin, interval)
st.plotly_chart(fig, use_container_width=True)

# ---------- AI信号与交易策略（双列玻璃卡）----------
colL, colR = st.columns(2)
with colL:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🎯 AI智能信号")
    if "强烈" in direction:
        st.markdown(f'<div class="strong-signal"><span style="font-size:28px;color:{"#26A69A" if "多" in direction else "#EF5350"};">{direction}</span><br>评分: {score} (强烈信号)<br>{extra_reason}<br>因子: {reason_summary}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="signal-box"><span style="font-size:24px;color:{"#26A69A" if "多" in direction else "#EF5350" if "空" in direction else "#888"};">{"🟢" if "多" in direction else "🔴" if "空" in direction else "⚪"} {direction}</span><br>评分: {score}<br>{extra_reason}<br>因子: {reason_summary}</div>', unsafe_allow_html=True)
    
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
        
        # 更新模拟净值曲线（简化）
        new_equity = st.session_state.equity_curve[-1] + pnl
        st.session_state.equity_curve.append(new_equity)
    else:
        st.info("输入有效入场价和止损价计算盈亏")
    st.markdown('</div>', unsafe_allow_html=True)

with colY:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("**📈 模拟账户净值曲线**")
    if len(st.session_state.equity_curve) > 1:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=list(range(len(st.session_state.equity_curve))),
            y=st.session_state.equity_curve,
            mode='lines',
            line=dict(color='#00D4FF', width=2),
            fill='tozeroy',
            fillcolor='rgba(0,212,255,0.1)'
        ))
        fig2.update_layout(
            template="plotly_dark",
            height=150,
            margin=dict(l=0, r=0, t=10, b=0),
            showlegend=False,
            xaxis=dict(showticklabels=False),
            yaxis=dict(title="净值")
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.write("暂无数据")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- 其他币种快照（增强版）----------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("📌 其他币种快照")
cols = st.columns(3)
for i, (coin_name, coin_info) in enumerate(COINS.items()):
    if coin_name == selected_coin:
        continue
    with cols[i % 3]:
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

# 页脚
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.caption("⚠️ 至尊AI信号基于技术指标和形态识别，不构成投资建议。100倍杠杆高风险，务必设止损。市场有风险，入市需谨慎。")
