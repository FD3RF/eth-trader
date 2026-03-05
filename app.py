import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import time
import random

# ------------------- 页面配置 -------------------
st.set_page_config(
    page_title="5分钟量价交易系统（演示版）",
    page_icon="📊",
    layout="wide"
)

st.title("📈 5分钟合约全自动量价盯盘系统（演示版）")
st.markdown("基于缩量/放量/横盘/突破口诀 + ATR动态风控 | 支持演示模式（无需交易所）")

# ------------------- 初始化session_state -------------------
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()
if 'error_count' not in st.session_state:
    st.session_state.error_count = 0

# ------------------- 侧边栏参数配置 -------------------
with st.sidebar:
    st.header("⚙️ 交易配置")
    
    # 演示模式开关
    demo_mode = st.checkbox("🎮 演示模式（使用模拟数据）", value=False,
                            help="开启后无需连接交易所，使用本地生成的随机K线")
    
    if not demo_mode:
        exchange_name = st.selectbox(
            "选择交易所",
            ["binance", "bybit", "okx", "bitget", "kucoin"],
            index=1,
            help="如果遇到地域屏蔽，请开启演示模式"
        )
        
        symbol = st.text_input("交易对", value="ETH/USDT")
    else:
        exchange_name = "demo"
        symbol = "DEMO/USDT"
    
    refresh_interval = st.slider(
        "自动刷新间隔（秒）", 
        min_value=5, 
        max_value=60, 
        value=10, 
        step=5,
        help="每隔N秒重新生成数据"
    )
    
    st.markdown("---")
    st.subheader("📊 策略参数")
    
    vol_window = st.number_input("成交量均线周期", value=5, min_value=3, max_value=20)
    
    atr_multiplier = st.slider(
        "横盘阈值 (ATR倍数)", 
        min_value=0.3, 
        max_value=1.5, 
        value=0.5, 
        step=0.1,
        help="当价格区间宽度 < ATR * 此倍数 时视为横盘"
    )
    
    confirm_with_close = st.checkbox("突破确认需收盘价在区间外", value=True, 
                                     help="勾选后以收盘价判断突破，减少插针干扰")
    
    risk_reward = st.slider("盈亏比", min_value=1.0, max_value=3.0, value=1.5, step=0.1)
    
    st.markdown("---")
    st.subheader("💰 账户设置")
    balance = st.number_input("账户余额 (U)", value=1000.0, step=100.0)
    risk_percent = st.slider("单笔风险 (%)", min_value=0.5, max_value=5.0, value=2.0, step=0.1) / 100.0

# ------------------- 初始化交易所（仅在非演示模式） -------------------
@st.cache_resource
def init_exchange(exchange_name):
    """初始化交易所客户端，并提供友好错误提示"""
    try:
        exchange_class = getattr(ccxt, exchange_name)
        params = {
            'enableRateLimit': True,
            'timeout': 30000,
        }
        # 现货 vs 期货
        if exchange_name == 'binance':
            params['options'] = {'defaultType': 'spot'}  # 现货更稳定
        else:
            params['options'] = {'defaultType': 'future'}
        
        exchange = exchange_class(params)
        exchange.fetch_time()  # 测试连接
        return exchange
    except Exception as e:
        error_msg = str(e).lower()
        if "restricted location" in error_msg or "cloudfront" in error_msg:
            st.error("🌍 当前 IP 被交易所 CDN 屏蔽，请开启「演示模式」使用模拟数据")
        else:
            st.error(f"交易所初始化失败: {e}")
        return None

exchange = None if demo_mode else init_exchange(exchange_name)

# ------------------- 数据获取函数（真实或模拟） -------------------
@st.cache_data(ttl=refresh_interval)
def fetch_market_data(symbol, limit=100):
    """获取K线数据，演示模式返回模拟数据"""
    if demo_mode:
        return generate_demo_data(limit)
    
    if exchange is None:
        return None
    try:
        # 调整交易对格式
        if exchange_name == 'okx' and '-' not in symbol:
            symbol = symbol.replace('/', '-')
        klines = exchange.fetch_ohlcv(
            symbol=symbol, 
            timeframe='5m', 
            limit=limit
        )
        if not klines or len(klines) < 20:
            st.warning("获取的数据不足20条，请稍后重试")
            return None
            
        df = pd.DataFrame(
            klines, 
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('datetime', inplace=True)
        st.session_state.error_count = 0
        return df
    except Exception as e:
        st.session_state.error_count += 1
        st.error(f"数据获取失败 (尝试 {st.session_state.error_count}/3): {e}")
        if st.session_state.error_count >= 3:
            st.warning("连续失败，建议开启「演示模式」")
        return None

def generate_demo_data(n_candles=100):
    """生成模拟K线数据"""
    np.random.seed(int(time.time()) % 1000)  # 每次刷新不同
    base_price = 2000 + np.random.randn() * 50
    closes = []
    for i in range(n_candles):
        change = np.random.randn() * 5
        if i == 0:
            closes.append(base_price)
        else:
            closes.append(closes[-1] + change)
    
    # 生成 OHLCV
    dates = pd.date_range(end=datetime.now(), periods=n_candles, freq='5min')
    df = pd.DataFrame(index=dates)
    df['close'] = closes
    df['open'] = df['close'].shift(1) + np.random.randn(n_candles) * 2
    df['high'] = df[['open', 'close']].max(axis=1) + abs(np.random.randn(n_candles) * 3)
    df['low'] = df[['open', 'close']].min(axis=1) - abs(np.random.randn(n_candles) * 3)
    df['volume'] = np.abs(np.random.randn(n_candles) * 1000 + 5000)
    df['timestamp'] = dates.astype(np.int64) // 10**6
    return df

# ------------------- 手动计算技术指标 -------------------
def calculate_atr(df, period=14):
    high, low, close = df['high'], df['low'], df['close']
    df['tr'] = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - close.shift()),
            np.abs(low - close.shift())
        )
    )
    df['atr'] = df['tr'].rolling(window=period, min_periods=1).mean()
    return df

def calculate_indicators(df):
    df = df.copy()
    df = calculate_atr(df, 14)
    df['ma_short'] = df['close'].rolling(5).mean()
    df['ma_long'] = df['close'].rolling(20).mean()
    df['vol_ma'] = df['volume'].rolling(vol_window).mean()
    return df

# ------------------- 状态识别函数（与之前相同） -------------------
def detect_volume_state(df):
    current_vol = df['volume'].iloc[-1]
    avg_vol = df['vol_ma'].iloc[-2] if len(df) > 1 and not pd.isna(df['vol_ma'].iloc[-2]) else current_vol
    if current_vol < avg_vol * 0.6:
        return "缩量"
    elif current_vol > avg_vol * 1.5:
        if current_vol > avg_vol * 3:
            return "巨量"
        else:
            return "放量"
    else:
        return "正常量"

def detect_range_market(df, lookback=10, atr_multiplier=0.5):
    if len(df) < lookback:
        return "非横盘"
    recent = df.iloc[-lookback:]
    high_max = recent['high'].max()
    low_min = recent['low'].min()
    range_width = high_max - low_min
    current_atr = df['atr'].iloc[-1]
    if pd.isna(current_atr) or current_atr == 0:
        return "非横盘"
    current_price = df['close'].iloc[-1]
    
    if range_width < current_atr * atr_multiplier:
        if current_price < low_min + range_width * 0.3:
            return "低位横盘"
        elif current_price > high_max - range_width * 0.3:
            return "高位横盘"
        else:
            return "横盘"
    else:
        return "非横盘"

def get_key_levels(df, lookback=20):
    if len(df) < lookback:
        lookback = len(df)
    recent = df.iloc[-lookback:]
    return {'high': recent['high'].max(), 'low': recent['low'].min()}

def detect_breakout(df, key_levels, use_close=False):
    current = df.iloc[-1]
    if use_close:
        price_for_high = current['close']
        price_for_low = current['close']
    else:
        price_for_high = current['high']
        price_for_low = current['low']
    
    results = []
    if price_for_high > key_levels['high']:
        results.append("向上突破")
    if price_for_low < key_levels['low']:
        results.append("向下突破")
    
    if abs(current['close'] - key_levels['low']) / key_levels['low'] < 0.005 and current['close'] >= key_levels['low']:
        results.append("回踩前低")
    if abs(current['close'] - key_levels['high']) / key_levels['high'] < 0.005 and current['close'] <= key_levels['high']:
        results.append("反弹前高")
    
    return results

def detect_volume_trap(df):
    current = df.iloc[-1]
    body = abs(current['close'] - current['open'])
    atr = current['atr']
    if pd.isna(atr) or atr == 0:
        return None
    
    if body < atr * 0.3:
        vol_state = detect_volume_state(df)
        if "放量" in vol_state or "巨量" in vol_state:
            if current['close'] > current['open']:
                return "放量滞涨（可能诱多）"
            else:
                return "放量滞跌（可能诱空）"
    return None

def detect_trend(df, lookback=30):
    if len(df) < lookback:
        lookback = len(df)
    close_prices = df['close'].values[-lookback:]
    if len(close_prices) < 5:
        return "震荡"
    x = np.arange(len(close_prices))
    slope = np.polyfit(x, close_prices, 1)[0]
    slope_pct = slope / close_prices.mean() * 100
    if slope_pct > 0.02:
        return "上升趋势"
    elif slope_pct < -0.02:
        return "下降趋势"
    else:
        return "震荡"

# ------------------- 信号生成函数 -------------------
def generate_signals(df, trend, range_state, trap_signal, breakout, vol_state):
    long_reasons = []
    short_reasons = []
    score = 0
    warnings = []
    
    # 做多条件
    if vol_state in ["缩量", "正常量"] and "回踩前低" in breakout:
        if "向上突破" in breakout:
            long_reasons.append("缩量回踩前低 + 向上突破 → 强做多信号")
            score += 4
        else:
            long_reasons.append("缩量回踩前低（等待放量突破）")
            score += 1
    
    if range_state == "低位横盘" and vol_state in ["缩量", "正常量"] and "向上突破" in breakout:
        long_reasons.append("低位缩量横盘 + 向上突破 → 平台突破做多")
        score += 3
    
    if "向上突破" in breakout and vol_state in ["放量", "巨量"]:
        long_reasons.append("放量向上突破")
        score += 2
    
    if trap_signal == "放量滞跌（可能诱空）" and "回踩前低" in breakout:
        long_reasons.append("放量暴跌不破 → 诱空陷阱，反向做多")
        score += 3
    
    # 做空条件
    if vol_state in ["缩量", "正常量"] and "反弹前高" in breakout:
        if "向下突破" in breakout:
            short_reasons.append("缩量反弹前高 + 向下突破 → 强做空信号")
            score += 4
        else:
            short_reasons.append("缩量反弹前高（等待放量跌破）")
            score += 1
    
    if range_state == "高位横盘" and vol_state in ["缩量", "正常量"] and "向下突破" in breakout:
        short_reasons.append("高位缩量横盘 + 向下突破 → 平台突破做空")
        score += 3
    
    if "向下突破" in breakout and vol_state in ["放量", "巨量"]:
        short_reasons.append("放量向下突破")
        score += 2
    
    if trap_signal == "放量滞涨（可能诱多）" and "反弹前高" in breakout:
        short_reasons.append("放量暴涨不破 → 诱多陷阱，反向做空")
        score += 3
    
    # 假突破预警
    if "向上突破" in breakout and vol_state not in ["放量", "巨量"]:
        warnings.append("⚠️ 无量向上突破，可能假突破")
    if "向下突破" in breakout and vol_state not in ["放量", "巨量"]:
        warnings.append("⚠️ 无量向下突破，可能假突破")
    
    # 趋势过滤
    if trend == "上升趋势":
        if not long_reasons:
            return "⏳ 观望（上升趋势无做多信号）", [], 0
        final_signal = "📈 做多"
        reasons = long_reasons
    elif trend == "下降趋势":
        if not short_reasons:
            return "⏳ 观望（下降趋势无做空信号）", [], 0
        final_signal = "📉 做空"
        reasons = short_reasons
    else:  # 震荡
        if long_reasons and not short_reasons:
            final_signal = "📈 做多"
            reasons = long_reasons
        elif short_reasons and not long_reasons:
            final_signal = "📉 做空"
            reasons = short_reasons
        elif long_reasons and short_reasons:
            final_signal = "🤔 观望（信号矛盾）"
            reasons = long_reasons + short_reasons
        else:
            final_signal = "⏳ 观望（无明确信号）"
            reasons = []
    
    strength = min(5, max(0, score))
    full_reasons = reasons + warnings
    return final_signal, full_reasons, strength

# ------------------- 仓位计算 -------------------
def calculate_position(entry, stop_loss, balance, risk_percent):
    risk_distance = abs(entry - stop_loss)
    if risk_distance <= 0:
        return 0, 0
    risk_amount = balance * risk_percent
    position_size = risk_amount / risk_distance
    return position_size, risk_amount

# ------------------- 主界面 -------------------
chart_placeholder = st.empty()
metrics_placeholder = st.empty()
signal_placeholder = st.empty()

# 获取数据
df = fetch_market_data(symbol)

if df is not None and len(df) > 20:
    df = calculate_indicators(df)
    
    vol_state = detect_volume_state(df)
    range_state = detect_range_market(df, atr_multiplier=atr_multiplier)
    trap_signal = detect_volume_trap(df)
    trend = detect_trend(df)
    key_levels = get_key_levels(df)
    breakout = detect_breakout(df, key_levels, use_close=confirm_with_close)
    
    signal, reasons, strength = generate_signals(
        df, trend, range_state, trap_signal, breakout, vol_state
    )
    
    current_price = df['close'].iloc[-1]
    current_atr = df['atr'].iloc[-1] if not pd.isna(df['atr'].iloc[-1]) else 0
    
    # 显示K线图
    with chart_placeholder.container():
        fig = go.Figure(data=[go.Candlestick(
            x=df.index[-30:],
            open=df['open'][-30:],
            high=df['high'][-30:],
            low=df['low'][-30:],
            close=df['close'][-30:],
            name='K线'
        )])
        
        fig.add_trace(go.Scatter(x=df.index[-30:], y=df['ma_short'][-30:],
                                  line=dict(color='orange', width=1), name='MA5'))
        fig.add_trace(go.Scatter(x=df.index[-30:], y=df['ma_long'][-30:],
                                  line=dict(color='blue', width=1), name='MA20'))
        
        fig.add_hline(y=key_levels['high'], line_dash="dash", line_color="red", annotation_text="前高")
        fig.add_hline(y=key_levels['low'], line_dash="dash", line_color="green", annotation_text="前低")
        
        if range_state != "非横盘" and len(df) >= 10:
            recent_high = df['high'].iloc[-10:].max()
            recent_low = df['low'].iloc[-10:].min()
            fig.add_hrect(y0=recent_low, y1=recent_high, line_width=0, fillcolor="gray", opacity=0.2, annotation_text="横盘区间")
        
        fig.update_layout(title=f"{symbol} 5分钟K线", height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # 显示实时指标
    with metrics_placeholder.container():
        cols = st.columns(6)
        cols[0].metric("当前价格", f"{current_price:.2f}")
        cols[1].metric("ATR(14)", f"{current_atr:.2f}" if current_atr > 0 else "N/A")
        cols[2].metric("成交量", vol_state)
        cols[3].metric("趋势", trend)
        cols[4].metric("横盘状态", range_state)
        cols[5].metric("突破信号", ", ".join(breakout) if breakout else "无")
    
    # 显示交易信号
    with signal_placeholder.container():
        st.markdown("---")
        if "📈 做多" in signal:
            st.success(f"### {signal}")
        elif "📉 做空" in signal:
            st.error(f"### {signal}")
        else:
            st.info(f"### {signal}")
        
        if signal not in ["🤔 观望（信号矛盾）", "⏳ 观望（无明确信号）",
                         "⏳ 观望（上升趋势无做多信号）", "⏳ 观望（下降趋势无做空信号）"]:
            stars = "★" * strength + "☆" * (5 - strength)
            st.markdown(f"**信号强度**：{stars} ({strength}/5)")
            st.progress(strength / 5)
            if strength >= 4:
                st.success("💪 强信号，可考虑进场")
            elif strength >= 3:
                st.info("👍 中强信号，谨慎尝试")
            else:
                st.warning("👀 一般信号，建议等待确认")
        
        if reasons:
            st.markdown("**信号依据**：")
            for r in reasons:
                st.markdown(f"- {r}")
        
        if trap_signal:
            st.warning(f"🔔 陷阱识别：{trap_signal}")
        
        # 交易计划
        if signal in ["📈 做多", "📉 做空"] and current_atr > 0:
            direction = 'long' if "📈 做多" in signal else 'short'
            if direction == 'long':
                stop_loss = current_price - current_atr * 1.0
                target1 = current_price + current_atr * risk_reward
                target2 = current_price + current_atr * 2.0
            else:
                stop_loss = current_price + current_atr * 1.0
                target1 = current_price - current_atr * risk_reward
                target2 = current_price - current_atr * 2.0
            
            pos_size, risk_amt = calculate_position(current_price, stop_loss, balance, risk_percent)
            
            st.markdown("---")
            st.subheader("📐 交易计划")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**入场**：{current_price:.2f}")
                st.markdown(f"**止损**：{stop_loss:.2f}")
            with col2:
                st.markdown(f"**目标1**：{target1:.2f} (1:{risk_reward})")
                st.markdown(f"**目标2**：{target2:.2f}")
            with col3:
                st.markdown(f"**建议仓位**：{pos_size:.2f} 张")
                st.markdown(f"**风险金额**：{risk_amt:.2f} U")
        
        # 内功心法
        st.markdown("---")
        st.subheader("🧘 内功心法")
        tips = []
        if range_state != "非横盘":
            tips.append("📌 当前处于横盘，等待突破再交易")
        if vol_state == "缩量" and "回踩前低" in breakout:
            tips.append("📌 缩量回踩前低，空头衰竭，等待放量做多")
        if vol_state == "缩量" and "反弹前高" in breakout:
            tips.append("📌 缩量反弹前高，多头乏力，等待放量做空")
        if "无量向上突破" in reasons:
            tips.append("📌 无量突破，警惕假突破，确认收盘价")
        if "放量滞涨" in str(trap_signal):
            tips.append("📌 放量滞涨，主力可能出货，避免追多")
        if "放量滞跌" in str(trap_signal):
            tips.append("📌 放量滞跌，主力可能吸筹，避免追空")
        if not tips:
            tips.append("📌 无明显警示，按信号执行，严格止损")
        for tip in tips:
            st.markdown(tip)

else:
    st.warning("数据不足，请检查连接或开启「演示模式」")
    if exchange is None and not demo_mode:
        st.error("交易所初始化失败，请尝试开启演示模式")

st.caption(f"最后更新：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 自动刷新
time.sleep(refresh_interval)
st.rerun()
