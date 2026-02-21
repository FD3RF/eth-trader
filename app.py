import streamlit as st
import ccxt
import pandas as pd
import pandas_ta as ta
import time
from datetime import datetime

# --- 配置区 ---
SYMBOL = 'ETH/USDT'
LEVERAGE = 100
ST_REFRESH = 5  # 刷新频率(秒)
STOP_PERCENT = 0.003  # 0.3% 固定止损
PROFIT_RATIO = 2.0     # 止盈为止损距离的2倍

st.set_page_config(page_title="ETH 100x AI Monitor", layout="wide")
st.title(f"🚀 {SYMBOL} 100x 短线监控器 (5分钟主周期)")
st.caption("数据源：Binance · 每5秒刷新 · 信号出现时弹窗提醒 · 无真实下单")

# 初始化交易所（公开数据，无需密钥）
exchange = ccxt.binance({
    'enableRateLimit': True,
    'options': {'defaultType': 'future'}
})

def fetch_data():
    """获取三个周期的K线数据并计算指标"""
    # 获取数据
    bars_5m = exchange.fetch_ohlcv(SYMBOL, timeframe='5m', limit=100)
    bars_15m = exchange.fetch_ohlcv(SYMBOL, timeframe='15m', limit=100)
    bars_1h = exchange.fetch_ohlcv(SYMBOL, timeframe='1h', limit=100)
    
    # 转换为 DataFrame
    df5 = pd.DataFrame(bars_5m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df15 = pd.DataFrame(bars_15m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df1h = pd.DataFrame(bars_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # 转换时间戳
    for df in [df5, df15, df1h]:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # 计算指标（使用 pandas_ta）
    # 5分钟指标
    df5['ema9'] = ta.ema(df5['close'], length=9)
    df5['ema21'] = ta.ema(df5['close'], length=21)
    df5['rsi'] = ta.rsi(df5['close'], length=14)
    df5['atr'] = ta.atr(df5['high'], df5['low'], df5['close'], length=14)
    df5['vwap'] = ta.vwap(df5['high'], df5['low'], df5['close'], df5['volume'])
    df5['volume_ma5'] = df5['volume'].rolling(5).mean()
    df5['volume_ratio'] = df5['volume'] / df5['volume_ma5'].shift(1)
    
    # 15分钟和1小时趋势指标（EMA50）
    df15['ema50'] = ta.ema(df15['close'], length=50)
    df1h['ema50'] = ta.ema(df1h['close'], length=50)
    
    return df5, df15, df1h

def calculate_confidence(df5, row, trend_up):
    """计算AI信心分（0-100）"""
    score = 0
    
    # 趋势得分（基于15分钟和1小时EMA50）
    if trend_up:
        score += 30
    else:
        score += 0  # 逆势不加分，但也不扣分
    
    # 动能得分（EMA金叉/死叉）
    if row['ema9'] > row['ema21']:
        score += 30
    
    # 价格位置（相对VWAP）
    if row['close'] > row['vwap']:
        score += 20
    
    # 成交量放大（>1.5倍）
    if row['volume_ratio'] > 1.5:
        score += 20
    elif row['volume_ratio'] > 1.2:
        score += 10
    
    # RSI 辅助（避免超买超卖）
    if 30 < row['rsi'] < 70:
        score += 10  # 健康区间加分
    
    return min(score, 100)

def get_signal():
    """主信号检测函数"""
    df5, df15, df1h = fetch_data()
    
    if len(df5) < 50 or len(df15) < 50 or len(df1h) < 50:
        return None, None, None, None, df5, df15, df1h
    
    last = df5.iloc[-1]
    prev = df5.iloc[-2]
    
    # 趋势判断（基于15分钟和1小时EMA50）
    trend_up = (df15['close'].iloc[-1] > df15['ema50'].iloc[-1]) and (df1h['close'].iloc[-1] > df1h['ema50'].iloc[-1])
    trend_down = (df15['close'].iloc[-1] < df15['ema50'].iloc[-1]) and (df1h['close'].iloc[-1] < df1h['ema50'].iloc[-1])
    
    # 计算信心分
    confidence = calculate_confidence(df5, last, trend_up)
    
    # 信号条件
    long_condition = (
        trend_up and
        prev['ema9'] <= prev['ema21'] and
        last['ema9'] > last['ema21'] and
        last['close'] > last['vwap'] and
        last['volume_ratio'] > 1.2
    )
    
    short_condition = (
        trend_down and
        prev['ema9'] >= prev['ema21'] and
        last['ema9'] < last['ema21'] and
        last['close'] < last['vwap'] and
        last['volume_ratio'] > 1.2
    )
    
    if long_condition and confidence >= 80:
        direction = '多'
        entry = last['close']
        stop_loss = entry * (1 - STOP_PERCENT)
        take_profit = entry * (1 + STOP_PERCENT * PROFIT_RATIO)
        return direction, entry, stop_loss, take_profit, confidence, df5, df15, df1h
    
    elif short_condition and confidence >= 80:
        direction = '空'
        entry = last['close']
        stop_loss = entry * (1 + STOP_PERCENT)
        take_profit = entry * (1 - STOP_PERCENT * PROFIT_RATIO)
        return direction, entry, stop_loss, take_profit, confidence, df5, df15, df1h
    
    else:
        return None, None, None, None, confidence, df5, df15, df1h

# 会话状态保存上一个信号，用于弹窗判断
if 'last_signal' not in st.session_state:
    st.session_state.last_signal = {'direction': None, 'entry': None, 'time': None}

# 主循环（Streamlit 会不断重新运行，我们用自动刷新）
# 但为了保持简洁，我们使用 st_autorefresh 并让代码每次运行都获取新数据

# 放置占位符
placeholder = st.empty()

# 获取最新信号
direction, entry, sl, tp, confidence, df5, df15, df1h = get_signal()

# 检查是否为新信号（用于弹窗）
if direction and entry:
    last_sig = st.session_state.last_signal
    if last_sig['direction'] != direction or abs(entry - (last_sig['entry'] or 0)) > 0.01:
        st.toast(f"🚨 新交易计划: {direction} 入场 {entry:.2f}", icon="💹")
        st.session_state.last_signal = {'direction': direction, 'entry': entry, 'time': datetime.now()}

# --- 渲染界面 ---
with placeholder.container():
    # 顶部指标卡片
    col1, col2, col3, col4, col5 = st.columns(5)
    last = df5.iloc[-1] if not df5.empty else None
    if last is not None:
        col1.metric("当前价", f"${last['close']:.2f}")
        col2.metric("ATR(14)", f"{last['atr']:.2f}")
        col3.metric("RSI", f"{last['rsi']:.1f}")
        col4.metric("成交量比", f"{last['volume_ratio']:.2f}")
        col5.metric("AI信心", f"{confidence}%")
    else:
        st.warning("等待数据...")

    # 趋势信息
    st.subheader("📊 趋势过滤 (15分钟 & 1小时 EMA50)")
    trend_col1, trend_col2, trend_col3 = st.columns(3)
    if not df15.empty:
        last15 = df15.iloc[-1]
        trend15 = "📈 多头" if last15['close'] > last15['ema50'] else "📉 空头"
        trend_col1.metric("15分钟价格", f"{last15['close']:.2f}")
        trend_col2.metric("15分钟EMA50", f"{last15['ema50']:.2f}")
        trend_col3.metric("15分钟趋势", trend15)
    if not df1h.empty:
        last1h = df1h.iloc[-1]
        trend1h = "📈 多头" if last1h['close'] > last1h['ema50'] else "📉 空头"
        st.metric("1小时趋势", trend1h)

    # 交易计划展示
    st.markdown("---")
    st.subheader("📋 最新交易计划")
    if direction and entry:
        st.success(f"**{direction}** | 信心分: {confidence}")
        col_e, col_sl, col_tp = st.columns(3)
        col_e.metric("入场价", f"{entry:.2f}")
        col_sl.metric("止损价", f"{sl:.2f}")
        col_tp.metric("止盈价", f"{tp:.2f}")
        
        # 风险警示
        risk_pct = abs(entry - sl) / entry * 100 * LEVERAGE
        st.error(f"⚠️ 100倍杠杆风险：若止损，本金损失约 {risk_pct:.1f}%")
    else:
        st.info("⏳ 暂无符合80%以上信心的交易计划")

    # 显示最近5根K线
    st.subheader("📈 最近5根5分钟K线")
    st.dataframe(df5[['timestamp', 'close', 'volume', 'rsi', 'vwap']].tail(5), use_container_width=True)

# 自动刷新
st_autorefresh = st.empty()  # 实际需要用 st_autorefresh 组件
# 由于 streamlit-autorefresh 需要安装，我们直接使用 time.sleep 不行，因为 streamlit 是脚本式执行。
# 正确做法是使用 st_autorefresh 组件
from streamlit_autorefresh import st_autorefresh
st_autorefresh(interval=ST_REFRESH * 1000, key="auto_refresh")
