import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import time
from datetime import datetime, timedelta
import pytz

# 页面配置
st.set_page_config(page_title="高频波动剥削监控系统", layout="wide")
st.title("📈 高频波动剥削监控系统 (OKX 永续合约)")

# 导入自动刷新组件
try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st.error("请安装 streamlit-autorefresh: pip install streamlit-autorefresh")
    st.stop()

# 常量配置
EXCHANGE_ID = 'okx'
DEFAULT_SYMBOLS = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT']
TIMEFRAME_1M = '1m'
TIMEFRAME_5M = '5m'
LIMIT = 100  # 获取K线数量
REFRESH_INTERVAL = 5000  # 毫秒

# 初始化交易所（只用于公共数据，无需API密钥）
exchange = ccxt.okx({
    'enableRateLimit': True,
    'options': {'defaultType': 'swap'},  # 永续合约
})

# 时区设置
tz = pytz.timezone('Asia/Shanghai')

# ==================== 数据获取模块 ====================
@st.cache_data(ttl=REFRESH_INTERVAL/1000, show_spinner=False)
def fetch_ohlcv(symbol, timeframe, limit=LIMIT):
    """获取K线数据，返回DataFrame"""
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(tz)
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        st.error(f"获取 {symbol} {timeframe} 数据失败: {e}")
        return pd.DataFrame()

def fetch_ticker(symbol):
    """获取当前最新价"""
    try:
        ticker = exchange.fetch_ticker(symbol)
        return ticker['last']
    except Exception as e:
        st.warning(f"获取 {symbol} 最新价失败: {e}")
        return None

# ==================== 指标计算模块 ====================
def calculate_indicators(df):
    """计算所需技术指标，返回更新后的df和最新值字典"""
    if df.empty or len(df) < 20:
        return df, {}
    
    df = df.copy()
    
    # 真实波幅 TR
    df['prev_close'] = df['close'].shift(1)
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['prev_close']),
            abs(df['low'] - df['prev_close'])
        )
    )
    
    # ATR(14) 和 ATR(20)
    df['atr14'] = df['tr'].rolling(window=14).mean()
    df['atr20'] = df['tr'].rolling(window=20).mean()
    
    # 成交量均值（10周期）
    df['volume_ma10'] = df['volume'].rolling(window=10).mean()
    
    # 最近10根最高价（用于突破）
    df['highest_10'] = df['high'].rolling(window=10).max().shift(1)  # 不包括当前K线
    
    # EMA
    df['ema5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['ema13'] = df['close'].ewm(span=13, adjust=False).mean()
    
    # RSI(7)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=7).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=7).mean()
    rs = gain / loss
    df['rsi7'] = 100 - (100 / (1 + rs))
    
    # 爆发倍数 (当前TR / ATR20)
    df['breakout_multiplier'] = df['tr'] / df['atr20']
    
    # 成交量放大倍数
    df['volume_ratio'] = df['volume'] / df['volume_ma10']
    
    # 获取最新一条数据（最近完成的K线）
    latest = df.iloc[-1].to_dict()
    
    # ATR向上拐头 (当前atr14 > 前一个atr14)
    latest['atr14_up'] = latest['atr14'] > df.iloc[-2]['atr14'] if len(df) >= 2 else False
    
    return df, latest

# ==================== 信号检测模块 ====================
def detect_breakout(latest, df_1m, df_5m=None, use_filters=False):
    """
    检测高频做多信号
    latest: 最新指标字典
    df_1m: 1m DataFrame (用于历史比较)
    use_filters: 是否启用可选过滤器
    """
    if not latest:
        return False, {}
    
    conditions = {}
    
    # 1. 波动爆发
    cond1 = latest['breakout_multiplier'] > 1.5
    cond2 = latest['atr14_up']
    conditions['波动爆发'] = cond1 and cond2
    
    # 2. 动量确认
    cond3 = latest['ema5'] > latest['ema13']
    cond4 = 55 < latest['rsi7'] < 80
    conditions['动量确认'] = cond3 and cond4
    
    # 3. 成交量异动
    cond5 = latest['volume_ratio'] > 1.8
    conditions['成交量异动'] = cond5
    
    # 4. 微结构突破
    cond6 = latest['close'] > latest['highest_10']
    conditions['微结构突破'] = cond6
    
    # 主信号
    signal = cond1 and cond2 and cond3 and cond4 and cond5 and cond6
    
    # 可选过滤器
    filters_passed = True
    if use_filters and signal:
        # 布林带压缩后爆发 (BB宽度小于近期均值)
        if 'bb_width' in latest:
            filters_passed = filters_passed and latest['bb_width'] < latest['bb_width_ma']
        # VWAP偏离 (价格在VWAP上方)
        if 'vwap' in latest:
            filters_passed = filters_passed and latest['close'] > latest['vwap']
        # 低波动禁止 (ATR/价格 > 0.001)
        atr_pct = latest['atr14'] / latest['close']
        filters_passed = filters_passed and atr_pct > 0.001  # 至少0.1%波动
    
    return signal and filters_passed, conditions

# ==================== 风控模型模块 ====================
def risk_model(entry_price, latest, df_1m):
    """计算止损、止盈、盈亏比等"""
    atr = latest['atr14']
    low = latest['low']
    
    # 止损 = min(当前K线最低价, ATR*0.8)
    stop_loss = min(low, entry_price - atr * 0.8)
    # 止盈 = 1.5R
    risk = entry_price - stop_loss
    take_profit = entry_price + 1.5 * risk
    
    # 盈亏比
    risk_reward = 1.5
    
    # 风险值 (R%)
    risk_pct = risk / entry_price * 100
    
    # 收益值 (预期收益%)
    reward_pct = (take_profit - entry_price) / entry_price * 100
    
    # 滑点警告
    slippage_warning = ""
    if atr / entry_price > 0.002:  # 0.2% 波动
        slippage_warning = "⚠️ 波幅过大，注意滑点"
    
    return {
        'entry': entry_price,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'risk_reward': risk_reward,
        'risk_pct': risk_pct,
        'reward_pct': reward_pct,
        'slippage_warning': slippage_warning
    }

# ==================== 仪表盘渲染模块 ====================
def render_dashboard(symbol, latest, signal_active, signal_info, conditions, risk_info, current_price):
    """渲染单个币种的仪表盘"""
    with st.container(border=True):
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        
        # 状态
        status = "🔴 等待" if not signal_active else "🟢 高频做多信号"
        col1.metric("状态", status)
        col1.metric("当前价格", f"{current_price:.2f}" if current_price else "N/A")
        
        # 指标
        col2.metric("爆发倍数", f"{latest.get('breakout_multiplier', 0):.2f}" if latest else "N/A")
        col2.metric("ATR(14)", f"{latest.get('atr14', 0):.4f}" if latest else "N/A")
        
        col3.metric("RSI(7)", f"{latest.get('rsi7', 0):.1f}" if latest else "N/A")
        col3.metric("EMA5/13", f"{latest.get('ema5', 0):.1f} / {latest.get('ema13', 0):.1f}" if latest else "N/A")
        
        col4.metric("成交量", f"{latest.get('volume', 0):.0f}" if latest else "N/A")
        col4.metric("成交量倍数", f"{latest.get('volume_ratio', 0):.2f}" if latest else "N/A")
        
        # 条件详情
        with st.expander("条件明细"):
            for cond, passed in conditions.items():
                st.write(f"{'✅' if passed else '❌'} {cond}")
        
        # 信号详情
        if signal_active and signal_info:
            st.subheader("📊 信号详情")
            cols = st.columns(5)
            cols[0].metric("入场价", f"{signal_info['entry']:.2f}")
            cols[1].metric("止损价", f"{signal_info['stop_loss']:.2f}")
            cols[2].metric("止盈价", f"{signal_info['take_profit']:.2f}")
            cols[3].metric("盈亏比", f"{signal_info['risk_reward']:.2f}")
            cols[4].metric("风险/收益", f"{signal_info['risk_pct']:.2f}% / {signal_info['reward_pct']:.2f}%")
            
            # 剩余时间
            remaining = signal_info.get('remaining_seconds', 0)
            mins, secs = divmod(int(remaining), 60)
            st.info(f"⏳ 信号剩余有效时间: {mins:02d}:{secs:02d}  {signal_info['slippage_warning']}")
        else:
            st.info("等待信号触发...")

# ==================== 主程序 ====================
def main():
    # 自动刷新
    count = st_autorefresh(interval=REFRESH_INTERVAL, key="auto_refresh")
    
    # 侧边栏配置
    with st.sidebar:
        st.header("⚙️ 配置")
        symbols = st.multiselect("选择交易对", DEFAULT_SYMBOLS, default=DEFAULT_SYMBOLS[:1])
        use_filters = st.checkbox("启用高级过滤器 (布林带/VWAP/低波动)", value=False)
        st.caption("高级过滤器需额外计算，可能降低信号频率")
    
    if not symbols:
        st.warning("请至少选择一个交易对")
        return
    
    # 初始化session state存储信号
    if 'signals' not in st.session_state:
        st.session_state.signals = {}
    
    current_time = datetime.now(tz)
    
    # 为每个币种处理
    for symbol in symbols:
        with st.spinner(f"加载 {symbol} 数据..."):
            # 获取数据
            df_1m = fetch_ohlcv(symbol, TIMEFRAME_1M)
            df_5m = fetch_ohlcv(symbol, TIMEFRAME_5M)  # 可选，用于高级过滤
            current_price = fetch_ticker(symbol)
            
            if df_1m.empty:
                st.error(f"{symbol} 数据为空，跳过")
                continue
            
            # 计算指标
            df_1m, latest = calculate_indicators(df_1m)
            if not latest:
                st.warning(f"{symbol} 指标计算失败")
                continue
            
            # 检测信号
            signal, conditions = detect_breakout(latest, df_1m, df_5m, use_filters)
            
            # 信号管理
            signal_key = f"{symbol}_signal"
            now_ts = current_time.timestamp()
            
            # 检查现有信号是否超时
            if signal_key in st.session_state.signals:
                signal_info = st.session_state.signals[signal_key]
                elapsed = now_ts - signal_info['timestamp']
                if elapsed > 15 * 60:  # 15分钟
                    del st.session_state.signals[signal_key]
                    signal_active = False
                    signal_info = None
                else:
                    signal_active = True
                    # 更新剩余时间
                    signal_info['remaining_seconds'] = 15 * 60 - elapsed
            else:
                signal_active = False
                signal_info = None
            
            # 如果检测到新信号且当前无有效信号，则生成新信号
            if signal and not signal_active:
                # 计算风控
                risk = risk_model(latest['close'], latest, df_1m)
                signal_info = {
                    'timestamp': now_ts,
                    'entry': latest['close'],
                    'stop_loss': risk['stop_loss'],
                    'take_profit': risk['take_profit'],
                    'risk_reward': risk['risk_reward'],
                    'risk_pct': risk['risk_pct'],
                    'reward_pct': risk['reward_pct'],
                    'slippage_warning': risk['slippage_warning'],
                    'remaining_seconds': 15 * 60
                }
                st.session_state.signals[signal_key] = signal_info
                signal_active = True
                st.success(f"🚀 {symbol} 高频做多信号触发！")
            
            # 渲染仪表盘
            st.markdown(f"## {symbol}")
            render_dashboard(symbol, latest, signal_active, signal_info, conditions, current_price)

if __name__ == "__main__":
    main()
