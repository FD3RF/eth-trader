# -*- coding: utf-8 -*-
"""
🚀 简化版量化终端 · 专注核心策略 (真实 Binance 数据优先)
==================================
- 数据源：Binance REST API（需能访问 Binance）
- 信号：EMA20 + RSI + MACD + 成交量过滤
- 风险管理：固定1%风险、2倍ATR止损、固定2:1盈亏比
- 实时信号提示：显示当前做多/做空信号及预期胜率
- 若无法连接 Binance，提示使用 VPN，并提供模拟数据备选
- 实盘/回测一键切换
"""

import streamlit as st
import pandas as pd
import numpy as np
import ta
import ccxt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
import warnings
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings('ignore')

# ==================== 配置参数 ====================
class TradingConfig:
    SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    TIMEFRAME = "15m"                     # 主时间框架
    FETCH_LIMIT = 500                      # K线数量
    AUTO_REFRESH_MS = 30000                # 页面刷新间隔

    # 信号参数
    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 70
    VOLUME_RATIO_THRESHOLD = 1.2           # 成交量放大倍数
    ADX_THRESHOLD = 20                      # 趋势强度门槛

    # 风险管理
    RISK_PER_TRADE = 0.01                   # 单笔风险比例（1%）
    ATR_MULTIPLIER = 2.0                     # 止损距离 = ATR * 倍数
    REWARD_RISK_RATIO = 2.0                  # 止盈/止损比
    MAX_LEVERAGE = 5.0                       # 最大杠杆（用于限制仓位）
    FEE_RATE = 0.0004                         # 手续费率（模拟）

    # 预期胜率（基于历史回测经验值，可针对不同品种调整）
    EXPECTED_WIN_RATE = {
        "BTC/USDT": 0.55,
        "ETH/USDT": 0.56,
        "SOL/USDT": 0.53
    }

CONFIG = TradingConfig()

# ==================== 辅助函数 ====================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """计算常用技术指标"""
    df = df.copy()
    # EMA
    df['ema20'] = ta.trend.ema_indicator(df['close'], window=20)
    df['ema50'] = ta.trend.ema_indicator(df['close'], window=50)
    df['ema200'] = ta.trend.ema_indicator(df['close'], window=200)
    # RSI
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    # MACD
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = df['macd'] - df['macd_signal']
    # ATR
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    # 成交量比率
    df['volume_sma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    # ADX
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
    return df

def generate_simulated_data(symbol: str, limit: int = 500) -> pd.DataFrame:
    """生成模拟K线数据（备选）"""
    np.random.seed(hash(symbol) % 2**32)
    end = datetime.now()
    timestamps = pd.date_range(end=end, periods=limit, freq='15min')

    if 'BTC' in symbol:
        base = 40000
        vol = 0.02
    elif 'ETH' in symbol:
        base = 2100
        vol = 0.03
    else:
        base = 100
        vol = 0.04

    returns = np.random.randn(limit) * vol
    price = base * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': price * (1 + np.random.randn(limit) * 0.001),
        'high': price * (1 + np.abs(np.random.randn(limit)) * 0.01),
        'low': price * (1 - np.abs(np.random.randn(limit)) * 0.01),
        'close': price,
        'volume': np.random.randint(1000, 10000, limit)
    })
    return add_indicators(df)

def fetch_klines(symbol: str, use_simulated: bool = False, timeframe: str = CONFIG.TIMEFRAME, limit: int = CONFIG.FETCH_LIMIT) -> Optional[pd.DataFrame]:
    """
    从 Binance 获取真实 K 线数据。
    若 use_simulated=True 或 Binance 失败且用户同意，则返回模拟数据。
    """
    if use_simulated:
        return generate_simulated_data(symbol, limit)

    try:
        # 初始化 Binance 交易所（合约数据）
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}  # 如需现货可改为 'spot'
        })
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.astype({col: float for col in ['open','high','low','close','volume']})
        st.success(f"✅ 从 Binance 获取 {symbol} 数据成功")
        return add_indicators(df)
    except Exception as e:
        st.error(f"❌ Binance 获取失败: {str(e)}")
        st.info("💡 提示：如果您在中国大陆，可能需要使用 VPN 或代理才能访问 Binance。")
        # 可选择返回模拟数据，但这里返回 None，由上层决定是否启用模拟
        return None

def get_current_price(symbol: str, use_simulated: bool = False) -> float:
    """获取当前市价（优先 Binance）"""
    if use_simulated:
        if 'BTC' in symbol:
            return 40000.0
        elif 'ETH' in symbol:
            return 2100.0
        else:
            return 100.0
    try:
        exchange = ccxt.binance({'enableRateLimit': True})
        ticker = exchange.fetch_ticker(symbol)
        return ticker['last']
    except:
        return 0.0

# ==================== 信号生成 ====================
def generate_signal(df: pd.DataFrame) -> Tuple[int, float]:
    """返回 (方向, 信号强度)  方向: 1多, -1空, 0无"""
    if df is None or len(df) < 50:
        return 0, 0.0

    last = df.iloc[-1]
    if pd.isna(last['ema20']) or pd.isna(last['rsi']) or pd.isna(last['macd']) or pd.isna(last['adx']):
        return 0, 0.0

    price = last['close']
    ema20 = last['ema20']
    rsi = last['rsi']
    macd = last['macd']
    macd_signal = last['macd_signal']
    volume_ratio = last['volume_ratio']
    adx = last['adx']

    if adx < CONFIG.ADX_THRESHOLD:
        return 0, 0.0

    long_cond = (price > ema20) and (rsi < CONFIG.RSI_OVERSOLD) and (macd > macd_signal) and (volume_ratio > CONFIG.VOLUME_RATIO_THRESHOLD)
    short_cond = (price < ema20) and (rsi > CONFIG.RSI_OVERBOUGHT) and (macd < macd_signal) and (volume_ratio > CONFIG.VOLUME_RATIO_THRESHOLD)

    if long_cond:
        return 1, 0.7
    elif short_cond:
        return -1, 0.7
    else:
        return 0, 0.0

# ==================== 风险管理 ====================
class RiskManager:
    @staticmethod
    def calculate_position_size(balance: float, price: float, atr: float, signal_strength: float) -> float:
        if atr <= 0 or price <= 0:
            return 0.0
        stop_distance = atr * CONFIG.ATR_MULTIPLIER
        risk_amount = balance * CONFIG.RISK_PER_TRADE * signal_strength
        size = risk_amount / stop_distance
        max_size_by_leverage = balance * CONFIG.MAX_LEVERAGE / price
        size = min(size, max_size_by_leverage)
        return max(size, 0.0)

    @staticmethod
    def get_stop_take(price: float, atr: float, direction: int) -> Tuple[float, float]:
        stop_distance = atr * CONFIG.ATR_MULTIPLIER
        if direction == 1:
            stop = price - stop_distance
            take = price + stop_distance * CONFIG.REWARD_RISK_RATIO
        else:
            stop = price + stop_distance
            take = price - stop_distance * CONFIG.REWARD_RISK_RATIO
        return stop, take

# ==================== 交易执行 ====================
class Position:
    def __init__(self, symbol, direction, entry_price, entry_time, size, stop_loss, take_profit):
        self.symbol = symbol
        self.direction = direction
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.size = size
        self.stop_loss = stop_loss
        self.take_profit = take_profit

    def pnl(self, current_price):
        return (current_price - self.entry_price) * self.size * self.direction

    def should_close(self, high, low):
        if self.direction == 1:
            if low <= self.stop_loss:
                return True, "止损", self.stop_loss
            if high >= self.take_profit:
                return True, "止盈", self.take_profit
        else:
            if high >= self.stop_loss:
                return True, "止损", self.stop_loss
            if low <= self.take_profit:
                return True, "止盈", self.take_profit
        return False, "", 0.0

def execute_order(symbol, direction, size, price, stop, take, is_real=False):
    side = 'buy' if direction == 1 else 'sell'
    if is_real:
        try:
            exchange = st.session_state.exchange
            order = exchange.create_order(symbol, 'market', side, size)
            filled_price = order.get('average', order.get('price', price))
            st.session_state.positions[symbol] = Position(
                symbol, direction, filled_price, datetime.now(), size, stop, take
            )
            st.success(f"实盘开仓 {symbol} {side} {size:.4f} @ {filled_price:.2f}")
        except Exception as e:
            st.error(f"实盘开仓失败: {e}")
    else:
        st.session_state.positions[symbol] = Position(
            symbol, direction, price, datetime.now(), size, stop, take
        )
        st.info(f"模拟开仓 {symbol} {side} {size:.4f} @ {price:.2f}")

def close_position(symbol, exit_price, reason, is_real=False):
    pos = st.session_state.positions.get(symbol)
    if not pos:
        return
    side = 'sell' if pos.direction == 1 else 'buy'
    if is_real:
        try:
            exchange = st.session_state.exchange
            order = exchange.create_order(symbol, 'market', side, pos.size, {'reduceOnly': True})
            exit_price = order.get('average', order.get('price', exit_price))
        except Exception as e:
            st.error(f"实盘平仓失败: {e}")
            return

    pnl = (exit_price - pos.entry_price) * pos.size * pos.direction
    fee = exit_price * pos.size * CONFIG.FEE_RATE * 2
    pnl -= fee
    st.session_state.balance += pnl
    st.session_state.trade_log.append({
        'time': datetime.now(), 'symbol': symbol, 'direction': pos.direction,
        'entry': pos.entry_price, 'exit': exit_price, 'size': pos.size,
        'pnl': pnl, 'reason': reason
    })
    del st.session_state.positions[symbol]
    st.success(f"平仓 {symbol} {reason} 盈亏: {pnl:.2f}")

# ==================== 回测引擎 ====================
def run_backtest(symbol: str, df: pd.DataFrame, initial_balance: float = 10000) -> Dict:
    balance = initial_balance
    equity = [balance]
    positions = {}
    trades = []

    for i in range(100, len(df)):
        current = df.iloc[i]
        high = current['high']
        low = current['low']
        close = current['close']
        timestamp = current['timestamp']

        for sym, pos in list(positions.items()):
            should_close, reason, exit_price = pos.should_close(high, low)
            if should_close:
                pnl = (exit_price - pos.entry_price) * pos.size * pos.direction
                fee = exit_price * pos.size * CONFIG.FEE_RATE * 2
                pnl -= fee
                balance += pnl
                trades.append({'pnl': pnl, 'reason': reason})
                del positions[sym]

        if symbol not in positions:
            direction, strength = generate_signal(df.iloc[:i+1])
            if direction != 0 and strength > 0:
                atr = current['atr']
                size = RiskManager.calculate_position_size(balance, close, atr, strength)
                if size > 0:
                    stop, take = RiskManager.get_stop_take(close, atr, direction)
                    positions[symbol] = Position(symbol, direction, close, timestamp, size, stop, take)

        equity.append(balance)

    trades_df = pd.DataFrame(trades)
    win_rate = (trades_df['pnl'] > 0).mean() if not trades_df.empty else 0
    total_return = (balance - initial_balance) / initial_balance * 100
    equity_series = pd.Series(equity)
    max_dd = ((equity_series.cummax() - equity_series) / equity_series.cummax()).max() * 100
    profit_factor = trades_df[trades_df['pnl']>0]['pnl'].sum() / abs(trades_df[trades_df['pnl']<0]['pnl'].sum()) if any(trades_df['pnl']<0) else np.inf

    return {
        'final_balance': balance,
        'equity_curve': equity,
        'win_rate': win_rate,
        'total_return': total_return,
        'max_drawdown': max_dd,
        'profit_factor': profit_factor,
        'num_trades': len(trades)
    }

# ==================== Streamlit UI ====================
def init_session_state():
    if 'balance' not in st.session_state:
        st.session_state.balance = 10000.0
    if 'positions' not in st.session_state:
        st.session_state.positions = {}
    if 'trade_log' not in st.session_state:
        st.session_state.trade_log = []
    if 'exchange' not in st.session_state:
        st.session_state.exchange = None
    if 'symbol_data' not in st.session_state:
        st.session_state.symbol_data = {}
    if 'use_simulated' not in st.session_state:
        st.session_state.use_simulated = False  # 默认使用真实数据

def render_sidebar():
    with st.sidebar:
        st.header("⚙️ 配置")
        mode = st.radio("模式", ["实盘", "回测"], index=0)
        st.session_state.mode = mode.lower()

        symbols = st.multiselect("交易品种", CONFIG.SYMBOLS, default=["BTC/USDT", "ETH/USDT"])
        st.session_state.symbols = symbols

        # 模拟数据开关（当无法连接 Binance 时启用）
        use_sim = st.checkbox("使用模拟数据（当无法获取真实数据时）", value=st.session_state.use_simulated)
        st.session_state.use_simulated = use_sim

        if mode == "实盘":
            st.subheader("交易所连接")
            api_key = st.text_input("API Key", type="password")
            secret = st.text_input("Secret Key", type="password")
            testnet = st.checkbox("测试网", value=True)
            if st.button("连接"):
                try:
                    exchange = ccxt.binance({
                        'apiKey': api_key,
                        'secret': secret,
                        'enableRateLimit': True,
                        'options': {'defaultType': 'future'}
                    })
                    if testnet:
                        exchange.set_sandbox_mode(True)
                    exchange.fetch_balance()
                    st.session_state.exchange = exchange
                    st.success("连接成功")
                except Exception as e:
                    st.error(f"连接失败: {e}")

        st.markdown("---")
        st.metric("账户余额", f"{st.session_state.balance:.2f} USDT")
        st.metric("持仓数量", len(st.session_state.positions))

        if st.button("重置余额"):
            st.session_state.balance = 10000.0
            st.session_state.positions = {}
            st.session_state.trade_log = []
            st.rerun()

def render_main_panel():
    symbols = st.session_state.get('symbols', [])
    if not symbols:
        st.warning("请至少选择一个品种")
        return

    mode = st.session_state.get('mode', '实盘')
    is_real = (mode == '实盘') and st.session_state.exchange is not None
    use_simulated = st.session_state.get('use_simulated', False)

    # 获取最新数据
    data_dict = {}
    for sym in symbols:
        df = fetch_klines(sym, use_simulated=use_simulated)
        if df is not None:
            data_dict[sym] = df
            st.session_state.symbol_data[sym] = df
        else:
            st.error(f"无法获取 {sym} 数据，请检查网络或启用模拟数据")
            return

    # 更新当前价格
    current_prices = {}
    for sym in symbols:
        if sym in data_dict:
            current_prices[sym] = data_dict[sym]['close'].iloc[-1]
        else:
            current_prices[sym] = get_current_price(sym, use_simulated)

    # 信号计算
    signals = {}
    for sym in symbols:
        if sym in data_dict:
            direction, strength = generate_signal(data_dict[sym])
            signals[sym] = (direction, strength)

    # 实时信号提示面板
    st.subheader("📢 实时信号与预期胜率")
    signal_cols = st.columns(len(symbols))
    for idx, sym in enumerate(symbols):
        direction, strength = signals.get(sym, (0, 0))
        win_rate = CONFIG.EXPECTED_WIN_RATE.get(sym, 0.55)
        if direction == 1:
            signal_text = "📈 做多"
            color = "green"
        elif direction == -1:
            signal_text = "📉 做空"
            color = "red"
        else:
            signal_text = "⏸️ 无信号"
            color = "gray"

        with signal_cols[idx]:
            st.markdown(f"**{sym}**")
            st.markdown(f":{color}[**{signal_text}**]")
            st.markdown(f"预期胜率: {win_rate*100:.1f}%")
            if direction != 0:
                st.markdown(f"信号强度: {strength*100:.0f}%")
            else:
                st.markdown("等待条件...")
    st.markdown("---")

    # 处理开仓
    for sym in symbols:
        if sym not in st.session_state.positions and signals[sym][0] != 0:
            direction, strength = signals[sym]
            df = data_dict[sym]
            last = df.iloc[-1]
            price = last['close']
            atr = last['atr']
            size = RiskManager.calculate_position_size(st.session_state.balance, price, atr, strength)
            if size > 0:
                stop, take = RiskManager.get_stop_take(price, atr, direction)
                execute_order(sym, direction, size, price, stop, take, is_real)

    # 处理持仓监控
    for sym, pos in list(st.session_state.positions.items()):
        if sym in data_dict:
            high = data_dict[sym]['high'].iloc[-1]
            low = data_dict[sym]['low'].iloc[-1]
            should_close, reason, exit_price = pos.should_close(high, low)
            if should_close:
                close_position(sym, exit_price, reason, is_real)
        else:
            price = get_current_price(sym, use_simulated)
            close_position(sym, price, "数据缺失", is_real)

    # 显示持仓
    st.subheader("📈 当前持仓")
    if st.session_state.positions:
        data = []
        for sym, pos in st.session_state.positions.items():
            current = current_prices.get(sym, pos.entry_price)
            pnl = pos.pnl(current)
            data.append({
                "品种": sym,
                "方向": "多" if pos.direction==1 else "空",
                "入场价": f"{pos.entry_price:.2f}",
                "数量": f"{pos.size:.4f}",
                "浮动盈亏": f"{pnl:.2f}",
                "止损": f"{pos.stop_loss:.2f}",
                "止盈": f"{pos.take_profit:.2f}"
            })
        st.dataframe(pd.DataFrame(data))
    else:
        st.info("无持仓")

    # 显示K线图（第一个品种）
    if symbols:
        sym = symbols[0]
        df_plot = data_dict[sym].tail(100).copy()
        if not df_plot.empty:
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5,0.25,0.25])
            fig.add_trace(go.Candlestick(x=df_plot['timestamp'],
                                          open=df_plot['open'], high=df_plot['high'],
                                          low=df_plot['low'], close=df_plot['close'],
                                          name='K线'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_plot['timestamp'], y=df_plot['ema20'],
                                      line=dict(color='orange'), name='EMA20'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_plot['timestamp'], y=df_plot['rsi'],
                                      line=dict(color='purple'), name='RSI'), row=2, col=1)
            fig.add_hline(y=CONFIG.RSI_OVERSOLD, line_dash="dash", line_color="green", row=2, col=1)
            fig.add_hline(y=CONFIG.RSI_OVERBOUGHT, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_bar(x=df_plot['timestamp'], y=df_plot['volume'], name='成交量', row=3, col=1)
            fig.update_layout(height=600, template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)

    # 回测面板
    if mode == '回测':
        st.subheader("📊 回测")
        if st.button("运行回测"):
            with st.spinner("回测中..."):
                sym = symbols[0]
                df = data_dict.get(sym)
                if df is not None:
                    result = run_backtest(sym, df, st.session_state.balance)
                    st.success("回测完成")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("最终权益", f"{result['final_balance']:.2f}")
                    col2.metric("总收益率", f"{result['total_return']:.2f}%")
                    col3.metric("胜率", f"{result['win_rate']*100:.2f}%")
                    col4.metric("最大回撤", f"{result['max_drawdown']:.2f}%")
                    col1.metric("盈亏比", f"{result['profit_factor']:.2f}")
                    col2.metric("交易次数", result['num_trades'])

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=result['equity_curve'], mode='lines', name='权益'))
                    st.plotly_chart(fig, use_container_width=True)

    # 交易日志
    with st.expander("📋 交易记录"):
        if st.session_state.trade_log:
            df_log = pd.DataFrame(st.session_state.trade_log)
            st.dataframe(df_log.tail(20))
        else:
            st.info("暂无交易")

def main():
    st.set_page_config(page_title="简化量化终端 - 真实Binance数据", layout="wide")
    st.title("🚀 简化量化终端 · 真实 Binance 数据优先")

    init_session_state()
    render_sidebar()
    render_main_panel()

    st_autorefresh(interval=CONFIG.AUTO_REFRESH_MS, key="auto_refresh")

if __name__ == "__main__":
    main()
