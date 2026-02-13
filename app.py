# -*- coding: utf-8 -*-
"""🚀 全中文智能交易监控中心 · 最终稳定版（带模拟数据回退与重试机制）"""

import streamlit as st
import pandas as pd
import numpy as np
import ta
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import asyncio
import aiohttp
import os
from streamlit_autorefresh import st_autorefresh
import random
import time

# -------------------- 从 Streamlit Secrets 读取必要配置 --------------------
# 注意：Binance K线是公开接口，不需要 API Key。此处只读取微信推送令牌（可选）
PUSHPLUS_TOKEN = st.secrets.get("PUSHPLUS_TOKEN", "")

# -------------------- 数据获取类（支持重试和模拟数据回退）--------------------
class DataManager:
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3/klines"
        self.symbol = "ETHUSDT"
        self.periods = ['1m', '5m', '15m', '1h', '4h', '1d']
        self.limit = 200
        self.use_mock = False          # 标记是否使用模拟数据
        self.max_retries = 3            # 最大重试次数
        self.retry_delay = 1            # 初始重试延迟（秒）

    def generate_mock_data(self, period, limit=200):
        """生成模拟K线数据（当API不可用时使用）"""
        np.random.seed(42)  # 固定种子，使数据可重复，便于调试
        end = datetime.now()
        freq_map = {
            '1m': '1min', '5m': '5min', '15m': '15min',
            '1h': '1H', '4h': '4H', '1d': '1D'
        }
        freq = freq_map.get(period, '1H')
        dates = pd.date_range(end=end, periods=limit, freq=freq)
        # 生成随机价格走势
        base = 2000 + random.randint(-100, 100)
        changes = np.random.randn(limit) * 10
        prices = base + np.cumsum(changes)
        opens = prices + np.random.randn(limit) * 2
        highs = np.maximum(prices, opens) + np.abs(np.random.randn(limit) * 5)
        lows = np.minimum(prices, opens) - np.abs(np.random.randn(limit) * 5)
        volumes = np.random.randint(1000, 5000, limit)

        df = pd.DataFrame({
            'timestamp': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': prices,
            'volume': volumes
        })
        return df

    async def fetch_period(self, session, period):
        """获取单个周期的数据，带重试机制"""
        if self.use_mock:
            return period, self.generate_mock_data(period, self.limit)

        params = {'symbol': self.symbol, 'interval': period, 'limit': self.limit}
        for attempt in range(self.max_retries):
            try:
                async with session.get(self.base_url, params=params, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if isinstance(data, list):
                            df = pd.DataFrame(data, columns=[
                                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                                'close_time', 'quote_asset_volume', 'num_trades',
                                'taker_buy_base', 'taker_buy_quote', 'ignore'
                            ])
                            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                            for col in ['open', 'high', 'low', 'close', 'volume']:
                                df[col] = df[col].astype(float)
                            return period, df
                        else:
                            break  # 数据格式错误，直接跳出重试
                    else:
                        # 非200状态码，重试前等待
                        await asyncio.sleep(self.retry_delay * (2 ** attempt))
            except Exception as e:
                print(f"获取 {period} 数据失败 (尝试 {attempt+1}/{self.max_retries}): {e}")
                await asyncio.sleep(self.retry_delay * (2 ** attempt))
        # 所有重试失败，返回None
        return period, None

    async def fetch_all(self):
        """并发获取所有周期数据，如有失败则切换到模拟数据"""
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_period(session, p) for p in self.periods]
            results = await asyncio.gather(*tasks)
            data_dict = {p: df for p, df in results if df is not None}

            # 如果任何一个周期数据获取失败，启用模拟数据模式并重新生成所有数据
            if len(data_dict) < len(self.periods):
                self.use_mock = True
                mock_dict = {}
                for p in self.periods:
                    mock_dict[p] = self.generate_mock_data(p, self.limit)
                return mock_dict, self.use_mock
            return data_dict, self.use_mock


# -------------------- 技术指标计算 --------------------
def add_indicators(df):
    """为DataFrame添加常用技术指标"""
    df = df.copy()
    # 移动平均线
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma60'] = df['close'].rolling(60).mean()
    # MACD
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    # RSI
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    # 布林带
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    # ATR
    df['atr'] = ta.volatility.AverageTrueRange(
        df['high'], df['low'], df['close'], window=14
    ).average_true_range()
    # 成交量相关
    df['volume_sma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    return df


# -------------------- 多周期融合信号 --------------------
class MultiPeriodFusion:
    def __init__(self):
        # 各周期权重（短周期权重低，长周期权重高，避免噪声）
        self.period_weights = {
            '1m': 0.05, '5m': 0.10, '15m': 0.15,
            '1h': 0.20, '4h': 0.25, '1d': 0.25
        }
        # 策略权重
        self.strategy_weights = {'trend': 0.5, 'oscillator': 0.3, 'volume': 0.2}

    def get_period_signal(self, df):
        """计算单个周期的多策略信号"""
        last = df.iloc[-1]
        signals = {}
        # 趋势信号（MA排列）
        if last['ma20'] > last['ma60']:
            signals['trend'] = 1
        elif last['ma20'] < last['ma60']:
            signals['trend'] = -1
        else:
            signals['trend'] = 0
        # 震荡信号（RSI）
        if last['rsi'] < 30:
            signals['oscillator'] = 1
        elif last['rsi'] > 70:
            signals['oscillator'] = -1
        else:
            signals['oscillator'] = 0
        # 成交量信号
        if last['volume_ratio'] > 1.2 and last['close'] > last['open']:
            signals['volume'] = 1
        elif last['volume_ratio'] > 1.2 and last['close'] < last['open']:
            signals['volume'] = -1
        else:
            signals['volume'] = 0
        return signals

    def fuse_periods(self, df_dict):
        """融合多周期信号，返回综合方向和置信度"""
        period_scores = {}
        for period, df in df_dict.items():
            if df is not None and len(df) > 20:
                signals = self.get_period_signal(df)
                score = sum(signals[s] * self.strategy_weights[s] for s in signals)
                period_scores[period] = score

        if not period_scores:
            return 0, 0.0

        total_score = 0.0
        total_weight = 0.0
        for period, score in period_scores.items():
            w = self.period_weights.get(period, 0)
            total_score += score * w
            total_weight += w

        if total_weight == 0:
            return 0, 0.0

        avg_score = total_score / total_weight
        # 方向判定
        if abs(avg_score) < 0.2:
            return 0, abs(avg_score)
        direction = 1 if avg_score > 0 else -1
        # 置信度映射（将分数映射到0.5~1.0之间）
        confidence = min(abs(avg_score) * 1.5, 1.0)
        return direction, confidence


# -------------------- 微信推送（带冷却）--------------------
_last_signal_time = None
_last_signal_direction = 0

def send_signal_alert(direction, confidence, price):
    """发送信号到微信，避免5分钟内重复相同方向"""
    global _last_signal_time, _last_signal_direction
    if not PUSHPLUS_TOKEN:
        return
    now = datetime.now()
    # 如果方向未变且上次推送在5分钟内，则跳过
    if direction == _last_signal_direction and _last_signal_time and (now - _last_signal_time).seconds < 300:
        return

    dir_str = "做多" if direction == 1 else "做空" if direction == -1 else "观望"
    content = f"""【交易信号】
方向: {dir_str}
置信度: {confidence:.1%}
当前价格: ${price:.2f}
时间: {now.strftime('%Y-%m-%d %H:%M:%S')}"""
    try:
        requests.post(
            "http://www.pushplus.plus/send",
            json={"token": PUSHPLUS_TOKEN, "title": "🤖 信号", "content": content},
            timeout=5
        )
        _last_signal_time = now
        _last_signal_direction = direction
    except Exception as e:
        print(f"微信推送失败: {e}")


# -------------------- 数据缓存（60秒）--------------------
@st.cache_data(ttl=60)
def fetch_all_data_cached():
    """封装的数据获取函数，供Streamlit缓存使用"""
    manager = DataManager()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    data_dict, use_mock = loop.run_until_complete(manager.fetch_all())
    # 为所有数据添加技术指标
    for p in data_dict:
        data_dict[p] = add_indicators(data_dict[p])
    return data_dict, use_mock


# -------------------- Streamlit 界面 --------------------
st.set_page_config(page_title="全中文交易监控中心", layout="wide")
st.markdown("""
<style>
    .stApp { background-color: #0B0E14; color: white; }
    .ai-box { background: #1A1D27; border-radius: 10px; padding: 20px; border-left: 6px solid #00F5A0; }
    .mock-warning { color: #FFAA00; font-size: 0.9em; margin-top: 10px; }
    .data-source { font-size: 0.8em; color: #888; }
</style>
""", unsafe_allow_html=True)

st.title("🧠 全中文智能交易监控中心 · 最终稳定版")
st.caption("数据60秒更新 | 多周期切换 | AI信号 | 模拟盈亏 | 微信提醒")

# 侧边栏控制
with st.sidebar:
    st.header("⚙️ 控制面板")
    period = st.selectbox("选择K线周期", ['1m', '5m', '15m', '1h', '4h', '1d'], index=4)  # 默认4h
    auto_refresh = st.checkbox("自动刷新", True)
    if auto_refresh:
        st_autorefresh(interval=60 * 1000, key="auto")  # 60秒自动刷新

    st.subheader("💰 模拟交易")
    entry_price = st.number_input("入场价 ($)", value=0.0, format="%.2f")
    stop_loss = st.number_input("止损价 ($)", value=0.0, format="%.2f")
    quantity = st.number_input("数量 (ETH)", value=0.01, format="%.4f")

# 获取数据
data_dict, use_mock = fetch_all_data_cached()

# 显示数据来源提示
if use_mock:
    st.sidebar.markdown('<p class="mock-warning">⚠️ 当前使用模拟数据（无法获取实时行情）</p>', unsafe_allow_html=True)
else:
    st.sidebar.markdown('<p class="data-source">✅ 数据源：Binance 实时</p>', unsafe_allow_html=True)

# 初始化信号变量
fusion = MultiPeriodFusion()
fusion_dir, fusion_conf = 0, 0.0
if data_dict:
    fusion_dir, fusion_conf = fusion.fuse_periods(data_dict)

# 微信推送（仅在非模拟数据且方向非0时推送）
if not use_mock and fusion_dir != 0 and data_dict.get('4h') is not None:
    current_price = data_dict['4h']['close'].iloc[-1]
    send_signal_alert(fusion_dir, fusion_conf, current_price)

# 主界面布局
col1, col2 = st.columns([2.2, 1.3])

with col1:
    st.subheader(f"📊 {period} K线图")
    if period in data_dict:
        df = data_dict[period].tail(100).copy()
        df['时间'] = df['timestamp']

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.7, 0.3],
            subplot_titles=(f"ETH/USDT {period}", "RSI (14)")
        )

        # 主图：K线 + 均线
        fig.add_trace(go.Candlestick(
            x=df['时间'], open=df['open'], high=df['high'],
            low=df['low'], close=df['close'], name="K线"
        ), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['时间'], y=df['ma20'], name="MA20",
                                  line=dict(color="orange")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['时间'], y=df['ma60'], name="MA60",
                                  line=dict(color="blue")), row=1, col=1)

        # 如果当前有明确方向，在最新K线位置标注箭头
        if fusion_dir != 0:
            last = df.iloc[-1]
            y_pos = last['close'] * 1.02 if fusion_dir == 1 else last['close'] * 0.98
            fig.add_annotation(
                x=last['时间'], y=y_pos,
                text="▲ 多" if fusion_dir == 1 else "▼ 空",
                showarrow=True, arrowhead=2, arrowsize=1,
                arrowcolor="green" if fusion_dir == 1 else "red"
            )

        # 副图：RSI
        fig.add_trace(go.Scatter(x=df['时间'], y=df['rsi'], name="RSI",
                                  line=dict(color="purple")), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)

        fig.update_layout(
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            height=600,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("等待数据...")

with col2:
    st.subheader("🧠 信号")
    dir_map = {1: "🔴 做多", -1: "🔵 做空", 0: "⚪ 观望"}
    signal_text = f"{dir_map[fusion_dir]}<br>置信度: {fusion_conf:.1%}"
    st.markdown(f'<div class="ai-box">{signal_text}</div>', unsafe_allow_html=True)

    # 模拟盈亏计算
    if entry_price > 0 and quantity > 0 and period in data_dict:
        current_price = data_dict[period]['close'].iloc[-1]
        pnl = (current_price - entry_price) * quantity
        pnl_pct = (current_price - entry_price) / entry_price * 100

        if pnl >= 0:
            st.markdown(f'**浮动盈亏**: <span style="color:#00F5A0">+${pnl:.2f} ({pnl_pct:.2f}%)</span>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'**浮动盈亏**: <span style="color:#FF5555">-${abs(pnl):.2f} ({pnl_pct:.2f}%)</span>',
                        unsafe_allow_html=True)

        # 止损提示
        if stop_loss > 0:
            if (entry_price < current_price <= stop_loss) or (entry_price > current_price >= stop_loss):
                st.warning("⚠️ 接近止损")
    else:
        st.info("输入入场价以计算盈亏")

    st.markdown("---")
    st.markdown("**📈 各周期快照**")
    if data_dict:
        for p in ['1m', '5m', '15m', '1h', '4h', '1d']:
            if p in data_dict and len(data_dict[p]) > 0:
                last = data_dict[p].iloc[-1]
                trend = "↑" if last['ma20'] > last['ma60'] else "↓" if last['ma20'] < last['ma60'] else "→"
                st.caption(f"{p}: {trend}  RSI {last['rsi']:.1f}  ${last['close']:.2f}")
    else:
        st.caption("暂无数据")

# 页脚
st.markdown("---")
st.caption("⚠️ 所有信号基于技术指标生成，不构成投资建议。杠杆交易风险极高，请自行控制仓位。")
