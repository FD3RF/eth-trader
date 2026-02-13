# -*- coding: utf-8 -*-
"""🚀 全中文智能交易监控中心 · 快速手动版"""
import streamlit as st
import pandas as pd
import numpy as np
import ta
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import asyncio
import aiohttp
import os
from streamlit_autorefresh import st_autorefresh

# -------------------- 密钥读取（从 Streamlit Secrets）--------------------
BINANCE_API_KEY = st.secrets.get("BINANCE_API_KEY", "")
BINANCE_SECRET_KEY = st.secrets.get("BINANCE_SECRET_KEY", "")
PUSHPLUS_TOKEN = st.secrets.get("PUSHPLUS_TOKEN", "")

# -------------------- 数据获取 --------------------
class AsyncDataFetcher:
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3/klines"
        self.symbol = "ETHUSDT"
        self.periods = ['1m', '5m', '15m', '1h', '4h', '1d']
        self.limit = 200

    async def fetch_period(self, session, period):
        params = {'symbol': self.symbol, 'interval': period, 'limit': self.limit}
        try:
            async with session.get(self.base_url, params=params, timeout=10) as resp:
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
                    return period, None
        except Exception:
            return period, None

    async def fetch_all(self):
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_period(session, p) for p in self.periods]
            results = await asyncio.gather(*tasks)
            return {p: df for p, df in results if df is not None}

# -------------------- 指标计算 --------------------
def add_indicators(df):
    df = df.copy()
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma60'] = df['close'].rolling(60).mean()
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    df['volume_sma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    return df

# -------------------- AI预测（规则模拟）--------------------
class AIPredictor:
    def predict(self, df_dict):
        signals = {}
        for period, df in df_dict.items():
            if df is not None and len(df) > 20:
                last = df.iloc[-1]
                if last['rsi'] < 30 and last['close'] > last['ma20']:
                    signals[period] = 1
                elif last['rsi'] > 70 and last['close'] < last['ma60']:
                    signals[period] = -1
                else:
                    signals[period] = 0
        if not signals:
            return 0, 0.5
        avg_signal = np.mean(list(signals.values()))
        confidence = abs(avg_signal)
        direction = 1 if avg_signal > 0.2 else -1 if avg_signal < -0.2 else 0
        return direction, confidence

# -------------------- 多周期融合 --------------------
class MultiPeriodFusion:
    def __init__(self):
        self.period_weights = {
            '1m': 0.05, '5m': 0.1, '15m': 0.15,
            '1h': 0.2, '4h': 0.25, '1d': 0.25
        }
        self.strategy_weights = {'trend': 0.5, 'oscillator': 0.3, 'volume': 0.2}

    def get_period_signal(self, df):
        last = df.iloc[-1]
        signals = {}
        if last['ma20'] > last['ma60']:
            signals['trend'] = 1
        elif last['ma20'] < last['ma60']:
            signals['trend'] = -1
        else:
            signals['trend'] = 0
        if last['rsi'] < 30:
            signals['oscillator'] = 1
        elif last['rsi'] > 70:
            signals['oscillator'] = -1
        else:
            signals['oscillator'] = 0
        if last['volume_ratio'] > 1.2 and last['close'] > last['open']:
            signals['volume'] = 1
        elif last['volume_ratio'] > 1.2 and last['close'] < last['open']:
            signals['volume'] = -1
        else:
            signals['volume'] = 0
        return signals

    def fuse_periods(self, df_dict):
        period_scores = {}
        for period, df in df_dict.items():
            if df is not None and len(df) > 20:
                signals = self.get_period_signal(df)
                score = sum(signals[s] * self.strategy_weights[s] for s in signals)
                period_scores[period] = score
        if not period_scores:
            return 0, 0
        total_score = sum(period_scores[p] * self.period_weights.get(p, 0) for p in period_scores)
        total_weight = sum(self.period_weights.get(p, 0) for p in period_scores)
        if total_weight == 0:
            return 0, 0
        avg_score = total_score / total_weight
        if abs(avg_score) < 0.2:
            return 0, abs(avg_score)
        direction = 1 if avg_score > 0 else -1
        confidence = min(abs(avg_score) * 1.2, 1.0)
        return direction, confidence

# -------------------- 微信推送（带冷却）--------------------
last_signal_time = None
last_signal_direction = 0
def send_signal_alert(direction, confidence, price):
    global last_signal_time, last_signal_direction
    if not PUSHPLUS_TOKEN:
        return
    now = datetime.now()
    if direction == last_signal_direction and last_signal_time and (now - last_signal_time).seconds < 300:
        return
    dir_str = "做多" if direction == 1 else "做空"
    content = f"""【交易信号】
方向: {dir_str}
置信度: {confidence:.1%}
当前价格: ${price:.2f}
时间: {now.strftime('%Y-%m-%d %H:%M:%S')}"""
    requests.post("http://www.pushplus.plus/send", json={
        "token": PUSHPLUS_TOKEN, "title": "🤖 信号", "content": content
    }, timeout=5)
    last_signal_time = now
    last_signal_direction = direction

# -------------------- 缓存数据 --------------------
@st.cache_data(ttl=60)
def fetch_all_data():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    fetcher = AsyncDataFetcher()
    data_dict = loop.run_until_complete(fetcher.fetch_all())
    for p in data_dict:
        data_dict[p] = add_indicators(data_dict[p])
    return data_dict

# -------------------- Streamlit 界面 --------------------
st.set_page_config(page_title="全中文交易监控中心", layout="wide")
st.markdown("""
<style>
    .stApp { background-color: #0B0E14; color: white; }
    .ai-box { background: #1A1D27; border-radius: 10px; padding: 20px; border-left: 6px solid #00F5A0; }
</style>
""", unsafe_allow_html=True)

st.title("🧠 全中文智能交易监控中心 · 快速版")
st.caption("数据60秒更新 | 多周期切换 | AI信号 | 模拟盈亏 | 微信提醒")

with st.sidebar:
    st.header("⚙️ 控制面板")
    period = st.selectbox("选择K线周期", ['1m','5m','15m','1h','4h','1d'], index=4)
    auto = st.checkbox("自动刷新", True)
    if auto:
        st_autorefresh(interval=10*1000, key="auto")
    st.subheader("💰 模拟交易")
    entry = st.number_input("入场价", value=0.0)
    stop = st.number_input("止损价", value=0.0)
    qty = st.number_input("数量", value=0.01)

data_dict = fetch_all_data()
if data_dict:
    ai = AIPredictor()
    fusion = MultiPeriodFusion()
    ai_dir, ai_conf = ai.predict(data_dict)
    fusion_dir, fusion_conf = fusion.fuse_periods(data_dict)
    if fusion_dir != 0 and data_dict.get('4h') is not None:
        price = data_dict['4h']['close'].iloc[-1]
        send_signal_alert(fusion_dir, fusion_conf, price)

col1, col2 = st.columns([2.2, 1.3])
with col1:
    st.subheader(f"📊 {period} K线图")
    if period in data_dict:
        df = data_dict[period].tail(100).copy()
        df['时间'] = df['timestamp']
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7,0.3])
        fig.add_trace(go.Candlestick(x=df['时间'], open=df['open'], high=df['high'],
                                      low=df['low'], close=df['close'], name="K线"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['时间'], y=df['ma20'], name="MA20", line=dict(color="orange")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['时间'], y=df['ma60'], name="MA60", line=dict(color="blue")), row=1, col=1)
        if fusion_dir != 0:
            last = df.iloc[-1]
            y_pos = last['close'] * 1.02 if fusion_dir == 1 else last['close'] * 0.98
            fig.add_annotation(x=last['时间'], y=y_pos, text="▲ 多" if fusion_dir==1 else "▼ 空", showarrow=True)
        fig.add_trace(go.Scatter(x=df['时间'], y=df['rsi'], name="RSI", line=dict(color="purple")), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2)
        fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=600)
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("🧠 信号")
    dir_map = {1:"🔴 做多", -1:"🔵 做空", 0:"⚪ 观望"}
    st.markdown(f'<div class="ai-box">{dir_map[fusion_dir]}<br>置信度: {fusion_conf:.1%}</div>', unsafe_allow_html=True)
    if entry > 0 and qty > 0 and period in data_dict:
        cur = data_dict[period]['close'].iloc[-1]
        pnl = (cur - entry) * qty
        pnl_pct = (cur - entry) / entry * 100
        st.metric("浮动盈亏", f"${pnl:.2f} ({pnl_pct:.2f}%)")
        if stop > 0 and ((entry < cur <= stop) or (entry > cur >= stop)):
            st.warning("⚠️ 接近止损")
    st.markdown("---")
    st.markdown("**📈 各周期**")
    for p in ['1m','5m','15m','1h','4h','1d']:
        if p in data_dict:
            last = data_dict[p].iloc[-1]
            trend = "↑" if last['ma20'] > last['ma60'] else "↓" if last['ma20'] < last['ma60'] else "→"
            st.caption(f"{p}: {trend} RSI {last['rsi']:.1f}")
