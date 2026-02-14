# -*- coding: utf-8 -*-
"""
🚀 合约智能监控中心 · 极简稳定版
数据源：MEXC + CryptoCompare (自动切换) | 多周期 | AI预测 | 强平分析
"""

import streamlit as st
import pandas as pd
import numpy as np
import ta
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
from streamlit_autorefresh import st_autorefresh
import warnings
warnings.filterwarnings('ignore')

# -------------------- 强平价格计算 --------------------
def calculate_liquidation_price(entry_price, side, leverage):
    if side == "long":
        return entry_price * (1 - 1/leverage)
    else:
        return entry_price * (1 + 1/leverage)

# -------------------- 极简数据获取器（仅 MEXC + CryptoCompare）--------------------
class SimpleDataFetcher:
    def __init__(self):
        self.symbol = "ETHUSDT"
        self.periods = ['1m', '5m', '15m', '1h', '4h', '1d']
        self.limit = 200
        self.timeout = 5

        # MEXC 配置
        self.mexc = {
            'name': 'MEXC',
            'url': 'https://api.mexc.com/api/v3/klines',
            'params': {'symbol': self.symbol, 'interval': None, 'limit': self.limit}
        }
        # CryptoCompare 配置（支持分钟级）
        self.cryptocompare = {
            'name': 'CryptoCompare',
            'base_url': 'https://min-api.cryptocompare.com/data/v2',
            'params': {'fsym': 'ETH', 'tsym': 'USD', 'limit': self.limit}
        }
        # 价格源（MEXC 最新价）
        self.price_url = 'https://api.mexc.com/api/v3/ticker/price'
        self.price_params = {'symbol': self.symbol}

    def fetch_kline(self, period):
        """尝试从 MEXC 获取 K线，失败则从 CryptoCompare 获取"""
        # 1. 尝试 MEXC
        params = self.mexc['params'].copy()
        params['interval'] = period
        try:
            resp = requests.get(self.mexc['url'], params=params, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                df = self._parse_mexc_kline(data)
                if df is not None:
                    return df, self.mexc['name']
        except:
            pass

        # 2. 尝试 CryptoCompare
        try:
            # 根据周期选择端点
            if period in ['1m', '5m', '15m']:
                endpoint = 'histominute'
                aggregate = {'1m':1, '5m':5, '15m':15}[period]
            elif period in ['1h', '4h']:
                endpoint = 'histohour'
                aggregate = 1 if period == '1h' else 4
            elif period == '1d':
                endpoint = 'histoday'
                aggregate = 1
            else:
                return None, None
            url = f"{self.cryptocompare['base_url']}/{endpoint}"
            params = self.cryptocompare['params'].copy()
            params['aggregate'] = aggregate
            resp = requests.get(url, params=params, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                if data.get('Response') == 'Success':
                    df = self._parse_cryptocompare_kline(data)
                    if df is not None:
                        return df, self.cryptocompare['name']
        except:
            pass

        return None, None

    def _parse_mexc_kline(self, data):
        """解析 MEXC K线数据（取前6列）"""
        if not isinstance(data, list) or len(data) == 0:
            return None
        rows = [row[:6] for row in data if isinstance(row, list) and len(row) >= 6]
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

    def _parse_cryptocompare_kline(self, data):
        items = data['Data']['Data']
        df = pd.DataFrame(items)
        df['timestamp'] = pd.to_datetime(df['time'], unit='s')
        df.rename(columns={'volumefrom': 'volume'}, inplace=True)
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        return df

    def fetch_price(self):
        """获取当前价格（MEXC 最新价）"""
        try:
            resp = requests.get(self.price_url, params=self.price_params, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                return float(data['price']), 'MEXC'
        except:
            pass
        return None, None

    def fetch_all(self):
        """获取所有周期数据"""
        data_dict = {}
        price = None
        price_source = None
        source_display = None
        errors = []

        for period in self.periods:
            df, src = self.fetch_kline(period)
            if df is not None:
                data_dict[period] = df
                if source_display is None:
                    source_display = src
            else:
                errors.append(f"{period} 数据获取失败")

        # 获取价格
        price, price_source = self.fetch_price()
        if price is None and data_dict:
            # 使用4h收盘价作为备用
            if '4h' in data_dict:
                price = data_dict['4h']['close'].iloc[-1]
                price_source = '4h收盘价(备用)'
            elif data_dict:
                first = next(iter(data_dict))
                price = data_dict[first]['close'].iloc[-1]
                price_source = f'{first}收盘价(备用)'

        return data_dict, price, price_source, errors, source_display or '无'

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

# -------------------- AI预测（简化规则版） --------------------
class SimpleAIPredictor:
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
        total_score = 0
        total_weight = 0
        for p, score in period_scores.items():
            w = self.period_weights.get(p, 0)
            total_score += score * w
            total_weight += w
        if total_weight == 0:
            return 0, 0
        avg_score = total_score / total_weight
        if abs(avg_score) < 0.2:
            return 0, abs(avg_score)
        direction = 1 if avg_score > 0 else -1
        confidence = min(abs(avg_score) * 1.2, 1.0)
        return direction, confidence

# -------------------- 缓存数据获取 --------------------
@st.cache_data(ttl=60)
def fetch_all_data():
    fetcher = SimpleDataFetcher()
    data_dict, price, price_source, errors, source_display = fetcher.fetch_all()
    if data_dict:
        for p in data_dict:
            data_dict[p] = add_indicators(data_dict[p])
    return data_dict, price, price_source, errors, source_display

# -------------------- Streamlit 界面 --------------------
st.set_page_config(page_title="合约智能监控·极简版", layout="wide")
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
.info-box { background: #1A2A3A; border-left: 6px solid #00F5A0; padding: 10px; border-radius: 5px; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("🧠 合约智能监控中心 · 极简稳定版")
st.caption("数据源：MEXC + CryptoCompare｜多周期｜AI预测｜强平分析")

# 初始化
if 'ai' not in st.session_state:
    st.session_state.ai = SimpleAIPredictor()
if 'fusion' not in st.session_state:
    st.session_state.fusion = MultiPeriodFusion()

# 侧边栏
with st.sidebar:
    st.header("⚙️ 控制面板")
    period_options = ['1m', '5m', '15m', '1h', '4h', '1d']
    selected_period = st.selectbox("选择K线周期", period_options, index=2)
    auto_refresh = st.checkbox("开启自动刷新", value=True)
    refresh_interval = st.number_input("刷新间隔(秒)", 5, 60, 10, disabled=not auto_refresh)
    if auto_refresh:
        st_autorefresh(interval=refresh_interval * 1000, key="auto_refresh")
    st.markdown("---")
    st.subheader("📈 模拟合约")
    sim_entry = st.number_input("开仓价", value=0.0, format="%.2f")
    sim_side = st.selectbox("方向", ["多单", "空单"])
    sim_leverage = st.slider("杠杆倍数", 1, 100, 10)
    sim_quantity = st.number_input("数量 (ETH)", value=0.01, format="%.4f")

# 获取数据
data_dict, current_price, price_source, errors, source_display = fetch_all_data()

# 显示数据源状态
if data_dict:
    st.markdown(f'<div class="info-box">✅ 当前数据源：{source_display} | 价格源：{price_source}</div>', unsafe_allow_html=True)

# 简单显示错误（仅当严重时）
if errors and len(errors) > 3:
    st.warning(f"⚠️ 部分周期数据不可用 ({len(errors)}个周期)，将使用可用周期计算信号")

# 计算信号
if data_dict:
    ai_dir, ai_conf = st.session_state.ai.predict(data_dict)
    fusion_dir, fusion_conf = st.session_state.fusion.fuse_periods(data_dict)
else:
    ai_dir, ai_conf = 0, 0.0
    fusion_dir, fusion_conf = 0, 0

# 主布局
col1, col2 = st.columns([2.2, 1.3])

with col1:
    st.subheader(f"📊 合约K线 ({selected_period})")
    if data_dict and selected_period in data_dict:
        df = data_dict[selected_period].tail(100).copy()
        df['日期'] = df['timestamp']
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.7, 0.3],
                            subplot_titles=(f"ETHUSDT {selected_period}", "RSI"))
        fig.add_trace(go.Candlestick(x=df['日期'], open=df['open'], high=df['high'],
                                      low=df['low'], close=df['close'], name="K线"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['日期'], y=df['ma20'], name="MA20", line=dict(color="orange")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['日期'], y=df['ma60'], name="MA60", line=dict(color="blue")), row=1, col=1)
        if fusion_dir != 0:
            last_date = df['日期'].iloc[-1]
            last_price = df['close'].iloc[-1]
            if fusion_dir == 1:
                fig.add_annotation(x=last_date, y=last_price * 1.02,
                                   text="▲ 融合多", showarrow=True, arrowhead=2, arrowcolor="green")
            else:
                fig.add_annotation(x=last_date, y=last_price * 0.98,
                                   text="▼ 融合空", showarrow=True, arrowhead=2, arrowcolor="red")
        fig.add_trace(go.Scatter(x=df['日期'], y=df['rsi'], name="RSI", line=dict(color="purple")), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
        fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=600)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("等待数据...")

with col2:
    st.subheader("🧠 即时决策")
    dir_map = {1: "🔴 做多", -1: "🔵 做空", 0: "⚪ 观望"}
    st.markdown(f'<div class="ai-box">{dir_map[fusion_dir]}<br>置信度: {fusion_conf:.1%}</div>', unsafe_allow_html=True)

    if current_price is not None:
        st.metric("当前价格", f"${current_price:.2f}", delta_color="off")
    else:
        st.metric("当前价格", "获取中...")

    if sim_entry > 0 and current_price is not None and selected_period in data_dict:
        if sim_side == "多单":
            pnl = (current_price - sim_entry) * sim_quantity
            pnl_pct = (current_price - sim_entry) / sim_entry * sim_leverage * 100
            liq_price = calculate_liquidation_price(sim_entry, "long", sim_leverage)
        else:
            pnl = (sim_entry - current_price) * sim_quantity
            pnl_pct = (sim_entry - current_price) / sim_entry * sim_leverage * 100
            liq_price = calculate_liquidation_price(sim_entry, "short", sim_leverage)

        color_class = "profit" if pnl >= 0 else "loss"
        distance_to_liq = abs(current_price - liq_price) / current_price * 100

        st.markdown(f"""
        <div class="metric">
            <h4>模拟合约持仓</h4>
            <p>方向: {sim_side} | 杠杆: {sim_leverage}x</p>
            <p>开仓: ${sim_entry:.2f}</p>
            <p class="{color_class}">盈亏: ${pnl:.2f} ({pnl_pct:.2f}%)</p>
            <p>强平价: <span class="warning">${liq_price:.2f}</span></p>
            <p>距强平: {distance_to_liq:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)

        if (sim_side == "多单" and current_price <= liq_price) or (sim_side == "空单" and current_price >= liq_price):
            st.error("🚨 强平风险！当前价格已触及强平线")
        elif distance_to_liq < 5:
            st.warning(f"⚠️ 距离强平仅 {distance_to_liq:.2f}%，请注意风险")
    else:
        st.info("请输入开仓价以查看模拟盈亏与强平分析")
