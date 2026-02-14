# -*- coding: utf-8 -*-
"""
🚀 全天候智能交易监控中心 · 公開數據版
多週期切換 | AI預測 | 模擬盈虧聯動 | 微信提醒 | 永久在線
完全使用幣安公開 API，無需任何金鑰設定。
"""

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
import time
from streamlit_autorefresh import st_autorefresh
import warnings
warnings.filterwarnings('ignore')

# -------------------- 公開數據獲取器（異步 + 同步備援） --------------------
class PublicDataFetcher:
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3/klines"
        self.symbol = "ETHUSDT"
        self.periods = ['1m', '5m', '15m', '1h', '4h', '1d']
        self.limit = 200

    async def fetch_period_async(self, session, period):
        """異步獲取單個週期"""
        params = {'symbol': self.symbol, 'interval': period, 'limit': self.limit}
        try:
            async with session.get(self.base_url, params=params, timeout=10) as resp:
                if resp.status != 200:
                    return period, None
                data = await resp.json()
                if isinstance(data, list):
                    df = self._convert_to_dataframe(data)
                    return period, df
                else:
                    return period, None
        except Exception as e:
            print(f"Async error {period}: {e}")
            return period, None

    def fetch_period_sync(self, period):
        """同步獲取單個週期（備用）"""
        params = {'symbol': self.symbol, 'interval': period, 'limit': self.limit}
        try:
            resp = requests.get(self.base_url, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list):
                    df = self._convert_to_dataframe(data)
                    return df
            return None
        except Exception as e:
            print(f"Sync error {period}: {e}")
            return None

    def _convert_to_dataframe(self, data):
        """將幣安原始數據轉為 DataFrame"""
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'num_trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        return df

    async def fetch_all_async(self):
        """嘗試異步獲取所有週期"""
        try:
            async with aiohttp.ClientSession() as session:
                tasks = [self.fetch_period_async(session, p) for p in self.periods]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                data_dict = {}
                for p, df in results:
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        data_dict[p] = df
                if data_dict:
                    return data_dict
        except Exception as e:
            print(f"Async fetch all failed: {e}")
        return None

    def fetch_all_sync(self):
        """同步獲取所有週期（備用）"""
        data_dict = {}
        for p in self.periods:
            df = self.fetch_period_sync(p)
            if df is not None:
                data_dict[p] = df
        return data_dict if data_dict else None

# -------------------- 指標計算 --------------------
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

# -------------------- AI 預測模組（簡化版，僅規則） --------------------
class SimpleAIPredictor:
    """純規則預測，不依賴任何模型檔案"""
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

# -------------------- 多週期策略融合 --------------------
class MultiPeriodFusion:
    def __init__(self):
        self.period_weights = {
            '1m': 0.05,
            '5m': 0.1,
            '15m': 0.15,
            '1h': 0.2,
            '4h': 0.25,
            '1d': 0.25
        }
        self.strategy_weights = {'trend': 0.5, 'oscillator': 0.3, 'volume': 0.2}

    def get_period_signal(self, df):
        last = df.iloc[-1]
        signals = {}
        # 趨勢
        if last['ma20'] > last['ma60']:
            signals['trend'] = 1
        elif last['ma20'] < last['ma60']:
            signals['trend'] = -1
        else:
            signals['trend'] = 0
        # 震盪
        if last['rsi'] < 30:
            signals['oscillator'] = 1
        elif last['rsi'] > 70:
            signals['oscillator'] = -1
        else:
            signals['oscillator'] = 0
        # 成交量
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

# -------------------- 微信推送（帶冷卻） --------------------
PUSHPLUS_TOKEN = ""  # 如需推送，請在 Streamlit Secrets 設定
last_signal_time = None
last_signal_direction = 0
signal_cooldown_minutes = 5

def send_signal_alert(direction, confidence, price, reason=""):
    global last_signal_time, last_signal_direction
    if not PUSHPLUS_TOKEN:
        return
    now = datetime.now()
    if direction == last_signal_direction and last_signal_time and (now - last_signal_time).total_seconds() < signal_cooldown_minutes * 60:
        return
    dir_str = "做多" if direction == 1 else "做空"
    content = f"""【交易訊號提醒】
方向: {dir_str}
置信度: {confidence:.1%}
當前價格: ${price:.2f}
時間: {now.strftime('%Y-%m-%d %H:%M:%S')}
{reason}"""
    url = "http://www.pushplus.plus/send"
    data = {"token": PUSHPLUS_TOKEN, "title": "🤖 交易訊號", "content": content, "template": "txt"}
    try:
        requests.post(url, json=data, timeout=5)
        last_signal_time = now
        last_signal_direction = direction
    except Exception as e:
        print(f"推送失敗: {e}")

# -------------------- 緩存數據獲取（自動異步 + 同步備援） --------------------
@st.cache_data(ttl=60)
def fetch_all_data():
    """嘗試異步獲取，若失敗則改用同步"""
    fetcher = PublicDataFetcher()
    
    # 先嘗試異步
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        data_dict = loop.run_until_complete(fetcher.fetch_all_async())
        if data_dict:
            for p in data_dict:
                data_dict[p] = add_indicators(data_dict[p])
            return data_dict
    except Exception as e:
        print(f"非同步獲取失敗，切換至同步模式: {e}")
    
    # 異步失敗，改用同步
    data_dict = fetcher.fetch_all_sync()
    if data_dict:
        for p in data_dict:
            data_dict[p] = add_indicators(data_dict[p])
        return data_dict
    else:
        st.error("無法獲取幣安數據，請檢查網路連線")
        return {}

# -------------------- Streamlit 介面 --------------------
st.set_page_config(page_title="全天候智能交易監控中心", layout="wide")
st.markdown("""
<style>
.stApp { background-color: #0B0E14; color: white; }
.ai-box { background: #1A1D27; border-radius: 10px; padding: 20px; border-left: 6px solid #00F5A0; }
.metric { background: #232734; padding: 15px; border-radius: 8px; }
.signal-buy { color: #00F5A0; font-weight: bold; }
.signal-sell { color: #FF5555; font-weight: bold; }
.profit { color: #00F5A0; }
.loss { color: #FF5555; }
</style>
""", unsafe_allow_html=True)

st.title("🧠 全天候智能交易監控中心 · 公開數據版")
st.caption("數據緩存60秒｜多週期切換｜AI預測｜盈虧聯動｜微信提醒")

# 初始化（使用簡化版 AI）
if 'ai' not in st.session_state:
    st.session_state.ai = SimpleAIPredictor()
if 'fusion' not in st.session_state:
    st.session_state.fusion = MultiPeriodFusion()

# 側邊欄
with st.sidebar:
    st.header("⚙️ 控制面板")
    period_options = ['1m', '5m', '15m', '1h', '4h', '1d']
    selected_period = st.selectbox("選擇K線週期", period_options, index=4)
    auto_refresh = st.checkbox("開啟自動刷新", value=True)
    refresh_interval = st.number_input("刷新間隔(秒)", 5, 60, 10, disabled=not auto_refresh)
    if auto_refresh:
        st_autorefresh(interval=refresh_interval * 1000, key="auto_refresh")
    st.markdown("---")
    st.subheader("💰 模擬交易")
    sim_entry = st.number_input("入場價", value=0.0, format="%.2f")
    sim_stop = st.number_input("止損價", value=0.0, format="%.2f")
    sim_quantity = st.number_input("數量 (ETH)", value=0.01, format="%.4f")

# 獲取數據
data_dict = fetch_all_data()

# 計算 AI 和融合訊號
if data_dict:
    ai_dir, ai_conf = st.session_state.ai.predict(data_dict)
    fusion_dir, fusion_conf = st.session_state.fusion.fuse_periods(data_dict)
    # 發送微信提醒（如有設定 token）
    if fusion_dir != 0 and selected_period in data_dict and PUSHPLUS_TOKEN:
        price_for_alert = data_dict[selected_period]['close'].iloc[-1]
        send_signal_alert(fusion_dir, fusion_conf, price_for_alert, "融合訊號觸發")
else:
    ai_dir, ai_conf = 0, 0.0
    fusion_dir, fusion_conf = 0, 0

# 主佈局
col1, col2 = st.columns([2.2, 1.3])

with col1:
    st.subheader(f"📊 實時K線 ({selected_period})")
    if data_dict and selected_period in data_dict:
        df = data_dict[selected_period].tail(100).copy()
        df['日期'] = df['timestamp']
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.7, 0.3],
                            subplot_titles=(f"ETH/USDT {selected_period}", "RSI"))
        # K線
        fig.add_trace(go.Candlestick(x=df['日期'], open=df['open'], high=df['high'],
                                      low=df['low'], close=df['close'], name="K線"), row=1, col=1)
        # 均線
        fig.add_trace(go.Scatter(x=df['日期'], y=df['ma20'], name="MA20", line=dict(color="orange")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['日期'], y=df['ma60'], name="MA60", line=dict(color="blue")), row=1, col=1)
        # 融合訊號箭頭
        if fusion_dir != 0:
            last_date = df['日期'].iloc[-1]
            last_price = df['close'].iloc[-1]
            if fusion_dir == 1:
                fig.add_annotation(x=last_date, y=last_price * 1.02,
                                   text="▲ 融合多", showarrow=True, arrowhead=2, arrowcolor="green")
            else:
                fig.add_annotation(x=last_date, y=last_price * 0.98,
                                   text="▼ 融合空", showarrow=True, arrowhead=2, arrowcolor="red")
        # RSI
        fig.add_trace(go.Scatter(x=df['日期'], y=df['rsi'], name="RSI", line=dict(color="purple")), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
        fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=600)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("等待數據...")

with col2:
    st.subheader("🧠 實時決策")
    dir_map = {1: "🔴 做多", -1: "🔵 做空", 0: "⚪ 觀望"}
    st.markdown(f'<div class="ai-box">{dir_map[fusion_dir]}<br>置信度: {fusion_conf:.1%}</div>', unsafe_allow_html=True)

    # 模擬盈虧顯示
    if sim_entry > 0 and selected_period in data_dict:
        current_price = data_dict[selected_period]['close'].iloc[-1]
        pnl = (current_price - sim_entry) * sim_quantity
        pnl_pct = (current_price - sim_entry) / sim_entry * 100
        color_class = "profit" if pnl >= 0 else "loss"
        st.markdown(f"""
        <div class="metric">
            <h4>模擬持倉</h4>
            <p>入場: ${sim_entry:.2f}</p>
            <p>當前: ${current_price:.2f}</p>
            <p class="{color_class}">盈虧: ${pnl:.2f} ({pnl_pct:.2f}%)</p>
            <p>止損: ${sim_stop:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
        if sim_stop > 0:
            if (sim_entry > sim_stop and current_price <= sim_stop) or (sim_entry < sim_stop and current_price >= sim_stop):
                st.warning("⚠️ 止損觸發！")
