# -*- coding: utf-8 -*-
"""
🚀 全天候智能交易监控中心 · 最終優化版
多週期切換 | AI預測 | 模擬盈虧聯動 | 微信提醒 | 永久在線

使用前請先在 Streamlit Cloud 的 Secrets 中設定：
BINANCE_API_KEY / BINANCE_SECRET_KEY (測試網可用任意值)
PUSHPLUS_TOKEN (可選)
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
import os
from streamlit_autorefresh import st_autorefresh
import warnings
warnings.filterwarnings('ignore')

# -------------------- 密鑰讀取 (從 Streamlit Secrets) --------------------
BINANCE_API_KEY = st.secrets.get("BINANCE_API_KEY", "")
BINANCE_SECRET_KEY = st.secrets.get("BINANCE_SECRET_KEY", "")
PUSHPLUS_TOKEN = st.secrets.get("PUSHPLUS_TOKEN", "")

# -------------------- 異步數據獲取器 --------------------
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
        except Exception as e:
            print(f"Error fetching {period}: {e}")
            return period, None

    async def fetch_all(self):
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_period(session, p) for p in self.periods]
            results = await asyncio.gather(*tasks)
            data_dict = {p: df for p, df in results if df is not None}
            return data_dict

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

# -------------------- AI 預測模組（可載入 LSTM 模型，無模型時用規則） --------------------
class AIPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_cols = ['ma20', 'ma60', 'rsi', 'macd', 'bb_high', 'bb_low', 'atr', 'volume_ratio']
        self.seq_len = 20
        self._load_model()

    def _load_model(self):
        """嘗試載入預訓練模型，若無則跳過"""
        try:
            # 若不安裝 tensorflow 或 joblib，此處會被捕獲，繼續使用規則
            import tensorflow as tf
            import joblib
            model_path = "models/lstm_model.h5"
            scaler_path = "models/scaler.pkl"
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.model = tf.keras.models.load_model(model_path)
                self.scaler = joblib.load(scaler_path)
                print("✅ 載入 LSTM 模型成功")
            else:
                print("⚠️ 未找到模型檔案，使用規則模擬")
        except Exception as e:
            print(f"⚠️ 模型載入失敗: {e}，使用規則模擬")

    def predict_with_model(self, df):
        """使用 LSTM 模型預測"""
        if len(df) < self.seq_len + 1:
            return 0, 0.5
        recent = df.iloc[-(self.seq_len+1):-1]  # 用前 seq_len 根預測下一根
        X_raw = recent[self.feature_cols].values
        X_scaled = self.scaler.transform(X_raw)
        X_input = X_scaled.reshape(1, self.seq_len, len(self.feature_cols))
        prob = self.model.predict(X_input, verbose=0)[0][0]
        if prob > 0.55:
            return 1, prob
        elif prob < 0.45:
            return -1, 1 - prob
        else:
            return 0, prob

    def predict_with_rules(self, df_dict):
        """規則模擬（備用）"""
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

    def predict(self, df_dict):
        """統一預測介面：優先使用模型（僅4h），否則規則"""
        if self.model is not None and '4h' in df_dict:
            return self.predict_with_model(df_dict['4h'])
        else:
            return self.predict_with_rules(df_dict)

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

# -------------------- 緩存數據獲取（加強錯誤處理） --------------------
@st.cache_data(ttl=60)
def fetch_all_data():
    """獲取所有週期數據並計算指標，若失敗則回傳空字典"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        fetcher = AsyncDataFetcher()
        data_dict = loop.run_until_complete(fetcher.fetch_all())
        if not data_dict:
            st.warning("無法獲取幣安數據，請檢查網路或 API 設定")
            return {}
        for p in data_dict:
            data_dict[p] = add_indicators(data_dict[p])
        return data_dict
    except Exception as e:
        st.error(f"數據獲取失敗: {e}")
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

st.title("🧠 全天候智能交易監控中心 · 最終優化版")
st.caption("數據緩存60秒｜多週期切換｜AI預測｜盈虧聯動｜微信提醒")

# 初始化 AI 和融合模組（只初始化一次）
if 'ai' not in st.session_state:
    st.session_state.ai = AIPredictor()
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
    # 盈虧價格源（預設使用顯示週期）
    use_display_period = st.radio("盈虧價格源", ["使用顯示週期", "使用實時價格 (需WebSocket)"], index=0) == "使用顯示週期"

# 獲取數據
data_dict = fetch_all_data()

# 計算 AI 和融合訊號
if data_dict:
    ai_dir, ai_conf = st.session_state.ai.predict(data_dict)
    fusion_dir, fusion_conf = st.session_state.fusion.fuse_periods(data_dict)
    # 發送微信提醒（當融合訊號非零且非冷卻）
    if fusion_dir != 0 and selected_period in data_dict:
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
        # 止損檢測
        if sim_stop > 0:
            if (sim_entry > sim_stop and current_price <= sim_stop) or (sim_entry < sim_stop and current_price >= sim_stop):
                st.warning("⚠️ 止損觸發！")
