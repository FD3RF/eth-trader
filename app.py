# -*- coding: utf-8 -*-
"""
🚀 全天候智能合約交易監控中心 · 終極故障轉移版
多端點自動切換（合約/現貨）｜HTTP 451 智能規避｜AI預測｜強平分析｜微信提醒
數據源：幣安公開 API（自動選擇可用節點）
"""

import streamlit as st
import pandas as pd
import numpy as np
import ta
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time
from streamlit_autorefresh import st_autorefresh
import warnings
warnings.filterwarnings('ignore')

# -------------------- 強平價格計算（逐倉，簡化版） --------------------
def calculate_liquidation_price(entry_price, side, leverage):
    if side == "long":
        return entry_price * (1 - 1/leverage)
    else:
        return entry_price * (1 + 1/leverage)

# -------------------- 智能數據獲取器（多端點故障轉移） --------------------
class SmartDataFetcher:
    def __init__(self):
        # 合約端點（優先）
        self.fapi_endpoints = [
            "https://fapi.binance.com",
            "https://fapi1.binance.com",
            "https://fapi2.binance.com",
            "https://fapi3.binance.com"
        ]
        # 現貨端點（備用）
        self.api_endpoints = [
            "https://api.binance.com",
            "https://api1.binance.com",
            "https://api2.binance.com",
            "https://api3.binance.com"
        ]
        self.symbol = "ETHUSDT"
        self.periods = ['1m', '5m', '15m', '1h', '4h', '1d']
        self.limit = 200
        self.timeout = 10
        self.retries = 2
        self.current_source = "合約"  # 用於界面顯示

    def _try_endpoints(self, base_urls, path, params, is_mark_price=False):
        """嘗試多個端點，返回 (response_json, success_endpoint)"""
        for base in base_urls:
            url = f"{base}{path}"
            for attempt in range(self.retries):
                try:
                    resp = requests.get(url, params=params, timeout=self.timeout)
                    if resp.status_code == 200:
                        return resp.json(), base
                    elif resp.status_code == 451:
                        # 地區封鎖，直接跳過此端點
                        break
                    # 其他錯誤，重試
                    time.sleep(1)
                except Exception:
                    time.sleep(1)
            # 端點重試失敗，嘗試下一個
        return None, None

    def fetch_kline(self, period):
        """獲取K線，優先合約，失敗則現貨"""
        # 先嚐試合約
        params = {'symbol': self.symbol, 'interval': period, 'limit': self.limit}
        data, base = self._try_endpoints(self.fapi_endpoints, "/fapi/v1/klines", params)
        if data is not None:
            self.current_source = "合約"
            return self._parse_kline(data), None

        # 合約失敗，嘗試現貨
        data, base = self._try_endpoints(self.api_endpoints, "/api/v3/klines", params)
        if data is not None:
            self.current_source = "現貨"
            return self._parse_kline(data), None

        return None, "所有端點K線獲取失敗"

    def _parse_kline(self, data):
        """將原始K線數據轉為DataFrame"""
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'num_trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        return df

    def fetch_mark_price(self):
        """獲取標記價格（合約專用），若失敗則返回None"""
        params = {'symbol': self.symbol}
        data, base = self._try_endpoints(self.fapi_endpoints, "/fapi/v1/premiumIndex", params)
        if data is not None:
            return float(data['markPrice']), None
        return None, "無法獲取標記價格（合約端點不可用）"

    def fetch_current_price(self):
        """獲取當前價格（現貨最新價）作為備用"""
        params = {'symbol': self.symbol}
        data, base = self._try_endpoints(self.api_endpoints, "/api/v3/ticker/price", params)
        if data is not None:
            return float(data['price']), None
        return None, "無法獲取現貨價格"

    def fetch_all(self):
        """獲取所有週期K線，並決定價格源"""
        data_dict = {}
        errors = []
        source_display = "未知"

        # 獲取所有週期K線（使用同一個source）
        first_period = True
        for p in self.periods:
            df, err = self.fetch_kline(p)
            if df is not None:
                data_dict[p] = df
                if first_period:
                    source_display = self.current_source
                    first_period = False
            else:
                errors.append(f"{p}: {err}")

        # 獲取價格（優先標記價格，否則用現貨最新價）
        price = None
        price_source = ""
        if data_dict:
            # 嘗試獲取標記價格
            mark, err = self.fetch_mark_price()
            if mark is not None:
                price = mark
                price_source = "標記價格(合約)"
            else:
                # 備用：使用現貨最新價
                spot_price, err2 = self.fetch_current_price()
                if spot_price is not None:
                    price = spot_price
                    price_source = "現貨最新價"
                else:
                    # 最後備用：使用所選週期K線最新收盤價
                    last_period = self.periods[-1]
                    if last_period in data_dict:
                        price = data_dict[last_period]['close'].iloc[-1]
                        price_source = f"{last_period}收盤價"
                        errors.append(f"價格源使用K線收盤價（{last_period}）")
                    else:
                        errors.append("無法獲取任何價格")

        return data_dict, price, price_source, errors, source_display

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

# -------------------- AI 預測（簡化規則版） --------------------
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

# -------------------- 多週期融合 --------------------
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

# -------------------- 微信推送（選用） --------------------
PUSHPLUS_TOKEN = st.secrets.get("PUSHPLUS_TOKEN", "")
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
    content = f"""【合約訊號提醒】
方向: {dir_str}
置信度: {confidence:.1%}
價格: ${price:.2f}
時間: {now.strftime('%Y-%m-%d %H:%M:%S')}
{reason}"""
    url = "http://www.pushplus.plus/send"
    data = {"token": PUSHPLUS_TOKEN, "title": "🤖 合約訊號", "content": content, "template": "txt"}
    try:
        requests.post(url, json=data, timeout=5)
        last_signal_time = now
        last_signal_direction = direction
    except:
        pass

# -------------------- 緩存數據獲取（智能故障轉移） --------------------
@st.cache_data(ttl=60)
def fetch_all_data():
    fetcher = SmartDataFetcher()
    data_dict, price, price_source, errors, source_display = fetcher.fetch_all()
    if data_dict:
        for p in data_dict:
            data_dict[p] = add_indicators(data_dict[p])
    return data_dict, price, price_source, errors, source_display

# -------------------- Streamlit 介面 --------------------
st.set_page_config(page_title="合約智能監控·100倍槓桿", layout="wide")
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
.error-box { background: #3A1F1F; border-left: 6px solid #FF5555; padding: 10px; border-radius: 5px; margin-bottom: 10px; }
.info-box { background: #1A2A3A; border-left: 6px solid #00F5A0; padding: 10px; border-radius: 5px; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("🧠 合約智能監控中心 · 終極故障轉移版")
st.caption("數據源：智能切換（合約/現貨）｜多週期｜AI預測｜強平分析｜微信提醒")

# 初始化
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
    st.subheader("📈 模擬合約")
    sim_entry = st.number_input("開倉價", value=0.0, format="%.2f")
    sim_side = st.selectbox("方向", ["多單", "空單"])
    sim_leverage = st.slider("槓桿倍數", 1, 100, 10)
    sim_quantity = st.number_input("數量 (ETH)", value=0.01, format="%.4f")

# 獲取數據
data_dict, current_price, price_source, errors, source_display = fetch_all_data()

# 顯示數據源狀態
if data_dict:
    st.markdown(f'<div class="info-box">✅ 當前數據源：{source_display} | 價格源：{price_source}</div>', unsafe_allow_html=True)

# 顯示錯誤訊息
if errors:
    with st.container():
        st.markdown('<div class="error-box">', unsafe_allow_html=True)
        st.error("⚠️ 部分數據獲取失敗，詳細錯誤：")
        for err in errors[:5]:
            st.write(f"- {err}")
        if len(errors) > 5:
            st.write(f"... 還有 {len(errors)-5} 條錯誤")
        st.markdown('</div>', unsafe_allow_html=True)

# 計算訊號
if data_dict:
    ai_dir, ai_conf = st.session_state.ai.predict(data_dict)
    fusion_dir, fusion_conf = st.session_state.fusion.fuse_periods(data_dict)
    # 推送
    if fusion_dir != 0 and selected_period in data_dict and PUSHPLUS_TOKEN:
        price_alert = data_dict[selected_period]['close'].iloc[-1]
        send_signal_alert(fusion_dir, fusion_conf, price_alert, "融合訊號")
else:
    ai_dir, ai_conf = 0, 0.0
    fusion_dir, fusion_conf = 0, 0

# 主佈局
col1, col2 = st.columns([2.2, 1.3])

with col1:
    st.subheader(f"📊 合約K線 ({selected_period})")
    if data_dict and selected_period in data_dict:
        df = data_dict[selected_period].tail(100).copy()
        df['日期'] = df['timestamp']
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.7, 0.3],
                            subplot_titles=(f"ETHUSDT {selected_period}", "RSI"))
        fig.add_trace(go.Candlestick(x=df['日期'], open=df['open'], high=df['high'],
                                      low=df['low'], close=df['close'], name="K線"), row=1, col=1)
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
        st.info("等待數據...")

with col2:
    st.subheader("🧠 即時決策")
    dir_map = {1: "🔴 做多", -1: "🔵 做空", 0: "⚪ 觀望"}
    st.markdown(f'<div class="ai-box">{dir_map[fusion_dir]}<br>置信度: {fusion_conf:.1%}</div>', unsafe_allow_html=True)

    # 價格顯示
    if current_price is not None:
        st.metric("當前價格", f"${current_price:.2f}", delta_color="off")
    else:
        st.metric("當前價格", "獲取中...")

    # 模擬合約盈虧與強平分析
    if sim_entry > 0 and current_price is not None and selected_period in data_dict:
        if sim_side == "多單":
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
            <h4>模擬合約持倉</h4>
            <p>方向: {sim_side} | 槓桿: {sim_leverage}x</p>
            <p>開倉: ${sim_entry:.2f}</p>
            <p class="{color_class}">盈虧: ${pnl:.2f} ({pnl_pct:.2f}%)</p>
            <p>強平價: <span class="warning">${liq_price:.2f}</span></p>
            <p>距強平: {distance_to_liq:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)

        if (sim_side == "多單" and current_price <= liq_price) or (sim_side == "空單" and current_price >= liq_price):
            st.error("🚨 強平風險！當前價格已觸及強平線")
        elif distance_to_liq < 5:
            st.warning(f"⚠️ 距離強平僅 {distance_to_liq:.2f}%，請注意風險")
    else:
        st.info("請輸入開倉價以查看模擬盈虧與強平分析")
