# -*- coding: utf-8 -*-
"""
🚀 全天候智能合約交易監控中心 · 终极绝望版
100倍槓桿 | 智能故障轉移（幣安/Bybit/OKX/CryptoCompare） | 模擬數據回退 | AI信號 | 強平分析 | 微信提醒
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

# -------------------- 強平價格計算 --------------------
def calculate_liquidation_price(entry_price, side, leverage):
    if side == "long":
        return entry_price * (1 - 1/leverage)
    else:
        return entry_price * (1 + 1/leverage)

# -------------------- 生成模拟K线数据（用于所有API均失败时） --------------------
def generate_simulated_data(periods, days=2):
    """生成模拟ETH/USDT K线数据"""
    data_dict = {}
    end_time = datetime.now()
    for period in periods:
        # 确定K线间隔（秒）
        interval_seconds = {
            '1m': 60, '5m': 300, '15m': 900,
            '1h': 3600, '4h': 14400, '1d': 86400
        }.get(period, 60)
        num_bars = 200
        timestamps = [end_time - timedelta(seconds=interval_seconds * (num_bars - i - 1)) for i in range(num_bars)]
        # 生成随机价格走势（带趋势）
        base_price = 2000
        price = base_price
        prices = []
        for i in range(num_bars):
            change = np.random.randn() * 10 + (i / num_bars) * 5  # 微弱上升趋势
            price += change
            prices.append(max(price, 10))
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices,
            'high': [p * (1 + abs(np.random.randn()*0.005)) for p in prices],
            'low': [p * (1 - abs(np.random.randn()*0.005)) for p in prices],
            'close': [p * (1 + np.random.randn()*0.002) for p in prices],
            'volume': np.random.randint(1000, 5000, num_bars)
        })
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        data_dict[period] = df
    return data_dict, 2000.0  # 返回模拟数据和模拟价格

# -------------------- 智能數據獲取器（含模拟回退） --------------------
class DesperateDataFetcher:
    def __init__(self):
        self.symbol = "ETHUSDT"
        self.periods = ['1m', '5m', '15m', '1h', '4h', '1d']
        self.limit = 200
        self.timeout = 5  # 缩短超时，避免卡死
        self.retries = 1
        self.current_source = "未知"

        # 交易所列表（优先级从高到低）
        self.exchanges = [
            # 币安合约镜像
            {'name': '币安合约', 'priority': 1, 'type': 'binance_fapi',
             'hosts': ['fapi.binance.com', 'fapi1.binance.com', 'fapi2.binance.com', 'fapi3.binance.com'],
             'url_path': '/fapi/v1/klines', 'params': {'symbol': self.symbol, 'interval': None, 'limit': self.limit}},
            # 币安现货镜像
            {'name': '币安现货', 'priority': 2, 'type': 'binance_spot',
             'hosts': ['api.binance.com', 'api1.binance.com', 'api2.binance.com', 'api3.binance.com'],
             'url_path': '/api/v3/klines', 'params': {'symbol': self.symbol, 'interval': None, 'limit': self.limit}},
            # Bybit
            {'name': 'Bybit', 'priority': 3, 'type': 'bybit',
             'hosts': ['api.bybit.com'],
             'url_path': '/v5/market/kline', 'params': {'category': 'linear', 'symbol': self.symbol, 'interval': None, 'limit': self.limit}},
            # OKX
            {'name': 'OKX', 'priority': 4, 'type': 'okx',
             'hosts': ['www.okx.com'],
             'url_path': '/api/v5/market/candles', 'params': {'instId': self.symbol + '-SWAP', 'bar': None, 'limit': self.limit}},
            # CryptoCompare（公共聚合API，一般国内可用）
            {'name': 'CryptoCompare', 'priority': 5, 'type': 'cryptocompare',
             'hosts': ['min-api.cryptocompare.com'],
             'url_path': '/data/v2/histoday', 'params': {'fsym': 'ETH', 'tsym': 'USD', 'limit': self.limit, 'aggregate': 1},
             'period_map': {'1d': 'day', '1h': 'hour', '4h': 'hour'}}  # 需特殊处理
        ]

        # 价格源列表
        self.price_sources = [
            {'name': '币安合約標記價', 'priority': 1,
             'hosts': ['fapi.binance.com', 'fapi1.binance.com', 'fapi2.binance.com', 'fapi3.binance.com'],
             'url_path': '/fapi/v1/premiumIndex', 'params': {'symbol': self.symbol},
             'parser': lambda data: float(data['markPrice'])},
            {'name': '币安現貨最新價', 'priority': 2,
             'hosts': ['api.binance.com', 'api1.binance.com', 'api2.binance.com', 'api3.binance.com'],
             'url_path': '/api/v3/ticker/price', 'params': {'symbol': self.symbol},
             'parser': lambda data: float(data['price'])},
            {'name': 'Bybit最新價', 'priority': 3,
             'hosts': ['api.bybit.com'],
             'url_path': '/v5/market/tickers', 'params': {'category': 'linear', 'symbol': self.symbol},
             'parser': lambda data: float(data['result']['list'][0]['markPrice'])},
            {'name': 'OKX最新價', 'priority': 4,
             'hosts': ['www.okx.com'],
             'url_path': '/api/v5/market/ticker', 'params': {'instId': self.symbol + '-SWAP'},
             'parser': lambda data: float(data['data'][0]['last'])},
            {'name': 'CryptoCompare價格', 'priority': 5,
             'hosts': ['min-api.cryptocompare.com'],
             'url_path': '/data/price', 'params': {'fsym': 'ETH', 'tsyms': 'USD'},
             'parser': lambda data: float(data['USD'])}
        ]

    def _fetch_from_exchange(self, exchange, period):
        """尝试从单个交易所获取K线"""
        for host in exchange['hosts']:
            url = f"https://{host}{exchange['url_path']}"
            params = exchange['params'].copy()
            # 处理周期参数
            if exchange['type'] in ('binance_fapi', 'binance_spot', 'bybit'):
                params['interval'] = period
            elif exchange['type'] == 'okx':
                params['bar'] = period
            elif exchange['type'] == 'cryptocompare':
                # CryptoCompare需要特殊处理：histoday/histohour
                if period == '1d':
                    url = f"https://{host}/data/v2/histoday"
                elif period in ('1h', '4h'):
                    url = f"https://{host}/data/v2/histohour"
                    params['limit'] = self.limit
                    if period == '4h':
                        params['aggregate'] = 4  # 4小时K线
                else:
                    return None, f"{exchange['name']} 不支持周期 {period}"
                params['fsym'] = 'ETH'
                params['tsym'] = 'USD'
                params.pop('interval', None)
                params.pop('bar', None)
            try:
                resp = requests.get(url, params=params, timeout=self.timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    # 解析
                    if exchange['type'] in ('binance_fapi', 'binance_spot'):
                        df = self._parse_binance_kline(data)
                        return df, None
                    elif exchange['type'] == 'bybit':
                        if data.get('retCode') == 0:
                            df = self._parse_bybit_kline(data)
                            return df, None
                        else:
                            return None, f"{exchange['name']} 业务错误: {data.get('retMsg')}"
                    elif exchange['type'] == 'okx':
                        if data.get('code') == '0':
                            df = self._parse_okx_kline(data)
                            return df, None
                        else:
                            return None, f"{exchange['name']} 业务错误: {data.get('msg')}"
                    elif exchange['type'] == 'cryptocompare':
                        if data.get('Response') == 'Success':
                            df = self._parse_cryptocompare_kline(data, period)
                            return df, None
                        else:
                            return None, f"{exchange['name']} 错误: {data.get('Message')}"
                elif resp.status_code == 451:
                    return None, f"{exchange['name']} HTTP 451 (被封鎖)"
                else:
                    return None, f"{exchange['name']} HTTP {resp.status_code}"
            except requests.exceptions.Timeout:
                return None, f"{exchange['name']} 超时"
            except requests.exceptions.ConnectionError:
                return None, f"{exchange['name']} 连接错误"
            except Exception as e:
                return None, f"{exchange['name']} 异常: {str(e)}"
        return None, f"{exchange['name']} 所有主机失败"

    def _fetch_price_from_source(self, source):
        """尝试从单个价格源获取价格"""
        for host in source['hosts']:
            url = f"https://{host}{source['url_path']}"
            params = source['params'].copy()
            try:
                resp = requests.get(url, params=params, timeout=self.timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    try:
                        price = source['parser'](data)
                        return price, None
                    except Exception as e:
                        return None, f"{source['name']} 解析失败: {e}"
                elif resp.status_code == 451:
                    return None, f"{source['name']} HTTP 451"
                else:
                    return None, f"{source['name']} HTTP {resp.status_code}"
            except Exception as e:
                return None, f"{source['name']} 请求异常: {str(e)}"
        return None, f"{source['name']} 所有主机失败"

    # ---------- 解析函数 ----------
    def _parse_binance_kline(self, data):
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'num_trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        return df

    def _parse_bybit_kline(self, data):
        items = data['result']['list']
        df = pd.DataFrame(items, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

    def _parse_okx_kline(self, data):
        items = data['data']
        df = pd.DataFrame(items, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'volCcy', 'volCcyQuote', 'confirm'])
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

    def _parse_cryptocompare_kline(self, data, period):
        # CryptoCompare返回格式：{"Data":{"Data":[{"time":...,"open":...,"high":...,"low":...,"close":...,"volumefrom":...}]}}
        items = data['Data']['Data']
        df = pd.DataFrame(items)
        df['timestamp'] = pd.to_datetime(df['time'], unit='s')
        df.rename(columns={'volumefrom': 'volume'}, inplace=True)
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        return df

    def fetch_all(self):
        """获取所有数据，失败时返回模拟数据"""
        all_errors = []
        data_dict = {}
        price = None
        price_source = "无"
        source_display = "无"

        # 按优先级尝试获取K线
        for period in self.periods:
            period_success = False
            for exch in sorted(self.exchanges, key=lambda x: x['priority']):
                df, err = self._fetch_from_exchange(exch, period)
                if df is not None:
                    data_dict[period] = df
                    source_display = exch['name']
                    period_success = True
                    break
                else:
                    all_errors.append(f"{period} {err}")
            if not period_success:
                all_errors.append(f"{period} 所有交易所失败")

        # 如果有至少一个周期成功，则尝试获取价格
        if data_dict:
            for src in sorted(self.price_sources, key=lambda x: x['priority']):
                p, err = self._fetch_price_from_source(src)
                if p is not None:
                    price = p
                    price_source = src['name']
                    break
                else:
                    all_errors.append(f"价格 {err}")
            # 如果价格仍未获取到，使用4h收盘价
            if price is None and '4h' in data_dict:
                price = data_dict['4h']['close'].iloc[-1]
                price_source = "4h收盘价(备用)"
            elif price is None and data_dict:
                first = next(iter(data_dict))
                price = data_dict[first]['close'].iloc[-1]
                price_source = f"{first}收盘价(备用)"
        else:
            # 所有周期都失败，生成模拟数据
            all_errors.append("所有外部数据源均失败，启用模拟数据")
            data_dict, price = generate_simulated_data(self.periods)
            source_display = "模拟数据(演示模式)"
            price_source = "模拟价格"

        return data_dict, price, price_source, all_errors, source_display

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

# -------------------- AI预测 --------------------
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

# -------------------- 微信推送 --------------------
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

# -------------------- 缓存数据获取 --------------------
@st.cache_data(ttl=60)
def fetch_all_data():
    fetcher = DesperateDataFetcher()
    data_dict, price, price_source, errors, source_display = fetcher.fetch_all()
    if data_dict:
        for p in data_dict:
            data_dict[p] = add_indicators(data_dict[p])
    return data_dict, price, price_source, errors, source_display

# -------------------- Streamlit 界面 --------------------
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
.demo-box { background: #2A2A1A; border-left: 6px solid #FFAA00; padding: 10px; border-radius: 5px; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("🧠 合約智能監控中心 · 终极绝望版")
st.caption("数据源：自动切换+模拟回退｜多周期｜AI预测｜强平分析｜微信提醒")

# 初始化
if 'ai' not in st.session_state:
    st.session_state.ai = SimpleAIPredictor()
if 'fusion' not in st.session_state:
    st.session_state.fusion = MultiPeriodFusion()

# 侧边栏
with st.sidebar:
    st.header("⚙️ 控制面板")
    period_options = ['1m', '5m', '15m', '1h', '4h', '1d']
    selected_period = st.selectbox("选择K线周期", period_options, index=4)
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
    if "模拟" in source_display:
        st.markdown(f'<div class="demo-box">⚠️ 当前处于演示模式（模拟数据） | 价格源：{price_source}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="info-box">✅ 当前数据源：{source_display} | 价格源：{price_source}</div>', unsafe_allow_html=True)

# 显示错误信息
if errors:
    with st.container():
        st.markdown('<div class="error-box">', unsafe_allow_html=True)
        st.error("⚠️ 部分数据获取失败，详细错误：")
        for err in errors[:8]:
            st.write(f"- {err}")
        if len(errors) > 8:
            st.write(f"... 还有 {len(errors)-8} 条错误")
        st.markdown('</div>', unsafe_allow_html=True)

# 计算信号
if data_dict:
    ai_dir, ai_conf = st.session_state.ai.predict(data_dict)
    fusion_dir, fusion_conf = st.session_state.fusion.fuse_periods(data_dict)
    # 推送
    if fusion_dir != 0 and selected_period in data_dict and PUSHPLUS_TOKEN:
        price_alert = data_dict[selected_period]['close'].iloc[-1]
        send_signal_alert(fusion_dir, fusion_conf, price_alert, "融合信号")
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
