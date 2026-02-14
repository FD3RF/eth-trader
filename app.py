- coding: utf-8 -  - 编码：UTF-8 --
"""🚀 全天候智能交易监控中心 · 最终完整版
多周期切换 | AI预测 | 模拟盈亏联动 | 微信提醒 | 永久在线
使用前请先在 Streamlit Cloud 的 Secrets 中设置：
BINANCE_API_KEY / BINANCE_SECRET_KEY (测试网可用任意值)
BINANCE_API_KEY/BINANCE_SECRET_KEY （测试网可用任意值）
PUSHPLUS_TOKEN (可选)  PUSHPLUS_TOKEN （可选）
"""
import streamlit as st  导入 Streamlit as ST
import pandas as pd  作为 PD 进口 Pandas。
import numpy as np
import ta  你的金额
import requests  导入请求
import plotly.graph_objects as go
导入 plotly.graph_objects 即使用权
from plotly.subplots import make_subplots
摘自 plotly.subplots 导入 make_subplots
from datetime import datetime, timedelta
来自 Datetime 导入 DateTime，Timedelta
import asyncio
import aiohttp  导入 AIOHTTP
import os  导入作系统
import time  进口时间
from streamlit_autorefresh import st_autorefresh
来自 streamlit_autorefresh 进口 st_autorefresh
import warnings  进口警告
warnings.filterwarnings('ignore')
警告.过滤器警告（“忽略”）
-------------------- 密钥读取 (从 Streamlit Secrets) --------------------
-------------------- 密钥读取 （从 Streamlit Secrets） --------------------
BINANCE_API_KEY = st.secrets.get("BINANCE_API_KEY", "")
BINANCE_API_KEY = st.secrets.get（“BINANCE_API_KEY”， “”）
BINANCE_SECRET_KEY = st.secrets.get("BINANCE_SECRET_KEY", "")
BINANCE_SECRET_KEY = st.secrets.get（“BINANCE_SECRET_KEY”， “”）
PUSHPLUS_TOKEN = st.secrets.get("PUSHPLUS_TOKEN", "")
PUSHPLUS_TOKEN = st.secrets.get（“PUSHPLUS_TOKEN”， “”）
-------------------- 异步数据获取器 --------------------
class AsyncDataFetcher:  class AsyncDataFetcher：
def init(self):  确定 init（self）：
self.base_url = "https://api.binance.com/api/v3/klines"
self.base_url = “https://api.binance.com/api/v3/klines”
self.symbol = "ETHUSDT"  self.symbol = “ETHUSDT”
self.periods = ['1m', '5m', '15m', '1h', '4h', '1d']
自周期 = ['1m'， '5m'， '15m'， '1h'， '4h'， '1d']
self.limit = 200  自限 = 200
code
Code  代码
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
-------------------- 指标计算 --------------------
def add_indicators(df):  防守 add_indicators（DF）：
df = df.copy()  df = df.copy（）
df['ma20'] = df['close'].rolling(20).mean()
df['ma20'] = df['close'].rolling（20）.mean（）
df['ma60'] = df['close'].rolling(60).mean()
DF['MA60'] = DF['close'].rolling（60）.mean（）
macd = ta.trend.MACD(df['close'])
MACD = ta.trend.MACD（df['close']）
df['macd'] = macd.macd()  df['macd'] = macd.macd（）
df['macd_signal'] = macd.macd_signal()
DF['macd_signal'] = macd.macd_signal（）
df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
df['rsi'] = ta.momentum.RSIIndicator（df['close']， window=14）.rsi（）
bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
bb = ta.volatility.BollingerBands（df['close']， window=20， window_dev=2）
df['bb_high'] = bb.bollinger_hband()
DF['bb_high'] = bb.bollinger_hband（）
df['bb_low'] = bb.bollinger_lband()
df['bb_low'] = bb.bollinger_lband（）
df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
df['atr'] = ta.volatility.AverageTrueRange（df['high']， df['low']， df['close']， window=14）.average_true_range（）
df['volume_sma'] = df['volume'].rolling(20).mean()
df['volume_sma'] = df['volume'].rolling（20）.mean（）
df['volume_ratio'] = df['volume'] / df['volume_sma']
return df  返回 DF
-------------------- AI 预测模块（可加载 LSTM 模型，无模型时用规则） --------------------
class AIPredictor:  类 AIPredictor：
def init(self):  确定 init（self）：
self.model = None  self.model = 无
self.scaler = None  self.scaler = 无
self.feature_cols = ['ma20', 'ma60', 'rsi', 'macd', 'bb_high', 'bb_low', 'atr', 'volume_ratio']
self.feature_cols = ['MA20'， 'ma60'， 'rsi'， 'macd'， 'bb_high'， 'bb_low'， 'atr'， 'volume_ratio']
self.seq_len = 20
self._load_model()  self._load_model（）
code
Code  代码
def _load_model(self):
    """尝试加载预训练模型，若无则跳过"""
    try:
        import tensorflow as tf
        import joblib
        model_path = "models/lstm_model.h5"
        scaler_path = "models/scaler.pkl"
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.model = tf.keras.models.load_model(model_path)
            self.scaler = joblib.load(scaler_path)
            print("✅ 加载 LSTM 模型成功")
        else:
            print("⚠️ 未找到模型文件，使用规则模拟")
    except Exception as e:
        print(f"⚠️ 模型加载失败: {e}，使用规则模拟")

def predict_with_model(self, df):
    """使用 LSTM 模型预测"""
    if len(df) < self.seq_len + 1:
        return 0, 0.5
    recent = df.iloc[-(self.seq_len+1):-1]  # 用前 seq_len 根预测下一根
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
    """规则模拟（备用）"""
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
    """统一预测接口：优先使用模型（仅4h），否则规则"""
    if self.model is not None and '4h' in df_dict:
        return self.predict_with_model(df_dict['4h'])
    else:
        return self.predict_with_rules(df_dict)
-------------------- 多周期策略融合 --------------------
class MultiPeriodFusion:  多时期融合课：
def init(self):  确定 init（self）：
self.period_weights = {
'1m': 0.05,  “1m”：0.05，
'5m': 0.1,  “5m”：0.1，
'15m': 0.15,  “15岁”：0.15，
'1h': 0.2,  “1小时”：0.2，
'4h': 0.25,  “4小时”：0.25，
'1d': 0.25  “1便士”：0.25
}
self.strategy_weights = {'trend': 0.5, 'oscillator': 0.3, 'volume': 0.2}
self.strategy_weights = {'趋势'： 0.5， '振荡器'： 0.3， 'volume'： 0.2}
code
Code  代码
def get_period_signal(self, df):
    last = df.iloc[-1]
    signals = {}
    # 趋势
    if last['ma20'] > last['ma60']:
        signals['trend'] = 1
    elif last['ma20'] < last['ma60']:
        signals['trend'] = -1
    else:
        signals['trend'] = 0
    # 震荡
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
-------------------- 微信推送（带冷却） --------------------
last_signal_time = None  last_signal_time = 无
last_signal_direction = 0
signal_cooldown_minutes = 5
def send_signal_alert(direction, confidence, price, reason=""):
def send_signal_alert（方向、信心、价格、理由=“）：
global last_signal_time, last_signal_direction
全球性 last_signal_time，last_signal_direction
if not PUSHPLUS_TOKEN:  如果不 PUSHPLUS_TOKEN：
return  回归
now = datetime.now()  现在 = DateTime。now（）
if direction == last_signal_direction and last_signal_time and (now - last_signal_time).total_seconds() < signal_cooldown_minutes * 60:
如果方向 == last_signal_direction 和 last_signal_time 以及 （现在 - last_signal_time）.total_seconds（） < signal_cooldown_minutes * 60：
return  回归
dir_str = "做多" if direction == 1 else "做空"
dir_str = “做多” 如果方向 == 1 否则 “做空”
content = f"""【交易信号提醒】
方向: {dir_str}  方向：{dir_str}
置信度: {confidence:.1%}  置信度： {confidence：.1%}
当前价格: ${price:.2f}
时间: {now.strftime('%Y-%m-%d %H:%M:%S')}
时间： {now.strftime（'%Y-%m-%d %H：%M：%S'）}
{reason}"""  {reason}“”
url = "http://www.pushplus.plus/send"
URL = “http://www.pushplus.plus/send”
data = {"token": PUSHPLUS_TOKEN, "title": "🤖 交易信号", "content": content, "template": "txt"}
data = {“token”： PUSHPLUS_TOKEN， “title”： “ 🤖 交易信号”， “content”： content， “template”： “txt”}
try:  试试：
requests.post(url, json=data, timeout=5)
requests.post（url， json=data， timeout=5）
last_signal_time = now  last_signal_time = 现在
last_signal_direction = direction
last_signal_direction = 方向
except Exception as e:  例外情况为 e：
print(f"推送失败: {e}")
-------------------- 缓存数据获取 --------------------
@st.cache_data(ttl=60)
def fetch_all_data():  防守 fetch_all_data（）：
loop = asyncio.new_event_loop()
循环 = asyncio.new_event_loop（）
asyncio.set_event_loop(loop)
asyncio.set_event_loop（循环）
fetcher = AsyncDataFetcher()
fetcher = AsyncDataFetcher（）
data_dict = loop.run_until_complete(fetcher.fetch_all())
data_dict = loop.run_until_complete（fetcher.fetch_all（））
for p in data_dict:  对于 p 在 data_dict 中：
data_dict[p] = add_indicators(data_dict[p])
data_dict[p] = add_indicators（data_dict[p]）
return data_dict  返回 data_dict
-------------------- Streamlit 界面 --------------------
st.set_page_config(page_title="全天候智能交易监控中心", layout="wide")
st.set_page_config（page_title=“全天候智能交易监控中心”， layout=“wide”）
st.markdown("""  圣马克当（“””
<style>
.stApp { background-color: #0B0E14; color: white; }
.stApp { 背景色：#0B0E14;颜色：白色; }
.ai-box { background: #1A1D27; border-radius: 10px; padding: 20px; border-left: 6px solid #00F5A0; }
.ai-box { 背景：#1A1D27;border-radius：10px;padding：20px;border-left：6px 实心 #00F5A0; }
.metric { background: #232734; padding: 15px; border-radius: 8px; }
.metric { 背景：#232734; 填充：15px;border-radius：8px; }
.signal-buy { color: #00F5A0; font-weight: bold; }
.signal-buy { color： #00F5A0; font-weight： borgan; }
.signal-sell { color: #FF5555; font-weight: bold; }
.signal-sell { 颜色：#FF5555;字体粗大：加粗; }
.profit { color: #00F5A0; }
.profit { 颜色：#00F5A0; }
.loss { color: #FF5555; }
.loss { 颜色：#FF5555; }
</style>
""", unsafe_allow_html=True)
“”“，unsafe_allow_html=真）
st.title("🧠 全天候智能交易监控中心 · 最终版")
st.caption("数据缓存60秒｜多周期切换｜AI预测｜盈亏联动｜微信提醒")
初始化 AI 和融合模块（只初始化一次）
if 'ai' not in st.session_state:
如果“ai”不在 st.session_state：
st.session_state.ai = AIPredictor()
st.session_state.ai = AIPredictor（）
if 'fusion' not in st.session_state:
如果“聚变”不在 st.session_state：
st.session_state.fusion = MultiPeriodFusion()
st.session_state.fusion = 多周期融合（）
侧边栏
with st.sidebar:  附圣侧栏：
st.header("⚙️ 控制面板")
period_options = ['1m', '5m', '15m', '1h', '4h', '1d']
period_options = ['1m'， '5m'， '15m'， '1h'， '4h'， '1d']
selected_period = st.selectbox("选择K线周期", period_options, index=4)
selected_period = st.selectbox（“选择 K 线周期”， period_options， index=4）
auto_refresh = st.checkbox("开启自动刷新", value=True)
auto_refresh = st.checkbox（“开启自动刷新”，value=True）
refresh_interval = st.number_input("刷新间隔(秒)", 5, 60, 10, disabled=not auto_refresh)
refresh_interval = st.number_input（“刷新间隔（秒）”，5， 60， 10， disabled=not auto_refresh）
if auto_refresh:  如果 auto_refresh：
st_autorefresh(interval=refresh_interval * 1000, key="auto_refresh")
st_autorefresh（interval=refresh_interval * 1000， key=“auto_refresh”）
st.markdown("---")  圣马克当（“---”）
st.subheader("💰 模拟交易")
sim_entry = st.number_input("入场价", value=0.0, format="%.2f")
sim_entry = st.number_input（“入场价”，value=0.0，格式=“%.2f”）
sim_stop = st.number_input("止损价", value=0.0, format="%.2f")
sim_stop = st.number_input（“止损价”，value=0.0，格式=“%.2f”）
sim_quantity = st.number_input("数量 (ETH)", value=0.01, format="%.4f")
sim_quantity = st.number_input（“数量 （ETH）”，值=0.01，格式=“%.4f”）
# 盈亏价格源（默认使用显示周期）
use_display_period = st.radio("盈亏价格源", ["使用显示周期", "使用实时价格 (需WebSocket)"], index=0) == "使用显示周期"
获取数据
data_dict = fetch_all_data()
data_dict = fetch_all_data（）
计算 AI 和融合信号
if data_dict:  如果 data_dict：
ai_dir, ai_conf = st.session_state.ai.predict(data_dict)
ai_dir，ai_conf = st.session_state.ai.predict（data_dict）
fusion_dir, fusion_conf = st.session_state.fusion.fuse_periods(data_dict)
fusion_dir，fusion_conf = st.session_state.fusion.fuse_periods（data_dict）
# 发送微信提醒（当融合信号非零且非冷却）
if fusion_dir != 0:  如果 fusion_dir！= 0：
price_for_alert = data_dict[selected_period]['close'].iloc[-1] if selected_period in data_dict else 0
price_for_alert = 如果 selected_period 属于 data_dict 否则 0 data_dict[selected_period]['close'].iloc[-1]
send_signal_alert(fusion_dir, fusion_conf, price_for_alert, "融合信号触发")
send_signal_alert（fusion_dir， fusion_conf， price_for_alert， “融合信号触发”）
else:  其他：
ai_dir, ai_conf = 0, 0
ai_dir，ai_conf = 0.0
fusion_dir, fusion_conf = 0, 0
fusion_dir，fusion_conf = 0， 0
主布局
col1, col2 = st.columns([2.2, 1.3])
col1， col2 = 柱数（[2.2， 1.3]）
with col1:  与 col1 合作：
st.subheader(f"📊 实时K线 ({selected_period})")
st.subheader（f“ 📊 实时 K 线 （{selected_period}）”）
if data_dict and selected_period in data_dict:
如果 data_dict 和 selected_period 在 data_dict：
df = data_dict[selected_period].tail(100).copy()
df = data_dict[selected_period].tail（100）.copy（）
df['日期'] = df['timestamp']
df['日期'] = df['timest 戳']
fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
fig = make_subplots（行=2，列=1，shared_xaxes=真，
row_heights=[0.7, 0.3],  row_heights=[0.7， 0.3]，
subplot_titles=(f"ETH/USDT {selected_period}", "RSI"))
subplot_titles=（f“ETH/USDT {selected_period}”， “RSI”））
# K线
fig.add_trace(go.Candlestick(x=df['日期'], open=df['open'], high=df['high'],
fig.add_trace（走吧。Candlestick（x=df['日期']，open=df['open']，high=df['high']，
low=df['low'], close=df['close'], name="K线"), row=1, col=1)
low=df['low']， close=df['close']， name=“K 线”）， row=1， col=1）
# 均线
fig.add_trace(go.Scatter(x=df['日期'], y=df['ma20'], name="MA20", line=dict(color="orange")), row=1, col=1)
fig.add_trace（走吧。Scatter（x=df['日期']， y=df['ma20']， name=“MA20”， line=dict（color=“orange”））， row=1， col=1）
fig.add_trace(go.Scatter(x=df['日期'], y=df['ma60'], name="MA60", line=dict(color="blue")), row=1, col=1)
fig.add_trace（走吧。Scatter（x=df['日期']， y=df['ma60']， name=“MA60”， line=dict（color=“blue”））， row=1， col=1）
# 融合信号箭头
if fusion_dir != 0:  如果 fusion_dir！= 0：
last_date = df['日期'].iloc[-1]
last_price = df['close'].iloc[-1]
if fusion_dir == 1:  如果 fusion_dir == 1：
fig.add_annotation(x=last_date, y=last_price1.02,
fig.add_annotation（x=last_date， y=last_price1.02，
text="▲ 融合多", showarrow=True, arrowhead=2, arrowcolor="green")
文本=“▲ 融合多”，showarrow=真，箭头=2，箭头颜色=“绿色”）
else:  其他：
fig.add_annotation(x=last_date, y=last_price0.98,
text="▼ 融合空", showarrow=True, arrowhead=2, arrowcolor="red")
text=“▼ 融合空”，showarrow=真，arrowhead=2，arrowcolor=“red”）
# RSI
fig.add_trace(go.Scatter(x=df['日期'], y=df['rsi'], name="RSI", line=dict(color="purple")), row=2, col=1)
fig.add_trace（走吧。Scatter（x=df['日期']， y=df['rsi']， name=“RSI”， line=dict（color=“purple”））， row=2， col=1）
fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
fig.add_hline（y=70，line_dash=“破折号”，line_color=“red”，不透明度=0.5，行=2，col=1）
fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
fig.add_hline（y=30，line_dash=“dash”，line_color=“green”，不透明度=0.5，row=2，col=1）
fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=600)
fig.update_layout（template=“plotly_dark”， xaxis_rangeslider_visible=False， height=600）
st.plotly_chart(fig, use_container_width=True)
st.plotly_chart（图，use_container_width=真）
else:  其他：
st.info("等待数据...")
fig.add_annotation（x=last_date， y=last_price0.98，

with col2:  使用 col2：
st.subheader("🧠 实时决策")
dir_map = {1: "🔴 做多", -1: "🔵 做空", 0: "⚪ 观望"}
st.markdown(f'<div class="ai-box">{dir_map[fusion_dir]}<br>置信度: {fusion_conf:.1%}</div>', unsafe_allow_html=True)
st.markdown（f'
{dir_map[fusion_dir]}
置信度： {fusion_conf：.1%}
'， unsafe_allow_html=True）
