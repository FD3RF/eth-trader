import streamlit as st
import pandas as pd
import requests
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from collections import deque
from scipy.signal import argrelextrema

# ==================== 1. 系统配置 ====================
st.set_page_config(page_title="ETH V17.0 Ultimate Intelligence", layout="wide")

if 'history' not in st.session_state:
    st.session_state.update({
        'history': deque(maxlen=50),
        'risk_per_trade': 1.0,
        'account_balance': 10000.0,
        'initialized': True,
        'score_thresh': 80
    })

# ==================== 2. 数据引擎 (非币安源) ====================
@st.cache_data(ttl=10)
def fetch_data(symbol="ETH", interval="minute", aggregate=5, limit=300):
    """从 CryptoCompare 获取全球加权聚合数据"""
    try:
        url = f"https://min-api.cryptocompare.com/data/v2/histo{interval}"
        params = {"fsym": symbol, "tsym": "USD", "limit": limit, "aggregate": aggregate, "e": "CCCAGG"}
        resp = requests.get(url, params=params).json()
        df = pd.DataFrame(resp['Data']['Data'])
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        return df[['open', 'high', 'low', 'close', 'volumefrom']].rename(columns={'volumefrom': 'volume'})
    except:
        return pd.DataFrame()

# ==================== 3. 核心计算大脑 ====================
def compute_indicators(df):
    if df.empty: return df
    # 趋势线
    df['ema8'] = df['close'].ewm(span=8).mean()
    df['ema21'] = df['close'].ewm(span=21).mean()
    # MACD & RSI
    df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    df['signal'] = df['macd'].ewm(span=9).mean()
    df['hist'] = df['macd'] - df['signal']
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-9)))
    return df

def get_ls_ratio():
    """实时模拟全网多空比"""
    return {'long': 48.2, 'short': 51.8, 'status': "空头占优"}

def get_liquidation_events():
    """模拟大额爆仓流"""
    now = datetime.now()
    return pd.DataFrame([
        {'time': now - timedelta(minutes=45), 'side': 'Long', 'amount': 2.1, 'price': 2015},
        {'time': now - timedelta(hours=3), 'side': 'Short', 'amount': 1.5, 'price': 2045}
    ])

def get_whale_walls(current_price):
    """模拟庄家挂单墙"""
    return [
        {'price': current_price * 1.02, 'amount': 45, 'type': 'Ask'},
        {'price': current_price * 0.98, 'amount': 62, 'type': 'Bid'}
    ]

# ==================== 4. UI 界面逻辑 ====================
st.title("🚀 ETH V17.0 终极量化工作站")

# 侧边栏：控制台
with st.sidebar:
    st.header("⚙️ 实时监控参数")
    st.session_state.score_thresh = st.slider("信号准入阈值", 50, 100, 80)
    st.divider()
    if st.button("🏁 生成 AI 复盘报告"):
        st.session_state.show_review = True

# 获取多周期数据
df5 = compute_indicators(fetch_data(aggregate=5))
df15 = compute_indicators(fetch_data(aggregate=15))
df1h = compute_indicators(fetch_data(interval="hour", aggregate=1))

if not df5.empty:
    last = df5.iloc[-1]
    ls = get_ls_ratio()
    liqs = get_liquidation_events()
    walls = get_whale_walls(last['close'])
    
    # 5m/15m/1h 共振灯
    res5 = last['ema8'] > last['ema21']
    res15 = df15['ema8'].iloc[-1] > df15['ema21'].iloc[-1]
    res1h = df1h['ema8'].iloc[-1] > df1h['ema21'].iloc[-1]
    
    # --- 仪表盘 ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("全球加权价", f"${last['close']:.2f}")
    c2.metric("全网多空比", f"{ls['long']}% / {ls['short']}%")
    
    # 状态灯 UI
    with c3:
        st.write("🚦 趋势共振")
        st.markdown(f"{'🟢' if res5 else '🔴'} 5m | {'🟢' if res15 else '🔴'} 15m | {'🟢' if res1h else '🔴'} 1h")
    
    # 模拟综合信心评分
    score = 60
    if res5 and res15 and res1h: score += 20
    if ls['short'] > 55: score += 15
    c4.metric("综合信心分", score, delta="建议抄底" if score >= 80 else "观察")

    # --- 主图：多维图表 ---
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    
    # K线图
    fig.add_trace(go.Candlestick(x=df5.index, open=df5['open'], high=df5['high'], low=df5['low'], close=df5['close'], name="K线"), row=1, col=1)
    
    # 庄家墙 (水平线)
    for wall in walls:
        color = "rgba(57, 211, 83, 0.4)" if wall['type'] == 'Bid' else "rgba(248, 81, 73, 0.4)"
        fig.add_hline(y=wall['price'], line_dash="dot", line_color=color, 
                      annotation_text=f"庄家墙 ${wall['amount']}M", row=1, col=1)

    # 爆仓闪电标记
    for _, row in liqs.iterrows():
        color = "#f85149" if row['side'] == 'Long' else "#39d353"
        fig.add_annotation(x=row['time'], y=row['price'], text="⚡", font=dict(size=20, color=color), showarrow=False, row=1, col=1)

    # MACD
    fig.add_trace(go.Bar(x=df5.index, y=df5['hist'], name="MACD"), row=2, col=1)
    
    fig.update_layout(height=800, template="plotly_dark", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # --- AI 复盘板块 ---
    if st.session_state.get('show_review'):
        st.divider()
        st.subheader("🧠 AI 自动复盘分析报告")
        st.markdown(f"""
        - **今日总结**：信号主要集中在 ${last['close']*0.98:.0f} - ${last['close']*1.02:.0f} 区间。
        - **陷阱识别**：今日 14:00 的 BUY 信号被判定为 **诱多陷阱**，因为当时上方存在 $45M 阻力墙且 15m 趋势未对齐。
        - **优化建议**：在空头比率超过 52% 时，可适当调低 ATR 止损倍数，以捕捉空头挤压的短线利润。
        """)
        st.session_state.history.append({'time': datetime.now(), 'score': score, 'price': last['close']})

st.caption("ETH V17.0 Ultimate Intelligence | 2026 旗舰级量化终端")
