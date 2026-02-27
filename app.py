import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# ==================== 1. 配置与数据引擎 ====================
st.set_page_config(page_title="ETH V55.0 终极统一指挥部", layout="wide")

def fetch_okx(endpoint, params=""):
    url = f"https://www.okx.com/api/v5/{endpoint}?instId=ETH-USDT{params}"
    try:
        with requests.Session() as s:
            s.trust_env = True # 穿透 LetsVPN
            return s.get(url, timeout=5).json()
    except: return None

# ==================== 2. AI 行情自动分类算法 ====================
def get_market_context(df):
    # 计算 ATR (波动率)
    tr = pd.concat([df['h']-df['l'], (df['h']-df['c'].shift()).abs(), (df['l']-df['c'].shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(12).mean().iloc[-1]
    
    # 判定模式
    if atr < 8.5: # 针对 1920 横盘的阈值
        return "🌀 震荡模式", "低波动，建议高抛低吸", 1.6
    else:
        return "📊 趋势模式", "动能爆发，建议顺势而为", 3.0

# ==================== 3. 核心计算层 ====================
def process_signals(k_data, d_raw, mode_factor):
    df = pd.DataFrame(k_data['data'], columns=['ts','o','h','l','c','v','volCcy','volCcyQ','confirm'])
    df[['o','h','l','c','v']] = df[['o','h','l','c','v']].astype(float)
    df = df[::-1].reset_index(drop=True)
    df['ema21'] = df['c'].ewm(span=21).mean()
    
    # 动态支撑压力
    asks = pd.DataFrame(d_raw['data'][0]['asks']).iloc[:, :2].astype(float)
    bids = pd.DataFrame(d_raw['data'][0]['bids']).iloc[:, :2].astype(float)
    
    res = asks[asks[1] > asks[1].mean() * mode_factor].iloc[0, 0] if not asks.empty else None
    sup = bids[bids[1] > bids[1].mean() * mode_factor].iloc[0, 0] if not bids.empty else None
    
    # 净流入
    t_raw = fetch_okx("market/trades", "&limit=100")
    if t_raw:
        tdf = pd.DataFrame(t_raw['data'])
        tdf['sz'] = tdf['sz'].astype(float)
        net_flow = tdf[tdf['side']=='buy']['sz'].sum() - tdf[tdf['side']=='sell']['sz'].sum()
    else: net_flow = 0
    
    return df, res, sup, net_flow, asks[1].sum(), bids[1].sum()

# ==================== 4. UI 渲染层 ====================
# 获取原始数据
k_raw = fetch_okx("market/candles", "&bar=5m&limit=100")
d_raw = fetch_okx("market/books", "&sz=20")

if k_raw and d_raw:
    # A. 预处理判断行情
    temp_df = pd.DataFrame(k_raw['data'], columns=['ts','o','h','l','c','v','volCcy','volCcyQ','confirm']).astype({'h':float,'l':float,'c':float})
    suggested_mode, reason, m_factor = get_market_context(temp_df[::-1])
    
    # B. 侧边栏控制
    with st.sidebar:
        st.header("🧠 自动识别系统")
        st.info(f"**建议模式:** {suggested_mode}\n\n{reason}")
        use_auto = st.checkbox("跟随 AI 建议", value=True)
        final_mode = suggested_mode if use_auto else st.radio("手动模式", ["🌀 震荡模式", "📊 趋势模式"])
        final_factor = 1.6 if "震荡" in final_mode else 3.0

        st.divider()
        st.header("🧮 仓位计算器")
        bal = st.number_input("余额", 1000.0)
        risk = st.slider("风险 %", 0.5, 5.0, 1.0)

    # C. 计算最终指标
    df, res, sup, net_flow, ask_v, bid_v = process_signals(k_raw, d_raw, final_factor)
    curr_p = df.iloc[-1]['c']
    
    # D. 顶部看板
    st.title(f"🛡️ ETH V55.0 {final_mode} 终端")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("现价", f"${curr_p}")
    c2.metric("1min 净流入", f"{net_flow:+.2f} ETH")
    c3.metric("买压占比", f"{(bid_v/(ask_v+bid_v)*100):.1f}%")
    
    # 实时动能条
    momentum = np.clip((net_flow + 50) / 100, 0.0, 1.0)
    c4.write("🚀 突破动能")
    c4.progress(momentum)

    # E. 图表渲染
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Candlestick(x=df.index, open=df['o'], high=df['h'], low=df['l'], close=df['c'], name="K线"))
    if res: fig.add_hline(y=res, line_dash="dash", line_color="red", annotation_text="拦截压力")
    if sup: fig.add_hline(y=sup, line_dash="dash", line_color="green", annotation_text="支撑防御")
    fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

else:
    st.error("数据引擎连接失败，请检查 LetsVPN...")
