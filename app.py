import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# ==================== 1. 初始化配置与数据引擎 ====================
st.set_page_config(page_title="ETH V57.0 实战终端", layout="wide")

def fetch_okx_data(endpoint, params=""):
    """100% OKX 官方原生 API 接口"""
    url = f"https://www.okx.com/api/v5/{endpoint}?instId=ETH-USDT{params}"
    try:
        with requests.Session() as s:
            s.trust_env = True  # 穿透 LetsVPN
            r = s.get(url, timeout=5).json()
            return r if r.get('code') == '0' else None
    except: return None

# ==================== 2. AI 行情自动识别算法 ====================
def analyze_market_context(df):
    # 计算 ATR (波动率)
    tr = pd.concat([df['h']-df['l'], (df['h']-df['c'].shift()).abs(), (df['l']-df['c'].shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(12).mean().iloc[-1]
    
    if atr < 8.5: # 识别 1920 附近的横盘
        return "🌀 震荡模式", 1.6, "低波动，建议高抛低吸"
    else:
        return "📊 趋势模式", 3.0, "动能爆发，建议顺势突破"

# ==================== 3. 指令生成逻辑 ====================
def get_trade_plan(curr_p, sup, res, mode, net_flow):
    risk = abs(curr_p - sup) if sup else 5
    reward = abs(res - curr_p) if res else 10
    rr = reward / risk if risk > 0 else 0
    
    if "震荡" in mode:
        action = "🟢 逢低买入" if curr_p < (sup + (res-sup)*0.4) else "🟡 观望中"
        target = res - 0.5
    else:
        action = "🚀 顺势追多" if net_flow > 20 else "📉 择机看空"
        target = curr_p + 15
        
    return {"action": action, "entry": curr_p, "sl": sup-2 if sup else curr_p-10, "tp": target, "rr": rr}

# ==================== 4. UI 渲染与主逻辑 ====================
k_raw = fetch_okx_data("market/candles", "&bar=5m&limit=150")
d_raw = fetch_okx_data("market/books", "&sz=20")
t_raw = fetch_okx_data("market/trades", "&limit=100")

if k_raw and d_raw:
    # 数据转换
    df = pd.DataFrame(k_raw['data'], columns=['ts','o','h','l','c','v','volCcy','volCcyQ','confirm'])
    df[['o','h','l','c','v']] = df[['o','h','l','c','v']].astype(float)
    df = df[::-1].reset_index(drop=True)
    df['ema21'] = df['c'].ewm(span=21).mean()
    curr_p = df.iloc[-1]['c']
    
    # 模式识别
    s_mode, m_factor, reason = analyze_market_context(df)
    
    # 侧边栏：自动识别 + 仓位计算
    with st.sidebar:
        st.header("🧠 自动识别与计划")
        st.success(f"建议：{s_mode}\n\n{reason}")
        
        # 实时作战指令卡片
        asks = pd.DataFrame(d_raw['data'][0]['asks']).iloc[:, :2].astype(float)
        bids = pd.DataFrame(d_raw['data'][0]['bids']).iloc[:, :2].astype(float)
        res_p = asks[asks[1] > asks[1].mean() * m_factor].iloc[0, 0] if not asks.empty else None
        sup_p = bids[bids[1] > bids[1].mean() * m_factor].iloc[0, 0] if not bids.empty else None
        
        tdf = pd.DataFrame(t_raw['data'])
        net_f = tdf[tdf['side']=='buy']['sz'].astype(float).sum() - tdf[tdf['side']=='sell']['sz'].astype(float).sum()
        
        plan = get_trade_plan(curr_p, sup_p, res_p, s_mode, net_f)
        st.markdown(f"""
            <div style="border:1px solid #00ffcc; padding:10px; border-radius:5px; background:rgba(0,255,204,0.1)">
                <h4 style="margin:0; color:#00ffcc">{plan['action']}</h4>
                <p style="margin:2px 0">进场位: ${plan['entry']:.2f}</p>
                <p style="margin:2px 0; color:#ff4b4b">止损位: ${plan['sl']:.2f}</p>
                <p style="margin:2px 0; color:#00ffcc">止盈位: ${plan['tp']:.2f}</p>
                <p style="margin:2px 0">盈亏比: {plan['rr']:.2f}R</p>
            </div>
        """, unsafe_allow_html=True)

    # 主面板渲染
    st.title(f"🛡️ ETH {s_mode} 终端")
    col1, col2, col3 = st.columns(3)
    col1.metric("现价", f"${curr_p}")
    col2.metric("1min 净流入", f"{net_f:+.2f} ETH")
    col3.metric("买压占比", f"{(bids[1].sum()/(asks[1].sum()+bids[1].sum())*100):.1f}%")

    # K线图
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Candlestick(x=df.index, open=df['o'], high=df['h'], low=df['l'], close=df['c']))
    if res_p: fig.add_hline(y=res_p, line_dash="dash", line_color="red", annotation_text="抛售拦截")
    if sup_p: fig.add_hline(y=sup_p, line_dash="dash", line_color="green", annotation_text="防御支撑")
    fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.error("数据连接失败，请检查 LetsVPN 是否已开启并连接至香港/海外节点。")
