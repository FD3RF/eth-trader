import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime

# ==========================================
# 1. 指挥部初始化与环境设置
# ==========================================
st.set_page_config(layout="wide", page_title="ETH V32000 终极指挥官", page_icon="🐋")

def init_commander():
    if 'battle_logs' not in st.session_state: st.session_state.battle_logs = []
    if 'last_cleanup_ts' not in st.session_state: st.session_state.last_cleanup_ts = time.time()
    if 'sentiment_score' not in st.session_state: st.session_state.sentiment_score = 50.0

init_commander()

# ==========================================
# 2. 核心算法库：爆仓热力图、阻力、模式识别
# ==========================================
def get_intel_matrix(df):
    """计算物理参考位、模式识别及爆仓生死线"""
    if df.empty: return {}
    
    last_p = df['c'].iloc[-1]
    # 物理支撑压力 (最近30根K线)
    res_px = df['h'].tail(30).max()
    sup_px = df['l'].tail(30).min()
    
    # ATR模式识别
    recent_atr = df['atr'].tail(60).mean()
    vol_ratio = (recent_atr / last_p) * 100
    if vol_ratio > 0.15: mode = ("趋势爆发", "🚀")
    elif vol_ratio < 0.05: mode = ("极窄震荡", "🦥")
    else: mode = ("标准波段", "⚖️")
    
    # 爆仓热力图逆推 (杠杆生死线)
    liq_zones = [
        {'type': '空头爆仓', 'lev': '50x', 'px': df['h'].tail(120).max() * 1.018, 'color': 'rgba(255, 0, 0, 0.4)'},
        {'type': '多头爆仓', 'lev': '50x', 'px': df['l'].tail(120).min() * 0.982, 'color': 'rgba(0, 255, 0, 0.4)'}
    ]
    
    return {"res": res_px, "sup": sup_px, "mode": mode, "liq": liq_zones}

# ==========================================
# 3. 数据层：多源情报采集 (K线、庄家流、多周期流)
# ==========================================
@st.cache_data(ttl=10)
def get_market_data():
    try:
        # 1min K线
        url = "https://www.okx.com/api/v5/market/candles?instId=ETH-USDT&bar=1m&limit=100"
        res = requests.get(url, timeout=5).json()
        df = pd.DataFrame(res['data'], columns=['ts','o','h','l','c','v','volC','volCQ','cf'])[::-1].reset_index(drop=True)
        df['time'] = pd.to_datetime(df['ts'].astype(float), unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')
        for c in ['o','h','l','c','v']: df[c] = df[c].astype(float)
        tr = pd.concat([df['h']-df['l'], abs(df['h']-df['c'].shift()), abs(df['l']-df['c'].shift())], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        
        # 庄家大单流 (L2 OrderBook)
        book_url = "https://www.okx.com/api/v5/market/books?instId=ETH-USDT&sz=20"
        book_res = requests.get(book_url, timeout=3).json()
        bids = pd.DataFrame(book_res['data'][0]['bids'], columns=['px', 'sz', 'cnt', 'ord'])
        asks = pd.DataFrame(book_res['data'][0]['asks'], columns=['px', 'sz', 'cnt', 'ord'])
        whale_bids = bids[bids['sz'].astype(float) > 500].copy()
        whale_asks = asks[asks['sz'].astype(float) > 500].copy()

        # 多周期资金流 (1h/4h)
        periods = {'1h': '1H', '4h': '4H'}
        multi_flow = {}
        for k, v in periods.items():
            f_url = f"https://www.okx.com/api/v5/market/candles?instId=ETH-USDT&bar={v}&limit=10"
            f_res = requests.get(f_url, timeout=3).json()
            net = sum([(float(x[4]) - float(x[1])) * float(x[5]) for x in f_res['data'][:5]])
            multi_flow[k] = net
            
        return df, whale_bids, whale_asks, multi_flow
    except: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {'1h':0, '4h':0}

# ==========================================
# 4. AI 裁决与复盘引擎：逻辑细化版
# ==========================================
def ai_commander_report(df, whale_asks, multi_flow):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    vol_avg = df['v'].tail(10).mean()
    
    # 状态识别逻辑
    is_price_up = last['c'] > prev['c']
    is_vol_push = last['v'] > vol_avg * 1.5
    
    if is_price_up and is_vol_push: status = "✅ 放量上涨：真实买盘推动，主力在进场。"
    elif is_price_up and not is_vol_push: status = "📉 缩量诱多：庄家虚拉吸引散户，警惕随时反手。"
    elif not is_price_up and is_vol_push: status = "🚨 放量下跌：主力恐慌砸盘，或大户强制平仓。"
    else: status = "⚠️ 缩量阴跌：市场流动性枯竭，磨灭散户耐心。"
    
    whale_info = f"上方巨墙拦截" if not whale_asks.empty else "上方暂无巨阻"
    trend_info = "看多" if multi_flow['4h'] > 0 else "看空"
    
    return f"{status}\n\n【博弈核心】\n主力大趋势：{trend_info}\n庄家流监测：{whale_info}"

# ==========================================
# 5. UI 渲染：大屏指挥中心
# ==========================================
def main():
    # 自动内存清理
    if time.time() - st.session_state.last_cleanup_ts > 14400:
        st.session_state.battle_logs = [f"【系统】{datetime.now().strftime('%H:%M:%S')} 内存防御完成"]
        st.session_state.last_cleanup_ts = time.time()
        st.cache_data.clear()

    df, w_bids, w_asks, m_flow = get_market_data()
    intel = get_intel_matrix(df)
    report = ai_commander_report(df, w_asks, m_flow)

    # 顶层状态栏
    st.markdown(f"### 🛰️ ETH 量子决策指挥官 (V32000) | {datetime.now().strftime('%H:%M:%S')}")
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("4H 资金流", "流入 🔥" if m_flow['4h']>0 else "流出 ❄️")
    h2.metric("模式识别", f"{intel['mode'][1]} {intel['mode'][0]}")
    h3.metric("实时价格", f"${df['c'].iloc[-1]}")
    h4.metric("波动率 ATR", f"{df['atr'].iloc[-1]:.2f}")

    st.divider()

    col_l, col_r = st.columns([1, 4])
    with col_l:
        # AI 文字裁决
        st.markdown(f"""<div style="padding:15px; border:2px solid #00ff00; border-radius:12px; background:rgba(0,0,0,0.5);">
            <h4 style="color:#00ff00; margin-top:0;">🤖 AI 自动复盘报告</h4>
            <p style="font-size:0.95em; color:white; line-height:1.6;">{report}</p>
        </div>""", unsafe_allow_html=True)
        
        # 爆仓地雷区
        st.markdown("#### ⚡ 爆仓地雷/燃料分布")
        for z in intel['liq']:
            st.caption(f"{z['type']} ({z['lev']}): **${z['px']:.1f}**")
            
        # 实时战术日志
        st.markdown("#### 📜 战术日志")
        if not w_bids.empty: st.session_state.battle_logs.insert(0, f"🐋 庄家护盘墙: ${w_bids['px'].iloc[0]}")
        for log in st.session_state.battle_logs[:5]: st.caption(log)

    with col_r:
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Candlestick(x=df['time'], open=df['o'], high=df['h'], low=df['l'], close=df['c'], name="Price"))
        
        # 渲染物理位与爆仓热力图
        fig.add_hline(y=intel['res'], line_dash="dash", line_color="red", annotation_text="阻力")
        fig.add_hline(y=intel['sup'], line_dash="dash", line_color="green", annotation_text="支撑")
        
        for z in intel['liq']:
            fig.add_hline(y=z['px'], line_dash="dot", line_color=z['color'])
            fig.add_hrect(y0=z['px']*0.9995, y1=z['px']*1.0005, fillcolor=z['color'], opacity=0.2, line_width=0)

        fig.update_layout(template="plotly_dark", height=800, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

    time.sleep(10); st.rerun()

if __name__ == "__main__":
    main()
