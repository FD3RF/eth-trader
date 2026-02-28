import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime

# ==========================================
# 1. 系统底层：环境锁与内存治理
# ==========================================
st.set_page_config(layout="wide", page_title="ETH V32000 终极指挥官", page_icon="🐋")

def init_system_state():
    """初始化全局状态与内存清理机制"""
    if 'battle_logs' not in st.session_state: st.session_state.battle_logs = []
    if 'last_cleanup' not in st.session_state: st.session_state.last_cleanup = time.time()
    
    # 4小时强制内存防御：防止 Streamlit 长时间运行导致的 OOM
    if time.time() - st.session_state.last_cleanup > 14400:
        st.cache_data.clear()
        st.session_state.last_cleanup = time.time()
        st.session_state.battle_logs.append(f"系统提示: {datetime.now().strftime('%H:%M:%S')} 内存防御治理完成")

# ==========================================
# 2. 核心算法：爆仓热力图与量价逻辑
# ==========================================
def calculate_liquidation_zones(df):
    """
    逆推杠杆爆仓生死线
    原理：识别高点/低点后的 50x 杠杆强制平仓点
    """
    if df is None or df.empty: return []
    
    high_24h = df['h'].tail(120).max()
    low_24h = df['l'].tail(120).min()
    
    return [
        {'label': '空头爆仓(50x)', 'px': high_24h * 1.018, 'color': 'rgba(255, 0, 0, 0.4)'},
        {'label': '多头爆仓(50x)', 'px': low_24h * 0.982, 'color': 'rgba(0, 255, 0, 0.4)'}
    ]

def ai_market_recap(df, flow_4h, whale_wall):
    """
    AI 深度复盘引擎：识别庄家阴谋
    """
    if df is None or df.empty: return "数据同步中..."
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    vol_ma = df['v'].tail(10).mean()
    
    # 核心判断逻辑
    is_vol_push = last['v'] > vol_ma * 1.5
    is_price_up = last['c'] > prev['c']
    
    if is_price_up and not is_vol_push:
        status = "📉 缩量诱多：庄家虚拉吸引散户，警惕随时反手。"
    elif not is_price_up and is_vol_push:
        status = "🚨 放量砸盘：主力抛售或大户强平，严禁盲目接刀。"
    else:
        status = "✅ 趋势运行：量价匹配，目前处于标准模式。"
        
    trend_desc = "看空 ❄️" if flow_4h < 0 else "看多 🔥"
    whale_desc = f"上方大单墙: ${whale_wall['px']}" if whale_wall is not None else "上方暂无巨阻"
    
    return f"{status}\n\n【博弈核心】主力趋势: {trend_desc} | 庄家流: {whale_desc}"

# ==========================================
# 3. 数据层：多源实时情报抓取
# ==========================================
@st.cache_data(ttl=10)
def get_intel_data():
    """抓取K线、庄家墙与多周期流"""
    try:
        # 1. 实时K线 (OKX API 示例)
        url_k = "https://www.okx.com/api/v5/market/candles?instId=ETH-USDT&bar=1m&limit=100"
        res_k = requests.get(url_k, timeout=5).json()
        df = pd.DataFrame(res_k['data'], columns=['ts','o','h','l','c','v','volC','volCQ','cf'])[::-1].reset_index(drop=True)
        for c in ['o','h','l','c','v']: df[c] = df[c].astype(float)
        df['atr'] = (df['h'] - df['l']).rolling(14).mean()
        
        # 2. 4H 资金流趋势
        url_f = "https://www.okx.com/api/v5/market/candles?instId=ETH-USDT&bar=4H&limit=5"
        res_f = requests.get(url_f, timeout=3).json()
        flow_4h = sum([(float(x[4]) - float(x[1])) for x in res_f['data']])
        
        # 3. L2 深度墙 (识别 >500 ETH 大单)
        url_b = "https://www.okx.com/api/v5/market/books?instId=ETH-USDT&sz=20"
        res_b = requests.get(url_b, timeout=3).json()
        asks = pd.DataFrame(res_b['data'][0]['asks'], columns=['px', 'sz', 'cnt', 'ord'])
        whale = asks[asks['sz'].astype(float) > 500].iloc[0] if not asks.empty else None
        
        return df, flow_4h, whale
    except Exception as e:
        return None, 0, None

# ==========================================
# 4. 渲染层：指挥大屏显示
# ==========================================
def main():
    init_system_state()
    df, flow_4h, whale = get_intel_data()
    
    if df is None:
        st.error("❌ 卫星连接失败：无法获取市场实时情报，请检查 API 网络。")
        return

    # 预计算
    liq_zones = calculate_liquidation_zones(df)
    report = ai_market_recap(df, flow_4h, whale)
    last_price = df['c'].iloc[-1]

    # --- 顶层状态栏 ---
    st.markdown(f"### 🛰️ ETH 量子决策指挥官 (V32000) | {datetime.now().strftime('%H:%M:%S')}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("4H 资金流", "流入 🔥" if flow_4h > 0 else "流出 ❄️")
    m2.metric("模式识别", "⚖️ 标准波段")
    m3.metric("实时价格", f"${last_price:.2f}")
    m4.metric("波动率 ATR", f"{df['atr'].iloc[-1]:.2f}")

    st.divider()

    # --- 核心交互区 ---
    col_l, col_r = st.columns([1, 4])
    
    with col_l:
        st.markdown(f"""<div style="padding:15px; border:2px solid #00ff00; border-radius:12px; background:rgba(0,0,0,0.5);">
            <h4 style="color:#00ff00; margin-top:0;">🤖 AI 自动复盘报告</h4>
            <p style="font-size:0.95em; color:#eee; line-height:1.6;">{report.replace('【', '<br>【')}</p>
        </div>""", unsafe_allow_html=True)
        
        st.markdown("#### ⚡ 爆仓地雷分布")
        for z in liq_zones:
            st.caption(f"{z['label']}: **${z['px']:.1f}**")
            
        st.markdown("#### 📜 战术日志")
        for log in st.session_state.battle_logs[-5:]: st.caption(log)

    with col_r:
        fig = make_subplots(rows=1, cols=1)
        # K线
        fig.add_trace(go.Candlestick(x=df.index, open=df['o'], high=df['h'], low=df['l'], close=df['c'], name="ETH-Price"))
        
        # 阻力/支撑线
        res_px = df['h'].tail(30).max()
        sup_px = df['l'].tail(30).min()
        fig.add_hline(y=res_px, line_dash="dash", line_color="red", annotation_text="压力")
        fig.add_hline(y=sup_px, line_dash="dash", line_color="green", annotation_text="支撑")
        
        # 爆仓热力区渲染
        for z in liq_zones:
            fig.add_hrect(y0=z['px']*0.9995, y1=z['px']*1.0005, fillcolor=z['color'], opacity=0.3, line_width=0)

        fig.update_layout(template="plotly_dark", height=750, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

    # 10秒心跳刷新
    time.sleep(10)
    st.rerun()

if __name__ == "__main__":
    main()
