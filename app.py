import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime

# ==========================================
# 1. 指挥部初始化与内存防御
# ==========================================
st.set_page_config(layout="wide", page_title="ETH V32000 终极指挥官", page_icon="🐋")

def init_system():
    """初始化全局状态及4小时强制内存治理"""
    if 'battle_logs' not in st.session_state: st.session_state.battle_logs = []
    if 'last_cleanup' not in st.session_state: st.session_state.last_cleanup = time.time()
    
    # 内存防御治理：防止 Streamlit 缓存堆积导致的页面卡死
    if time.time() - st.session_state.last_cleanup > 14400:
        st.cache_data.clear()
        st.session_state.last_cleanup = time.time()
        st.session_state.battle_logs.append(f"【系统】{datetime.now().strftime('%H:%M:%S')} 内存防御治理已完成")

# ==========================================
# 2. 核心算法：爆仓地雷分布逆推
# ==========================================
def calculate_liquidation_zones(df):
    """
    逆推 50x 杠杆生死线
    原理：基于 24h 高低点计算强制平仓触发区
    """
    if df is None or df.empty: return []
    
    # 获取最近密集交易区的高低点
    h_24 = df['h'].tail(120).max()
    l_24 = df['l'].tail(120).min()
    
    return [
        {'label': '空头爆仓(50x)', 'px': h_24 * 1.018, 'color': 'rgba(255, 0, 0, 0.4)'},
        {'label': '多头爆仓(50x)', 'px': l_24 * 0.982, 'color': 'rgba(0, 255, 0, 0.4)'}
    ]

# ==========================================
# 3. AI 复盘报告逻辑：识别量价阴谋
# ==========================================
def generate_ai_recap(df, flow_4h, whale):
    """
    自动识别“放量砸盘”或“缩量诱多”
    """
    if df is None or df.empty: return "数据同步中..."
    
    last, prev = df.iloc[-1], df.iloc[-2]
    vol_ma = df['v'].tail(10).mean()
    
    # 核心博弈判断
    is_price_up = last['c'] > prev['c']
    is_vol_push = last['v'] > vol_ma * 1.5
    
    if is_price_up and not is_vol_push:
        status = "📉 缩量诱多：庄家虚拉吸引散户，警惕随时反手。"
    elif not is_price_up and is_vol_push:
        status = "🚨 放量砸盘：主力恐慌抛售或大户强平，严禁接刀。"
    else:
        status = "✅ 趋势运行：目前量价匹配，处于标准波段模式。"
        
    trend = "看多 🔥" if flow_4h > 0 else "看空 ❄️"
    whale_info = f"上方大单墙: ${whale['px']}" if whale is not None else "上方暂无巨阻"
    
    return f"{status}\n\n【博弈核心】主力趋势: {trend} | 庄家流: {whale_info}"

# ==========================================
# 4. 数据采集：卫星多源情报
# ==========================================
@st.cache_data(ttl=10)
def fetch_market_intel():
    """抓取K线、4H流及L2大单墙"""
    try:
        # 1. 1m K线抓取
        url_k = "https://www.okx.com/api/v5/market/candles?instId=ETH-USDT&bar=1m&limit=100"
        res_k = requests.get(url_k, timeout=5).json()
        df = pd.DataFrame(res_k['data'], columns=['ts','o','h','l','c','v','volC','volCQ','cf'])[::-1].reset_index(drop=True)
        for c in ['o','h','l','c','v']: df[c] = df[c].astype(float)
        df['atr'] = (df['h'] - df['l']).rolling(14).mean()
        
        # 2. 4H 趋势资金流
        url_f = "https://www.okx.com/api/v5/market/candles?instId=ETH-USDT&bar=4H&limit=5"
        res_f = requests.get(url_f, timeout=3).json()
        flow_4h = sum([(float(x[4]) - float(x[1])) for x in res_f['data']])
        
        # 3. L2 大单墙监测 (识别 >500 ETH 巨单)
        url_b = "https://www.okx.com/api/v5/market/books?instId=ETH-USDT&sz=20"
        res_b = requests.get(url_b, timeout=3).json()
        asks = pd.DataFrame(res_b['data'][0]['asks'], columns=['px', 'sz', 'cnt', 'ord'])
        whale = asks[asks['sz'].astype(float) > 500].iloc[0] if not asks.empty else None
        
        return df, flow_4h, whale
    except:
        return None, 0, None

# ==========================================
# 5. UI 渲染：指挥中心显示
# ==========================================
def main():
    init_system()
    df, flow_4h, whale = fetch_market_intel()
    
    if df is None:
        st.error("❌ 卫星连接失败：无法获取市场实时情报，请检查 API 网络或服务器状态")
        return

    # 计算衍生指标
    liq_zones = calculate_liquidation_zones(df)
    ai_report = generate_ai_recap(df, flow_4h, whale)
    last_p = df['c'].iloc[-1]

    # --- 顶层仪表盘 ---
    st.markdown(f"### 🛰️ ETH 量子决策指挥官 (V32000) | {datetime.now().strftime('%H:%M:%S')}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("4H 资金流", "流入 🔥" if flow_4h > 0 else "流出 ❄️")
    m2.metric("模式识别", "⚖️ 标准波段")
    m3.metric("实时价格", f"${last_p:.2f}")
    m4.metric("波动率 ATR", f"{df['atr'].iloc[-1]:.2f}")

    st.divider()

    # --- 交互侧边栏与主图表 ---
    col_l, col_r = st.columns([1, 4])
    
    with col_l:
        st.markdown(f"""<div style="padding:15px; border:2px solid #00ff00; border-radius:12px; background:rgba(0,0,0,0.5);">
            <h4 style="color:#00ff00; margin-top:0;">🤖 AI 自动复盘报告</h4>
            <p style="font-size:0.95em; color:#eee; line-height:1.6;">{ai_report.replace('【', '<br>【')}</p>
        </div>""", unsafe_allow_html=True)
        
        st.markdown("#### ⚡ 爆仓地雷/燃料分布")
        for z in liq_zones:
            st.caption(f"{z['label']}: **${z['px']:.1f}**")
            
        st.markdown("#### 📜 战术日志")
        if whale is not None: st.session_state.battle_logs.insert(0, f"🐋 发现大单墙: ${whale['px']}")
        for log in st.session_state.battle_logs[:6]: st.caption(log)

    with col_r:
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Candlestick(x=df.index, open=df['o'], high=df['h'], low=df['l'], close=df['c'], name="Price"))
        
        # 物理压力支撑位
        res_px = df['h'].tail(30).max()
        sup_px = df['l'].tail(30).min()
        fig.add_hline(y=res_px, line_dash="dash", line_color="red", annotation_text="压力")
        fig.add_hline(y=sup_px, line_dash="dash", line_color="green", annotation_text="支撑")
        
        # 爆仓热力区渲染
        for z in liq_zones:
            fig.add_hrect(y0=z['px']*0.9997, y1=z['px']*1.0003, fillcolor=z['color'], opacity=0.3, line_width=0)

        fig.update_layout(template="plotly_dark", height=750, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

    time.sleep(10)
    st.rerun()

if __name__ == "__main__":
    main()
