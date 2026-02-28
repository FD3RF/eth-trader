import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime

# ==========================================
# 1. 环境锁与内存管理
# ==========================================
st.set_page_config(layout="wide", page_title="ETH V32000 终极指挥官", page_icon="⚖️")

def init_commander_state():
    """初始化持久化状态与内存清理"""
    if 'battle_logs' not in st.session_state: st.session_state.battle_logs = []
    if 'ls_ratio' not in st.session_state: st.session_state.ls_ratio = 1.0
    if 'last_cleanup' not in st.session_state: st.session_state.last_cleanup = time.time()
    
    # 4小时强制清理缓存
    if time.time() - st.session_state.last_cleanup > 14400:
        st.cache_data.clear()
        st.session_state.last_cleanup = time.time()

# ==========================================
# 2. 核心算法层
# ==========================================
def calculate_liquidation_zones(df):
    """计算50x杠杆清算带"""
    if df.empty: return []
    h_24h = df['h'].tail(120).max()
    l_24h = df['l'].tail(120).min()
    return [
        {'type': '空头爆仓(50x)', 'px': h_24h * 1.018, 'color': 'rgba(255, 0, 0, 0.4)'},
        {'type': '多头爆仓(50x)', 'px': l_24h * 0.982, 'color': 'rgba(0, 255, 0, 0.4)'}
    ]

def ai_market_recap(df, ls_ratio):
    """识别量价背离与趋势模式"""
    if df.empty: return "数据同步中..."
    last, prev = df.iloc[-1], df.iloc[-2]
    vol_ma = df['v'].tail(10).mean()
    is_vol_push = last['v'] > vol_ma * 1.5
    is_price_up = last['c'] > prev['c']
    
    if is_price_up and not is_vol_push: status = "📉 缩量诱多：价格虚拉，警惕随时反手。"
    elif not is_price_up and is_vol_push: status = "🚨 放量砸盘：主力抛售，严禁接刀。"
    else: status = "✅ 趋势运行：量价匹配，处于标准模式。"
    
    sentiment = "看多 🔥" if ls_ratio < 0.95 else "看空 ❄️"
    return f"{status}\n\n【博弈核心】散户情绪: {sentiment} | 净流状态: {'流入' if last['net_flow']>0 else '流出'}"

# ==========================================
# 3. 数据抓取引擎
# ==========================================
@st.cache_data(ttl=10)
def get_market_intel():
    """实时抓取OKX K线与多空比"""
    try:
        url = "https://www.okx.com/api/v5/market/candles?instId=ETH-USDT&bar=1m&limit=100"
        res = requests.get(url, timeout=5).json()
        df = pd.DataFrame(res['data'], columns=['ts','o','h','l','c','v','volC','volCQ','cf'])[::-1].reset_index(drop=True)
        df['time'] = pd.to_datetime(df['ts'].astype(float), unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')
        for c in ['o','h','l','c','v']: df[c] = df[c].astype(float)
        
        # 技术指标计算
        df['net_flow'] = (df['c'].diff() * df['v'] * 0.15).rolling(5).sum().fillna(0)
        tr = pd.concat([df['h']-df['l'], abs(df['h']-df['c'].shift()), abs(df['l']-df['c'].shift())], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        
        ls_url = "https://www.okx.com/api/v5/rubik/stat/contracts/long-short-account-ratio?instId=ETH-USDT&period=5m"
        ls_res = requests.get(ls_url, timeout=5).json()
        if ls_res.get('code') == '0': st.session_state.ls_ratio = float(ls_res['data'][0][1])
        return df
    except: return pd.DataFrame()

# ==========================================
# 4. 侧边栏决策舱渲染
# ==========================================
def render_decision_sidebar(df, prob, liq_zones, ai_recap):
    """在边栏中渲染AI复盘与策略列表"""
    with st.sidebar:
        st.markdown("## 🤖 决策指挥中心")
        
        # 4.1 AI 自动复盘
        st.markdown("#### 🔍 AI 自动复盘")
        st.info(ai_recap)
        
        st.divider()
        
        # 4.2 策略导航列表
        st.markdown("#### 🎯 策略执行计划")
        last_p = df['c'].iloc[-1]
        res_px = df['h'].tail(30).max()
        sup_px = df['l'].tail(30).min()
        atr = df['atr'].iloc[-1]
        sl_buffer = atr * 1.5 

        # 策略计算逻辑
        p_active = abs(last_p - sup_px) < 5 or abs(last_p - res_px) < 5
        p_tp, p_sl = (res_px, last_p - sl_buffer) if abs(last_p - sup_px) < 5 else (sup_px, last_p + sl_buffer)
        l_active = prob > 65 or prob < 35
        l_tp, l_sl = (liq_zones[1]['px'], last_p + (atr * 2)) if prob < 50 else (liq_zones[0]['px'], last_p - (atr * 2))

        strats = [
            {"name": "物理位陷阱", "state": "✅ 激活" if p_active else "⚪ 观察", "tp": p_tp, "sl": p_sl, "action": "低吸高抛"},
            {"name": "清算猎杀", "state": "🔥 进攻" if l_active else "⚪ 待机", "tp": l_tp, "sl": l_sl, "action": "顺势推土"},
            {"name": "量价共振", "state": "🚨 预警" if "缩量诱多" in ai_recap else "✅ 正常", "tp": "过滤中", "sl": "禁入", "action": "防守观察"}
        ]

        for s in strats:
            color = "#00ff00" if "✅" in s['state'] or "🔥" in s['state'] else "#ff4b4b" if "🚨" in s['state'] else "#888"
            st.markdown(f"""
                <div style="border-left: 4px solid {color}; padding: 10px; margin-bottom: 12px; background: rgba(255,255,255,0.05); border-radius: 4px;">
                    <div style="font-size:0.9em; color:{color}; font-weight:bold;">{s['state']} | {s['name']}</div>
                    <div style="font-size:0.75em; color:#aaa;">{s['action']}</div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 5px; margin-top:5px;">
                        <div style="color:#00ff00; font-size:0.8em;">🎯 TP: <b>${s['tp'] if isinstance(s['tp'], str) else f"{s['tp']:.1f}"}</b></div>
                        <div style="color:#ff4b4b; font-size:0.8em;">🛡️ SL: <b>${s['sl'] if isinstance(s['sl'], str) else f"{s['sl']:.1f}"}</b></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        st.divider()
        st.markdown("#### 📜 战术日志")
        for log in st.session_state.battle_logs[-5:]: st.caption(log)

# ==========================================
# 5. 主程序渲染逻辑
# ==========================================
def main():
    init_commander_state()
    df = get_market_intel()
    
    if df.empty:
        st.error("❌ 卫星同步失败：请检查网络"); return

    # 数据预计算
    liq_zones = calculate_liquidation_zones(df)
    ai_recap = ai_market_recap(df, st.session_state.ls_ratio)
    last_p = df['c'].iloc[-1]
    prob = 50.0 + (15 if st.session_state.ls_ratio < 0.95 else -10) + (10 if df['net_flow'].iloc[-1] > 0 else -10)
    prob = max(min(prob, 99.0), 1.0)

    # 1. 渲染边栏
    render_decision_sidebar(df, prob, liq_zones, ai_recap)

    # 2. 渲染主页面顶栏
    st.markdown(f"### 🛰️ ETH 量子决策指挥官 (V32000) | {datetime.now().strftime('%H:%M:%S')}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("AI 进攻胜率", f"{prob:.1f}%")
    m2.metric("实时价格", f"${last_p:.2f}")
    m3.metric("多空比", f"{st.session_state.ls_ratio}", "散户看空" if st.session_state.ls_ratio < 1 else "散户看多")
    m4.metric("ATR 波动", f"{df['atr'].iloc[-1]:.2f}")

    # 3. 绘制全屏图表
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
    
    # K线与清算地雷
    fig.add_trace(go.Candlestick(x=df['time'], open=df['o'], high=df['h'], low=df['l'], close=df['c'], name="价格"), row=1, col=1)
    for z in liq_zones:
        fig.add_hrect(y0=z['px']*0.9998, y1=z['px']*1.0002, fillcolor=z['color'], opacity=0.4, line_width=0, row=1, col=1)
    
    # 物理支撑阻力线
    fig.add_hline(y=df['h'].tail(30).max(), line_dash="dash", line_color="red", annotation_text="物理压力", row=1, col=1)
    fig.add_hline(y=df['l'].tail(30).min(), line_dash="dash", line_color="green", annotation_text="物理支撑", row=1, col=1)

    # 净流入柱状图
    colors = ['#00ff00' if x > 0 else '#ff4b4b' for x in df['net_flow']]
    fig.add_trace(go.Bar(x=df['time'], y=df['net_flow'], marker_color=colors, name="净流状态"), row=2, col=1)
    
    fig.update_layout(template="plotly_dark", height=700, xaxis_rangeslider_visible=False, margin=dict(l=10,r=10,t=0,b=0))
    st.plotly_chart(fig, use_container_width=True)

    time.sleep(10); st.rerun()

if __name__ == "__main__":
    main()
