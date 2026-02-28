import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime

# ==========================================
# 1. 指挥部底层：环境锁与内存治理
# ==========================================
st.set_page_config(layout="wide", page_title="ETH V32000 终极指挥官", page_icon="⚖️")

def init_commander_state():
    """初始化持久化战术状态与内存防御机制"""
    if 'battle_logs' not in st.session_state: st.session_state.battle_logs = []
    if 'ls_ratio' not in st.session_state: st.session_state.ls_ratio = 1.0
    if 'last_cleanup' not in st.session_state: st.session_state.last_cleanup = time.time()
    
    # 每4小时强制内存清理，防止长时间盯盘卡顿
    if time.time() - st.session_state.last_cleanup > 14400:
        st.cache_data.clear()
        st.session_state.last_cleanup = time.time()
        st.session_state.battle_logs.append(f"【系统】{datetime.now().strftime('%H:%M:%S')} 内存治理完成")

# ==========================================
# 2. 核心算法层：策略、位置与清算逆推
# ==========================================
def calculate_liquidation_zones(df):
    """逆推 50x 杠杆生死线，识别价格磁吸区"""
    if df.empty: return []
    h_24h = df['h'].tail(120).max()
    l_24h = df['l'].tail(120).min()
    return [
        {'type': '空头爆仓(50x)', 'px': h_24h * 1.018, 'color': 'rgba(255, 0, 0, 0.4)'},
        {'type': '多头爆仓(50x)', 'px': l_24h * 0.982, 'color': 'rgba(0, 255, 0, 0.4)'}
    ]

def ai_market_recap(df, ls_ratio):
    """AI 深度复盘：识别“缩量诱多”或“放量砸盘”"""
    if df.empty: return "数据同步中..."
    last, prev = df.iloc[-1], df.iloc[-2]
    vol_ma = df['v'].tail(10).mean()
    is_vol_push = last['v'] > vol_ma * 1.5
    is_price_up = last['c'] > prev['c']
    
    if is_price_up and not is_vol_push: status = "📉 缩量诱多：庄家虚拉吸引散户，警惕随时反手。"
    elif not is_price_up and is_vol_push: status = "🚨 放量砸盘：主力抛售或大户强平，严禁接刀。"
    else: status = "✅ 趋势运行：目前量价匹配，处于标准模式。"
    
    sentiment = "看多 🔥" if ls_ratio < 0.95 else "看空 ❄️"
    return f"{status}\n\n【博弈核心】散户情绪: {sentiment} | 净流状态: {'流入' if last['net_flow']>0 else '流出'}"

# ==========================================
# 3. UI 组件层：策略导航列表 (同步止盈止损)
# ==========================================
def render_strategy_navigator(df, prob, liq_zones, ai_recap):
    """带止盈止损参考的策略 UI 阵列"""
    st.markdown("#### 🎯 实时策略执行计划")
    last_p = df['c'].iloc[-1]
    res_px = df['h'].tail(30).max()
    sup_px = df['l'].tail(30).min()
    atr = df['atr'].iloc[-1]
    sl_buffer = atr * 1.5 

    # 策略 1: 物理位陷阱
    p_active = abs(last_p - sup_px) < 5 or abs(last_p - res_px) < 5
    p_tp = res_px if abs(last_p - sup_px) < 5 else sup_px
    p_sl = last_p - sl_buffer if abs(last_p - sup_px) < 5 else last_p + sl_buffer

    # 策略 2: 清算猎杀
    l_active = prob > 65 or prob < 35
    l_tp = liq_zones[1]['px'] if prob < 50 else liq_zones[0]['px'] 
    l_sl = last_p + (atr * 2) if prob < 50 else last_p - (atr * 2)

    strats = [
        {"name": "物理位陷阱", "state": "✅ 激活" if p_active else "⚪ 观察", "tp": p_tp, "sl": p_sl, "action": "低吸高抛"},
        {"name": "清算猎杀", "state": "🔥 进攻" if l_active else "⚪ 待机", "tp": l_tp, "sl": l_sl, "action": "顺势推土"},
        {"name": "量价共振", "state": "🚨 预警" if "缩量诱多" in ai_recap else "✅ 正常", "tp": "过滤中", "sl": "禁入", "action": "防守观察"}
    ]

    for s in strats:
        color = "#00ff00" if "✅" in s['state'] or "🔥" in s['state'] else "#ff4b4b" if "🚨" in s['state'] else "#888"
        st.markdown(f"""
            <div style="border-left: 4px solid {color}; padding: 10px; margin-bottom: 12px; background: rgba(255,255,255,0.03); border-radius: 0 8px 8px 0;">
                <div style="display: flex; justify-content: space-between;">
                    <b style="color:{color};">{s['state']} | {s['name']}</b>
                    <i style="color:#aaa; font-size:0.8em;">{s['action']}</i>
                </div>
                <div style="margin-top: 8px; display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                    <div style="color:#00ff00; font-size:0.85em;">🎯 止盈: <b>${s['tp'] if isinstance(s['tp'], str) else f"{s['tp']:.1f}"}</b></div>
                    <div style="color:#ff4b4b; font-size:0.85em;">🛡️ 止损: <b>${s['sl'] if isinstance(s['sl'], str) else f"{s['sl']:.1f}"}</b></div>
                </div>
            </div>
        """, unsafe_allow_html=True)

# ==========================================
# 4. 数据层：多源实时情报引擎
# ==========================================
@st.cache_data(ttl=10)
def get_market_intel():
    """抓取K线、ATR及多空比"""
    try:
        url = "https://www.okx.com/api/v5/market/candles?instId=ETH-USDT&bar=1m&limit=100"
        res = requests.get(url, timeout=5).json()
        df = pd.DataFrame(res['data'], columns=['ts','o','h','l','c','v','volC','volCQ','cf'])[::-1].reset_index(drop=True)
        df['time'] = pd.to_datetime(df['ts'].astype(float), unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')
        for c in ['o','h','l','c','v']: df[c] = df[c].astype(float)
        
        # 净流与ATR计算
        df['net_flow'] = (df['c'].diff() * df['v'] * 0.15).rolling(5).sum().fillna(0)
        tr = pd.concat([df['h']-df['l'], abs(df['h']-df['c'].shift()), abs(df['l']-df['c'].shift())], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        
        ls_url = "https://www.okx.com/api/v5/rubik/stat/contracts/long-short-account-ratio?instId=ETH-USDT&period=5m"
        ls_res = requests.get(ls_url, timeout=5).json()
        if ls_res.get('code') == '0': st.session_state.ls_ratio = float(ls_res['data'][0][1])
        return df
    except: return pd.DataFrame()

# ==========================================
# 5. 主程序：渲染与逻辑编排
# ==========================================
def main():
    init_commander_state()
    df = get_market_intel()
    
    if df.empty:
        st.error("❌ 卫星同步失败：请检查 API 网络连接"); return

    # 预计算
    liq_zones = calculate_liquidation_zones(df)
    ai_recap = ai_market_recap(df, st.session_state.ls_ratio)
    last_p = df['c'].iloc[-1]
    prob = 50.0 + (15 if st.session_state.ls_ratio < 0.95 else -10) + (10 if df['net_flow'].iloc[-1] > 0 else -10)
    prob = max(min(prob, 99.0), 1.0)

    # UI 顶栏
    st.markdown(f"### 🛰️ ETH 量子决策指挥官 (V32000) | {datetime.now().strftime('%H:%M:%S')}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("AI 进攻胜率", f"{prob:.1f}%")
    m2.metric("实时价格", f"${last_p:.2f}")
    m3.metric("多空比", f"{st.session_state.ls_ratio}", "散户看空" if st.session_state.ls_ratio < 1 else "散户看多")
    m4.metric("ATR 波动", f"{df['atr'].iloc[-1]:.2f}")

    st.divider()

    # 指挥中心布局
    col_l, col_r = st.columns([1, 4])
    with col_l:
        # 1. 自动复盘报告
        st.info(f"🤖 **AI 自动复盘**\n\n{ai_recap}")
        
        # 2. 策略导航列表 (核心增强)
        render_strategy_navigator(df, prob, liq_zones, ai_recap)
        
        st.markdown("#### 📜 战术日志")
        for log in st.session_state.battle_logs[-5:]: st.caption(log)

    with col_r:
        # K线与清算地雷可视化
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df['time'], open=df['o'], high=df['h'], low=df['l'], close=df['c'], name="Price"), row=1, col=1)
        
        # 压力/支撑/地雷区
        fig.add_hline(y=df['h'].tail(30).max(), line_dash="dash", line_color="red", row=1, col=1)
        fig.add_hline(y=df['l'].tail(30).min(), line_dash="dash", line_color="green", row=1, col=1)
        for z in liq_zones:
            fig.add_hrect(y0=z['px']*0.9998, y1=z['px']*1.0002, fillcolor=z['color'], opacity=0.3, line_width=0, row=1, col=1)

        # 净流图表
        colors = ['#00ff00' if x > 0 else '#ff4b4b' for x in df['net_flow']]
        fig.add_trace(go.Bar(x=df['time'], y=df['net_flow'], marker_color=colors, name="NetFlow"), row=2, col=1)
        
        fig.update_layout(template="plotly_dark", height=800, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

    time.sleep(10); st.rerun()

if __name__ == "__main__":
    main()
