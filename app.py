import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime

# ==================== 1. 系统配置 ====================
st.set_page_config(page_title="ETH V160 分析师终端", layout="wide")

def fetch_okx(endpoint, params=""):
    url = f"https://www.okx.com/api/v5/{endpoint}?instId=ETH-USDT{params}"
    try:
        r = requests.get(url, timeout=5).json()
        return r if r.get('code') == '0' else None
    except: return None

# ==================== 2. 核心战术引擎 (整合 V15-V135) ====================
def tactical_engine_v160(df1, df5, net_flow, buy_ratio):
    # 指标计算
    for d in [df1, df5]:
        for col in ['o', 'h', 'l', 'c', 'v']: d[col] = d[col].astype(float)
        d['ema20'] = d['c'].ewm(span=20, adjust=False).mean()
        d['ema60'] = d['c'].ewm(span=60, adjust=False).mean()
        d['atr'] = (d['h'] - d['l']).rolling(14).mean()
        d['std'] = d['c'].rolling(20).std()
        d['upper'] = d['ema20'] + (d['std'] * 2.2)
        d['lower'] = d['ema20'] - (d['std'] * 2.2)

    curr1, curr5 = df1.iloc[-1], df5.iloc[-1]
    atr_val = curr1['atr']
    
    # 动能品质判定 (修复 V135 过于严苛的问题)
    vol_avg = df1['v'].rolling(20).mean().iloc[-1]
    v_ratio = curr1['v'] / vol_avg if vol_avg > 0 else 1.0
    quality = "💎 优良" if v_ratio > 1.1 else "🌪️ 一般"
    
    # 趋势判定逻辑
    slope = (df5['ema20'].iloc[-1] - df5['ema20'].iloc[-5]) / 5
    is_trend = abs(slope) > (atr_val * 0.1)

    # 策略生成计划
    plan = {"status": "🔭 扫描中", "action": "观望", "entry": None, "tp": None, "sl": None, "strategy": "等待高胜率形态"}
    
    # 逻辑 A：趋势突击 (V80 风格)
    if is_trend:
        if curr1['c'] > curr1['ema20'] and net_flow > 0.5:
            plan = {
                "status": "🚀 趋势多头", "action": "做多 (LONG)", 
                "entry": curr1['c'], "tp": curr1['c'] + 2.5*atr_val, "sl": curr1['c'] - 1.5*atr_val,
                "strategy": "追随 EMA20 趋势，博取波段拉升"
            }
        elif curr1['c'] < curr1['ema20'] and net_flow < -0.5:
            plan = {
                "status": "🔥 破位空头", "action": "做空 (SHORT)", 
                "entry": curr1['c'], "tp": curr1['c'] - 2.5*atr_val, "sl": curr1['c'] + 1.5*atr_val,
                "strategy": "恐慌盘释放，顺势截断下跌"
            }
    # 逻辑 B：震荡反抽 (V75 风格)
    else:
        if curr1['c'] < curr1['lower']:
            plan = {
                "status": "🏹 底部背离", "action": "博多 (BOUNCE)", 
                "entry": curr1['c'], "tp": curr1['ema20'], "sl": curr1['c'] - 1.0*atr_val,
                "strategy": "超卖反弹，目标中轨回归"
            }
        elif curr1['c'] > curr1['upper']:
            plan = {
                "status": "🎯 顶部承压", "action": "博空 (REJECT)", 
                "entry": curr1['c'], "tp": curr1['ema20'], "sl": curr1['c'] + 1.0*atr_val,
                "strategy": "超买回调，高位空单介入"
            }

    return plan, is_trend, quality, v_ratio, atr_val

# ==================== 3. 实时界面展示 ====================
# 获取数据
k1_raw = fetch_okx("market/candles", "&bar=1m&limit=100")
k5_raw = fetch_okx("market/candles", "&bar=5m&limit=100")
t_raw = fetch_okx("market/trades", "&limit=50")

if k1_raw and k5_raw and t_raw:
    df1 = pd.DataFrame(k1_raw['data'], columns=['ts','o','h','l','c','v','volC','volCQ','cf'])[::-1].reset_index(drop=True)
    df5 = pd.DataFrame(k5_raw['data'], columns=['ts','o','h','l','c','v','volC','volCQ','cf'])[::-1].reset_index(drop=True)
    tdf = pd.DataFrame(t_raw['data'])
    net_f = tdf[tdf['side']=='buy']['sz'].astype(float).sum() - tdf[tdf['side']=='sell']['sz'].astype(float).sum()
    buy_r = (tdf[tdf['side']=='buy']['sz'].astype(float).sum() / tdf['sz'].astype(float).sum()) * 100
    
    # 运行引擎
    plan, is_trend, quality, vol_ratio, atr = tactical_engine_v160(df1, df5, net_f, buy_r)

    st.title("🛡️ ETH 策略分析师终端 V160")
    
    # 指标看板
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("1m 净流入", f"{net_f:+.2f} ETH")
    c2.metric("实时买压", f"{buy_r:.1f}%")
    c3.metric("市场模式", "🌊 趋势" if is_trend else "⚖️ 震荡")
    c4.metric("信号品质", quality)

    st.markdown("---")
    
    # 核心作战计划与图表
    col_plan, col_chart = st.columns([1, 2])
    
    with col_plan:
        st.subheader("📋 实时详细交易计划")
        color = "#00FFCC" if "多" in plan['action'] or "BOUNCE" in plan['action'] else "#FF4B4B" if "空" in plan['action'] or "REJECT" in plan['action'] else "#888"
        st.markdown(f"""
            <div style="border:2px solid {color}; padding:15px; border-radius:10px; background-color:rgba(0,0,0,0.2)">
                <h2 style="color:{color}; margin-top:0">{plan['status']}</h2>
                <p style="font-size:20px"><b>行动建议：</b> {plan['action']}</p>
                <hr>
                <p>📍 <b>入场参考:</b> {f"${plan['entry']:.2f}" if plan['entry'] else "等待信号"}</p>
                <p style="color:#00FFCC">💰 <b>止盈目标:</b> {f"${plan['tp']:.2f}" if plan['tp'] else "---"}</p>
                <p style="color:#FF4B4B">❌ <b>硬性止损:</b> {f"${plan['sl']:.2f}" if plan['sl'] else "---"}</p>
                <p>📖 <b>策略核心:</b> {plan['strategy']}</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col_chart:
        fig = go.Figure(data=[go.Candlestick(x=df1.index, open=df1['o'], high=df1['h'], low=df1['l'], close=df1['c'], name="1m K")])
        fig.add_trace(go.Scatter(x=df1.index, y=df1['ema20'], line=dict(color='yellow', width=1), name="EMA20"))
        fig.add_trace(go.Scatter(x=df1.index, y=df1['upper'], line=dict(color='rgba(255,255,255,0.2)', dash='dot'), name="BB上"))
        fig.add_trace(go.Scatter(x=df1.index, y=df1['lower'], line=dict(color='rgba(255,255,255,0.2)', dash='dot'), name="BB下"))
        fig.update_layout(template="plotly_dark", height=450, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

    # 底部状态栏 (修复报错点)
    st.markdown(f"**🕵️ 系统诊断** | 量比: `{vol_ratio:.2f}` | 波动率 ATR: `{atr:.2f}` | 建议仓位: `5%-10%` | 刷新时间: {datetime.now().strftime('%H:%M:%S')}")

else:
    st.error("数据连接异常，请检查网络或 OKX API 状态")
