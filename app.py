import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go

# ==================== 1. 系统核心配置 ====================
st.set_page_config(page_title="ETH V81 战略指挥版", layout="wide")

def fetch_okx_data(endpoint, params=""):
    url = f"https://www.okx.com/api/v5/{endpoint}?instId=ETH-USDT{params}"
    try:
        r = requests.get(url, timeout=5).json()
        return r if r.get('code') == '0' else None
    except: return None

# ==================== 2. 深度分析函数 (恢复 AI 建议模块) ====================
def generate_ai_strategy(sig, curr_p, atr, f_score, net_flow, buy_ratio, is_real):
    if not sig['type']:
        return None
    
    # 计算风控位
    sl_multiplier = 1.5 if sig['dir'] == "LONG" else 1.5
    tp_multiplier = 3.0
    sl = curr_p - (sl_multiplier * atr) if sig['dir'] == "LONG" else curr_p + (sl_multiplier * atr)
    tp = curr_p + (tp_multiplier * atr) if sig['dir'] == "LONG" else curr_p - (tp_multiplier * atr)
    
    # 策略话术库
    strat = {
        "dir_text": "反手做多" if sig['dir'] == "LONG" else "反手做空",
        "logic": f"当前1min买压 {buy_ratio:.1f}%，且动能验证为 {('真实' if is_real else '虚假')}。资金净流入 {net_flow:.2f} ETH，显示主力{'正在扫货' if net_flow > 0 else '正在砸盘'}。",
        "action": f"建议在 ${curr_p:.2f} 附近{'入场跟进' if is_real else '轻仓试探'}，跟随主力步伐。",
        "stop_loss": f"若价格回落/反弹至 ${sl:.2f}，说明{'多头' if sig['dir'] == 'LONG' else '空头'}动能衰竭，需止损离场。",
        "target": f"第一目标点位看到 ${tp:.2f}，此处存在{'压力' if sig['dir'] == 'LONG' else '支撑'}。"
    }
    return strat

# ==================== 3. 核心辨伪引擎 (V80 逻辑) ====================
def engine_v81(df1, net_flow, buy_ratio):
    df1['ema20'] = df1['c'].ewm(span=20, adjust=False).mean()
    df1['atr'] = (df1['h'] - df1['l']).rolling(14).mean()
    curr = df1.iloc[-1]
    vol_mean = df1['v'].rolling(20).mean().iloc[-1]
    
    # 辨伪因子
    price_move_ratio = abs(curr['c'] - curr['o']) / curr['atr']
    vol_surge_ratio = curr['v'] / vol_mean
    is_real = vol_surge_ratio > 1.1 or price_move_ratio < 1.5
    is_spoofing = buy_ratio > 90 and net_flow < 2.0
    
    f_score = 0
    if net_flow > 4 and not is_spoofing: f_score += 3
    if net_flow < -4: f_score -= 3
    if buy_ratio > 60: f_score += 1
    if buy_ratio < 40: f_score -= 1

    sig = {"type": None, "dir": None, "score": 0, "warning": ""}
    if is_real and abs(f_score) >= 3:
        if curr['c'] > df1['ema20'].iloc[-1] and f_score > 0:
            sig = {"type": "真·趋势突破", "dir": "LONG", "score": 7 + f_score}
        elif curr['c'] < df1['ema20'].iloc[-1] and f_score < 0:
            sig = {"type": "真·趋势崩塌", "dir": "SHORT", "score": 7 + abs(f_score)}
    
    if is_spoofing: sig["warning"] = "🚨 监测到虚假对敲诱多！"
    elif not is_real and price_move_ratio > 2.0: sig["warning"] = "⚠️ 无量异动，谨防假突破！"
    
    return sig, curr['atr'], f_score, is_real

# ==================== 4. 渲染 ====================
k1 = fetch_okx_data("market/candles", "&bar=1m&limit=100")
t_data = fetch_okx_data("market/trades", "&limit=50")

if k1 and t_data:
    df = pd.DataFrame(k1['data'], columns=['ts','o','h','l','c','v','volCcy','volCcyQ','confirm'])
    df[['o','h','l','c','v']] = df[['o','h','l','c','v']].astype(float)
    df = df[::-1].reset_index(drop=True)
    
    tdf = pd.DataFrame(t_data['data'])
    buy_v = tdf[tdf['side']=='buy']['sz'].astype(float).sum()
    sell_v = tdf[tdf['side']=='sell']['sz'].astype(float).sum()
    net_f = buy_v - sell_v
    buy_r = (buy_v / (buy_v + sell_v)) * 100 if (buy_v + sell_v) > 0 else 50
    
    sig, atr, f_score, is_real = engine_v81(df, net_f, buy_r)
    curr_p = df.iloc[-1]['c']
    ai_plan = generate_ai_strategy(sig, curr_p, atr, f_score, net_f, buy_r, is_real)

    with st.sidebar:
        st.header("🧠 AI 参谋部 V81")
        if sig['warning']: st.error(sig['warning'])
        
        if ai_plan:
            st.markdown(f"""
                <div style="background:rgba(255,75,75,0.1); padding:15px; border-radius:10px; border-left:5px solid #ff4b4b">
                    <h3 style="margin:0; color:#ff4b4b">🤖 AI 建议：{ai_plan['dir_text']}</h3>
                    <p style="font-size:14px; margin-top:10px"><b>● 大白话理由：</b>{ai_plan['logic']}</p>
                    <p style="font-size:14px"><b>📍 在哪进场：</b>{ai_plan['action']}</p>
                    <p style="font-size:14px"><b>❌ 在哪认输：</b>{ai_plan['stop_loss']}</p>
                    <p style="font-size:14px"><b>💰 目标收钱：</b>{ai_plan['target']}</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.info("🔭 正在深度扫描资金动能，暂无确定性策略。建议空仓观察主力下一步动作。")

    st.title("🛡️ ETH 战略执行终端 V81")
    col1, col2, col3 = st.columns(3)
    col1.metric("现价", f"${curr_p}")
    col2.metric("1min 净流", f"{net_f:+.2f} ETH")
    col3.metric("动能品质", "✅ 真实" if is_real else "❌ 虚假")

    fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['o'], high=df['h'], low=df['l'], close=df['c'])])
    fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
