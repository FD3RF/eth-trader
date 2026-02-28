import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime

# ==================== 1. 系统核心配置 ====================
st.set_page_config(page_title="ETH V65.0 终极全功能终端", layout="wide")

def fetch_okx_data(endpoint, params=""):
    url = f"https://www.okx.com/api/v5/{endpoint}?instId=ETH-USDT{params}"
    try:
        with requests.Session() as s:
            s.trust_env = True  
            r = s.get(url, timeout=5).json()
            return r if r.get('code') == '0' else None
    except: return None

# ==================== 2. 计算引擎：战绩/模式/转向 ====================
def master_engine(df):
    delta = df['c'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain/loss))
    df['atr'] = (df['h'] - df['l']).rolling(12).mean()
    
    lookback = 288
    recent_df = df.tail(lookback).copy()
    success, total = 0, 0
    replay_data = {'buy': [], 'sell': []}
    
    for i in range(20, len(recent_df) - 6):
        if recent_df['rsi'].iloc[i] < 35:
            total += 1
            entry_p = recent_df['c'].iloc[i]
            future = recent_df.iloc[i+1 : i+7]
            if future['h'].max() > entry_p * 1.005:
                success += 1
                replay_data['buy'].append((recent_df.index[i], entry_p))
                replay_data['sell'].append((future['h'].idxmax(), future['h'].max()))
                
    win_rate = (success / total * 100) if total > 0 else 0
    return win_rate, total, replay_data

# ==================== 3. AI 大白话决策系统 ====================
def generate_smart_plan(curr_p, sup, res, win_rate, net_f, buy_ratio):
    PANIC_ZONE = 35.0 
    tips = []
    
    # 诊断提醒
    if net_f < -15: tips.append("🔴 庄家正在撤退：大单砸得很猛，别去当炮灰接飞刀！")
    elif net_f > 15: tips.append("🟢 主力在偷偷买：资金在流入，反弹可能要来了。")
    if buy_ratio < 40: tips.append("💀 阵地不稳：卖的人多买的人少，小心下面还有坑。")

    if win_rate < PANIC_ZONE:
        # 【反手做空模式】
        p = {
            "title": "🔴 AI 建议：反手做空",
            "tag": "多头已死，顺势而为",
            "color": "#ff4b4b",
            "why": f"最近抄底的 10 人里有 8 个都亏了（胜率才 {win_rate:.1f}%），说明市场在走下坡路。",
            "how_in": f"等价格稍微反弹到 ${res:.2f} 左右（也就是主力原本想跑路的位置）再进场做空。",
            "how_out": f"万一价格冲回 ${res+10:.2f} 就要认怂止损，说明多头还没断气。",
            "money": f"目标看到 ${sup:.2f}，那里可能才有支撑。",
            "entry": res, "sl": res+10, "tp": sup
        }
    elif net_f < -20:
        # 【拦截观望模式】
        p = {
            "title": "🟡 AI 建议：空仓看戏",
            "tag": "现在的钱不好赚，保命要紧",
            "color": "#ffd700",
            "why": "虽然看着像到了底部，但大户跑路速度太快，这种‘假支撑’一踩就破。",
            "how_in": "系统已经帮你封印了买入键。等这波风头过去再说。",
            "how_out": "空仓是最好的防守。",
            "money": "无",
            "entry": 0, "sl": 0, "tp": 0
        }
    else:
        # 【标准抄底模式】
        p = {
            "title": "🟢 AI 建议：低吸波段",
            "tag": "行情稳住了，捡点便宜货",
            "color": "#00ffcc",
            "why": "现在是震荡节奏，主力在护盘，胜率也靠谱，适合捞一把就跑。",
            "how_in": f"在 ${sup:.2f} 附近蹲点买入，这里是大佬们的护盘阵地。",
            "how_out": f"跌破 ${sup-10:.2f} 赶紧撤，说明大佬们也守不住了。",
            "money": f"涨到 ${res:.2f} 就把它卖掉，落袋为安。",
            "entry": sup, "sl": sup-10, "tp": res
        }
    return p, tips

# ==================== 4. 主逻辑渲染 ====================
k_raw = fetch_okx_data("market/candles", "&bar=5m&limit=400")
d_raw = fetch_okx_data("market/books", "&sz=20")
t_raw = fetch_okx_data("market/trades", "&limit=100")

if k_raw and d_raw:
    df = pd.DataFrame(k_raw['data'], columns=['ts','o','h','l','c','v','volCcy','volCcyQ','confirm'])
    df[['o','h','l','c','v']] = df[['o','h','l','c','v']].astype(float)
    df = df[::-1].reset_index(drop=True)
    curr_p = df.iloc[-1]['c']
    
    wr, total_s, replay = master_engine(df)
    
    with st.sidebar:
        st.header("🧠 AI 指挥官 V65.0")
        st.metric("24H 模拟胜率", f"{wr:.1f}%")
        show_replay = st.toggle("🔍 开启一键复盘标记", value=False)
        st.divider()
        
        asks = pd.DataFrame(d_raw['data'][0]['asks']).iloc[:, :2].astype(float)
        bids = pd.DataFrame(d_raw['data'][0]['bids']).iloc[:, :2].astype(float)
        res_p = asks[asks[1] > asks[1].mean() * 1.8].iloc[0, 0] if not asks.empty else None
        sup_p = bids[bids[1] > bids[1].mean() * 1.8].iloc[0, 0] if not bids.empty else None
        tdf = pd.DataFrame(t_raw['data'])
        net_f = tdf[tdf['side']=='buy']['sz'].astype(float).sum() - tdf[tdf['side']=='sell']['sz'].astype(float).sum()
        buy_r = (bids[1].sum()/(asks[1].sum()+bids[1].sum())*100)
        
        plan, tip_list = generate_smart_plan(curr_p, sup_p, res_p, wr, net_f, buy_r)
        
        for t in tip_list:
            st.info(t)
            
        st.markdown(f"""
            <div style="border:4px solid {plan['color']}; padding:15px; border-radius:12px; background:rgba(0,0,0,0.1)">
                <h2 style="color:{plan['color']}; margin:0">{plan['title']}</h2>
                <p style="color:{plan['color']}"><b>「{plan['tag']}」</b></p>
                <p style="font-size:14px">💬 <b>大白话理由：</b>{plan['why']}</p>
                <hr style="border:0.5px solid #444">
                <p>📍 <b>在哪进场：</b>{plan['how_in']}</p>
                <p>❌ <b>在哪认输：</b>{plan['how_out']}</p>
                <p style="color:#00ffcc">💰 <b>目标收钱：</b>{plan['money']}</p>
            </div>
        """, unsafe_allow_html=True)

    st.title(f"🛡️ ETH 决策终端 V65.0")
    col1, col2, col3 = st.columns(3)
    col1.metric("现价", f"${curr_p}")
    col2.metric("1min 净流入", f"{net_f:+.2f} ETH")
    col3.metric("盘口买压", f"{buy_r:.1f}%")

    fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['o'], high=df['h'], low=df['l'], close=df['c'], name="K线")])
    if show_replay:
        if replay['buy']:
            b_idx, b_val = zip(*replay['buy'])
            fig.add_trace(go.Scatter(x=b_idx, y=[v*0.998 for v in b_val], mode='markers', marker=dict(symbol='triangle-up', size=10, color='#00ffcc')))
        if replay['sell']:
            s_idx, s_val = zip(*replay['sell'])
            fig.add_trace(go.Scatter(x=s_idx, y=[v*1.002 for v in s_val], mode='markers', marker=dict(symbol='triangle-down', size=10, color='#ff00ff')))
    
    if res_p: fig.add_hline(y=res_p, line_dash="dash", line_color="red", annotation_text="压力/做空区")
    if sup_p: fig.add_hline(y=sup_p, line_dash="dash", line_color="green", annotation_text="支撑/做多区")
    fig.update_layout(template="plotly_dark", height=700, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.error("数据连接异常，请检查网络...")
