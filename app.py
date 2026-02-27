import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==================== 1. 配置与数据引擎 ====================
st.set_page_config(page_title="ETH V61.0 趋势反转终端", layout="wide")

def fetch_okx_data(endpoint, params=""):
    url = f"https://www.okx.com/api/v5/{endpoint}?instId=ETH-USDT{params}"
    try:
        with requests.Session() as s:
            s.trust_env = True  
            r = s.get(url, timeout=5).json()
            return r if r.get('code') == '0' else None
    except: return None

# ==================== 2. 战绩回溯（包含空头逻辑回测） ====================
def get_backtest_results(df):
    lookback = 288
    recent_df = df.tail(lookback).copy()
    delta = recent_df['c'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    recent_df['rsi'] = 100 - (100 / (1 + gain/loss))
    
    success, total = 0, 0
    for i in range(20, len(recent_df) - 6):
        if recent_df['rsi'].iloc[i] < 35: # 抄底信号回测
            total += 1
            if recent_df['h'].iloc[i+1 : i+7].max() > recent_df['c'].iloc[i] * 1.005:
                success += 1
    win_rate = (success / total * 100) if total > 0 else 0
    return total, win_rate

# ==================== 3. 核心：空头转向指令引擎 ====================
def get_flipper_plan(curr_p, sup, res, mode, net_flow, win_rate):
    # 阈值定义
    PANIC_THRESHOLD = 35.0 # 胜率低于35%开启空头转向
    
    # --- 逻辑 A：空头转向模式 (针对阴跌) ---
    if win_rate < PANIC_THRESHOLD:
        action = "🔴 逢高做空 (Trend Flip)"
        color = "#ff4b4b"
        reason = f"多头胜率过低({win_rate:.1f}%)，系统已自动反转逻辑：将阻力位视为做空点。"
        # 空头计划：在阻力位附近进场，支撑位止盈
        entry = res if res else curr_p * 1.002
        sl = entry * 1.005 # 止损设在进场位上方 0.5%
        tp = sup if sup else curr_p * 0.99
        return {"action": action, "color": color, "reason": reason, "entry": entry, "sl": sl, "tp": tp}
    
    # --- 逻辑 B：正常逻辑 (震荡/趋势多头) ---
    if net_flow < -20:
        return {"action": "🟡 观望 (资金出逃)", "color": "#ffd700", "reason": "资金净流出过大，暂不建议操作。", "entry":0,"sl":0,"tp":0}
    
    action = "🟢 逢低买入" if curr_p < (sup + 2 if sup else curr_p) else "⚪ 待机中"
    return {
        "action": action, "color": "#00ffcc", "reason": "多头胜率健康，维持正常抄底逻辑。",
        "entry": curr_p, "sl": sup-2 if sup else curr_p-10, "tp": res if res else curr_p+15
    }

# ==================== 4. UI 渲染 ====================
k_raw = fetch_okx_data("market/candles", "&bar=5m&limit=400")
d_raw = fetch_okx_data("market/books", "&sz=20")
t_raw = fetch_okx_data("market/trades", "&limit=100")

if k_raw and d_raw:
    df = pd.DataFrame(k_raw['data'], columns=['ts','o','h','l','c','v','volCcy','volCcyQ','confirm'])
    df[['o','h','l','c','v']] = df[['o','h','l','c','v']].astype(float)
    df = df[::-1].reset_index(drop=True)
    curr_p = df.iloc[-1]['c']
    
    total_s, wr = get_backtest_results(df)
    
    with st.sidebar:
        st.header("📈 逻辑状态")
        st.metric("多头 24H 胜率", f"{wr:.1f}%")
        status_text = "🛡️ 保护中: 逻辑已反转" if wr < 35 else "✅ 正常: 多头逻辑"
        st.write(status_text)
        
        st.divider()
        st.header("🎯 自动转向指令")
        
        # 盘口与净流
        asks = pd.DataFrame(d_raw['data'][0]['asks']).iloc[:, :2].astype(float)
        bids = pd.DataFrame(d_raw['data'][0]['bids']).iloc[:, :2].astype(float)
        res_p = asks[asks[1] > asks[1].mean() * 1.8].iloc[0, 0] if not asks.empty else None
        sup_p = bids[bids[1] > bids[1].mean() * 1.8].iloc[0, 0] if not bids.empty else None
        tdf = pd.DataFrame(t_raw['data'])
        net_f = tdf[tdf['side']=='buy']['sz'].astype(float).sum() - tdf[tdf['side']=='sell']['sz'].astype(float).sum()
        
        # 获取反转计划
        plan = get_flipper_plan(curr_p, sup_p, res_p, "震荡", net_f, wr)
        st.markdown(f"""<div style="border:3px solid {plan['color']}; padding:15px; border-radius:10px; background:rgba(0,0,0,0.3)">
            <h2 style="margin:0; color:{plan['color']}">{plan['action']}</h2>
            <p style="margin:10px 0; font-size:14px; color:#ddd">{plan['reason']}</p>
            {f'<p style="margin:5px 0"><b>空单进场：</b>${plan["entry"]:.2f}</p>' if plan['entry'] > 0 else ''}
            {f'<p style="margin:5px 0; color:#ffbcbc"><b>空单止损：</b>${plan["sl"]:.2f}</p>' if plan['sl'] > 0 else ''}
            {f'<p style="margin:5px 0; color:#bcffbc"><b>目标平空：</b>${plan["tp"]:.2f}</p>' if plan['tp'] > 0 else ''}
        </div>""", unsafe_allow_html=True)

    st.title(f"🛡️ ETH V61.0 {'反转模式' if wr < 35 else '标准模式'}")
    # 绘图逻辑同上
    fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['o'], high=df['h'], low=df['l'], close=df['c'])])
    if res_p: fig.add_hline(y=res_p, line_dash="dash", line_color="#ff4b4b", annotation_text="空头入场区")
    if sup_p: fig.add_hline(y=sup_p, line_dash="dash", line_color="#00ffcc", annotation_text="空头止盈区")
    fig.update_layout(template="plotly_dark", height=700, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.error("正在同步反转算法...")
