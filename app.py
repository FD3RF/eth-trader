import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime

# ==========================================
# 1. 物理层：环境硬锁与指挥部初始化
# ==========================================
st.set_page_config(layout="wide", page_title="ETH V32000 决策指挥官", page_icon="⚖️")

def init_quantum_commander():
    """确保所有战术状态在刷新中保持锁定"""
    if 'battle_logs' not in st.session_state: st.session_state.battle_logs = []
    if 'sentiment_score' not in st.session_state: st.session_state.sentiment_score = 50.0
    if 'ls_ratio' not in st.session_state: st.session_state.ls_ratio = 1.0
    if 'prev_depth' not in st.session_state: st.session_state.prev_depth = {'bids': None, 'asks': None}

init_quantum_commander()

# ==========================================
# 2. 侧边栏：量子控制与复盘导出中心
# ==========================================
st.sidebar.markdown("### 🛸 量子实时控制")
auto_refresh = st.sidebar.toggle("开启全球全维度监控", value=True)
refresh_rate = st.sidebar.slider("心跳频率 (秒)", 5, 60, 10)

st.sidebar.divider()
ema_f = st.sidebar.number_input("快线 EMA", 5, 30, 12)
ema_s = st.sidebar.number_input("慢线 EMA", 20, 100, 26)

st.sidebar.divider()
st.sidebar.markdown("### 📊 战役复盘中心")
if st.sidebar.button("💾 生成今日战术报告"):
    if st.session_state.battle_logs:
        logs_df = pd.DataFrame([{"时间": l.split("】")[0][1:], "战术详情": l.split("】")[1]} for l in st.session_state.battle_logs])
        st.sidebar.download_button("📥 下载复盘报告 (CSV)", data=logs_df.to_csv(index=False).encode('utf-8-sig'), file_name="ETH_Combat_Report.csv")
    else:
        st.sidebar.error("暂无战斗日志")

# ==========================================
# 3. 核心引擎：多源情报采集与逻辑计算
# ==========================================
@st.cache_data(ttl=refresh_rate)
def get_commander_intel():
    """获取 K线、多空比、ATR 波动率"""
    try:
        # K线与指标
        k_url = "https://www.okx.com/api/v5/market/candles?instId=ETH-USDT&bar=1m&limit=100"
        kr = requests.get(k_url, timeout=5).json()
        df = pd.DataFrame(kr['data'], columns=['ts','o','h','l','c','v','volC','volCQ','cf'])[::-1].reset_index(drop=True)
        df['time'] = pd.to_datetime(df['ts'].astype(float), unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')
        for col in ['o','h','l','c','v']: df[col] = df[col].astype(float)
        
        df['ema_f'] = df['c'].ewm(span=ema_f, adjust=False).mean()
        df['ema_s'] = df['c'].ewm(span=ema_s, adjust=False).mean()
        df['net_flow'] = (df['c'].diff() * df['v'] * 0.15).rolling(5).sum().fillna(0)
        
        # ATR 波动率计算
        tr = pd.concat([df['h']-df['l'], abs(df['h']-df['c'].shift()), abs(df['l']-df['c'].shift())], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        
        # 全网多空比
        ls_url = "https://www.okx.com/api/v5/rubik/stat/contracts/long-short-account-ratio?instId=ETH-USDT&period=5m"
        lsr = requests.get(ls_url, timeout=5).json()
        if lsr.get('code') == '0': st.session_state.ls_ratio = float(lsr['data'][0][1])
        
        return df
    except: return st.session_state.df

def analyze_win_and_rr(df, score, ratio):
    """AI 胜率与 R/R 盈亏比核心评估逻辑"""
    last = df.iloc[-1]
    # AI 胜率预判
    prob = 50.0 + (10 if last['ema_f'] > last['ema_s'] else -5) + (15 if ratio < 0.95 else -10) + (score - 50) * 0.35
    prob = max(min(prob, 99.0), 1.0)
    
    # ATR 动态建议线计算
    atr = last['atr']
    tp_multiplier = 1.5 if prob < 55 else 2.5 if prob < 75 else 3.5
    sl_multiplier = 1.5
    
    tp = last['c'] + (atr * tp_multiplier) if prob >= 50 else last['c'] - (atr * tp_multiplier)
    sl = last['c'] - (atr * sl_multiplier) if prob >= 50 else last['c'] + (atr * sl_multiplier)
    rr = abs(tp - last['c']) / abs(last['c'] - sl)
    
    return prob, tp, sl, rr

# ==========================================
# 4. 渲染层：指挥部大屏
# ==========================================
def main():
    df = get_commander_intel()
    if df.empty: return
    
    # 模拟盘口撤单侦测逻辑 (整合自截图日志)
    now = datetime.now().strftime('%H:%M:%S')
    if len(st.session_state.battle_logs) < 1:
         st.session_state.battle_logs.insert(0, f"【{now}】🛰️ 卫星同步成功: 物理状态 Active Sync")

    win_p, tp_line, sl_line, rr_val = analyze_win_and_rr(df, st.session_state.sentiment_score, st.session_state.ls_ratio)
    
    # 顶部状态看板
    st.markdown(f"### 🛰️ ETH 量子决策指挥官 (V32000) | {datetime.now().strftime('%H:%M:%S')}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AI 进攻胜率", f"{win_p:.1f}%", f"{win_p-50:+.1f}%")
    c2.metric("建议盈亏比 (R/R)", f"1 : {rr_val:.2f}")
    c3.metric("全网多空比", f"{st.session_state.ls_ratio}", "散户看空" if st.session_state.ls_ratio < 1 else "散户看多")
    c4.metric("庄家情绪", f"{st.session_state.sentiment_score:.0f}", "庄家拉升" if st.session_state.sentiment_score > 60 else "博弈中")

    st.divider()

    # 指挥部双栏布局
    col_l, col_r = st.columns([1, 4])
    with col_l:
        # 决策信号盒
        decision_color = "#00ff00" if (win_p > 60 and rr_val > 1.8) else "#FFD700" if win_p > 50 else "#ff4b4b"
        st.markdown(f"""<div style="border:2px solid {decision_color}; padding:20px; border-radius:15px; text-align:center; background:rgba(0,0,0,0.5); box-shadow: 0 0 15px {decision_color};">
            <h2 style="color:{decision_color}; margin:0;">{'🔥 黄金机会' if win_p > 65 else '⚖️ 策略博弈'}</h2>
            <p style="color:{decision_color}; font-size:2.2em; font-weight:bold; margin:10px 0;">{win_p:.1f}%</p>
            <p style="color:#888; font-size:0.8em;">盈亏比指引: 1 : {rr_val:.2f}</p>
        </div>""", unsafe_allow_html=True)
        
        st.markdown("#### 📜 战术日志")
        for log in st.session_state.battle_logs[:12]:
            st.caption(log)

    with col_r:
        # 深度集成图表 (K线 + ATR 建议线 + 资金流)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
        
        # 1. K线层
        fig.add_trace(go.Candlestick(x=df['time'], open=df['o'], high=df['h'], low=df['l'], close=df['c'], name="ETH"), row=1, col=1)
        
        # 2. 动态止盈止损线 (决策指挥官核心)
        fig.add_shape(type="line", x0=df['time'].iloc[0], y0=tp_line, x1=df['time'].iloc[-1], y1=tp_line, line=dict(color="Lime", width=2, dash="dash"), row=1, col=1)
        fig.add_shape(type="line", x0=df['time'].iloc[0], y0=sl_line, x1=df['time'].iloc[-1], y1=sl_line, line=dict(color="Red", width=2), row=1, col=1)
        fig.add_annotation(x=df['time'].iloc[-1], y=tp_line, text=f"🎯 建议止盈: ${tp_line:.1f}", showarrow=False, xanchor="left", row=1, col=1)
        
        # 3. 资金流向层
        flow_colors = ['#00ff00' if x > 0 else '#ff4b4b' for x in df['net_flow']]
        fig.add_trace(go.Bar(x=df['time'], y=df['net_flow'], marker_color=flow_colors, name="资金净流"), row=2, col=1)
        
        fig.update_layout(template="plotly_dark", height=800, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

    if auto_refresh:
        time.sleep(refresh_rate)
        st.rerun()

if __name__ == "__main__":
    main()
