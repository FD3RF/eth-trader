import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime

# ==========================================
# 1. 物理层：环境硬锁与 Session 持久化
# ==========================================
st.set_page_config(layout="wide", page_title="ETH V32000 决策指挥官", page_icon="⚖️")

def hard_lock_environment():
    initial_states = {
        'df': pd.DataFrame(),
        'battle_logs': [],
        'prev_bids': pd.DataFrame(),
        'prev_asks': pd.DataFrame(),
        'sentiment_score': 50.0,
        'ls_ratio': 1.0
    }
    for key, value in initial_states.items():
        if key not in st.session_state:
            st.session_state[key] = value

hard_lock_environment()

# ==========================================
# 2. 侧边栏：量子控制与复盘中心
# ==========================================
st.sidebar.markdown("### 🛸 量子实时控制")
auto_refresh = st.sidebar.toggle("开启全球监控", value=True)
refresh_rate = st.sidebar.slider("心跳频率 (秒)", 5, 60, 10)

st.sidebar.divider()
ema_f_val = st.sidebar.number_input("快线 EMA", 5, 30, 12)
ema_s_val = st.sidebar.number_input("慢线 EMA", 20, 100, 26)

if st.sidebar.button("💾 生成今日战术报告"):
    if st.session_state.battle_logs:
        report_df = pd.DataFrame([{"时间": l.split("】")[0][1:], "详情": l.split("】")[1]} for l in st.session_state.battle_logs])
        st.sidebar.download_button("📥 下载 CSV 报告", data=report_df.to_csv().encode('utf-8-sig'), file_name="ETH_Combat_Report.csv")

# ==========================================
# 3. 数据与战术计算引擎
# ==========================================
@st.cache_data(ttl=refresh_rate)
def get_market_intel():
    """获取 K线与多空比"""
    try:
        # K线数据
        k_url = "https://www.okx.com/api/v5/market/candles?instId=ETH-USDT&bar=1m&limit=100"
        kr = requests.get(k_url, timeout=5).json()
        df = pd.DataFrame(kr['data'], columns=['ts','o','h','l','c','v','volC','volCQ','cf'])[::-1]
        df = df.reset_index(drop=True)
        df['time'] = pd.to_datetime(df['ts'].astype(float), unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')
        for col in ['o','h','l','c','v']: df[col] = df[col].astype(float)
        
        # EMA 与 净流
        df['ema_f'] = df['c'].ewm(span=ema_f_val, adjust=False).mean()
        df['ema_s'] = df['c'].ewm(span=ema_s_val, adjust=False).mean()
        df['net_flow'] = (df['c'].diff() * df['v'] * 0.15).rolling(5).sum().fillna(0)
        
        # ATR 计算 (14周期)
        tr = pd.concat([df['h']-df['l'], abs(df['h']-df['c'].shift()), abs(df['l']-df['c'].shift())], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        
        # 多空持仓比
        ls_url = "https://www.okx.com/api/v5/rubik/stat/contracts/long-short-account-ratio?instId=ETH-USDT&period=5m"
        lsr = requests.get(ls_url, timeout=5).json()
        st.session_state.ls_ratio = float(lsr['data'][0][1]) if lsr.get('code') == '0' else 1.0
        
        return df
    except: return st.session_state.df

@st.cache_data(ttl=2)
def get_combat_intel():
    """获取盘口与爆仓"""
    try:
        d_url = "https://www.okx.com/api/v5/market/books?instId=ETH-USDT&sz=20"
        l_url = "https://www.okx.com/api/v5/public/liquidation-orders?instId=ETH-USDT-SWAP&mgnMode=cross"
        dr, lr = requests.get(d_url).json(), requests.get(l_url).json()
        bids = pd.DataFrame(dr['data'][0]['bids'], columns=['px','sz','l','c']).astype(float)
        asks = pd.DataFrame(dr['data'][0]['asks'], columns=['px','sz','l','c']).astype(float)
        return bids, asks, lr['data'][0]['details'] if lr.get('code') == '0' else []
    except: return None, None, []

# ==========================================
# 4. 决策逻辑：AI 胜率 + R/R 评估
# ==========================================
def calculate_commander_logic(df, score, ratio):
    last = df.iloc[-1]
    # AI 胜率
    win_prob = 50.0 + (10 if last['ema_f'] > last['ema_s'] else -5) + (15 if ratio < 0.9 else -10) + (score-50)*0.3
    win_prob = max(min(win_prob, 98.0), 2.0)
    
    # 盈亏比 (ATR 驱动)
    atr = last['atr']
    tp_mult = 1.5 if win_prob < 55 else 2.5 if win_prob < 70 else 3.5
    tp_dist, sl_dist = atr * tp_mult, atr * 1.5
    rr_ratio = tp_dist / sl_dist
    
    return win_prob, last['c']+tp_dist, last['c']-sl_dist, last['c']-tp_dist, last['c']+sl_dist, rr_ratio

def process_events(bids, asks, liqs):
    now = datetime.now().strftime('%H:%M:%S')
    # 爆仓侦测
    for l in liqs[:2]:
        if float(l['sz']) > 40:
            side = "多头" if l['posSide'] == 'long' else "空头"
            msg = f"💀 [爆仓] {side}清算! ${l['bkPx']}"
            if msg not in st.session_state.battle_logs[:5]:
                st.session_state.battle_logs.insert(0, f"【{now}】{msg}")
                st.session_state.sentiment_score += (15 if side == "空头" else -15)
    # 撤单侦测
    if bids is not None and not st.session_state.prev_bids.empty:
        th = (bids['sz'].mean() + asks['sz'].mean()) * 2.5
        for _, p in st.session_state.prev_bids.iterrows():
            if p['sz'] > th and bids[bids['px'] == p['px']].empty:
                st.session_state.battle_logs.insert(0, f"【{now}】⚠️ 诱多撤单: ${p['px']}")
                st.session_state.sentiment_score -= 5
    st.session_state.prev_bids, st.session_state.prev_asks = bids, asks
    st.session_state.sentiment_score = max(min(st.session_state.sentiment_score, 100), 0)

# ==========================================
# 5. UI 渲染：指挥部大屏
# ==========================================
def main():
    df = get_market_intel()
    bids, asks, liqs = get_combat_intel()
    process_events(bids, asks, liqs)
    
    win_p, tp_l, sl_l, tp_s, sl_s, rr = calculate_commander_logic(df, st.session_state.sentiment_score, st.session_state.ls_ratio)
    
    # 看板
    st.markdown(f"### 🛰️ ETH 量子决策指挥官 (V32000) | {datetime.now().strftime('%H:%M:%S')}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AI 进攻胜率", f"{win_p:.1f}%")
    c2.metric("实时盈亏比", f"1 : {rr:.2f}")
    c3.metric("全网多空比", f"{st.session_state.ls_ratio}")
    c4.metric("庄家情绪", f"{st.session_state.sentiment_score:.0f}")

    st.divider()

    col_l, col_r = st.columns([1, 4])
    with col_l:
        # 决策建议盒
        is_gold = win_p > 65 and rr >= 2.0
        box_c = "#00ff00" if is_gold else "#FFD700" if win_p > 50 else "#ff4b4b"
        st.markdown(f"""<div style="border:2px solid {box_c}; padding:20px; border-radius:15px; text-align:center; background:rgba(0,0,0,0.5);">
            <h2 style="color:{box_c}; margin:0;">{'🔥 黄金机会' if is_gold else '⚖️ 观察博弈' if win_p>50 else '❌ 放弃猎杀'}</h2>
            <p style="color:{box_c}; font-size:2em; font-weight:bold; margin:10px 0;">{win_p:.1f}%</p>
            <p style="color:#888; font-size:0.8em;">建议盈亏比 > 1.5</p>
        </div>""", unsafe_allow_html=True)
        
        st.markdown("#### 📜 战术日志")
        for log in st.session_state.battle_logs[:10]:
            if "💀" in log: st.warning(log)
            elif "⚠️" in log: st.error(log)
            else: st.caption(log)

    with col_r:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
        # K线与建议线
        fig.add_trace(go.Candlestick(x=df['time'], open=df['o'], high=df['h'], low=df['l'], close=df['c'], name="ETH"), row=1, col=1)
        target_tp = tp_l if win_p >= 50 else tp_s
        target_sl = sl_l if win_p >= 50 else sl_s
        fig.add_shape(type="line", x0=df['time'].iloc[0], y0=target_tp, x1=df['time'].iloc[-1], y1=target_tp, line=dict(color="Lime", width=2, dash="dash"), row=1, col=1)
        fig.add_shape(type="line", x0=df['time'].iloc[0], y0=target_sl, x1=df['time'].iloc[-1], y1=target_sl, line=dict(color="Red", width=2), row=1, col=1)
        
        # 净流
        flow_c = ['#00ff00' if x > 0 else '#ff4b4b' for x in df['net_flow']]
        fig.add_trace(go.Bar(x=df['time'], y=df['net_flow'], marker_color=flow_c, name="资金净流"), row=2, col=1)
        
        fig.update_layout(template="plotly_dark", height=800, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

    if auto_refresh:
        time.sleep(refresh_rate); st.rerun()

if __name__ == "__main__": main()
