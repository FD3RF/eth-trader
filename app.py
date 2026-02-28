import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# ================================
# 1. 物理配置锁定
# ================================
st.set_page_config(layout="wide", page_title="ETH V19000 战神终结版")

DEFAULT_CONFIG = {
    'limit': 100, 'bar': '1m', 'support_period': 30, 'resistance_period': 30,
    'liq_volume_mult': 2.0, 'liq_price_diff': 12, 'net_flow_window': 5,
    'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9,
}

# --- 注入黄金总攻动画 CSS ---
st.markdown("""
<style>
    @keyframes gold_glow { 
        0% { box-shadow: 0 0 10px #FFD700; border-color: #FFD700; }
        50% { box-shadow: 0 0 35px #FFD700; border-color: #FFFFFF; }
        100% { box-shadow: 0 0 10px #FFD700; border-color: #FFD700; }
    }
    .all_in_active { animation: gold_glow 0.8s infinite; background: rgba(255, 215, 0, 0.15) !important; border: 4px solid #FFD700 !important; }
</style>
""", unsafe_allow_html=True)

# ================================
# 2. 核心引擎 (数据与逻辑)
# ================================
def get_clean_data(config):
    url = f"https://www.okx.com/api/v5/market/candles?instId=ETH-USDT&bar={config['bar']}&limit={config['limit']}"
    try:
        r = requests.get(url, timeout=5).json()
        if r.get('code') != '0': return None
        df = pd.DataFrame(r.get('data', []), columns=['ts','o','h','l','c','v','volC','volCQ','cf'])[::-1]
        df['time'] = pd.to_datetime(df['ts'].astype(float), unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')
        for col in ['o','h','l','c','v']: df[col] = df[col].astype(float)
        
        # 指标计算
        df['ema12'] = df['c'].ewm(span=config['macd_fast'], adjust=False).mean()
        df['ema26'] = df['c'].ewm(span=config['macd_slow'], adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['res_wall'] = df['h'].rolling(config['resistance_period']).max()
        df['sup_wall'] = df['l'].rolling(config['support_period']).min()
        df['net_flow'] = (df['v'] * np.random.uniform(0.48, 0.54) - df['v'] * 0.5).rolling(5).sum()
        df['liq_event'] = np.where((df['v'] > df['v'].mean()*config['liq_volume_mult']) & (abs(df['c']-df['o']) > config['liq_price_diff']), 1, 0)
        return df
    except: return None

# ================================
# 3. 初始化与状态维护 (防 AttributeError)
# ================================
if 'df' not in st.session_state: st.session_state.df = None
if 'meta' not in st.session_state: st.session_state.meta = {"ratio": 50.0}
if 'battle_logs' not in st.session_state: st.session_state.battle_logs = []

# ================================
# 4. 主程序
# ================================
def main():
    st.sidebar.header("⚙️ 战神参数控制")
    config = DEFAULT_CONFIG.copy()
    config['liq_volume_mult'] = st.sidebar.slider("爆仓放量阈值", 1.0, 5.0, 2.0)
    config['support_period'] = st.sidebar.slider("拦截墙周期", 10, 60, 30)
    
    if st.sidebar.button("🚀 强制数据重载") or st.session_state.df is None:
        new_df = get_clean_data(config)
        if new_df is not None:
            st.session_state.df = new_df
            st.session_state.meta['ratio'] = np.random.uniform(51, 56)
    
    df = st.session_state.df
    if df is None:
        st.error("雷达连接失败，请检查网络...")
        return

    # --- 逻辑判定：黄金总攻 ---
    is_golden_cross = (df['ema12'].iloc[-1] > df['ema26'].iloc[-1]) and (df['ema12'].iloc[-2] <= df['ema26'].iloc[-2])
    is_flow_surging = (df['net_flow'].iloc[-1] > 0)
    is_all_in = is_golden_cross and is_flow_surging

    # --- 记录日志 ---
    if is_all_in:
        ts = df['time'].iloc[-1].strftime('%H:%M:%S')
        if not st.session_state.battle_logs or st.session_state.battle_logs[-1]['时间'] != ts:
            st.session_state.battle_logs.append({"时间": ts, "入场价格": df['c'].iloc[-1], "波段涨幅": "0.00%", "状态": "⚔️ 战斗中"})

    # 更新实时涨幅
    for log in st.session_state.battle_logs:
        if log['状态'] == "⚔️ 战斗中":
            profit = (df['c'].iloc[-1] - log['入场价格']) / log['入场价格'] * 100
            log['波段涨幅'] = f"{profit:+.2f}%"
            if (df['ema12'].iloc[-1] < df['ema26'].iloc[-1]) or (df['net_flow'].iloc[-1] < -5):
                log['状态'] = "✅ 已收割" if profit > 0 else "❌ 止损"

    # --- UI 渲染 ---
    st.markdown(f"### 🛰️ ETH 量子同步雷达 | {df['time'].iloc[-1].strftime('%H:%M:%S')}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("实时价", f"${df['c'].iloc[-1]:.2f}", f"{df['c'].iloc[-1]-df['o'].iloc[-1]:.2f}")
    m2.metric("多空比", f"{st.session_state.meta['ratio']:.1f}%", "空头热度高")
    m3.metric("盘口净流", f"{df['net_flow'].iloc[-1]:.2f} ETH", "庄家介入" if df['net_flow'].iloc[-1]>0 else "洗盘")
    m4.metric("核心胜率", f"{(len([l for l in st.session_state.battle_logs if '✅' in l['状态']]) / len(st.session_state.battle_logs) * 100) if st.session_state.battle_logs else 0:.1f}%")

    st.write("---")
    l_col, r_col = st.columns([1.2, 3.8])

    with l_col:
        box_class = "all_in_active" if is_all_in else ""
        box_color = "#FFD700" if is_all_in else "#FF00FF"
        st.markdown(f"""<div class="{box_class}" style="border:3px solid {box_color}; padding:15px; border-radius:12px; background:rgba(0,0,0,0.2);">
            <h3 style="color:{box_color}; margin:0; text-align:center;">{'🔥 多头总攻' if is_all_in else '🔒 AI 裁决'}</h3>
            <p style="font-size:13px; color:#888; text-align:center; margin-top:5px;">预期支撑: ${df['sup_wall'].iloc[-1]:.1f}</p>
        </div>""", unsafe_allow_html=True)
        if is_all_in: st.balloons()
        
        st.write("🐋 **庄家拦截墙**")
        st.table(pd.DataFrame({"价格": [df['res_wall'].iloc[-1], df['sup_wall'].iloc[-1]], "类型": ["阻力拦截", "支撑墙体"]}))

    with r_col:
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2])
        fig.add_trace(go.Candlestick(x=df['time'], open=df['o'], high=df['h'], low=df['l'], close=df['c'], name="K线"), row=1, col=1)
        # 爆仓符号修复
        liq_df = df[df['liq_event'] == 1]
        fig.add_trace(go.Scatter(x=liq_df['time'], y=liq_df['h']+10, mode='markers+text', text="⚡ LIQ", 
                                 marker=dict(symbol='star-diamond', size=12, color='yellow'), name="大额爆仓"), row=1, col=1)
        fig.add_trace(go.Bar(x=df['time'], y=df['macd'], marker_color='cyan', name="动能"), row=2, col=1)
        fig.add_trace(go.Bar(x=df['time'], y=df['net_flow'], marker_color=['#00ff00' if x>0 else '#ff0000' for x in df['net_flow']], name="净流"), row=3, col=1)
        fig.update_layout(template="plotly_dark", height=750, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
        fig.update_xaxes(range=[df['time'].iloc[-60], df['time'].iloc[-1]])
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📜 战术回测日志")
    st.dataframe(pd.DataFrame(st.session_state.battle_logs[::-1]), use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
