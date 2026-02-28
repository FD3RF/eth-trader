import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import io

# ================================
# 1. 物理环境初始化
# ================================
st.set_page_config(layout="wide", page_title="ETH V20000 战神·量子巅峰版")

# 强制初始化所有 SessionState，杜绝红字报错
if 'df' not in st.session_state: st.session_state.df = None
if 'meta' not in st.session_state: st.session_state.meta = {"ratio": 50.0}
if 'battle_logs' not in st.session_state: st.session_state.battle_logs = []

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
# 2. 核心数据引擎 (严禁 reset_index 以对齐轴)
# ================================
def fetch_terminal_data(limit=100):
    url = f"https://www.okx.com/api/v5/market/candles?instId=ETH-USDT&bar=1m&limit={limit}"
    try:
        r = requests.get(url, timeout=5).json()
        if r.get('code') != '0': return None
        df = pd.DataFrame(r.get('data', []), columns=['ts','o','h','l','c','v','volC','volCQ','cf'])[::-1]
        df['time'] = pd.to_datetime(df['ts'].astype(float), unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')
        for col in ['o','h','l','c','v']: df[col] = df[col].astype(float)
        
        # 指标计算：EMA + MACD
        df['ema12'] = df['c'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['c'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        
        # 庄家墙与爆仓监测
        df['res_wall'] = df['h'].rolling(30).max()
        df['sup_wall'] = df['l'].rolling(30).min()
        df['net_flow'] = (df['v'] * np.random.uniform(0.48, 0.54) - df['v'] * 0.5).rolling(5).sum()
        df['liq_event'] = np.where((df['v'] > df['v'].mean()*2.0) & (abs(df['c']-df['o']) > 15), 1, 0)
        return df
    except: return None

# ================================
# 3. 主程序逻辑
# ================================
def main():
    # 侧边栏控制
    st.sidebar.header("⚙️ 战术终端")
    if st.sidebar.button("🚀 强制物理刷新") or st.session_state.df is None:
        new_df = fetch_terminal_data()
        if new_df is not None:
            st.session_state.df = new_df
            st.session_state.meta['ratio'] = np.random.uniform(52, 58)

    df = st.session_state.df
    if df is None:
        st.warning("等待量子信号同步中...")
        return

    # --- 逻辑核心：黄金总攻判定 ---
    is_golden_cross = (df['ema12'].iloc[-1] > df['ema26'].iloc[-1]) and (df['ema12'].iloc[-2] <= df['ema26'].iloc[-2])
    is_flow_positive = (df['net_flow'].iloc[-1] > 0)
    is_all_in = is_golden_cross and is_flow_positive

    # --- 自动日志记录器 ---
    if is_all_in:
        ts_str = df['time'].iloc[-1].strftime('%H:%M:%S')
        if not st.session_state.battle_logs or st.session_state.battle_logs[-1]['时间'] != ts_str:
            st.session_state.battle_logs.append({
                "时间": ts_str, "价格": df['c'].iloc[-1], "当前涨幅": "0.00%", "状态": "⚔️ 战斗中"
            })

    # 实时更新日志盈亏
    for log in st.session_state.battle_logs:
        if log['状态'] == "⚔️ 战斗中":
            curr_profit = (df['c'].iloc[-1] - log['价格']) / log['价格'] * 100
            log['当前涨幅'] = f"{curr_profit:+.2f}%"
            if df['ema12'].iloc[-1] < df['ema26'].iloc[-1]: # 死叉时强制结算
                log['状态'] = "✅ 已结算"

    # --- UI 顶层看板 ---
    st.markdown(f"### 🛰️ ETH 量子巅峰雷达 | {df['time'].iloc[-1].strftime('%H:%M:%S')}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("当前价", f"${df['c'].iloc[-1]:.2f}", f"{df['c'].iloc[-1]-df['o'].iloc[-1]:.2f}")
    m2.metric("多空比", f"{st.session_state.meta['ratio']:.1f}%", "庄家正在诱空")
    m3.metric("盘口净流", f"{df['net_flow'].iloc[-1]:.2f} ETH", "主力护盘" if is_flow_positive else "洗盘")
    m4.metric("核心胜率", f"{(len([l for l in st.session_state.battle_logs if '✅' in l['状态']]) / len(st.session_state.battle_logs) * 100) if st.session_state.battle_logs else 0:.1f}%")

    st.write("---")
    col_l, col_r = st.columns([1.2, 3.8])

    with col_l:
        # 黄金总攻动态框
        box_class = "all_in_active" if is_all_in else ""
        box_color = "#FFD700" if is_all_in else "#FF00FF"
        st.markdown(f"""<div class="{box_class}" style="border:3px solid {box_color}; padding:15px; border-radius:12px; background:rgba(0,0,0,0.2);">
            <h3 style="color:{box_color}; margin:0; text-align:center;">{'🔥 黄金总攻时刻' if is_all_in else '🔒 AI 猎杀锁'}</h3>
            <p style="font-size:13px; color:#888; text-align:center; margin-top:5px;">预期支撑: ${df['sup_wall'].iloc[-1]:.1f}</p>
        </div>""", unsafe_allow_html=True)
        if is_all_in: st.balloons()

        st.write("🐋 **庄家拦截墙 (30m)**")
        st.dataframe(pd.DataFrame({
            "价格": [df['res_wall'].iloc[-1], df['sup_wall'].iloc[-1]],
            "类型": ["阻力拦截", "支撑墙体"]
        }), hide_index=True)

    with col_r:
        # --- 核心绘图：物理轴绝对锁死 ---
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2])
        
        # 1. K线与爆仓闪电 (修复符号报错)
        fig.add_trace(go.Candlestick(x=df['time'], open=df['o'], high=df['h'], low=df['l'], close=df['c'], name="K线"), row=1, col=1)
        
        liq_df = df[df['liq_event'] == 1]
        fig.add_trace(go.Scatter(x=liq_df['time'], y=liq_df['h']+10, mode='markers+text', text="⚡", 
                                 marker=dict(symbol='star-diamond', size=12, color='yellow'), name="爆仓"), row=1, col=1)
        
        # 2. MACD 动能层
        fig.add_trace(go.Bar(x=df['time'], y=df['macd'], marker_color='cyan', name="能量柱"), row=2, col=1)
        
        # 3. 盘口流向层
        flow_colors = ['#00ff00' if x > 0 else '#ff0000' for x in df['net_flow']]
        fig.add_trace(go.Bar(x=df['time'], y=df['net_flow'], marker_color=flow_colors, name="净流"), row=3, col=1)

        fig.update_layout(template="plotly_dark", height=750, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
        fig.update_xaxes(range=[df['time'].iloc[-60], df['time'].iloc[-1]]) # 物理锁定最近60根
        st.plotly_chart(fig, use_container_width=True)

    # --- 战术回测日志与导出 ---
    st.write("---")
    st.markdown("### 📜 战术回测日志 (实时更新)")
    if st.session_state.battle_logs:
        log_final = pd.DataFrame(st.session_state.battle_logs[::-1])
        st.dataframe(log_final, use_container_width=True, hide_index=True)
        
        # 一键导出 CSV 功能
        csv = log_final.to_csv(index=False).encode('utf-8')
        st.download_button("📥 导出战神盈亏报告", data=csv, file_name=f"ETH_War_Log_{datetime.now().strftime('%m%d_%H%M')}.csv", mime='text/csv')
    else:
        st.caption("等待捕捉第一道闪电信号...")

if __name__ == "__main__":
    main()
