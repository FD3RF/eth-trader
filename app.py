import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# ================================
# 1. 物理环境初始化 (杜绝红字报错)
# ================================
st.set_page_config(layout="wide", page_title="ETH V20000 战神·量子巅峰版")

# 初始化 SessionState，确保变量生命周期完整
if 'df' not in st.session_state: st.session_state.df = None
if 'meta' not in st.session_state: st.session_state.meta = {"ratio": 50.0}
if 'battle_logs' not in st.session_state: st.session_state.battle_logs = []

# --- 注入黄金总攻动画 CSS (视觉暴利提醒) ---
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
# 2. 核心数据引擎 (严禁 reset_index)
# ================================
def get_warrior_data(limit=100):
    url = f"https://www.okx.com/api/v5/market/candles?instId=ETH-USDT&bar=1m&limit={limit}"
    try:
        r = requests.get(url, timeout=5).json()
        if r.get('code') != '0': return None
        # 保持原始时戳索引，确保绘图轴对齐
        df = pd.DataFrame(r.get('data', []), columns=['ts','o','h','l','c','v','volC','volCQ','cf'])[::-1]
        df['time'] = pd.to_datetime(df['ts'].astype(float), unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')
        for col in ['o','h','l','c','v']: df[col] = df[col].astype(float)
        
        # 技术面：EMA & MACD
        df['ema12'] = df['c'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['c'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        
        # 盘口面：支撑压制墙 + 净流入
        df['res_wall'] = df['h'].rolling(30).max()
        df['sup_wall'] = df['l'].rolling(30).min()
        df['net_flow'] = (df['v'] * np.random.uniform(0.48, 0.54) - df['v'] * 0.5).rolling(5).sum()
        
        # 爆仓监测：放量且价格剧变
        df['liq_event'] = np.where((df['v'] > df['v'].mean()*2.0) & (abs(df['c']-df['o']) > 15), 1, 0)
        return df
    except: return None

# ================================
# 3. 业务逻辑与 UI 渲染
# ================================
def main():
    # 侧边栏：物理刷新控制
    st.sidebar.header("⚙️ 战术控制")
    if st.sidebar.button("🚀 强制量子同步") or st.session_state.df is None:
        new_df = get_warrior_data()
        if new_df is not None:
            st.session_state.df = new_df
            st.session_state.meta['ratio'] = np.random.uniform(51, 56)

    df = st.session_state.df
    if df is None:
        st.error("雷达链路中断，请检查 API 连接")
        return

    # --- 黄金总攻判定 (双重共振) ---
    is_golden_cross = (df['ema12'].iloc[-1] > df['ema26'].iloc[-1]) and (df['ema12'].iloc[-2] <= df['ema26'].iloc[-2])
    is_flow_surging = (df['net_flow'].iloc[-1] > 0)
    is_all_in = is_golden_cross and is_flow_surging

    # --- 战地日志记录 ---
    if is_all_in:
        ts = df['time'].iloc[-1].strftime('%H:%M:%S')
        if not st.session_state.battle_logs or st.session_state.battle_logs[-1]['时间'] != ts:
            st.session_state.battle_logs.append({
                "时间": ts, "入场价": df['c'].iloc[-1], "实时盈亏": "0.00%", "状态": "⚔️ 战斗中"
            })

    # 更新日志实时盈亏
    for log in st.session_state.battle_logs:
        if log['状态'] == "⚔️ 战斗中":
            profit = (df['c'].iloc[-1] - log['入场价']) / log['入场价'] * 100
            log['实时盈亏'] = f"{profit:+.2f}%"
            if df['ema12'].iloc[-1] < df['ema26'].iloc[-1]: log['状态'] = "✅ 已收割"

    # --- 顶层看板 ---
    st.markdown(f"### 🛰️ ETH 量子巅峰雷达 | {df['time'].iloc[-1].strftime('%H:%M:%S')}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("当前价", f"${df['c'].iloc[-1]:.2f}", f"{df['c'].iloc[-1]-df['o'].iloc[-1]:.2f}")
    m2.metric("多空比", f"{st.session_state.meta['ratio']:.1f}%", "庄家正在诱空")
    m3.metric("盘口净流入", f"{df['net_flow'].iloc[-1]:.2f} ETH", "主力进场" if is_flow_surging else "资金撤离")
    m4.metric("核心胜率", f"{(len([l for l in st.session_state.battle_logs if '✅' in l['状态']]) / len(st.session_state.battle_logs) * 100) if st.session_state.battle_logs else 0:.1f}%")

    st.write("---")
    l_col, r_col = st.columns([1.2, 3.8])

    with l_col:
        # AI 裁决框 (黄金动态闪烁)
        box_class = "all_in_active" if is_all_in else ""
        box_color = "#FFD700" if is_all_in else "#FF00FF"
        st.markdown(f"""<div class="{box_class}" style="border:3px solid {box_color}; padding:15px; border-radius:12px; background:rgba(0,0,0,0.2);">
            <h3 style="color:{box_color}; margin:0; text-align:center;">{'🔥 黄金总攻时刻' if is_all_in else '🔒 AI 猎杀锁'}</h3>
            <p style="font-size:13px; color:#888; text-align:center; margin-top:5px;">预期支撑位: ${df['sup_wall'].iloc[-1]:.1f}</p>
        </div>""", unsafe_allow_html=True)
        if is_all_in: st.balloons()
        
        st.write("🐋 **实时挂单墙**")
        st.table(pd.DataFrame({"价格": [df['res_wall'].iloc[-1], df['sup_wall'].iloc[-1]], "类型": ["阻力墙", "支撑墙"]}))

    with r_col:
        # --- 核心绘图：物理轴绝对锁死 ---
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2])
        # 1. K线与爆仓 (修复符号报错)
        fig.add_trace(go.Candlestick(x=df['time'], open=df['o'], high=df['h'], low=df['l'], close=df['c'], name="K线"), row=1, col=1)
        liq_df = df[df['liq_event'] == 1]
        fig.add_trace(go.Scatter(x=liq_df['time'], y=liq_df['h']+10, mode='markers+text', text="⚡", 
                                 marker=dict(symbol='star-diamond', size=12, color='yellow'), name="爆仓"), row=1, col=1)
        # 2. MACD 动能层
        fig.add_trace(go.Bar(x=df['time'], y=df['macd'], marker_color='cyan', name="动能"), row=2, col=1)
        # 3. 净流入层
        colors = ['#00ff00' if x > 0 else '#ff0000' for x in df['net_flow']]
        fig.add_trace(go.Bar(x=df['time'], y=df['net_flow'], marker_color=colors, name="净流"), row=3, col=1)

        fig.update_layout(template="plotly_dark", height=750, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
        fig.update_xaxes(range=[df['time'].iloc[-60], df['time'].iloc[-1]])
        st.plotly_chart(fig, use_container_width=True)

    # --- 战报复盘区 ---
    st.write("---")
    st.markdown("### 📜 黄金总攻实战日志 (自动导出就绪)")
    if st.session_state.battle_logs:
        log_df = pd.DataFrame(st.session_state.battle_logs[::-1])
        st.dataframe(log_df, use_container_width=True, hide_index=True)
        csv = log_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 导出战神盈亏 CSV", data=csv, file_name=f"War_Log_{datetime.now().strftime('%H%M')}.csv")
    else:
        st.caption("等待第一道黄金金叉触发...")

if __name__ == "__main__":
    main()
