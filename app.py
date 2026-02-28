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
    """初始化并锁定战术状态"""
    if 'battle_logs' not in st.session_state: st.session_state.battle_logs = []
    if 'sentiment_score' not in st.session_state: st.session_state.sentiment_score = 50.0
    if 'ls_ratio' not in st.session_state: st.session_state.ls_ratio = 1.0
    if 'prev_depth' not in st.session_state: st.session_state.prev_depth = {'bids': None, 'asks': None}
    if 'last_cleanup_time' not in st.session_state: st.session_state.last_cleanup_time = time.time()

init_quantum_commander()

# ==========================================
# 2. 内存防御层：4小时自动清理引擎 (新增)
# ==========================================
def auto_memory_cleanup():
    """检测运行时间，每 4 小时自动清理冗余缓存防止浏览器崩溃"""
    current_time = time.time()
    elapsed = current_time - st.session_state.last_cleanup_time
    if elapsed > 14400:  # 4 小时阈值
        st.session_state.battle_logs = [f"【系统】{datetime.now().strftime('%H:%M:%S')} 内存治理完成：历史日志已归档。"]
        st.session_state.last_cleanup_time = current_time
        st.cache_data.clear()

# ==========================================
# 3. 核心情报引擎：多源数据采集
# ==========================================
@st.cache_data(ttl=10)
def get_commander_intel(ema_f, ema_s):
    """抓取K线、多空比并计算波动率指标"""
    try:
        # K线与 EMA
        k_url = "https://www.okx.com/api/v5/market/candles?instId=ETH-USDT&bar=1m&limit=100"
        kr = requests.get(k_url, timeout=5).json()
        df = pd.DataFrame(kr['data'], columns=['ts','o','h','l','c','v','volC','volCQ','cf'])[::-1].reset_index(drop=True)
        df['time'] = pd.to_datetime(df['ts'].astype(float), unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')
        for col in ['o','h','l','c','v']: df[col] = df[col].astype(float)
        
        df['ema_f'] = df['c'].ewm(span=ema_f, adjust=False).mean()
        df['ema_s'] = df['c'].ewm(span=ema_s, adjust=False).mean()
        df['net_flow'] = (df['c'].diff() * df['v'] * 0.15).rolling(5).sum().fillna(0)
        
        # ATR 波动率
        tr = pd.concat([df['h']-df['l'], abs(df['h']-df['c'].shift()), abs(df['l']-df['c'].shift())], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        
        # 全网多空比 (反向指标)
        ls_url = "https://www.okx.com/api/v5/rubik/stat/contracts/long-short-account-ratio?instId=ETH-USDT&period=5m"
        lsr = requests.get(ls_url, timeout=5).json()
        if lsr.get('code') == '0': st.session_state.ls_ratio = float(lsr['data'][0][1])
        
        return df
    except: return pd.DataFrame()

# ==========================================
# 4. 决策逻辑：AI胜率与 R/R 评估
# ==========================================
def analyze_combat_logic(df, score, ratio):
    """基于共振模型计算胜率与动态出场位"""
    if df.empty: return 0, 0, 0, 0
    last = df.iloc[-1]
    
    # AI 进攻胜率计算逻辑
    prob = 50.0 + (10 if last['ema_f'] > last['ema_s'] else -5) + (15 if ratio < 0.95 else -10) + (score - 50) * 0.35
    prob = max(min(prob, 99.0), 1.0)
    
    # ATR 动态出场位
    atr = last['atr']
    tp_mult = 1.5 if prob < 55 else 2.5 if prob < 75 else 3.5
    sl_mult = 1.5
    
    is_long = prob >= 50
    tp = last['c'] + (atr * tp_mult) if is_long else last['c'] - (atr * tp_mult)
    sl = last['c'] - (atr * sl_mult) if is_long else last['c'] + (atr * sl_mult)
    rr = abs(tp - last['c']) / abs(last['c'] - sl)
    
    return prob, tp, sl, rr

# ==========================================
# 5. UI 渲染：量子指挥大屏
# ==========================================
def main():
    auto_memory_cleanup() # 执行内存防御
    
    # 侧边栏控制
    st.sidebar.markdown("### 🛸 量子实时控制")
    auto_refresh = st.sidebar.toggle("开启全球监控", value=True)
    refresh_rate = st.sidebar.slider("心跳频率 (秒)", 5, 60, 10)
    ema_f_val = st.sidebar.number_input("快线 EMA", 5, 30, 12)
    ema_s_val = st.sidebar.number_input("慢线 EMA", 20, 100, 26)
    
    if st.sidebar.button("💾 手动生成战术报告"):
        report_df = pd.DataFrame([{"详情": l} for l in st.session_state.battle_logs])
        st.sidebar.download_button("下载报告", data=report_df.to_csv().encode('utf-8-sig'), file_name="Combat.csv")

    # 数据获取与逻辑运行
    df = get_commander_intel(ema_f_val, ema_s_val)
    win_p, tp_line, sl_line, rr_val = analyze_combat_logic(df, st.session_state.sentiment_score, st.session_state.ls_ratio)
    
    # 顶部面板
    st.markdown(f"### 🛰️ ETH 量子决策指挥官 (V32000) | {datetime.now().strftime('%H:%M:%S')}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("AI 进攻胜率", f"{win_p:.1f}%", f"{win_p-50:+.1f}%")
    m2.metric("建议盈亏比 (R/R)", f"1 : {rr_val:.2f}")
    m3.metric("全网多空比", f"{st.session_state.ls_ratio}", "散户看空" if st.session_state.ls_ratio < 1 else "散户看多")
    m4.metric("庄家情绪", f"{st.session_state.sentiment_score:.0f}")

    st.divider()

    # 战场核心渲染
    col_l, col_r = st.columns([1, 4])
    with col_l:
        # 实时决策信号盒
        signal_color = "#00ff00" if (win_p > 60 and rr_val > 1.8) else "#FFD700" if win_p > 50 else "#ff4b4b"
        st.markdown(f"""<div style="border:2px solid {signal_color}; padding:20px; border-radius:15px; text-align:center; background:rgba(0,0,0,0.5); box-shadow: 0 0 15px {signal_color};">
            <h2 style="color:{signal_color}; margin:0;">{'🔥 黄金机会' if win_p > 65 else '⚖️ 策略博弈'}</h2>
            <p style="color:{signal_color}; font-size:2.2em; font-weight:bold; margin:10px 0;">{win_p:.1f}%</p>
            <p style="color:#888; font-size:0.8em;">盈亏比指引: 1 : {rr_val:.2f}</p>
        </div>""", unsafe_allow_html=True)
        
        st.markdown("#### 📜 实时战术日志")
        if not st.session_state.battle_logs: st.session_state.battle_logs.append(f"【{datetime.now().strftime('%H:%M:%S')}】🛰️ 卫星同步成功：物理状态 Active Sync")
        for log in st.session_state.battle_logs[:10]: st.caption(log)

    with col_r:
        # 主战场图表渲染
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df['time'], open=df['o'], high=df['h'], low=df['l'], close=df['c'], name="ETH"), row=1, col=1)
        
        # 绘制 ATR 建议线
        fig.add_shape(type="line", x0=df['time'].iloc[0], y0=tp_line, x1=df['time'].iloc[-1], y1=tp_line, line=dict(color="Lime", width=2, dash="dash"), row=1, col=1)
        fig.add_shape(type="line", x0=df['time'].iloc[0], y0=sl_line, x1=df['time'].iloc[-1], y1=sl_line, line=dict(color="Red", width=2), row=1, col=1)
        fig.add_annotation(x=df['time'].iloc[-1], y=tp_line, text=f"🎯 建议止盈", showarrow=False, xanchor="left", row=1, col=1)
        
        # 资金流向
        flow_colors = ['#00ff00' if x > 0 else '#ff4b4b' for x in df['net_flow']]
        fig.add_trace(go.Bar(x=df['time'], y=df['net_flow'], marker_color=flow_colors, name="资金净流"), row=2, col=1)
        
        fig.update_layout(template="plotly_dark", height=800, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

    if auto_refresh:
        time.sleep(refresh_rate)
        st.rerun()

if __name__ == "__main__":
    main()
