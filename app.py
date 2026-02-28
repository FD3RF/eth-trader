import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# ==========================================
# 1. 底层防护：物理状态强行焊死
# ==========================================
st.set_page_config(layout="wide", page_title="ETH V60000 战神主宰", page_icon="⚡")

def secure_init():
    """使用原生 setdefault，确保 session_state 绝对不会出现 KeyMissing"""
    s = st.session_state
    s.setdefault('df', pd.DataFrame())
    s.setdefault('peak_equity', 10000.0)
    s.setdefault('equity_history', [{"time": "00:00", "equity": 10000.0, "eth_pnl": 0.0}])
    s.setdefault('battle_logs', [])

secure_init()

# ==========================================
# 2. 数据引擎：协议级内存重构
# ==========================================
def fetch_bulletproof():
    url = "https://www.okx.com/api/v5/market/candles?instId=ETH-USDT&bar=1m&limit=100"
    try:
        r = requests.get(url, timeout=3).json()
        data = r.get('data', [])
        if not data: return st.session_state.df
        
        # 基础转换
        df = pd.DataFrame(data, columns=['ts','o','h','l','c','v','volC','volCQ','cf'])[::-1]
        df['time'] = pd.to_datetime(df['ts'].astype(float), unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')
        for col in ['o','h','l','c','v']: df[col] = df[col].astype(float)
        
        # --- 暴力对冲：直接覆盖所有可能缺失的指标列 ---
        df = df.assign(
            ema12=df['c'].ewm(span=12, adjust=False).mean(),
            ema26=df['c'].ewm(span=26, adjust=False).mean(),
            liq=0.0,
            net_flow=0.0
        )
        df['macd'] = df['ema12'] - df['ema26']
        # 随机流向模拟 (作为保底，确保 Flow 图表不为空)
        df['net_flow'] = (df['v'] * np.random.uniform(-0.05, 0.05)).rolling(5).sum().fillna(0)
        
        # 爆仓判定 (直接在 DataFrame 内完成)
        v_threshold = df['v'].mean() * 3
        df.loc[df['v'] > v_threshold, 'liq'] = 1
        
        return df
    except Exception:
        return st.session_state.df

# ==========================================
# 3. 战神终端核心
# ==========================================
def main():
    secure_init()
    
    # 强制刷新/加载逻辑
    if st.sidebar.button("💎 物理重铸系统 (RESET)") or st.session_state.df.empty:
        with st.spinner("量子对冲中..."):
            st.session_state.df = fetch_bulletproof()

    df = st.session_state.df
    
    # --- 最终防线：物理阻断 ---
    if df.empty or 'liq' not in df.columns:
        st.warning("📡 数据链路握手中，请点击左侧‘物理重铸’...")
        return

    # 提取核心数据点 (使用原生 python 变量，避免频繁触碰 df)
    last_row = df.iloc[-1]
    curr_price = last_row['c']
    curr_equity = st.session_state.equity_history[-1]['equity']
    
    if curr_equity > st.session_state.peak_equity:
        st.session_state.peak_equity = curr_equity
    
    # --- UI 渲染 (原生精简版) ---
    st.markdown(f"### 🛰️ ETH 战神·量子终结终端 V60000")
    
    m = st.columns(4)
    m[0].metric("金库净值", f"${curr_equity:.2f}")
    m[1].metric("历史巅峰", f"${st.session_state.peak_equity:.2f}")
    m[2].metric("实时币价", f"${curr_price:.2f}")
    m[3].metric("动能状态", "🌊 趋势偏多" if last_row['macd'] > 0 else "📉 趋势偏空")

    st.divider()

    # --- 绘图引擎 (原生保护模式) ---
    try:
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.05, row_heights=[0.5, 0.2, 0.3])
        
        # 1. K线图
        fig.add_trace(go.Candlestick(x=df['time'], open=df['o'], high=df['h'], 
                                   low=df['l'], close=df['c'], name="行情"), row=1, col=1)
        
        # 2. 爆仓散点 (原生列表推导，绝对不报 KeyError)
        liq_points = df[df['liq'] == 1]
        if not liq_points.empty:
            fig.add_trace(go.Scatter(x=liq_points['time'], y=liq_points['h'] + 5, 
                                   mode='markers', marker=dict(color='yellow', size=12, symbol='star'),
                                   name="爆仓预警"), row=1, col=1)
        
        # 3. MACD 能量柱
        fig.add_trace(go.Bar(x=df['time'], y=df['macd'], name="MACD", marker_color='cyan'), row=2, col=1)
        
        # 4. 净值曲线
        eq_df = pd.DataFrame(st.session_state.equity_history)
        fig.add_trace(go.Scatter(x=eq_df['time'], y=eq_df['equity'], 
                               line=dict(color='#00ff00', width=4), name="收益曲线"), row=3, col=1)

        fig.update_layout(template="plotly_dark", height=800, xaxis_rangeslider_visible=False,
                         margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"🎨 绘图链路波动: {e}")

    # 实时日志
    with st.expander("📜 查看底层数据流 (无死角监控)"):
        st.write(df.tail(5))

if __name__ == "__main__":
    main()
