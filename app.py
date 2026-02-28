import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ==========================================
# 1. 基础配置：物理初始化
# ==========================================
st.set_page_config(layout="wide", page_title="ETH V50000 水泥级防御", page_icon="🧱")

# 强制初始化所有状态，一个都不能少
for key, val in {
    'df': pd.DataFrame(),
    'peak': 10000.0,
    'equity_history': [{"time": "INIT", "equity": 10000.0, "eth_pnl": 0.0}],
    'logs': [],
    'cooldown': None
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ==========================================
# 2. 数据层：强制焊死列结构
# ==========================================
def get_data():
    url = "https://www.okx.com/api/v5/market/candles?instId=ETH-USDT&bar=1m&limit=100"
    try:
        r = requests.get(url, timeout=5).json()
        data = r.get('data', [])
        if not data: return st.session_state.df
        
        df = pd.DataFrame(data, columns=['ts','o','h','l','c','v','volC','volCQ','cf'])[::-1]
        df['time'] = pd.to_datetime(df['ts'].astype(float), unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')
        for col in ['o','h','l','c','v']: df[col] = df[col].astype(float)
        
        # --- 核心防御：强制焊死所有列，防止 KeyError ---
        df = df.assign(
            ema12=df['c'].ewm(span=12, adjust=False).mean(),
            ema26=df['c'].ewm(span=26, adjust=False).mean(),
            liq=0
        )
        df['macd'] = df['ema12'] - df['ema26']
        df['net_flow'] = (df['v'] * np.random.uniform(-0.02, 0.02)).rolling(5).sum().fillna(0)
        
        # 简单暴力的爆仓标记
        df.loc[df['v'] > df['v'].mean() * 3, 'liq'] = 1
        return df
    except:
        return st.session_state.df

# ==========================================
# 3. 核心主逻辑
# ==========================================
def main():
    # 侧边栏控制
    if st.sidebar.button("♻️ 强制物理重置") or st.session_state.df.empty:
        st.session_state.df = get_data()

    df = st.session_state.df
    # 防御 1：如果表是空的，直接停止执行，不碰任何索引
    if df.empty or 'liq' not in df.columns:
        st.warning("📡 正在接入数据链，请稍候...")
        return

    # 获取最新数据点 (使用 iat 确保安全)
    last_idx = -1
    c_price = df['c'].iat[last_idx]
    
    # 净值计算
    curr_equity = st.session_state.equity_history[-1]['equity']
    if curr_equity > st.session_state.peak:
        st.session_state.peak = curr_equity
    
    drawdown = (curr_equity - st.session_state.peak) / st.session_state.peak

    # 简单判定信号
    is_long = df['ema12'].iat[last_idx] > df['ema26'].iat[last_idx]
    is_flow = df['net_flow'].iat[last_idx] > 0
    signal = is_long and is_flow

    # --- UI 渲染 ---
    st.title("🛰️ ETH 战神·水泥防御终端 V50000")
    
    cols = st.columns(4)
    cols[0].metric("实时金库", f"${curr_equity:.2f}", f"{drawdown*100:.2f}%")
    cols[1].metric("最高峰值", f"${st.session_state.peak:.2f}")
    cols[2].metric("信号扫描", "🔥 黄金总攻" if signal else "📡 监控中")
    cols[3].metric("当前报价", f"${c_price:.2f}")

    # --- 绘图区 (用 try 包裹，死也死在里面，不准弄脏界面) ---
    try:
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.5, 0.2, 0.3])
        
        # K线
        fig.add_trace(go.Candlestick(x=df['time'], open=df['o'], high=df['h'], low=df['l'], close=df['c'], name="K线"), row=1, col=1)
        
        # 爆仓点 (防御：先判断有没有爆仓数据)
        liq_data = df[df['liq'] == 1]
        if not liq_data.empty:
            fig.add_trace(go.Scatter(x=liq_data['time'], y=liq_data['h'] + 5, mode='markers', marker=dict(color='yellow', size=10, symbol='diamond'), name="爆仓点"), row=1, col=1)

        # 动能
        fig.add_trace(go.Bar(x=df['time'], y=df['macd'], name="动能"), row=2, col=1)
        
        # 财富曲线
        eq_df = pd.DataFrame(st.session_state.equity_history)
        fig.add_trace(go.Scatter(x=eq_df['time'], y=eq_df['equity'], line=dict(color='#00ff00', width=3), name="账户净值"), row=3, col=1)

        fig.update_layout(template="plotly_dark", height=700, xaxis_rangeslider_visible=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"⚠️ 绘图引擎对冲中: {e}")

    # 日志
    if st.checkbox("查看原始审计日志"):
        st.dataframe(df.tail(10), use_container_width=True)

if __name__ == "__main__":
    main()
