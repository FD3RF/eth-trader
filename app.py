import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime

# ==================== 1. 物理账本：绝对初始化 ====================
st.set_page_config(page_title="ETH 最终实战终端", layout="wide")

def init_vault():
    if 'bal' not in st.session_state: st.session_state.bal = 10000.0
    if 'entry' not in st.session_state: st.session_state.entry = 1850.0 # 预设入场点让曲线动起来
    if 'history' not in st.session_state: 
        st.session_state.history = [{"t": datetime.now().strftime('%H:%M:%S'), "v": 10000.0}]

init_vault()

# ==================== 2. 暴力数据：只要最稳的价格 ====================
def get_market_data():
    try:
        # 使用最稳的同步 requests
        url = "https://www.okx.com/api/v5/market/candles?instId=ETH-USDT&bar=1m&limit=100"
        r = requests.get(url, timeout=5).json()
        d = r.get('data', [])
        if not d: return None
        
        df = pd.DataFrame(d, columns=['ts','o','h','l','c','v','volC','volCQ','cf'])[::-1]
        df['c'] = df['c'].astype(float)
        df['time'] = pd.to_datetime(df['ts'].astype(float), unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')
        
        # 强制补齐指标，防止绘图时 KeyError
        df['ema20'] = df['c'].ewm(span=20, adjust=False).mean()
        return df
    except:
        return None

# ==================== 3. 结算逻辑：每一秒都在动 ====================
def main():
    init_vault()
    df = get_market_data()
    
    if df is None:
        st.warning("📡 正在尝试穿透防火墙接入手续，请稍候...")
        return

    # 物理计算：拒绝死线
    last_p = df['c'].iloc[-1]
    pnl = (last_p - st.session_state.entry) / st.session_state.entry
    curr_val = st.session_state.bal * (1 + pnl)
    
    # 记录时间序列
    now_t = datetime.now().strftime('%H:%M:%S')
    if st.session_state.history[-1]['t'] != now_t:
        st.session_state.history.append({"t": now_t, "v": curr_val})
        if len(st.session_state.history) > 60: st.session_state.history.pop(0)

    # UI 渲染
    st.title("⚔️ ETH 战神修复终端")
    c1, c2, c3 = st.columns(3)
    c1.metric("账户净值", f"${curr_val:.2f}", f"{pnl*100:+.2f}%")
    c2.metric("实时币价", f"${last_p:.2f}")
    c3.metric("入场基准", f"${st.session_state.entry:.2f}")

    # --- 绘图：双轴物理对齐 ---
    fig = go.Figure()
    # 价格线
    fig.add_trace(go.Scatter(x=df['time'], y=df['c'], name="ETH价格", line=dict(color='cyan', width=2)))
    # 净值线 (绘制在底部的历史点)
    hist_df = pd.DataFrame(st.session_state.history)
    fig.add_trace(go.Scatter(x=df['time'].tail(len(hist_df)), y=hist_df['v'], 
                             name="收益曲线", fill='tozeroy', line=dict(color='#00ff00', width=3)))

    fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False, margin=dict(t=30,b=10,l=10,r=10))
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
