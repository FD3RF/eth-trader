import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from scipy.signal import argrelextrema
from datetime import datetime

# ==================== 1. 核心配置 ====================
OKX_CONFIG = {
    "api_key": "a2a2a452-49e6-4e76-95f3-fb54eb982e7b",
    "secret_key": "330FABB2CAD3585677716686C2BF3872",
    "passphrase": "123321aA@", 
}

# 初始化笔记存储
if 'trade_logs' not in st.session_state:
    st.session_state.trade_logs = []

def fetch_okx(endpoint, params=""):
    url = f"https://www.okx.com/api/v5/{endpoint}?instId=ETH-USDT{params}"
    try:
        with requests.Session() as s:
            s.trust_env = True 
            r = s.get(url, timeout=5)
            return r.json()
    except: return None

# ==================== 2. 数据处理与 ATR ====================
def get_processed_data():
    k_data = fetch_okx("market/candles", "&bar=5m&limit=150")
    if not k_data: return pd.DataFrame()
    df = pd.DataFrame(k_data['data'], columns=['ts','o','h','l','c','v','volCcy','volCcyQ','confirm'])
    df[['o','h','l','c','v']] = df[['o','h','l','c','v']].astype(float)
    df = df[::-1].reset_index(drop=True)
    
    # ATR 计算
    tr = pd.concat([df['h']-df['l'], np.abs(df['h']-df['c'].shift()), np.abs(df['l']-df['c'].shift())], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    df['ema21'] = df['c'].ewm(span=21).mean()
    return df

# ==================== 3. 界面布局 ====================
st.set_page_config(page_title="ETH V38.0 职业笔记版", layout="wide")

# --- 侧边栏：计算器 + 交易笔记 ---
st.sidebar.title("🧮 战斗准备区")
balance = st.sidebar.number_input("本金 (USDT)", value=10000.0)
risk_pct = st.sidebar.slider("风险度 (%)", 0.5, 5.0, 1.5) / 100
atr_mult = st.sidebar.slider("止损倍数", 1.0, 3.0, 1.8)

st.sidebar.divider()

st.sidebar.subheader("📝 交易笔记")
note_content = st.sidebar.text_area("输入当前操作理由...", placeholder="例如：5m底背离+1h共振灯绿，回踩POC买入")
if st.sidebar.button("💾 保存当前快照"):
    log_entry = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "note": note_content,
        "price": st.session_state.get('last_price', 0)
    }
    st.session_state.trade_logs.insert(0, log_entry)
    st.sidebar.success("快照已保存")

# --- 主界面 ---
df = get_processed_data()
depth_data = fetch_okx("market/books", "&sz=10")

if not df.empty and depth_data:
    curr_p = df.iloc[-1]['c']
    st.session_state.last_price = curr_p
    curr_atr = df.iloc[-1]['atr']
    sl_price = curr_p - (curr_atr * atr_mult)
    
    # 计算仓位
    pos_eth = (balance * risk_pct) / (curr_p - sl_price)
    
    st.title("🛡️ ETH V38.0 职业实战指挥部")
    
    # 顶部状态栏
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ETH 现价", f"${curr_p:.2f}")
    c2.metric("建议买入", f"{pos_eth:.3f} ETH")
    c3.metric("硬性止损", f"${sl_price:.2f}")
    risk_val = 98.9 if curr_p < df.iloc[-1]['ema21'] and df.iloc[-1]['v'] < df['v'].mean()*0.7 else 12.0
    c4.metric("阴跌风险", f"{risk_val}", delta_color="inverse")

    # 左右排版
    main_view, log_view = st.columns([3, 1])
    
    with main_view:
        # K线与 VPVR
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['o'], high=df['h'], low=df['l'], close=df['c'], name="K线"))
        fig.add_hline(y=sl_price, line_dash="dash", line_color="red", annotation_text="止损线")
        fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)
    
    with log_view:
        st.subheader("📋 历史快照记录")
        if not st.session_state.trade_logs:
            st.info("暂无记录，请在左侧点击保存")
        else:
            for log in st.session_state.trade_logs[:5]: # 只显示最近5条
                with st.expander(f"⏰ {log['time']} | 价: {log['price']}"):
                    st.write(log['note'])

    # 庄家大单 (放在底部)
    st.divider()
    st.subheader("📊 庄家实时大单 (盘口压力)")
    asks = pd.DataFrame(depth_data['data'][0]['asks'], columns=['p', 's', 'o']).astype(float)
    bids = pd.DataFrame(depth_data['data'][0]['bids'], columns=['p', 's', 'o']).astype(float)
    bc1, bc2 = st.columns(2)
    bc1.write("🔴 压盘 (Asks)")
    bc1.dataframe(asks.head(5), use_container_width=True)
    bc2.write("🟢 护盘 (Bids)")
    bc2.dataframe(bids.head(5), use_container_width=True)

else:
    st.warning("🔄 正在穿透网络隧道...")
