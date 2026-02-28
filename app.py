import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# 1. 物理锁：最顶层强制初始化 (严禁改动)
# ==========================================
st.set_page_config(layout="wide", page_title="ETH V70000 战神终端")

# 强制焊死状态机，防止刷新瞬间的 KeyError
def force_init():
    for k, v in {
        'df': pd.DataFrame(),
        'p_equity': 10000.0,
        'e_hist': [{"time": "00:00", "equity": 10000.0, "pnl": 0.0}],
        'init_p': 0.0
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

force_init()

# ==========================================
# 2. 数据层：物理补齐协议
# ==========================================
def get_data_v7():
    url = "https://www.okx.com/api/v5/market/candles?instId=ETH-USDT&bar=1m&limit=100"
    cols = ['ts','o','h','l','c','v','volC','volCQ','cf']
    # 物理必填列，缺少这些列程序就不准往下走
    must_have = ['ema12', 'ema26', 'macd', 'liq', 'flow']
    
    try:
        r = requests.get(url, timeout=3).json()
        raw = r.get('data', [])
        if not raw: return st.session_state.df
        
        df = pd.DataFrame(raw, columns=cols)[::-1]
        df['time'] = pd.to_datetime(df['ts'].astype(float), unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')
        for c in ['o','h','l','c','v']: df[c] = df[c].astype(float)
        
        # --- 物理补齐：无论如何，先造出这些列 ---
        df = df.reindex(columns=list(df.columns) + must_have, fill_value=0.0)
        
        df['ema12'] = df['c'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['c'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['flow'] = (df['v'] * np.random.uniform(-0.02, 0.02)).rolling(5).sum().fillna(0)
        
        # 爆仓判定：只要成交量翻倍即标记
        v_avg = df['v'].mean()
        df.loc[df['v'] > v_avg * 3, 'liq'] = 1
        
        return df
    except:
        return st.session_state.df

# ==========================================
# 3. 终端主界面
# ==========================================
def main():
    force_init()
    
    if st.sidebar.button("♻️ 物理链路重置") or st.session_state.df.empty:
        st.session_state.df = get_data_v7()

    df = st.session_state.df
    
    # --- 最终物理关卡：不达标不显示 ---
    if df.empty or len(df) < 10:
        st.info("📡 链路数据同步中... 若长时间无响应请检查网络或点击‘重置’")
        return

    # 安全取值，绝不直接读 index
    last_c = df['c'].values[-1]
    curr_eq = st.session_state.e_hist[-1]['equity']
    
    # UI 标题与指标
    st.markdown(f"### 🛡️ ETH 战神·最终裁决终端 (V70000)")
    c1, c2, c3 = st.columns(3)
    c1.metric("账户净值", f"${curr_eq:.2f}")
    c2.metric("实时币价", f"${last_c:.2f}")
    c3.metric("趋势状态", "📈 多头向上" if df['macd'].values[-1] > 0 else "📉 空头压制")

    # --- 绘图保护：用 try-except 物理包裹 ---
    try:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
        
        # 1. K线与爆仓点
        fig.add_trace(go.Candlestick(x=df['time'], open=df['o'], high=df['h'], low=df['l'], close=df['c'], name="行情"), row=1, col=1)
        
        liq_df = df[df['liq'] == 1]
        if not liq_df.empty:
            fig.add_trace(go.Scatter(x=liq_df['time'], y=liq_df['h'] + 5, mode='markers', 
                                   marker=dict(color='yellow', size=12, symbol='star'), name="爆仓"), row=1, col=1)
        
        # 2. 动能
        fig.add_trace(go.Bar(x=df['time'], y=df['macd'], marker_color='cyan', name="动能"), row=2, col=1)

        fig.update_layout(template="plotly_dark", height=700, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.warning(f"⚠️ 绘图引擎临时挂起: {e}")

    # 日志审查
    with st.expander("🔍 物理内存数据审查"):
        st.dataframe(df.tail(5), use_container_width=True)

if __name__ == "__main__":
    main()
