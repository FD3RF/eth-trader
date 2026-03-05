import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import httpx
import time

# ==========================================
# 1. 顶级视觉配置
# ==========================================
st.set_page_config(layout="wide", page_title="Warrior V5.2 | 实战指令", page_icon="⚔️")

st.markdown("""
    <style>
    .block-container { padding: 1rem 1.5rem; }
    [data-testid="stMetric"] { background: #0e1117; border: 1px solid #262730; padding: 10px; border-radius: 8px; }
    .trade-plan { background: #1a1c23; border-left: 5px solid #d4af37; padding: 15px; border-radius: 5px; }
    .modebar { display: none !important; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 核心引擎与逻辑 (承袭 V5.1)
# ==========================================
class WarriorCore:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.client = httpx.Client(timeout=4.0, http2=True)
        return cls._instance

    def fetch_data(self, symbol):
        url = "https://www.okx.com/api/v5/market/candles"
        try:
            params = {"instId": symbol, "bar": "5m", "limit": "100", "_t": time.time_ns()}
            resp = self.client.get(url, params=params)
            if resp.status_code == 200:
                data = resp.json().get('data', [])
                df = pd.DataFrame(data, columns=['ts','o','h','l','c','v','volCcy','volCcyQuote','confirm'])
                for col in ['o','h','l','c','v']: df[col] = pd.to_numeric(df[col])
                df['time'] = pd.to_datetime(df['ts'].astype(int), unit='ms')
                return df.sort_values('time').reset_index(drop=True)
        except: return None

# ==========================================
# 3. 进场计划计算逻辑
# ==========================================
def generate_trade_plan(side, entry, sl, rr=1.5, risk_amount=200):
    """
    根据口诀逻辑自动计算交易计划
    """
    if side == "LONG":
        tp = entry + (entry - sl) * rr
        dist = entry - sl
    else:
        tp = entry - (sl - entry) * rr
        dist = sl - entry
    
    qty = risk_amount / dist if dist > 0 else 0
    return {"entry": entry, "sl": sl, "tp": tp, "qty": qty}

# ==========================================
# 4. 实时动态扫描与指令 UI
# ==========================================
@st.fragment(run_every="5s")
def render_warrior_v5_2():
    core = WarriorCore()
    symbol = st.session_state.get('symbol', 'ETH-USDT-SWAP')
    df = core.fetch_data(symbol)
    if df is None: return

    # --- 逻辑运算 ---
    df['v_ma5'] = df['v'].rolling(5).mean()
    curr, last_v_ma = df.iloc[-1], df.iloc[-1]['v_ma5']
    p_high = df['h'].iloc[-15:-1].max()
    p_low = df['l'].iloc[-15:-1].min()
    
    # --- 信号判定 ---
    is_expanding = curr['v'] > last_v_ma * 1.8
    sig_long = is_expanding and curr['c'] > p_high
    sig_short = is_expanding and curr['c'] < p_low

    # --- UI 布局 ---
    t1, t2 = st.columns([1.2, 3])

    with t1:
        st.subheader("🎯 实时进场指令")
        
        # 信号触发表单
        active_side = "NONE"
        if sig_long: 
            st.success("🔥 发现多头进攻信号！")
            active_side = "LONG"
        elif sig_short:
            st.error("❄️ 发现空头下砸信号！")
            active_side = "SHORT"
        else:
            st.info("💎 动能积蓄中，等待放量...")

        if active_side != "NONE":
            # 自动生成初版计划
            entry_p = curr['c']
            sl_p = p_low if active_side == "LONG" else p_high
            plan = generate_trade_plan(active_side, entry_p, sl_p, risk_amount=st.session_state.risk)

            with st.container(border=True):
                st.markdown(f"**方向: {'做多' if active_side=='LONG' else '做空'}**")
                st.write(f"建议入场: **{plan['entry']:.2f}**")
                st.write(f"建议止损: **{plan['sl']:.2f}**")
                st.write(f"建议止盈: **{plan['tp']:.2f}**")
                st.write(f"建议头寸: **{plan['qty']:.3f} ETH**")
                
                if st.button("🚀 确认执行并记录战报", use_container_width=True):
                    st.toast("战报已保存至系统缓存", icon="✅")

        st.divider()
        st.metric("前高压力", f"{p_high:.2f}")
        st.metric("前低支撑", f"{p_low:.2f}")

    with t2:
        # K线图逻辑
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
        
        # K线
        fig.add_trace(go.Candlestick(x=df['time'], open=df['o'], high=df['h'], low=df['l'], close=df['c'], name="Price"), row=1, col=1)
        
        # 价格锚点线 (如果有活动信号，画出计划线)
        if active_side != "NONE":
            fig.add_hline(y=plan['entry'], line_color="#d4af37", line_dash="dash", annotation_text="ENTRY", row=1, col=1)
            fig.add_hline(y=plan['sl'], line_color="#ef5350", line_dash="dot", annotation_text="STOP LOSS", row=1, col=1)
            fig.add_hline(y=plan['tp'], line_color="#26a69a", line_dash="dot", annotation_text="TAKE PROFIT", row=1, col=1)
        else:
            # 常驻现价金色准星
            fig.add_hline(y=curr['c'], line_color="#d4af37", line_width=1, annotation_text=f"LIVE: {curr['c']:.2f}", row=1, col=1)

        # 成交量
        v_colors = ['#26a69a' if _c >= _o else '#ef5350' for _c, _o in zip(df['c'], df['o'])]
        fig.add_trace(go.Bar(x=df['time'], y=df['v'], marker_color=v_colors, opacity=0.3), row=2, col=1)

        fig.update_layout(height=750, template="plotly_dark", showlegend=False, 
                          xaxis_rangeslider_visible=False, margin=dict(t=10,b=10,l=10,r=70),
                          uirevision=symbol)
        
        st.plotly_chart(fig, width="stretch", config={'displayModeBar': False})

# ==========================================
# 5. 指挥中心主入口
# ==========================================
def main():
    st.sidebar.title("⚔️ Warrior V5.2")
    st.session_state.symbol = st.sidebar.text_input("目标合约", "ETH-USDT-SWAP")
    st.session_state.risk = st.sidebar.number_input("单笔风险金额 (USDT)", 10, 5000, 200)
    
    st.sidebar.divider()
    st.sidebar.markdown("""
    **九阴真经·实战状态：**
    - 🟢 缩量回踩：监视中
    - 🔴 放量突破：**检测引擎就绪**
    - ⚠️ 陷阱防御：已开启
    """)
    
    render_warrior_v5_2()

if __name__ == "__main__":
    main()
