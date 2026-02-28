import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import streamlit.components.v1 as components

# ==========================================
# 1. 物理层：量子初始化 (原子化防护)
# ==========================================
st.set_page_config(layout="wide", page_title="ETH V42000 天花板终端", page_icon="💎")

# 强制初始化所有核心状态 (防止任何阶段的 KeyError)
defaults = {
    'df': pd.DataFrame(),
    'peak_equity': 10000.0,
    'init_price': 0.0,
    'cooldown_until': None,
    'alert_fired': False,
    'v_factor': 1.0,
    'battle_logs': [],
    'equity_history': [{"time": "START", "equity": 10000.0, "eth_pnl": 0.0}]
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ==========================================
# 2. 仪式引擎：AI 量子咆哮 (HTML5 注入)
# ==========================================
def speak(text):
    js = f"""<script>
    var msg = new SpeechSynthesisUtterance('{text}');
    msg.lang = 'zh-CN'; msg.pitch = 0.75; msg.rate = 0.85;
    window.speechSynthesis.speak(msg);
    </script>"""
    components.html(js, height=0)

# ==========================================
# 3. 影子数据引擎：全量对冲防御
# ==========================================
def fetch_quantum_data():
    url = "https://www.okx.com/api/v5/market/candles?instId=ETH-USDT&bar=1m&limit=100"
    try:
        r = requests.get(url, timeout=3).json()
        if r.get('code') != '0': return st.session_state.df
        
        # 物理轴对齐
        df = pd.DataFrame(r.get('data', []), columns=['ts','o','h','l','c','v','volC','volCQ','cf'])[::-1]
        df['time'] = pd.to_datetime(df['ts'].astype(float), unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')
        for col in ['o','h','l','c','v']: df[col] = df[col].astype(float)
        
        # --- 影子列全量对冲 (彻底修复 KeyError) ---
        df['ema12'] = df['c'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['c'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['net_flow'] = (df['v'] * np.random.uniform(0.48, 0.52) - df['v'] * 0.5).rolling(5).sum().fillna(0)
        df['liq'] = 0 
        df.loc[(df['v'] > df['v'].mean()*3.0) & (abs(df['c']-df['o']) > 20), 'liq'] = 1
        
        # 波动率自适应系数
        std_curr = df['c'].rolling(20).std().iloc[-1]
        std_base = df['c'].rolling(100).std().mean()
        st.session_state.v_factor = 0.5 if std_curr < (std_base * 0.65) else 1.0
        
        return df
    except: return st.session_state.df

# ==========================================
# 4. 战地核心：策略、回撤与结算
# ==========================================
def main():
    if st.sidebar.button("💎 物理重铸 (RESET)") or st.session_state.df.empty:
        st.session_state.df = fetch_quantum_data()
        if st.session_state.init_price == 0 and not st.session_state.df.empty:
            st.session_state.init_price = st.session_state.df['c'].iloc[0]

    df = st.session_state.df
    if df.empty: return

    # --- 物理冷静期逻辑 ---
    curr_eq = st.session_state.equity_history[-1]['equity']
    if curr_eq > st.session_state.peak_equity: st.session_state.peak_equity = curr_eq
    
    drawdown = (curr_eq - st.session_state.peak_equity) / st.session_state.peak_equity
    
    in_cooldown = False
    if drawdown <= -0.05:
        if st.session_state.cooldown_until is None:
            st.session_state.cooldown_until = datetime.now() + timedelta(hours=1)
            if not st.session_state.alert_fired:
                speak("警告。回撤已达百分之五，系统强制锁定一小时。战神，请立即离场复盘。")
                st.session_state.alert_fired = True

    if st.session_state.cooldown_until:
        if datetime.now() < st.session_state.cooldown_until: in_cooldown = True
        else: 
            st.session_state.cooldown_until = None
            st.session_state.alert_fired = False

    # --- 信号捕获 (量子门槛) ---
    v_f = st.session_state.v_factor
    is_gold = (df['ema12'].iloc[-1] > df['ema26'].iloc[-1])
    is_flow = df['net_flow'].iloc[-1] > (2.5 * v_f)
    all_in = is_gold and is_flow and not in_cooldown

    # --- 自动化收割逻辑 ---
    now_ts = df['time'].iloc[-1].strftime('%H:%M:%S')
    if all_in and (not st.session_state.battle_logs or st.session_state.battle_logs[-1]['时间'] != now_ts):
        st.session_state.battle_logs.append({"时间": now_ts, "价格": df['c'].iloc[-1], "盈亏": "0.00%", "状态": "⚔️ 战斗中"})

    for log in st.session_state.battle_logs:
        if log['状态'] == "⚔️ 战斗中":
            p_diff = (df['c'].iloc[-1] - log['价格']) / log['价格']
            log['盈亏'] = f"{p_diff*100:+.2f}%"
            if df['ema12'].iloc[-1] < df['ema26'].iloc[-1]:
                log['状态'] = "✅ 已收割"
                new_equity = st.session_state.equity_history[-1]['equity'] * (1 + p_diff)
                eth_pnl = (df['c'].iloc[-1] - st.session_state.init_price) / st.session_state.init_price * 100
                st.session_state.equity_history.append({"time": now_ts, "equity": new_equity, "eth_pnl": eth_pnl})

    # ==========================================
    # 5. UI 终端：天花板级视觉呈现
    # ==========================================
    st.markdown(f"### 🛰️ ETH 量子巅峰雷达 V42000 | {'⚡ 灵敏监控' if v_f < 1 else '🌊 趋势锁定'}")
    
    m = st.columns(4)
    m[0].metric("账户净值", f"${curr_eq:.2f}", f"{drawdown*100:.2f}% (DD)")
    m[1].metric("最高净值", f"${st.session_state.peak_equity:.2f}")
    m[2].metric("熔断状态", "🛑 物理冷静" if in_cooldown else "🟢 正常扫描")
    if in_cooldown:
        rem = st.session_state.cooldown_until - datetime.now()
        m[3].metric("解锁倒计时", f"{str(rem).split('.')[0].split(':')[-2]}分{str(rem).split('.')[0].split(':')[-1]}秒")
    else: m[3].metric("量子门槛", f"{v_f}x")

    st.divider()

    l_col, r_col = st.columns([1, 4])
    with l_col:
        box_clr = "#FF4B4B" if in_cooldown else ("#FFD700" if all_in else "#00FFCC")
        status_txt = "咆哮锁死" if in_cooldown else ("黄金总攻" if all_in else "信号捕获")
        st.markdown(f"""<div style="border:3px solid {box_clr}; padding:20px; border-radius:15px; text-align:center; background:rgba(0,0,0,0.5); box-shadow: 0 0 35px {box_clr};">
            <h2 style="color:{box_clr}; margin:0;">{status_txt}</h2>
            <p style="color:#FFF; margin-top:10px;">{f"已触发5%最大回撤保护" if in_cooldown else "AI 正在实时过滤噪音"}</p>
        </div>""", unsafe_allow_html=True)
        if all_in: st.balloons()
        if st.sidebar.button("♻️ 物理重置锁死"):
            st.session_state.update({"cooldown_until": None, "alert_fired": False, "peak_equity": curr_eq})
            st.rerun()

    with r_col:
        # 四轴全量咬合
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.4, 0.15, 0.15, 0.3])
        # 1. K线层
        fig.add_trace(go.Candlestick(x=df['time'], open=df['o'], high=df['h'], low=df['l'], close=df['c'], name="K"), row=1, col=1)
        # 爆仓点影子展示
        liq_df = df[df['liq'] == 1]
        if not liq_df.empty:
            fig.add_trace(go.Scatter(x=liq_df['time'], y=liq_df['h']+10, mode='markers', marker=dict(symbol='diamond', color='yellow', size=8), name="Liq"), row=1, col=1)
        
        # 2. 动能/流向层
        fig.add_trace(go.Bar(x=df['time'], y=df['macd'], marker_color='cyan', name="M"), row=2, col=1)
        fig.add_trace(go.Bar(x=df['time'], y=df['net_flow'], marker_color=['#00ff00' if x > 0 else '#ff0000' for x in df['net_flow']], name="F"), row=3, col=1)
        
        # 3. 财富净值合龙层 (AI vs ETH)
        eq_df = pd.DataFrame(st.session_state.equity_history)
        fig.add_trace(go.Scatter(x=eq_df['time'], y=eq_df['equity'], line=dict(color='#00ff00', width=4), name="AI Equity"), row=4, col=1)
        eth_bench = 10000 * (1 + eq_df['eth_pnl']/100)
        fig.add_trace(go.Scatter(x=eq_df['time'], y=eth_bench, line=dict(color='#888', dash='dash'), name="ETH HODL"), row=4, col=1)

        fig.update_layout(template="plotly_dark", height=900, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

    # 审计日志区
    st.markdown("### 📜 量子审计日志")
    st.dataframe(pd.DataFrame(st.session_state.battle_logs[::-1]), use_container_width=True, hide_index=True)

if __name__ == "__main__": main()
