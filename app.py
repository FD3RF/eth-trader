import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import streamlit.components.v1 as components

# ==========================================
# 1. 物理层：系统初始化与环境锁
# ==========================================
st.set_page_config(layout="wide", page_title="ETH V40000 战神·不灭终端", page_icon="⚡")

def init_warrior_system():
    """初始化所有核心变量，确保系统稳定性"""
    if 'df' not in st.session_state: st.session_state.df = pd.DataFrame()
    if 'meta' not in st.session_state:
        st.session_state.meta = {
            "v_factor": 1.0, 
            "circuit_break": False, 
            "init_price": 0.0,
            "peak_equity": 10000.0,
            "cooldown_until": None,
            "alert_fired": False,
            "mode": "Normal"
        }
    if 'battle_logs' not in st.session_state: st.session_state.battle_logs = []
    if 'equity_history' not in st.session_state:
        # 初始资金 $10,000
        st.session_state.equity_history = [{"time": datetime.now().strftime('%H:%M:%S'), "equity": 10000.0, "eth_pnl": 0.0}]

init_warrior_system()

# ==========================================
# 2. 仪式感：量子语音引擎
# ==========================================
def speak(text):
    """注入 HTML5 语音合成脚本"""
    js = f"""<script>
    var msg = new SpeechSynthesisUtterance('{text}');
    msg.lang = 'zh-CN'; msg.pitch = 0.7; msg.rate = 0.9;
    window.speechSynthesis.speak(msg);
    </script>"""
    components.html(js, height=0)

# ==========================================
# 3. 数据层：影子防御引擎
# ==========================================
def fetch_bulletproof_data():
    url = "https://www.okx.com/api/v5/market/candles?instId=ETH-USDT&bar=1m&limit=100"
    try:
        r = requests.get(url, timeout=5).json()
        if r.get('code') != '0': return st.session_state.df
        
        # 数据转换与影子补齐
        df = pd.DataFrame(r.get('data', []), columns=['ts','o','h','l','c','v','volC','volCQ','cf'])[::-1]
        df['time'] = pd.to_datetime(df['ts'].astype(float), unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')
        for col in ['o','h','l','c','v']: df[col] = df[col].astype(float)
        
        # 指标计算
        df['ema12'] = df['c'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['c'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['net_flow'] = (df['v'] * np.random.uniform(0.48, 0.52) - df['v'] * 0.5).rolling(5).sum().fillna(0)
        
        # 影子列防御：确保 'liq' 永远存在
        df['liq'] = 0
        df.loc[(df['v'] > df['v'].mean()*2.5) & (abs(df['c']-df['o']) > 15), 'liq'] = 1
        
        # 波动率自适应计算
        std_20 = df['c'].rolling(20).std().iloc[-1]
        std_avg = df['c'].rolling(100).std().mean()
        if std_20 < (std_avg * 0.6) and not st.session_state.meta['circuit_break']:
            st.session_state.meta['v_factor'] = 0.5
            st.session_state.meta['mode'] = "⚡ 灵敏模式"
        else:
            st.session_state.meta['v_factor'] = 1.0
            st.session_state.meta['mode'] = "🌊 标准趋势"
            
        return df
    except:
        return st.session_state.df

# ==========================================
# 4. 逻辑层：熔断与结算
# ==========================================
def main():
    # 物理重载
    if st.sidebar.button("💎 物理重铸系统") or st.session_state.df.empty:
        st.session_state.df = fetch_bulletproof_data()
    
    df = st.session_state.df
    if df.empty: return

    # --- 冷静期判定 ---
    curr_equity = st.session_state.equity_history[-1]['equity']
    if curr_equity > st.session_state.meta['peak_equity']:
        st.session_state.meta['peak_equity'] = curr_equity
    
    drawdown = (curr_equity - st.session_state.meta['peak_equity']) / st.session_state.meta['peak_equity']
    
    in_cooldown = False
    if drawdown <= -0.05:
        if st.session_state.meta['cooldown_until'] is None:
            st.session_state.meta['cooldown_until'] = datetime.now() + timedelta(hours=1)
            if not st.session_state.meta['alert_fired']:
                speak("警告。回撤已达百分之五，系统强制锁定一小时。战神，请立即离场复盘。")
                st.session_state.meta['alert_fired'] = True
    
    if st.session_state.meta['cooldown_until']:
        if datetime.now() < st.session_state.meta['cooldown_until']:
            in_cooldown = True
        else:
            st.session_state.meta['cooldown_until'] = None
            st.session_state.meta['alert_fired'] = False

    # --- 信号判定 ---
    v_f = st.session_state.meta['v_factor']
    is_gold = (df['ema12'].iloc[-1] > df['ema26'].iloc[-1])
    is_flow = df['net_flow'].iloc[-1] > (3.0 * v_f)
    all_in = is_gold and is_flow and not in_cooldown

    # --- 结算逻辑 ---
    curr_ts = df['time'].iloc[-1].strftime('%H:%M:%S')
    if all_in and (not st.session_state.battle_logs or st.session_state.battle_logs[-1]['时间'] != curr_ts):
        st.session_state.battle_logs.append({"时间": curr_ts, "价格": df['c'].iloc[-1], "盈亏": "0.00%", "状态": "⚔️ 战斗中"})

    for log in st.session_state.battle_logs:
        if log['状态'] == "⚔️ 战斗中":
            pnl = (df['c'].iloc[-1] - log['价格']) / log['价格']
            log['盈亏'] = f"{pnl*100:+.2f}%"
            if df['ema12'].iloc[-1] < df['ema26'].iloc[-1]:
                log['状态'] = "✅ 已收割"
                # 更新财富曲线
                new_eq = st.session_state.equity_history[-1]['equity'] * (1 + pnl)
                # 计算基准
                if st.session_state.meta['init_price'] == 0: st.session_state.meta['init_price'] = df['c'].iloc[0]
                eth_pnl = (df['c'].iloc[-1] - st.session_state.meta['init_price']) / st.session_state.meta['init_price'] * 100
                st.session_state.equity_history.append({"time": curr_ts, "equity": new_eq, "eth_pnl": eth_pnl})

    # ==========================================
    # 5. UI 层：战神可视化终端
    # ==========================================
    st.markdown(f"### 🛰️ ETH 量子巅峰雷达 V40000 | {st.session_state.meta['mode']}")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("账户净值", f"${curr_equity:.2f}", f"{drawdown*100:.2f}%")
    m2.metric("最高净值", f"${st.session_state.meta['peak_equity']:.2f}")
    m3.metric("熔断冷静期", "🛑 强制锁死" if in_cooldown else "🟢 状态安全")
    if in_cooldown:
        rem = st.session_state.meta['cooldown_until'] - datetime.now()
        m4.metric("锁死倒计时", f"{str(rem).split('.')[0].split(':')[-2]}分")
    else:
        m4.metric("灵敏因子", f"{v_f}x")

    st.divider()

    l_panel, r_panel = st.columns([1, 4])
    with l_panel:
        box_clr = "#FF4B4B" if in_cooldown else ("#FFD700" if all_in else "#00FFCC")
        st.markdown(f"""<div style="border:3px solid {box_clr}; padding:20px; border-radius:15px; text-align:center; background:rgba(0,0,0,0.4); box-shadow: 0 0 20px {box_clr};">
            <h2 style="color:{box_clr}; margin:0;">{'咆哮冷静中' if in_cooldown else ('黄金总攻' if all_in else '监控扫描')}</h2>
            <p style="color:#888; margin-top:10px;">{'5%最大回撤熔断已触发' if in_cooldown else '回撤监控系统已就绪'}</p>
        </div>""", unsafe_allow_html=True)
        if all_in: st.balloons()
        if st.sidebar.button("♻️ 强制重置熔断"):
            st.session_state.meta.update({"cooldown_until": None, "alert_fired": False, "peak_equity": curr_equity})
            st.rerun()

    with r_panel:
        # 四层合龙绘图
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02, 
                            row_heights=[0.4, 0.15, 0.15, 0.3],
                            subplot_titles=("K线图", "动能", "资金流", "💰 财富净值对比"))
        
        # K线 + 爆仓点 (影子防御)
        fig.add_trace(go.Candlestick(x=df['time'], open=df['o'], high=df['h'], low=df['l'], close=df['c'], name="Price"), row=1, col=1)
        if 'liq' in df.columns:
            liq_df = df[df['liq'] == 1]
            fig.add_trace(go.Scatter(x=liq_df['time'], y=liq_df['h']+10, mode='markers', marker=dict(symbol='diamond', color='yellow', size=8), name="Liq"), row=1, col=1)
        
        # 指标层
        fig.add_trace(go.Bar(x=df['time'], y=df['macd'], marker_color='cyan', name="MACD"), row=2, col=1)
        fig.add_trace(go.Bar(x=df['time'], y=df['net_flow'], marker_color=['#00ff00' if x > 0 else '#ff0000' for x in df['net_flow']], name="Flow"), row=3, col=1)
        
        # 财富曲线层
        eq_df = pd.DataFrame(st.session_state.equity_history)
        fig.add_trace(go.Scatter(x=eq_df['time'], y=eq_df['equity'], line=dict(color='#00ff00', width=3), name="Strategy"), row=4, col=1)
        eth_bench = 10000 * (1 + eq_df['eth_pnl']/100)
        fig.add_trace(go.Scatter(x=eq_df['time'], y=eth_bench, line=dict(color='grey', dash='dash'), name="ETH HODL"), row=4, col=1)

        fig.update_layout(template="plotly_dark", height=900, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig, use_container_width=True)

    # 日志区
    st.markdown("### 📜 实时猎杀审计日志")
    st.dataframe(pd.DataFrame(st.session_state.battle_logs[::-1]), use_container_width=True)

if __name__ == "__main__":
    main()
