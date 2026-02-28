import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime

st.set_page_config(layout="wide", page_title="ETH V32007 终极指挥官", page_icon="⚖️")

# ==========================================
# 1. 状态管理与内存治理
# ==========================================
def init_commander_state():
    if 'ls_ratio' not in st.session_state: st.session_state.ls_ratio = 1.0
    if 'last_cleanup' not in st.session_state: st.session_state.last_cleanup = time.time()

def light_cleanup():
    if time.time() - st.session_state.last_cleanup > 7200:
        st.cache_data.clear()
        st.session_state.last_cleanup = time.time()

# ==========================================
# 2. 情报引擎（支持更多框架：1m/5m/15m/30m/1H）
# ==========================================
@st.cache_data(ttl=10)
def get_candles(f_ema, s_ema, bar="1m"):
    try:
        url = f"https://www.okx.com/api/v5/market/candles?instId=ETH-USDT&bar={bar}&limit=100"
        res = requests.get(url, timeout=6).json()
        df = pd.DataFrame(res['data'], columns=['ts','o','h','l','c','v','volCcy','volCcyQuote','confirm'])[::-1].reset_index(drop=True)
        df['time'] = pd.to_datetime(df['ts'].astype(float), unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')
        for c in ['o','h','l','c','v']: df[c] = df[c].astype(float)
        
        df['ema_f'] = df['c'].ewm(span=f_ema, adjust=False).mean()
        df['ema_s'] = df['c'].ewm(span=s_ema, adjust=False).mean()
        df['net_flow'] = (df['c'].diff() * df['v'] * 0.15).rolling(5).sum().fillna(0)
        tr = pd.concat([df['h']-df['l'], abs(df['h']-df['c'].shift()), abs(df['l']-df['c'].shift())], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        return df
    except: 
        return pd.DataFrame()

@st.cache_data(ttl=15)
def get_ls_ratio():
    try:
        url = "https://www.okx.com/api/v5/rubik/stat/contracts/long-short-account-ratio?instId=ETH-USDT&period=5m"
        res = requests.get(url, timeout=6).json()
        return float(res['data'][0][1]) if res.get('code') == '0' else 1.0
    except: 
        return 1.0

# ==========================================
# 3. 侧边栏：新增多TF共振提示 + 详细策略
# ==========================================
def render_sidebar(df):
    with st.sidebar:
        st.markdown("### 🛸 量子实时控制 V32007")
        hb = st.slider("刷新频率 (秒)", 5, 60, 10)
        
        tf_options = ["1m", "5m", "15m", "30m", "1H"]
        tf = st.selectbox("时间框架", tf_options, index=2)  # 默认15m
        
        f_e = st.number_input("快线 EMA", 5, 30, 12)
        s_e = st.number_input("慢线 EMA", 20, 100, 26)
        st.divider()
        
        last_p = df['c'].iloc[-1]
        atr = df['atr'].iloc[-1]
        ls = st.session_state.ls_ratio
        is_golden_cross = df['ema_f'].iloc[-1] > df['ema_s'].iloc[-1]

        prob = 50.0
        prob += 18 if is_golden_cross else -10
        prob += 12 if ls > 1.08 else -14 if ls < 0.92 else 0
        prob += 8 if df['net_flow'].iloc[-1] > 0 else -6
        prob = max(min(prob, 89.0), 22.0)

        direction = 1 if prob > 50 else -1
        tp_sugg = last_p + direction * atr * 2.6
        sl_sugg = last_p - direction * atr * 1.4
        rr = abs(tp_sugg - last_p) / abs(last_p - sl_sugg) if abs(last_p - sl_sugg) > 0.01 else 1.8
        
        box_color = "#00ff88" if prob > 65 else "#FFD700" if prob > 48 else "#ff4b4b"
        st.markdown(f"""
            <div style="border:2px solid {box_color}; padding:18px; border-radius:14px; background:rgba(0,255,136,0.08); text-align:center;">
                <div style="color:{box_color}; font-size:0.9em; font-weight:bold;">AI 实时胜率</div>
                <div style="color:{box_color}; font-size:2.6em; font-weight:bold;">{prob:.1f}%</div>
                <div style="color:#aaa; font-size:0.8em;">建议 R/R: 1 : {rr:.2f}</div>
            </div>
        """, unsafe_allow_html=True)
        st.divider()

        sentiment = "看多 🔥" if ls > 1.0 else "看空 ❄️"
        recap = "✅ 趋势量价匹配良好" if prob > 55 else "📉 警惕缩量虚拉"
        st.info(f"🔍 AI 自动复盘：{recap}\n\n散户情绪: {sentiment}")

        # 新增：多时间框架共振提示
        st.markdown("#### 🔄 多时间框架共振提示")
        if tf in ["15m", "30m"] and df['net_flow'].iloc[-1] > 1000:
            st.success("✅ **15m/30m 共振强信号**：主力吸筹，适合轻仓布局")
        elif tf == "1H" and is_golden_cross:
            st.success("✅ **1H 金叉确认**：大趋势反转，可中仓追涨")
        elif tf == "1m" and prob < 45:
            st.warning("⚠️ 1m 噪音大，切换15m/30m确认信号")
        else:
            st.info("⚖️ 当前框架共振中性，尝试切换15m/30m查看净流")

        # 详细交易策略执行计划
        st.markdown("#### 🎯 详细交易策略执行计划")
        st.caption(f"当前市场判断：{'EMA金叉+净流偏多' if is_golden_cross and df['net_flow'].iloc[-1]>0 else '震荡中性+轻微流出'}")

        if prob < 45:
            st.error("🚨 **指挥官最优决策**：空仓观望！胜率过低，保本第一，等待15m金叉。")
        elif prob > 65:
            st.success("🚀 **指挥官最优决策**：中仓进攻清算猎杀！")
        else:
            st.warning("⚖️ **指挥官最优决策**：轻仓物理位陷阱 或 观望。")

        strats = [
            {
                "name": "物理位陷阱",
                "state": "⚪ 观察中",
                "color": "#FFD700",
                "entry": f"价格回踩 EMA26（约{last_p - atr*0.8:.1f}）后反弹",
                "tp": f"${last_p + atr*2.0:.1f}",
                "sl": f"${last_p - atr*1.2:.1f}",
                "rr": "1:1.7",
                "position": "轻仓 20%",
                "reason": "当前震荡箱体，防假突破。适合防御型交易者。",
                "risk": "若净流持续转负，立即取消。"
            },
            {
                "name": "清算猎杀",
                "state": "🔥 进攻中" if prob > 65 else "⚪ 待机",
                "color": "#00ff88" if prob > 65 else "#ff4b4b",
                "entry": f"突破近期高点（约{df['h'].tail(30).max():.1f}）",
                "tp": f"${tp_sugg:.1f}",
                "sl": f"${sl_sugg:.1f}",
                "rr": f"1:{rr:.2f}",
                "position": "中仓 40%" if prob > 65 else "观望",
                "reason": f"胜率{prob:.0f}% + EMA金叉，主力净流倾向多头，适合猎杀空头止损。",
                "risk": "ATR扩大时严格执行SL。"
            },
            {
                "name": "EMA金叉追涨",
                "state": "✅ 已激活" if is_golden_cross else "⚪ 未触发",
                "color": "#00ff88" if is_golden_cross else "#888888",
                "entry": "15m框架下金叉确认 + 量能放大",
                "tp": f"${last_p + atr*3.5:.1f}",
                "sl": f"${last_p - atr*1.0:.1f}",
                "rr": "1:3.5",
                "position": "重仓 60%" if is_golden_cross and prob > 70 else "轻仓",
                "reason": "多时间框架共振，金叉后趋势加速概率高。当前15m框架最强。",
                "risk": "若15m回踩EMA26失效，立即减仓。"
            }
        ]

        for s in strats:
            st.markdown(f"""
                <div style="border-left:4px solid {s['color']}; padding:12px; margin:8px 0; background:rgba(255,255,255,0.05); border-radius:8px;">
                    <b style="color:{s['color']};">{s['state']} | {s['name']}</b><br>
                    <span style="color:#00ff88;">📍 入场：{s['entry']}</span><br>
                    <span style="color:#00ff88;">🎯 TP：{s['tp']}　</span>
                    <span style="color:#ff4b4b;">🛡️ SL：{s['sl']}</span><br>
                    <span style="color:#aaa;">R/R {s['rr']}　仓位 {s['position']}</span><br>
                    <span style="font-size:0.8em; color:#ccc;">理由：{s['reason']}</span><br>
                    <span style="font-size:0.75em; color:#ff4b4b;">⚠️ 风险：{s['risk']}</span>
                </div>
            """, unsafe_allow_html=True)

        st.divider()
        st.markdown(f"【{datetime.now().strftime('%H:%M:%S')}】卫星同步成功 | 当前框架：{tf}")
        return hb, f_e, s_e, tf

# ==========================================
# 4. 主界面
# ==========================================
def main():
    init_commander_state()
    light_cleanup()
    
    ls = get_ls_ratio()
    st.session_state.ls_ratio = ls
    
    df_init = get_candles(12, 26, "15m")
    if df_init.empty:
        st.error("❌ 卫星连接断开，正在重试...")
        time.sleep(2)
        st.rerun()
    
    hb, f_e, s_e, tf = render_sidebar(df_init)
    df = get_candles(f_e, s_e, tf)
    
    st.markdown(f"### 🛰️ ETH 量子决策指挥官 (V32007) | {tf} | {datetime.now().strftime('%H:%M:%S')}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("实时价格", f"${df['c'].iloc[-1]:.2f}")
    
    if ls > 1.02:
        delta_text = "↑ 多头占优"
    elif ls < 0.98:
        delta_text = "↓ 空头占优"
    else:
        delta_text = "↔ 中性均衡"
    c2.metric("全网多空比", f"{ls:.2f}", delta_text)
    
    c3.metric("ATR 动态波幅", f"{df['atr'].iloc[-1]:.2f}")
    c4.metric("庄家净流", f"{df['net_flow'].iloc[-1]:.0f}")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=df['time'], open=df['o'], high=df['h'], low=df['l'], close=df['c'], name="K线"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['ema_f'], line=dict(color='#00ff88', width=2.5), name="EMA快线"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['ema_s'], line=dict(color='#ff4b4b', width=2.5), name="EMA慢线"), row=1, col=1)

    h_max = df['h'].tail(60).max()
    l_min = df['l'].tail(60).min()
    fig.add_hrect(y0=h_max*1.018, y1=h_max*1.02, fillcolor="red", opacity=0.35, annotation_text="空头燃料", row=1, col=1)
    fig.add_hrect(y0=l_min*0.98, y1=l_min*0.982, fillcolor="green", opacity=0.35, annotation_text="多头燃料", row=1, col=1)

    flow_max = abs(df['net_flow']).max() or 1
    scaled_flow = df['net_flow'] / flow_max * 800
    colors = ['#00ff88' if x > 0 else '#ff4b4b' for x in df['net_flow']]
    fig.add_trace(go.Bar(x=df['time'], y=scaled_flow, marker_color=colors, name="庄家净流"), row=2, col=1)

    fig.update_layout(template="plotly_dark", height=830, xaxis_rangeslider_visible=False, margin=dict(l=10,r=10,t=15,b=10))
    st.plotly_chart(fig, use_container_width=True)

    time.sleep(hb)
    st.rerun()

if __name__ == "__main__":
    main()
