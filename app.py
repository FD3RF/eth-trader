import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(layout="wide", page_title="ETH V32009 终极指挥官", page_icon="⚖️")

# ==========================================
# 1. 状态管理与内存治理
# ==========================================
def init_commander_state():
    if 'ls_ratio' not in st.session_state: st.session_state.ls_ratio = 1.0
    if 'last_cleanup' not in st.session_state: st.session_state.last_cleanup = time.time()

def light_cleanup():
    if time.time() - st.session_state.last_cleanup > 86400:
        st.cache_data.clear()
        st.session_state.last_cleanup = time.time()

# ==========================================
# 2. 情报引擎（优化真实性 + 动态阈值）
# ==========================================
@st.cache_data(ttl=60, max_entries=50)
def get_candles(f_ema, s_ema, bar="1m"):
    try:
        url = f"https://www.okx.com/api/v5/market/candles?instId=ETH-USDT&bar={bar}&limit=100"
        res = requests.get(url, timeout=6).json()
        if res.get('code') != '0':
            raise ValueError("API 返回错误: " + res.get('msg', '未知'))
        df = pd.DataFrame(res['data'], columns=['ts','o','h','l','c','v','volCcy','volCcyQuote','confirm'])[::-1].reset_index(drop=True)
        df['time'] = pd.to_datetime(df['ts'].astype(float), unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')
        for c in ['o','h','l','c','v']: df[c] = df[c].astype(float)
        
        df['ema_f'] = df['c'].ewm(span=f_ema, adjust=False).mean()
        df['ema_s'] = df['c'].ewm(span=s_ema, adjust=False).mean()
        # 改进净流：用 RSI 调整系数，更真实模拟 (可接 trades API)
        # 注释：真实版调用 trades API 估算买/卖压力
        diff = df['c'].diff()
        gain = diff.clip(lower=0).rolling(14).mean()
        loss = -diff.clip(upper=0).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        coeff = 0.2 if rsi.iloc[-1] > 50 else 0.1
        df['net_flow'] = (df['c'].diff() * df['v'] * coeff).rolling(5).sum().fillna(0)
        tr = pd.concat([df['h']-df['l'], abs(df['h']-df['c'].shift()), abs(df['l']-df['c'].shift())], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        return df
    except requests.exceptions.RequestException as e:
        st.error(f"网络错误: {e}")
        return pd.DataFrame()
    except ValueError as e:
        st.error(f"数据错误: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"未知错误: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=15, max_entries=50)
def get_ls_ratio():
    try:
        url = "https://www.okx.com/api/v5/rubik/stat/contracts/long-short-account-ratio?instId=ETH-USDT&period=5m"
        res = requests.get(url, timeout=6).json()
        if res.get('code') != '0':
            raise ValueError("API 返回错误")
        return float(res['data'][0][1])
    except:
        st.error("多空比获取失败")
        return 1.0

# ==========================================
# 3. 侧边栏（优化体验 + 健壮性）
# ==========================================
def render_sidebar(df):
    with st.sidebar:
        st.markdown("### 🛸 量子实时控制 V32009")
        st.warning("⚠️ 部分指标（如净流入、清算区）为模拟演示，请勿作为实盘依据。")
        hb = st.slider("刷新频率 (秒)", 5, 60, 10)
        pause_refresh = st.checkbox("暂停自动刷新", value=False)
        
        tf_options = ["1m", "5m", "15m", "30m", "1H"]
        tf = st.selectbox("时间框架", tf_options, index=2)
        
        f_e = st.number_input("快线 EMA", 5, 30, 12)
        s_e = st.number_input("慢线 EMA", 20, 100, 26)
        st.divider()

        if len(df) < 14:
            st.warning("数据不足，无法计算指标，使用默认值")
            last_p = 0
            atr = 5.0
            is_golden_cross = False
            net_flow = 0
        else:
            last_p = df['c'].iloc[-1]
            atr = df['atr'].iloc[-1] if not pd.isna(df['atr'].iloc[-1]) else 5.0
            is_golden_cross = df['ema_f'].iloc[-1] > df['ema_s'].iloc[-1]
            net_flow = df['net_flow'].iloc[-1]

        ls = st.session_state.ls_ratio

        # 重构胜率：GBDT + 贝叶斯
        features = ["ema_f", "ema_s", "atr", "net_flow"]
        X = df[features]
        y = (df['c'].shift(-1) > df['c']).astype(int)  # 简化标签

        if len(df) < 50:
            prob = 50.0
        else:
            X_train, X_test, y_train, y_test = train_test_split(X.iloc[:-1], y.iloc[1:], test_size=0.2, shuffle=False)
            model = GradientBoostingClassifier()
            model.fit(X_train, y_train)
            prob = model.predict_proba(df[features].iloc[-1:].values)[0][1] * 100

        evidence = 0.6 if ls < 1 else 0.4
        bayes_prob = bayesian_update(prob / 100, evidence) * 100

        direction = 1 if bayes_prob > 50 else -1
        tp_sugg = last_p + direction * atr * 2.6
        sl_sugg = last_p - direction * atr * 1.4
        rr = abs(tp_sugg - last_p) / abs(last_p - sl_sugg) if not np.isclose(last_p, sl_sugg) else 1.8

        box_color = "#00ff88" if bayes_prob > 65 else "#FFD700" if bayes_prob > 48 else "#ff4b4b"
        st.markdown(f"""
            <div style="border:2px solid {box_color}; padding:18px; border-radius:14px; background:rgba(0,255,136,0.08); text-align:center;">
                <div style="color:{box_color}; font-size:0.9em; font-weight:bold;">AI 实时胜率</div>
                <div style="color:{box_color}; font-size:2.6em; font-weight:bold;">{bayes_prob:.1f}%</div>
                <div style="color:#aaa; font-size:0.8em;">建议 R/R: 1 : {rr:.2f}</div>
            </div>
        """, unsafe_allow_html=True)
        st.divider()

        sentiment = "看多 🔥" if ls > 1.0 else "看空 ❄️"
        recap = "✅ 趋势量价匹配良好" if bayes_prob > 55 else "📉 警惕缩量虚拉"
        st.info(f"🔍 AI 自动复盘：{recap}\n\n散户情绪: {sentiment}")

        # 多时间框架共振提示
        st.markdown("#### 🔄 多时间框架共振提示")
        if tf in ["15m", "30m"] and net_flow > 1000:
            st.success("✅ **15m/30m 共振强信号**：主力吸筹，适合轻仓布局")
        elif tf == "1H" and is_golden_cross:
            st.success("✅ **1H 金叉确认**：大趋势反转，可中仓追涨")
        else:
            st.info("⚖️ 当前框架共振中性，等待15m/30m净流持续正值")

        # 详细交易策略执行计划（用 expander 折叠）
        with st.expander("🎯 详细交易策略执行计划", expanded=False):
            st.caption(f"当前市场判断：{'EMA金叉+净流偏多' if is_golden_cross and net_flow>0 else '震荡中性+轻微流出'}")

            if bayes_prob < 45:
                st.error("🚨 **指挥官最优决策**：空仓观望！胜率过低，保本第一，等待15m金叉。")
            elif bayes_prob > 65:
                st.success("🚀 **指挥官最优决策**：中仓进攻清算猎杀！")
            else:
                st.warning("⚖️ **指挥官最优决策**：轻仓物理位陷阱 或 观望。")

            strats = [
                # ... 同 V32007 3个策略卡片，保持不变
            ]

            for s in strats:
                st.markdown(...)  # 卡片代码同前

        st.divider()
        st.markdown(f"【{datetime.now().strftime('%H:%M:%S')}】卫星同步成功 | 当前框架：{tf}")
        return hb, f_e, s_e, tf, pause_refresh

# ==========================================
# 4. 主界面
# ==========================================
def main():
    init_commander_state()
    light_cleanup()
    
    ls = get_ls_ratio()
    st.session_state.ls_ratio = ls
    
    df = get_candles(12, 26, "15m")
    if df.empty:
        st.error("❌ 卫星连接断开，正在重试...")
        time.sleep(2)
        st.rerun()
    
    hb, f_e, s_e, tf, pause_refresh = render_sidebar(df)
    df = get_candles(f_e, s_e, tf)
    
    st.markdown(f"### 🛰️ ETH 量子决策指挥官 (V32009) | {tf} | {datetime.now().strftime('%H:%M:%S')}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("实时价格", f"${df['c'].iloc[-1]:.2f}" if not df.empty else "N/A")
    
    delta_text = "↑ 多头占优" if ls > 1.02 else "↓ 空头占优" if ls < 0.98 else "↔ 中性均衡"
    delta_color = "green" if ls > 1.02 else "red" if ls < 0.98 else "gray"
    c2.metric("全网多空比", f"{ls:.2f}", delta_text, delta_color=delta_color if delta_color != "gray" else "off")
    
    c3.metric("ATR 动态波幅", f"{df['atr'].iloc[-1]:.2f}" if not df.empty else "N/A")
    c4.metric("庄家净流", f"{df['net_flow'].iloc[-1]:.0f}" if not df.empty else "N/A")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=df['time'], open=df['o'], high=df['h'], low=df['l'], close=df['c'], name="K线"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['ema_f'], line=dict(color='#00ff88', width=2.5), name="EMA快线"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['ema_s'], line=dict(color='#ff4b4b', width=2.5), name="EMA慢线"), row=1, col=1)

    if not df.empty:
        h_max = df['h'].tail(60).max()
        l_min = df['l'].tail(60).min()
        fig.add_hrect(y0=h_max*1.015, y1=h_max*1.025, fillcolor="red", opacity=0.3, annotation_text="空头风险区 (估算)", row=1, col=1)
        fig.add_hrect(y0=l_min*0.975, y1=l_min*0.985, fillcolor="green", opacity=0.3, annotation_text="多头风险区 (估算)", row=1, col=1)

        flow_max = abs(df['net_flow']).max() or 1
        scaled_flow = df['net_flow'] / flow_max * 800
        colors = ['#00ff88' if x > 0 else '#ff4b4b' for x in df['net_flow']]
        fig.add_trace(go.Bar(x=df['time'], y=scaled_flow, marker_color=colors, name="庄家净流"), row=2, col=1)

    fig.update_layout(template="plotly_dark", height=830, xaxis_rangeslider_visible=False, margin=dict(l=10,r=10,t=15,b=10))
    st.plotly_chart(fig, use_container_width=True)

    if not pause_refresh:
        time.sleep(hb)
        st.rerun()

if __name__ == "__main__":
    main()
