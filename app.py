import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
import numpy as np
import hmac
import base64
import hashlib
from urllib.parse import urlencode
import json
import os

st.set_page_config(layout="wide", page_title="ETH V32009 终极指挥官", page_icon="⚖️")

# ==========================================
# OKX API 配置（使用您的密钥）
# ==========================================
API_KEY = "a2a2a452-49e6-4e76-95f3-fb54e98e2e7b"
SECRET_KEY = "330FABDB2CAD3585677716686C2BF382"
PASSPHRASE = "YYDS"
BASE_URL = "https://www.okx.com"

def okx_signed_request(method, endpoint, params=None):
    timestamp = datetime.utcnow().isoformat()[:-3] + 'Z'
    request_path = endpoint
    if params:
        request_path += '?' + urlencode(params)
    
    message = timestamp + method.upper() + request_path
    signature = hmac.new(SECRET_KEY.encode('utf-8'), message.encode('utf-8'), hashlib.sha256).digest()
    signature_b64 = base64.b64encode(signature).decode('utf-8')
    
    headers = {
        'OK-ACCESS-KEY': API_KEY,
        'OK-ACCESS-SIGN': signature_b64,
        'OK-ACCESS-TIMESTAMP': timestamp,
        'OK-ACCESS-PASSPHRASE': PASSPHRASE,
        'Content-Type': 'application/json'
    }
    return headers

# ==========================================
# 状态管理与内存治理
# ==========================================
def init_commander_state():
    if 'ls_ratio' not in st.session_state: st.session_state.ls_ratio = 1.0
    if 'last_cleanup' not in st.session_state: st.session_state.last_cleanup = time.time()
    if 'theme' not in st.session_state: st.session_state.theme = 'dark'
    if 'alarm_on' not in st.session_state: st.session_state.alarm_on = False

def light_cleanup():
    if time.time() - st.session_state.last_cleanup > 86400:
        st.cache_data.clear()
        st.session_state.last_cleanup = time.time()

# ==========================================
# 2. 情报引擎（使用真实 OKX API + RSI + MACD）
# ==========================================
@st.cache_data(ttl=60, max_entries=50)
def get_candles(f_ema, s_ema, bar="1m"):
    try:
        endpoint = "/api/v5/market/candles"
        params = {
            'instId': 'ETH-USDT',
            'bar': bar,
            'limit': '100'
        }
        headers = okx_signed_request('GET', endpoint, params)
        res = requests.get(BASE_URL + endpoint, headers=headers, params=params).json()
        if res.get('code') != '0':
            raise ValueError("API 返回错误: " + res.get('msg', '未知'))
        df = pd.DataFrame(res['data'], columns=['ts','o','h','l','c','v','volCcy','volCcyQuote','confirm'])[::-1].reset_index(drop=True)
        df['time'] = pd.to_datetime(df['ts'].astype(float), unit='ms')
        for c in ['o','h','l','c','v']: df[c] = df[c].astype(float)
        
        df['ema_f'] = df['c'].ewm(span=f_ema, adjust=False).mean()
        df['ema_s'] = df['c'].ewm(span=s_ema, adjust=False).mean()
        
        # RSI
        diff = df['c'].diff()
        gain = diff.clip(lower=0).rolling(14).mean()
        loss = -diff.clip(upper=0).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['c'].ewm(span=12, adjust=False).mean()
        ema26 = df['c'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # 净流（改进，用 RSI 调整）
        coeff = 0.2 if df['rsi'].iloc[-1] > 50 else 0.1
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
        endpoint = "/api/v5/rubik/stat/contracts/long-short-account-ratio"
        params = {'instId': 'ETH-USDT', 'period': '5m'}
        headers = okx_signed_request('GET', endpoint, params)
        res = requests.get(BASE_URL + endpoint, headers=headers, params=params).json()
        if res.get('code') != '0':
            raise ValueError("API 返回错误")
        return float(res['data'][0][1])
    except:
        st.error("多空比获取失败")
        return 1.0

# ==========================================
# 胜率计算（规则 + 动态阈值）
# ==========================================
def calculate_prob(df):
    if len(df) < 14:
        return 50.0
    
    is_golden_cross = df['ema_f'].iloc[-1] > df['ema_s'].iloc[-1]
    rsi = df['rsi'].iloc[-1]
    macd_cross = df['macd'].iloc[-1] > df['macd_signal'].iloc[-1]
    net_flow = df['net_flow'].iloc[-1]
    
    prob = 50.0
    prob += 20 if is_golden_cross else -15
    prob += 15 if rsi > 60 else -15 if rsi < 40 else 0
    prob += 10 if macd_cross else -10
    prob += 10 if net_flow > 0 else -10
    return max(min(prob, 95.0), 5.0)

# 贝叶斯更新
def bayesian_update(prior, evidence):
    likelihood = evidence
    posterior = (prior * likelihood) / ((prior * likelihood) + ((1 - prior) * (1 - likelihood))) if ((prior * likelihood) + ((1 - prior) * (1 - likelihood))) != 0 else 0.5
    return posterior * 100

# ==========================================
# 3. 侧边栏
# ==========================================
def render_sidebar(df):
    with st.sidebar:
        st.markdown("### 🛸 量子实时控制 V32009")
        st.warning("⚠️ 部分指标为模拟演示，请勿作为实盘唯一依据。")
        hb = st.slider("刷新频率 (秒)", 5, 60, 10)
        pause_refresh = st.checkbox("暂停自动刷新", value=False)
        st.session_state.alarm_on = st.checkbox("启用声音报警 (胜率>70% or <30%)", value=st.session_state.alarm_on)
        
        tf_options = ["1m", "5m", "15m", "30m", "1H", "4H"]
        tf = st.selectbox("时间框架", tf_options, index=2)
        
        f_e = st.number_input("快线 EMA", 5, 30, 12)
        s_e = st.number_input("慢线 EMA", 20, 100, 26)
        st.session_state.theme = st.selectbox("主题", ['dark', 'light'], index=0 if st.session_state.theme == 'dark' else 1)
        st.divider()

        if len(df) < 14:
            st.warning("数据不足，使用默认值")
            last_p = 0
            atr = 5.0
            is_golden_cross = False
            net_flow = 0
            rsi = 50.0
            macd = 0
            macd_signal = 0
        else:
            last_p = df['c'].iloc[-1]
            atr = df['atr'].iloc[-1] if not pd.isna(df['atr'].iloc[-1]) else 5.0
            is_golden_cross = df['ema_f'].iloc[-1] > df['ema_s'].iloc[-1]
            net_flow = df['net_flow'].iloc[-1]
            rsi = df['rsi'].iloc[-1]
            macd = df['macd'].iloc[-1]
            macd_signal = df['macd_signal'].iloc[-1]

        ls = st.session_state.ls_ratio

        prob = calculate_prob(df)
        evidence = 0.6 if ls < 1 else 0.4
        bayes_prob = bayesian_update(prob / 100, evidence)

        if st.session_state.alarm_on:
            if bayes_prob > 70 or bayes_prob < 30:
                st.markdown('<audio autoplay="true"><source src="https://www.w3schools.com/html/horse.mp3" type="audio/mpeg"></audio>', unsafe_allow_html=True)  # 替换为您的音频 URL

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
                    "state": "🔥 进攻中" if bayes_prob > 65 else "⚪ 待机",
                    "color": "#00ff88" if bayes_prob > 65 else "#ff4b4b",
                    "entry": f"突破近期高点（约{df['h'].tail(30).max():.1f}）",
                    "tp": f"${tp_sugg:.1f}",
                    "sl": f"${sl_sugg:.1f}",
                    "rr": f"1:{rr:.2f}",
                    "position": "中仓 40%" if bayes_prob > 65 else "观望",
                    "reason": f"胜率{bayes_prob:.0f}% + EMA金叉，主力净流倾向多头，适合猎杀空头止损。",
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
                    "position": "重仓 60%" if is_golden_cross and bayes_prob > 70 else "轻仓",
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
    
    st.markdown(f"### 🛰️ ETH 量子决策指挥官 (V32008) | {tf} | {datetime.now().strftime('%H:%M:%S')}")
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

    fig.update_layout(template=st.session_state.theme, height=830, xaxis_rangeslider_visible=False, margin=dict(l=10,r=10,t=15,b=10))
    st.plotly_chart(fig, use_container_width=True)

    if not pause_refresh:
        time.sleep(hb)
        st.rerun()

if __name__ == "__main__":
    main()
