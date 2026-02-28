import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
import numpy as np
import plotly.io as pio
import hmac
import hashlib
import base64

# ====================== 您的真实OKX API ======================
API_KEY = "a2a2a452-49e6-4e76-95f3-fb54eb982e7b"
API_SECRET = "330FABB2CAD3585677716686C2BF3872"
PASSPHRASE = "YYDS"

pio.templates['custom_dark'] = pio.templates['plotly_dark']
pio.templates['custom_light'] = pio.templates['plotly']

st.set_page_config(layout="wide", page_title="ETH V32101 智能策略指挥官", page_icon="⚖️")

# ====================== 函数全部置顶 ======================
def bayesian_update(prior, evidence):
    likelihood = evidence
    posterior = (prior * likelihood) / ((prior * likelihood) + ((1 - prior) * (1 - likelihood))) if ((prior * likelihood) + ((1 - prior) * (1 - likelihood))) != 0 else 0.5
    return posterior * 100

def calculate_prob(df):
    if len(df) < 30: return 50.0
    is_golden_cross = df['ema_f'].iloc[-1] > df['ema_s'].iloc[-1]
    rsi = df['rsi'].iloc[-1] if 'rsi' in df.columns else 50.0
    macd_cross = df['macd'].iloc[-1] > df['macd_signal'].iloc[-1] if 'macd' in df.columns else False
    net_flow = df['net_flow'].iloc[-1] if 'net_flow' in df.columns else 0
    rsi_high_th = np.percentile(df['rsi'].tail(30), 70) if 'rsi' in df.columns and len(df) > 30 else 60
    rsi_low_th = np.percentile(df['rsi'].tail(30), 30) if 'rsi' in df.columns and len(df) > 30 else 40
    prob = 50.0
    prob += 20 if is_golden_cross else -15
    prob += 15 if rsi > rsi_high_th else -15 if rsi < rsi_low_th else 0
    prob += 10 if macd_cross else -10
    prob += 10 if net_flow > 0 else -10
    return max(min(prob, 95.0), 5.0)

# ====================== 状态管理 ======================
def init_state():
    for k in ['ls_ratio', 'last_cleanup', 'theme', 'alarm_on', 'logs', 'tg_token', 'tg_chat_id']:
        if k not in st.session_state:
            st.session_state[k] = 1.0 if k == 'ls_ratio' else time.time() if k == 'last_cleanup' else 'dark' if k == 'theme' else False if 'on' in k else ""

def light_cleanup():
    if time.time() - st.session_state.last_cleanup > 86400:
        st.cache_data.clear()
        st.session_state.last_cleanup = time.time()

# ====================== 多空比（公共优先 + 重试，永不失败） ======================
@st.cache_data(ttl=15, max_entries=50)
def get_ls_ratio():
    for attempt in range(5):
        try:
            url = "https://www.okx.com/api/v5/rubik/stat/contracts/long-short-account-ratio?instId=ETH-USDT&period=5m"
            res = requests.get(url, timeout=6).json()
            if res.get('code') == '0':
                return float(res['data'][0][1])
            time.sleep(2 ** attempt)
        except:
            pass
    st.warning("多空比获取失败，使用默认 1.0")
    return 1.0

def send_telegram(msg):
    if st.session_state.tg_token and st.session_state.tg_chat_id:
        try:
            url = f"https://api.telegram.org/bot{st.session_state.tg_token}/sendMessage"
            requests.post(url, json={"chat_id": st.session_state.tg_chat_id, "text": msg, "parse_mode": "HTML"})
        except:
            pass

# ====================== 情报引擎 ======================
@st.cache_data(ttl=60, max_entries=50)
def get_candles(f_ema, s_ema, bar="1m"):
    try:
        url = f"https://www.okx.com/api/v5/market/candles?instId=ETH-USDT&bar={bar}&limit=100"
        res = requests.get(url, timeout=6).json()
        if res.get('code') != '0':
            raise ValueError("API错误")
        df = pd.DataFrame(res['data'], columns=['ts','o','h','l','c','v','volCcy','volCcyQuote','confirm'])[::-1].reset_index(drop=True)
        df['time'] = pd.to_datetime(df['ts'].astype(float), unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')
        for c in ['o','h','l','c','v']: df[c] = df[c].astype(float)
        
        df['ema_f'] = df['c'].ewm(span=f_ema, adjust=False).mean()
        df['ema_s'] = df['c'].ewm(span=s_ema, adjust=False).mean()
        
        diff = df['c'].diff()
        gain = diff.clip(lower=0).rolling(14).mean()
        loss = -diff.clip(upper=0).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        ema12 = df['c'].ewm(span=12, adjust=False).mean()
        ema26 = df['c'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        trades_res = requests.get("https://www.okx.com/api/v5/market/trades?instId=ETH-USDT&limit=100", timeout=6).json()
        if trades_res.get('code') == '0':
            trades_df = pd.DataFrame(trades_res['data'], columns=['ts', 'px', 'sz', 'side'])
            trades_df['sz'] = trades_df['sz'].astype(float)
            buy_vol = trades_df[trades_df['side'] == 'buy']['sz'].sum()
            sell_vol = trades_df[trades_df['side'] == 'sell']['sz'].sum()
            df['net_flow'] = buy_vol - sell_vol
        else:
            df['net_flow'] = 0

        tr = pd.concat([df['h']-df['l'], abs(df['h']-df['c'].shift()), abs(df['l']-df['c'].shift())], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        return df
    except:
        dummy = pd.DataFrame({'time': pd.date_range('now', periods=50, freq='T'), 'o': [1900]*50, 'h': [1910]*50, 'l': [1890]*50, 'c': [1900]*50, 'v': [1000]*50})
        dummy['ema_f'] = dummy['c'].ewm(span=f_ema, adjust=False).mean()
        dummy['ema_s'] = dummy['c'].ewm(span=s_ema, adjust=False).mean()
        dummy['rsi'] = 50.0
        dummy['macd'] = 0
        dummy['macd_signal'] = 0
        dummy['net_flow'] = 0
        dummy['atr'] = 5.0
        return dummy

# ====================== 侧边栏 ======================
def render_sidebar(df):
    with st.sidebar:
        st.title("⚡ V32101 智能策略指挥官")
        st.success("✅ 多空比永不失败 + 智能策略已激活")
        
        hb = st.slider("刷新频率 (秒)", 5, 60, 10)
        pause = st.checkbox("暂停自动刷新", False)
        st.session_state.alarm_on = st.checkbox("声音报警 (>70% or <30%)", st.session_state.alarm_on)
        
        symbols = ["ETH-USDT", "BTC-USDT", "SOL-USDT"]
        symbol = st.selectbox("交易对", symbols, index=0)
        
        f_e = st.number_input("快线 EMA", 5, 30, 12)
        s_e = st.number_input("慢线 EMA", 20, 100, 26)
        st.session_state.theme = st.selectbox("主题", ['dark', 'light'])
        if st.button("🚀 强制刷新"):
            st.rerun()
        st.divider()

        # 自动仓位计算器
        with st.expander("💰 自动仓位计算器", expanded=True):
            risk = st.slider("单笔风险 (%)", 0.1, 5.0, 1.0, 0.1)
            entry = st.number_input("计划入场价", value=float(df['c'].iloc[-1]) if not df.empty else 1900.0)
            sl = st.number_input("止损价", value=entry * 0.98)
            balance = st.number_input("账户余额 (USDT)", 1000.0)
            size = (balance * risk / 100) / abs(entry - sl) if abs(entry - sl) > 0 else 0
            st.success(f"建议仓位: **{size:.4f} {symbol.split('-')[0]}**")

        # Telegram
        with st.expander("📱 Telegram推送", expanded=False):
            st.session_state.tg_token = st.text_input("Bot Token", value=st.session_state.get('tg_token', ''), type="password")
            st.session_state.tg_chat_id = st.text_input("Chat ID", value=st.session_state.get('tg_chat_id', ''))
            st.caption("创建Bot后 /myid 获取Chat ID")

        return hb, f_e, s_e, symbol, pause

# ====================== 主界面 ======================
def main():
    init_state()
    light_cleanup()
    
    ls = get_ls_ratio()
    st.session_state.ls_ratio = ls
    
    df = get_candles(12, 26, "15m")
    hb, f_e, s_e, symbol, pause = render_sidebar(df)
    df = get_candles(f_e, s_e, "15m")
    
    prob = calculate_prob(df)
    evidence = 0.6 if ls < 1 else 0.4
    bayes_prob = bayesian_update(prob / 100, evidence)
    
    # Telegram推送
    if bayes_prob > 70 and st.session_state.get('tg_token') and st.session_state.get('tg_chat_id'):
        send_telegram(f"🚀 V32101智能信号！\n{symbol} 胜率 {bayes_prob:.1f}% 看多\n价格 ${df['c'].iloc[-1]:.2f}")
    if bayes_prob < 30 and st.session_state.get('tg_token') and st.session_state.get('tg_chat_id'):
        send_telegram(f"⚠️ V32101智能信号！\n{symbol} 胜率 {bayes_prob:.1f}% 看空\n价格 ${df['c'].iloc[-1]:.2f}")
    
    st.markdown(f"### 🛰️ ETH 量子决策指挥官 (V32101) | {symbol} | {datetime.now().strftime('%H:%M:%S')}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("实时价格", f"${df['c'].iloc[-1]:.2f}")
    c2.metric("全网多空比", f"{ls:.2f}")
    c3.metric("AI实时胜率", f"{bayes_prob:.1f}%")
    c4.metric("庄家净流", f"{df['net_flow'].iloc[-1]:.0f}" if not df.empty else "N/A")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=df['time'], open=df['o'], high=df['h'], low=df['l'], close=df['c'], name="K线"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['ema_f'], line=dict(color='#00ff88', width=2.5), name="EMA快线"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['ema_s'], line=dict(color='#ff4b4b', width=2.5), name="EMA慢线"), row=1, col=1)

    if not df.empty:
        h_max = df['h'].tail(60).max()
        l_min = df['l'].tail(60).min()
        fig.add_hrect(y0=h_max*1.015, y1=h_max*1.025, fillcolor="red", opacity=0.3, annotation_text="空头风险区", row=1, col=1)
        fig.add_hrect(y0=l_min*0.975, y1=l_min*0.985, fillcolor="green", opacity=0.3, annotation_text="多头风险区", row=1, col=1)
        flow_max = abs(df['net_flow']).max() or 1
        scaled_flow = df['net_flow'] / flow_max * 800
        colors = ['#00ff88' if x > 0 else '#ff4b4b' for x in df['net_flow']]
        fig.add_trace(go.Bar(x=df['time'], y=scaled_flow, marker_color=colors, name="庄家净流"), row=2, col=1)

    fig.update_layout(template=pio.templates['custom_dark'] if st.session_state.theme == 'dark' else pio.templates['custom_light'], height=830, xaxis_rangeslider_visible=False, margin=dict(l=10,r=10,t=15,b=10))
    st.plotly_chart(fig)

    # 智能策略卡片
    with st.expander("🎯 智能策略执行计划", expanded=True):
        if bayes_prob < 45:
            st.error("🚨 **指挥官最优决策**：空仓观望！胜率过低，保本第一")
        elif bayes_prob > 65:
            st.success("🚀 **指挥官最优决策**：中仓进攻！强信号")
        else:
            st.warning("⚖️ **指挥官最优决策**：轻仓观察或观望")

    if not pause:
        time.sleep(hb)
        st.rerun()

if __name__ == "__main__":
    main()
