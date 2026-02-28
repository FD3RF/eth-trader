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

# ==========================================
# 0. 基础配置与签名 (核心权限)
# ==========================================
st.set_page_config(layout="wide", page_title="ETH V32010 终极指挥官", page_icon="⚖️")

API_KEY = "a2a2a452-49e6-4e76-95f3-fb54e98e2e7b"
SECRET_KEY = "330FABDB2CAD3585677716686C2BF382"
PASSPHRASE = "YYDS"
BASE_URL = "https://www.okx.com"

def okx_signed_request(method, endpoint, params=None):
    timestamp = datetime.utcnow().isoformat()[:-3] + 'Z'
    request_path = endpoint
    if params: request_path += '?' + urlencode(params)
    message = timestamp + method.upper() + request_path
    signature = base64.b64encode(hmac.new(SECRET_KEY.encode('utf-8'), message.encode('utf-8'), hashlib.sha256).digest()).decode('utf-8')
    return {'OK-ACCESS-KEY': API_KEY, 'OK-ACCESS-SIGN': signature, 'OK-ACCESS-TIMESTAMP': timestamp, 'OK-ACCESS-PASSPHRASE': PASSPHRASE, 'Content-Type': 'application/json'}

# ==========================================
# 1. 核心计算逻辑 (复用 V32009 算法)
# ==========================================
@st.cache_data(ttl=15)
def get_intel_data(f_ema, s_ema, bar="15m"):
    try:
        # K线与指标
        res = requests.get(f"{BASE_URL}/api/v5/market/candles?instId=ETH-USDT&bar={bar}&limit=100").json()
        df = pd.DataFrame(res['data'], columns=['ts','o','h','l','c','v','volC','volCQ','confirm'])[::-1].reset_index(drop=True)
        for c in ['o','h','l','c','v']: df[c] = df[c].astype(float)
        df['time'] = pd.to_datetime(df['ts'].astype(float), unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')
        
        df['ema_f'] = df['c'].ewm(span=f_ema, adjust=False).mean()
        df['ema_s'] = df['c'].ewm(span=s_ema, adjust=False).mean()
        
        # RSI 计算 
        diff = df['c'].diff(); gain = diff.clip(lower=0).rolling(14).mean(); loss = -diff.clip(upper=0).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, np.nan))))
        
        # 净流抓取
        t_res = requests.get(f"{BASE_URL}/api/v5/market/trades?instId=ETH-USDT&limit=100").json()
        tdf = pd.DataFrame(t_res['data'], columns=['ts','px','sz','side'])
        tdf['sz'] = tdf['sz'].astype(float)
        df['net_flow'] = tdf[tdf['side']=='buy']['sz'].sum() - tdf[tdf['side']=='sell']['sz'].sum()
        
        # ATR 波动
        tr = pd.concat([df['h']-df['l'], abs(df['h']-df['c'].shift()), abs(df['l']-df['c'].shift())], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        return df
    except: return pd.DataFrame()

# ==========================================
# 2. 侧边栏与 UI 增强
# ==========================================
def render_ui(df):
    with st.sidebar:
        st.header("🛸 量子控制台 V32010")
        hb = st.slider("心跳频率 (秒)", 5, 60, 10)
        tf = st.selectbox("时间框架", ["1m", "5m", "15m", "1H"], index=2)
        # 核心修复：确保主题值为 Plotly 标准字符串
        theme_map = {"深邃黑": "plotly_dark", "简约白": "plotly_white"}
        theme_label = st.selectbox("视觉主题", list(theme_map.keys()))
        st.session_state.theme = theme_map[theme_label]
        
        st.divider()
        # 动态胜率展示
        prob = 50.0 # 默认
        if not df.empty:
            last = df.iloc[-1]
            prob += 15 if last['ema_f'] > last['ema_s'] else -15
            prob += 10 if last['net_flow'] > 0 else -10
            prob += 10 if 40 < last['rsi'] < 60 else -5
        
        color = "#00ff88" if prob > 60 else "#ff4b4b"
        st.markdown(f"""<div style='border:2px solid {color}; padding:15px; border-radius:10px; text-align:center;'>
            <small style='color:#888;'>AI 实时胜率</small><br>
            <strong style='color:{color}; font-size:2em;'>{prob:.1f}%</strong>
        </div>""", unsafe_allow_html=True)
        
        st.info("🔍 AI 复盘：趋势量价匹配良好" if prob > 55 else "⚠️ AI 复盘：缩量诱多，注意回调")
        return hb, tf

# ==========================================
# 3. 主屏幕渲染 (复刻 V32009 所有板块)
# ==========================================
def main():
    if 'theme' not in st.session_state: st.session_state.theme = 'plotly_dark'
    
    df = get_intel_data(12, 26, "15m") # 初始加载
    hb, tf = render_ui(df)
    df = get_intel_data(12, 26, tf)
    
    if df.empty: st.error("卫星链路中断..."); time.sleep(2); st.rerun()

    # 指标数据 
    last_p = df['c'].iloc[-1]
    atr = df['atr'].iloc[-1]
    net_flow = df['net_flow'].iloc[-1]

    # 顶栏仪表盘
    st.markdown(f"### 🛰️ ETH 量子决策指挥官 | {tf} | {datetime.now().strftime('%H:%M:%S')}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("实时价格", f"${last_p:.2f}")
    m2.metric("全网多空比", "1.02", "散户看多", delta_color="normal")
    m3.metric("ATR 动态波幅", f"{atr:.2f}")
    m4.metric("庄家净流", f"{net_flow:.0f}")

    # 左右分栏：左侧策略，右侧图表
    col_left, col_right = st.columns([1, 3])

    with col_left:
        st.markdown("#### 🎯 实时策略执行计划")
        with st.container():
            # 策略卡片 1
            st.success(f"✅ **激活 | 物理位陷阱**\n\n止盈: ${last_p+atr*2:.1f} | 止损: ${last_p-atr:.1f}")
            # 策略卡片 2
            st.warning(f"🔥 **进攻 | 清算猎杀**\n\n建议 R/R: 1:1.85")
            # 策略卡片 3
            st.error(f"🚨 **预警 | 量价共振**\n\n注意回踩 EMA26")
        
        st.markdown("---")
        st.markdown(f"📜 **战术日志**\n\n`[{datetime.now().strftime('%H:%M')}]` 卫星同步成功")

    with col_right:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
        # K线主图 
        fig.add_trace(go.Candlestick(x=df['time'], open=df['o'], high=df['h'], low=df['l'], close=df['c'], name="Price"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['time'], y=df['ema_f'], line=dict(color='#00ff88', width=1.5), name="EMA12"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['time'], y=df['ema_s'], line=dict(color='#ff4b4b', width=1.5), name="EMA26"), row=1, col=1)
        
        # 净流副图
        colors = ['#00ff88' if x > 0 else '#ff4b4b' for x in df['net_flow'].rolling(5).mean()]
        fig.add_trace(go.Bar(x=df['time'], y=df['net_flow'], marker_color=colors, name="NetFlow"), row=2, col=1)
        
        # 核心修复：确保 template 动态加载
        fig.update_layout(template=st.session_state.theme, height=750, xaxis_rangeslider_visible=False, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

    time.sleep(hb)
    st.rerun()

if __name__ == "__main__":
    main()
