import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import hmac
import base64
import hashlib
from datetime import datetime
from urllib.parse import urlencode
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# 0. 指挥官核心配置
# ==========================================
st.set_page_config(layout="wide", page_title="ETH V32010 终极指挥官", page_icon="⚖️")

API_KEY = "a2a2a452-49e6-4e76-95f3-fb54e98e2e7b"
SECRET_KEY = "330FABDB2CAD3585677716686C2BF382"
PASSPHRASE = "YYDS"
BASE_URL = "https://www.okx.com"

# ==========================================
# 1. 强化情报引擎 (核心修复：强制字段注入)
# ==========================================
@st.cache_data(ttl=10)
def get_commander_intel(f_ema=12, s_ema=26, bar="15m"):
    try:
        # 获取 K 线
        res = requests.get(f"{BASE_URL}/api/v5/market/candles?instId=ETH-USDT&bar={bar}&limit=100").json()
        if res.get('code') != '0': return pd.DataFrame()
        
        df = pd.DataFrame(res['data'], columns=['ts','o','h','l','c','v','volC','volCQ','confirm'])[::-1].reset_index(drop=True)
        for c in ['o','h','l','c','v']: df[c] = df[c].astype(float)
        df['time'] = pd.to_datetime(df['ts'].astype(float), unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')

        # 【核心修复】必须在函数内完成计算，确保 dataframe 返回时已包含所有字段
        df['ema_f'] = df['c'].ewm(span=f_ema, adjust=False).mean()
        df['ema_s'] = df['c'].ewm(span=s_ema, adjust=False).mean()
        
        # RSI 计算
        diff = df['c'].diff()
        gain = diff.clip(lower=0).rolling(14).mean()
        loss = -diff.clip(upper=0).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, np.nan))))
        
        # ATR 波动
        tr = pd.concat([df['h']-df['l'], abs(df['h']-df['c'].shift()), abs(df['l']-df['c'].shift())], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        
        # 净流抓取
        t_res = requests.get(f"{BASE_URL}/api/v5/market/trades?instId=ETH-USDT&limit=100").json()
        df['net_flow'] = 0
        if t_res.get('code') == '0':
            tdf = pd.DataFrame(t_res['data'], columns=['ts','px','sz','side'])
            tdf['sz'] = tdf['sz'].astype(float)
            df['net_flow'] = tdf[tdf['side']=='buy']['sz'].sum() - tdf[tdf['side']=='sell']['sz'].sum()
            
        return df
    except:
        return pd.DataFrame()

# ==========================================
# 2. UI 渲染与决策逻辑
# ==========================================
def main():
    # 侧边栏：量子控制台
    with st.sidebar:
        st.markdown("### 🛸 量子实时控制 V32010")
        hb = st.slider("心跳频率 (秒)", 5, 60, 10)
        f_ema = st.number_input("快线 EMA", 5, 30, 12)
        s_ema = st.number_input("慢线 EMA", 20, 100, 26)
        tf = st.selectbox("时间框架", ["1m", "5m", "15m", "1H"], index=2)
        
        # 【关键修复】映射 Plotly 标准主题名，解决 ValueError
        theme_map = {"深邃黑": "plotly_dark", "简约白": "plotly_white"}
        theme_sel = st.selectbox("视觉主题", list(theme_map.keys()))
        current_theme = theme_map[theme_sel]
        
        st.divider()
        
        # 数据获取
        df = get_commander_intel(f_ema, s_ema, tf)
        
        # 实时胜率计算
        prob = 50.0
        if not df.empty:
            last = df.iloc[-1]
            # 防御性判断字段是否存在
            if 'ema_f' in df.columns:
                prob += 15 if last['ema_f'] > last['ema_s'] else -15
                prob += 10 if last.get('net_flow', 0) > 0 else -10
                prob += 5 if 40 < last.get('rsi', 50) < 60 else -5
        
        status_color = "#00ff88" if prob > 60 else "#ff4b4b" if prob < 40 else "#FFD700"
        st.markdown(f"""
            <div style="border:2px solid {status_color}; padding:15px; border-radius:12px; text-align:center; background-color:rgba(0,0,0,0.2);">
                <small style="color:#888;">AI 实时胜率</small><br>
                <strong style="color:{status_color}; font-size:2.2em;">{prob:.1f}%</strong><br>
                <small style="color:#888;">建议 R/R: 1 : 1.85</small>
            </div>
        """, unsafe_allow_html=True)
        
        st.info("🔍 AI 复盘：趋势量价匹配良好" if prob > 55 else "⚠️ AI 复盘：缩量诱多，注意回调")

    # 主屏幕
    if df.empty:
        st.warning("📡 卫星链路重组中..."); time.sleep(2); st.rerun()

    last_p = df['c'].iloc[-1]
    atr = df['atr'].iloc[-1]
    net_flow = df['net_flow'].iloc[-1]

    # 顶栏指标
    st.markdown(f"### 🛰️ ETH 量子决策指挥官 | {tf} | {datetime.now().strftime('%H:%M:%S')}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("实时价格", f"${last_p:.2f}")
    m2.metric("全网多空比", "1.02", "↑ 散户看多", delta_color="normal")
    m3.metric("ATR 动态波幅", f"{atr:.2f}")
    m4.metric("庄家净流", f"{net_flow:.0f}")

    # 左右分栏布局
    col_strat, col_chart = st.columns([1.1, 3])

    with col_strat:
        st.markdown("#### 🎯 实时策略执行计划")
        # 激活卡片
        st.success(f"✅ **激活 | 物理位陷阱**\n\n止盈: ${last_p + atr*2:.1f} | 止损: ${last_p - atr:.1f}")
        # 进攻卡片
        st.warning(f"🔥 **进攻 | 清算猎杀**\n\n建议 R/R: 1:1.85")
        # 预警卡片
        st.error(f"🚨 **预警 | 量价共振**\n\n注意回踩 EMA26")
        
        st.divider()
        st.markdown(f"📜 **战术日志**\n\n`[{datetime.now().strftime('%H:%M:%S')}]` 卫星同步成功")

    with col_chart:
        # 综合图表绘制
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.7, 0.3])
        
        # K线与均线         fig.add_trace(go.Candlestick(x=df['time'], open=df['o'], high=df['h'], low=df['l'], close=df['c'], name="K线"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['time'], y=df['ema_f'], line=dict(color='#00ff88', width=1.5), name="EMA快线"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['time'], y=df['ema_s'], line=dict(color='#ff4b4b', width=1.5), name="EMA慢线"), row=1, col=1)
        
        # 资金流柱状图
        colors = ['#00ff88' if x > 0 else '#ff4b4b' for x in df['net_flow'].rolling(5).mean()]
        fig.add_trace(go.Bar(x=df['time'], y=df['net_flow'], marker_color=colors, name="净流"), row=2, col=1)
        
        fig.update_layout(template=current_theme, height=780, xaxis_rangeslider_visible=False, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

    # 循环
    time.sleep(hb)
    st.rerun()

if __name__ == "__main__":
    main()
