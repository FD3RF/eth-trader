import streamlit as st
import pandas as pd
import ccxt
import plotly.graph_objects as go
from datetime import datetime
import time

# --- 1. 系统核心配置与状态锁 ---
st.set_page_config(page_title="ETH AI 智能播报系统", layout="wide")

# 初始化信号记忆锁：记录 (K线时间戳, 信号类型)
if 'signal_memory' not in st.session_state:
    st.session_state.signal_memory = {"ts": None, "action": None}

def ai_voice_broadcast(text, k_ts):
    """
    精准播报控制：同一根K线下的同一动作只播报一次
    """
    if st.session_state.signal_memory["ts"] == k_ts and st.session_state.signal_memory["action"] == text:
        return # 拦截重复播报
    
    js_code = f"""
    <script>
    var msg = new SpeechSynthesisUtterance("{text}");
    msg.lang = 'zh-CN';
    msg.rate = 1.15; // 优化语速，更具实战感
    window.speechSynthesis.speak(msg);
    </script>
    """
    st.components.v1.html(js_code, height=0)
    st.session_state.signal_memory = {"ts": k_ts, "action": text}

# --- 2. 真实数据接入 (OKX API) ---
@st.cache_resource
def init_exchange():
    return ccxt.okx({
        'apiKey': 'a2a2a452-49e6-4e76-95f3-fb54eb982e7b',
        'secret': '330FABB2CAD3585677716686C2BF3872',
        'password': '123321aA@',
        'enableRateLimit': True,
    })

def fetch_market_data():
    exchange = init_exchange()
    try:
        # 获取100条数据，计算20均量和30支撑压力
        bars = exchange.fetch_ohlcv('ETH/USDT:USDT', timeframe='5m', limit=100)
        df = pd.DataFrame(bars, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
        df['ts_dt'] = pd.to_datetime(df['ts'], unit='ms')
        return df
    except Exception as e:
        st.error(f"网络异常，自动重连中... {e}")
        return pd.DataFrame()

# --- 3. 核心引擎：精准口诀计算 ---
def ai_logic_engine(df):
    curr = df.iloc[-1]
    price = curr['close']
    
    # 算法排查：使用前20根已收盘K线的均量，避免当前跳动量能干扰
    avg_vol = df['vol'].iloc[-21:-1].mean()
    vol_ratio = curr['vol'] / avg_vol
    
    # 支撑压力：取前30根K线波段，加入0.02%的缓冲区
    res_line = df['high'].iloc[-30:-1].max()
    sup_line = df['low'].iloc[-30:-1].min()
    
    # 盈亏比预计算
    long_r = (res_line - price) / (price - sup_line) if (price - sup_line) > 0.1 else 0
    short_r = (price - sup_line) / (res_line - price) if (res_line - price) > 0.1 else 0
    
    status = {"action": "AI 扫描中", "motto": "缩量是提醒，放量是信号", "color": "#121212", "voice": "", "tri": None}

    # --- 精准口诀对齐判定 ---
    
    # A. 做多系列
    if vol_ratio < 0.5 and price <= sup_line * 1.005 and curr['close'] < curr['open']:
        status.update({"action": "准备动手(多)", "motto": "缩量回踩，低点不破", "color": "#0D47A1", "voice": "缩量回踩，低点不破，准备动手"})
    elif vol_ratio > 1.6 and price > res_line:
        status.update({"action": "直接开多", "motto": "放量起涨，突破前高", "color": "#1B5E20", "voice": "放量起涨，突破前高，直接开多", "tri": "buy"})
    elif vol_ratio > 2.2 and curr['low'] <= sup_line and price > curr['low'] * 1.001:
        status.update({"action": "机会点(多)", "motto": "放量急跌，底部不破", "color": "#006064", "voice": "放量急跌，底部不破，这是机会", "tri": "buy"})
    
    # B. 做空系列
    elif vol_ratio < 0.5 and price >= res_line * 0.995 and curr['close'] > curr['open']:
        status.update({"action": "准备动手(空)", "motto": "缩量反弹，高点不破", "color": "#E65100", "voice": "缩量反弹，高点不破，准备动手"})
    elif vol_ratio > 1.6 and price < sup_line:
        status.update({"action": "直接开空", "motto": "放量下跌，跌破前低", "color": "#B71C1C", "voice": "放量下跌，跌破前低，直接开空", "tri": "sell"})
    elif vol_ratio > 2.2 and curr['high'] >= res_line and price < curr['high'] * 0.999:
        status.update({"action": "机会点(空)", "motto": "放量急涨，顶部不破", "color": "#4A148C", "voice": "放量急涨，顶部不破，这是机会", "tri": "sell"})

    return status, vol_ratio, res_line, sup_line, long_r, short_r

# --- 4. UI 界面 ---
st.markdown("<h1 style='text-align: center; color: #FFD700;'>🛡️ ETH AI 智能播报系统</h1>", unsafe_allow_html=True)

main_view = st.empty()

while True:
    df = fetch_market_data()
    if not df.empty:
        status, vr, res, sup, lr, sr = ai_logic_engine(df)
        k_ts = df['ts'].iloc[-1]
        
        with main_view.container():
            # AI 视觉看板
            st.markdown(f"""
                <div style="background-color:{status['color']}; padding:25px; border-radius:15px; text-align:center; border: 4px solid #FFD700; box-shadow: 0px 0px 20px {status['color']};">
                    <h1 style="color:white; font-size:55px; margin:0; letter-spacing:2px;">{status['action']}</h1>
                    <h2 style="color:#FFD700; margin-top:10px;">“{status['motto']}”</h2>
                    
                    <div style="display: flex; justify-content: space-around; margin-top: 20px; padding: 10px; background: rgba(255,255,255,0.05); border-radius: 10px;">
                        <div>
                            <div style="font-size: 12px; color: #aaa;">做多盈亏比</div>
                            <div style="font-size: 26px; font-weight: bold; color: {'#00FF00' if lr >= 1.5 else '#666'};">{lr:.2f}</div>
                        </div>
                        <div style="width: 1px; background: #444;"></div>
                        <div>
                            <div style="font-size: 12px; color: #aaa;">做空盈亏比</div>
                            <div style="font-size: 26px; font-weight: bold; color: {'#FF3D00' if sr >= 1.5 else '#666'};">{sr:.2f}</div>
                        </div>
                    </div>
                    <p style="color:#888; font-size:14px; margin-top:10px;">量比: {vr:.2f}x | 支撑: {sup} | 压力: {res}</p>
                </div>
            """, unsafe_allow_html=True)

            # 触发 AI 播报 (含状态锁)
            if status['voice']:
                ai_voice_broadcast(status['voice'], k_ts)

            # 绘制专业级K线图
            
            fig = go.Figure(data=[go.Candlestick(
                x=df['ts_dt'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
                increasing_line_color='#00ff88', decreasing_line_color='#ff3344', name="ETH/USDT"
            )])
            fig.add_hline(y=res, line_dash="dash", line_color="#FF00FF", opacity=0.4, annotation_text="AI压力位")
            fig.add_hline(y=sup, line_dash="dash", line_color="#00FFFF", opacity=0.4, annotation_text="AI支撑位")
            
            if status['tri'] == "buy":
                fig.add_trace(go.Scatter(x=[df['ts_dt'].iloc[-1]], y=[df['low'].iloc[-1]*0.998], mode="markers", marker=dict(symbol="triangle-up", size=20, color="#00FF00"), name="入场多"))
            elif status['tri'] == "sell":
                fig.add_trace(go.Scatter(x=[df['ts_dt'].iloc[-1]], y=[df['high'].iloc[-1]*1.002], mode="markers", marker=dict(symbol="triangle-down", size=20, color="#FF0000"), name="入场空"))

            fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig, use_container_width=True)
            
            st.caption(f"系统运行稳定 | 刷新时间: {datetime.now().strftime('%H:%M:%S')} | 5M周期监测中")

    time.sleep(8) # 8秒刷新一次，平衡实时性与稳定性
    st.rerun()
