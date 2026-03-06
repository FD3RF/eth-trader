import streamlit as st
import pandas as pd
import ccxt
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# --- 1. 系统状态锁与内存优化 ---
st.set_page_config(page_title="ETH AI 智能播报系统", layout="wide")

# 信号锁：记录 (类型, 时间戳)
if 'signal_memory' not in st.session_state:
    st.session_state.signal_memory = {"last_key": None}
# 极值播报频率限制（防止在极值点反复喊话，设置5分钟冷却）
if 'extreme_cooldown' not in st.session_state:
    st.session_state.extreme_cooldown = datetime.now()

def ai_voice_broadcast(text, k_ts, priority=False):
    """
    智能语音引擎
    priority: 是否忽略冷却（如放量突破信号）
    """
    current_key = f"{text}_{k_ts}"
    if not priority and st.session_state.signal_memory.get("last_key") == current_key:
        return 
    
    # 极值冷却逻辑
    if "24H" in text:
        if datetime.now() < st.session_state.extreme_cooldown:
            return
        st.session_state.extreme_cooldown = datetime.now() + timedelta(minutes=5)

    js_code = f"""
    <script>
    var msg = new SpeechSynthesisUtterance("{text}");
    msg.lang = 'zh-CN';
    msg.rate = 1.15;
    window.speechSynthesis.speak(msg);
    </script>
    """
    st.components.v1.html(js_code, height=0)
    st.session_state.signal_memory["last_key"] = current_key

# --- 2. 数据获取 (免授权公共模式) ---
@st.cache_resource
def init_exchange():
    return ccxt.okx({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})

def fetch_safe_data():
    exchange = init_exchange()
    try:
        # 获取 K 线用于口诀计算 (5m周期)
        bars = exchange.fetch_ohlcv('ETH/USDT:USDT', timeframe='5m', limit=100)
        # 获取 24H 聚合数据
        ticker = exchange.fetch_ticker('ETH/USDT:USDT')
        
        df = pd.DataFrame(bars, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
        df['ts_dt'] = pd.to_datetime(df['ts'], unit='ms')
        return df, ticker
    except Exception as e:
        st.error(f"网络同步中... {e}")
        return pd.DataFrame(), None

# --- 3. 核心引擎：八大口诀与极值联动 ---
def ai_logic_engine(df, ticker):
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    price = curr['close']
    k_ts = df['ts'].iloc[-1]
    
    # 指标精准排查：使用前20根均量（已固定量）作为参考
    avg_vol_20 = df['vol'].iloc[-21:-1].mean()
    vol_ratio = curr['vol'] / avg_vol_20
    
    # 支撑压力判定 (30周期)
    res_30 = df['high'].iloc[-31:-1].max()
    sup_30 = df['low'].iloc[-31:-1].min()
    
    # 24H 极值数据
    h24 = ticker['high']
    l24 = ticker['low']
    
    # 盈亏比计算 (Risk/Reward)
    long_r = (res_30 - price) / (price - sup_30) if (price - sup_30) > 0.1 else 0
    short_r = (price - sup_30) / (res_30 - price) if (res_30 - price) > 0.1 else 0
    
    status = {"action": "AI 扫描中", "motto": "静如处子，动如脱兔", "color": "#121212", "voice": "", "tri": None}

    # --- 排查口诀：优先级判定 ---
    
    # 1. 24H 极值播报 (警示最高优先级)
    if price >= h24 * 0.9999:
        status.update({"action": "24H新高附近", "motto": "极值区域，注意放量", "color": "#FFD700", "voice": "注意，价格触及二十四小时最高点，谨防冲高回落"})
    elif price <= l24 * 1.0001:
        status.update({"action": "24H新低附近", "motto": "极值区域，观察支撑", "color": "#D50000", "voice": "注意，价格触及二十四小时最低点，观察支撑强度"})
    
    # 2. 核心成交量口诀判定
    else:
        # 做多：缩量回踩不破底 / 放量突破前高 / 放量插针
        if vol_ratio < 0.5 and price <= sup_30 * 1.002 and price < curr['open']:
            status.update({"action": "准备动手(多)", "motto": "缩量回踩，低点不破", "color": "#0D47A1", "voice": "缩量回踩，低点不破，准备动手"})
        elif vol_ratio > 1.6 and price > res_30:
            status.update({"action": "直接开多", "motto": "放量起涨，突破前高", "color": "#1B5E20", "voice": "放量起涨，突破前高，直接开多", "tri": "buy"})
        elif vol_ratio > 2.2 and curr['low'] <= sup_30 and price > curr['low'] * 1.001:
            status.update({"action": "机会点(多)", "motto": "放量急跌，底部不破", "color": "#006064", "voice": "放量急跌，底部不破，这是机会", "tri": "buy"})
        
        # 做空：缩量反弹不过顶 / 放量跌破前低 / 放量急涨
        elif vol_ratio < 0.5 and price >= res_30 * 0.998 and price > curr['open']:
            status.update({"action": "准备动手(空)", "motto": "缩量反弹，高点不破", "color": "#E65100", "voice": "缩量反弹，高点不破，准备动手"})
        elif vol_ratio > 1.6 and price < sup_30:
            status.update({"action": "直接开空", "motto": "放量下跌，跌破前低", "color": "#B71C1C", "voice": "放量下跌，跌破前低，直接开空", "tri": "sell"})

    return status, vol_ratio, res_30, sup_30, long_r, short_r, h24, l24

# --- 4. 界面渲染 ---
st.markdown("<h1 style='text-align: center; color: #FFD700;'>🛡️ ETH AI 智能播报系统</h1>", unsafe_allow_html=True)

container = st.empty()

while True:
    df, ticker = fetch_safe_data()
    if not df.empty and ticker:
        status, vr, res, sup, lr, sr, h24, l24 = ai_logic_engine(df, ticker)
        k_ts = df['ts'].iloc[-1]
        
        with container.container():
            # AI 播报看板
            st.markdown(f"""
                <div style="background-color:{status['color']}; padding:25px; border-radius:15px; text-align:center; border: 4px solid #FFD700; box-shadow: 0px 0px 20px {status['color']};">
                    <h1 style="color:white; font-size:55px; margin:0;">{status['action']}</h1>
                    <h2 style="color:#FFD700; margin-top:10px;">“{status['motto']}”</h2>
                    
                    <div style="display: flex; justify-content: space-around; margin-top: 20px; background: rgba(0,0,0,0.3); border-radius: 10px; padding: 10px;">
                        <div>
                            <div style="font-size: 11px; color: #aaa;">24H 最高价</div>
                            <div style="font-size: 20px; font-weight: bold; color: #FFD700;">{h24}</div>
                        </div>
                        <div style="width: 1px; background: #444;"></div>
                        <div>
                            <div style="font-size: 11px; color: #aaa;">24H 最低价</div>
                            <div style="font-size: 20px; font-weight: bold; color: #ff5252;">{l24}</div>
                        </div>
                    </div>

                    <div style="display: flex; justify-content: space-around; margin-top: 15px;">
                        <div>
                            <div style="font-size: 11px; color: #aaa;">做多盈亏比</div>
                            <div style="font-size: 26px; font-weight: bold; color: {'#00FF00' if lr >= 1.5 else '#666'};">{lr:.2f}</div>
                        </div>
                        <div>
                            <div style="font-size: 11px; color: #aaa;">做空盈亏比</div>
                            <div style="font-size: 26px; font-weight: bold; color: {'#FF3D00' if sr >= 1.5 else '#666'};">{sr:.2f}</div>
                        </div>
                    </div>
                    <p style="color:#888; font-size:14px; margin-top:10px;">量比: {vr:.2f}x | 5M压力: {res} | 5M支撑: {sup}</p>
                </div>
            """, unsafe_allow_html=True)

            # 触发 AI 播报
            if status['voice']:
                ai_voice_broadcast(status['voice'], k_ts)

            # 绘制 K 线
            
            fig = go.Figure(data=[go.Candlestick(
                x=df['ts_dt'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
                increasing_line_color='#00ff88', decreasing_line_color='#ff3344', name="ETH"
            )])
            fig.add_hline(y=h24, line_dash="dash", line_color="#FFD700", opacity=0.3, annotation_text="24H高点")
            fig.add_hline(y=l24, line_dash="dash", line_color="#ff5252", opacity=0.3, annotation_text="24H低点")
            fig.add_hline(y=res, line_dash="dot", line_color="#FF00FF", opacity=0.5, annotation_text="5M压力")
            fig.add_hline(y=sup, line_dash="dot", line_color="#00FFFF", opacity=0.5, annotation_text="5M支撑")
            
            if status['tri'] == "buy":
                fig.add_trace(go.Scatter(x=[df['ts_dt'].iloc[-1]], y=[df['low'].iloc[-1]*0.998], mode="markers", marker=dict(symbol="triangle-up", size=20, color="#00FF00")))
            elif status['tri'] == "sell":
                fig.add_trace(go.Scatter(x=[df['ts_dt'].iloc[-1]], y=[df['high'].iloc[-1]*1.002], mode="markers", marker=dict(symbol="triangle-down", size=20, color="#FF0000")))

            fig.update_layout(template="plotly_dark", height=450, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig, use_container_width=True)
            
            st.caption(f"系统运行稳定 | 极值冷却监测中 | 刷新: {datetime.now().strftime('%H:%M:%S')}")

    time.sleep(8)
    st.rerun()
