import streamlit as st
import pandas as pd
import ccxt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from streamlit_autorefresh import st_autorefresh

# --- 1. 环境初始化 ---
st.set_page_config(page_title="ETH AI 终极盯盘", layout="wide")
st_autorefresh(interval=5000, key="eth_stable_refresh")

if "signal_memory" not in st.session_state:
    st.session_state.signal_memory = {"last_key": None, "last_time": 0}

# --- 2. 语音播报：防空值加固 ---
def safe_broadcast(text):
    """修复报错：通过 str() 强制转换，确保 text 永远不为 None"""
    if not text: return
    safe_text = str(text).replace('"', '').replace("'", "")
    components_code = f"""
    <script>
    (function() {{
        window.speechSynthesis.cancel();
        var msg = new SpeechSynthesisUtterance("{safe_text}");
        msg.lang='zh-CN'; msg.rate=1.2;
        window.speechSynthesis.speak(msg);
    }})();
    </script>
    """
    st.components.v1.html(components_code, height=0)

# --- 3. 数据获取：异常拦截 ---
@st.cache_resource
def get_exchange():
    return ccxt.okx({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})

def fetch_market_data():
    ex = get_exchange()
    try:
        # 增加 limit 确保计算分位数时有足够样本
        bars = ex.fetch_ohlcv('ETH/USDT:USDT', '5m', limit=200)
        if not bars: return None, None
        
        df = pd.DataFrame(bars, columns=['ts','open','high','low','close','vol'])
        df['ts_dt'] = pd.to_datetime(df['ts'], unit='ms')
        
        # 量能进化计算
        df['vol_ma'] = df['vol'].rolling(window=24).median().fillna(method='bfill')
        df['vol_ratio'] = df['vol'] / df['vol_ma']
        
        ticker = ex.fetch_ticker('ETH/USDT:USDT')
        return df.dropna(), ticker
    except Exception as e:
        # 记录错误但不中断渲染
        return None, None

# --- 4. 核心判定：确保返回默认字典 ---
def analyze_signal(df):
    if df is None or df.empty:
        return {"action": "连接中", "motto": "等待行情数据...", "color": "#121212", "voice": ""}, 1.0, 0, 0
    
    curr = df.iloc[-1]
    ratio = curr.get('vol_ratio', 1.0)
    
    # 动态支撑压力
    res = df['high'].iloc[-40:-1].quantile(0.95)
    sup = df['low'].iloc[-40:-1].quantile(0.05)
    
    status = {"action": "AI 扫描中", "motto": "量价合一，顺势而为", "color": "#121212", "voice": "", "tri": None}

    # 口诀判定逻辑
    if ratio < 0.6 and curr['low'] <= sup * 1.002:
        status.update({"action": "准备多", "motto": "缩量回踩，低点不破", "color": "#0D47A1", "voice": "缩量回踩，低点不破"})
    elif ratio > 1.6 and curr['close'] > res:
        status.update({"action": "直接开多", "motto": "放量起涨，突破前高", "color": "#1B5E20", "voice": "放量起涨，直接开多", "tri": "buy"})
    elif ratio < 0.6 and curr['high'] >= res * 0.998:
        status.update({"action": "准备空", "motto": "缩量反弹，高点不破", "color": "#E65100", "voice": "缩量反弹，高点不破"})
    elif ratio > 1.6 and curr['close'] < sup:
        status.update({"action": "直接开空", "motto": "放量下跌，跌破前低", "color": "#B71C1C", "voice": "放量下跌，直接开空", "tri": "sell"})

    return status, ratio, res, sup

# --- 5. 渲染引擎 ---
def main():
    df, ticker = fetch_market_data()
    
    # 如果数据暂未获取到，显示加载状态而非崩溃
    if df is None or ticker is None:
        st.warning("🔄 正在尝试连接 OKX 交易所数据流...")
        return

    status, ratio, res, sup = analyze_signal(df)
    
    # 语音播报调度
    now = time.time()
    if status.get("voice") and st.session_state.signal_memory["last_key"] != status["action"]:
        if now - st.session_state.signal_memory.get("last_time", 0) > 25:
            safe_broadcast(status["voice"])
            st.session_state.signal_memory["last_key"] = status["action"]
            st.session_state.signal_memory["last_time"] = now

    # 看板渲染
    st.markdown(f"""
    <div style="background:{status['color']}; padding:25px; border-radius:15px; text-align:center; border: 2px solid #FFD700; color: white;">
        <h1 style="margin:0; font-size:45px;">{status['action']} ({ratio:.2f}x)</h1>
        <h3 style="color:#FFD700; margin:10px 0;">“{status['motto']}”</h3>
        <p style="font-size:14px; opacity:0.7;">当前价格: {ticker.get('last', '---')} | 24H高: {ticker.get('high', '---')}</p>
    </div>
    """, unsafe_allow_html=True)

    # 成交量进化双图 
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])

    fig.add_trace(go.Candlestick(x=df['ts_dt'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="ETH"), row=1, col=1)
    fig.add_hline(y=res, line_dash="dash", line_color="#FF00FF", row=1, col=1)
    fig.add_hline(y=sup, line_dash="dash", line_color="#00FFFF", row=1, col=1)

    fig.add_trace(go.Scatter(x=df['ts_dt'], y=df['vol_ratio'], fill='tozeroy', line=dict(color='gold', width=2), name="量能进化"), row=2, col=1)
    
    fig.update_layout(template="plotly_dark", height=700, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=20,b=0))
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
