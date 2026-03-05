import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import httpx
import time
from datetime import datetime
import streamlit.components.v1 as components

# ==========================================
# 1. HTTP 连接池与语音冷却 (解决一、六)
# ==========================================
if 'http_client' not in st.session_state:
    st.session_state.http_client = httpx.Client(
        timeout=10.0, 
        http2=True,
        limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
    )

if "last_voice_time" not in st.session_state:
    st.session_state.last_voice_time = 0

def voice_alert(text):
    now = time.time()
    # 增加60秒冷却时间，防止10秒一次的复读机效应
    if now - st.session_state.last_voice_time > 60:
        components.html(f"""
            <script>
            var msg = new SpeechSynthesisUtterance('{text}');
            msg.rate = 1.1; window.speechSynthesis.speak(msg);
            </script>
        """, height=0)
        st.session_state.last_voice_time = now

# ==========================================
# 2. 顶级视觉配置
# ==========================================
st.set_page_config(layout="wide", page_title="Warrior V6.3 | 金融级版", page_icon="⚔️")
st.markdown("""<style>.status-card { background: #1a1c23; border-left: 8px solid #d4af37; padding: 20px; border-radius: 12px; }</style>""", unsafe_allow_html=True)

# ==========================================
# 3. 核心引擎（含自动重试与数据清洗）(解决二、九)
# ==========================================
class WarriorEngine:
    def get_market_data(self, symbol):
        url = "https://www.okx.com/api/v5/market/candles"
        params = {"instId": symbol, "bar": "5m", "limit": "100"}
        for i in range(3): # 自动重试机制
            try:
                resp = st.session_state.http_client.get(url, params=params)
                if resp.status_code == 200:
                    data = resp.json().get('data', [])
                    if not data: continue
                    df = pd.DataFrame(data, columns=['ts','o','h','l','c','v','volCcy','volCcyQuote','confirm'])
                    # 强制类型转换，防止API返回字符串 (解决二)
                    df = df.astype({'o':float,'h':float,'l':float,'c':float,'v':float})
                    df = df[df['confirm'] == '1'].copy()
                    df['time'] = pd.to_datetime(df['ts'].astype(int), unit='ms')
                    # 确保顺序正确 (解决二)
                    return df.sort_values('time').reset_index(drop=True)
            except Exception:
                time.sleep(1)
        return None

def apply_warrior_logic(df, p):
    # 解决三：bfill处理前几根NaN，确保信号不丢失
    df['ma_v'] = df['v'].rolling(p['ma_len']).mean().bfill()
    
    # 解决四：防止极端K线导致的实体比例溢出
    rng = (df['h'] - df['l']).clip(lower=0.001)
    df['body_ratio'] = abs(df['c'] - df['o']) / rng
    df['vol_ratio'] = df['v'] / df['ma_v'].replace(0, 1e-9)
    
    # 解决八：趋势过滤 (EMA20)
    df['ema20'] = df['c'].ewm(span=20).mean()
    df['is_expand'] = df['v'] > df['ma_v'] * (p['expand_p'] / 100.0)
    
    # 最终信号：加入 EMA 趋势保护
    df['buy_sig'] = df['is_expand'] & (df['c'] > df['o']) & (df['body_ratio'] > p['body_r']) & (df['c'] > df['ema20'])
    df['sell_sig'] = df['is_expand'] & (df['c'] < df['o']) & (df['body_ratio'] > p['body_r']) & (df['c'] < df['ema20'])
    
    # 解决五：局部锚点空值逻辑补丁
    window = df.tail(30)
    local_down = window[window['c'] < window['o']].nlargest(1, 'v')
    local_up = window[window['c'] > window['o']].nlargest(1, 'v')
    
    anchors = {
        'down_high': local_down['h'].iloc[0] if len(local_down) > 0 else window['h'].max(),
        'up_low': local_up['l'].iloc[0] if len(local_up) > 0 else window['l'].min()
    }
    return df, anchors

# ==========================================
# 4. 渲染循环 (解决七：性能优化)
# ==========================================
@st.fragment(run_every="10s")
def dashboard_loop():
    engine = WarriorEngine()
    df_raw = engine.get_market_data(st.session_state.symbol)
    if df_raw is None: st.stop()

    df, anchors = apply_warrior_logic(df_raw, st.session_state.params)
    # 解决七：限制数据量提升性能，并保持UI位置
    df_plot = df.tail(80) 
    curr = df_plot.iloc[-1]
    
    # 状态战报渲染
    upper, lower = anchors['down_high'], anchors['up_low']
    if curr['buy_sig'] and curr['c'] > upper:
        status, detail, color = "🚀 核心突破", "EMA上方放量突破局部锚点，多头总攻！", "#26a69a"
        voice_alert("放量起涨，突破前高")
    elif curr['sell_sig'] or curr['c'] < lower:
        status, detail, color = "❄️ 趋势转弱", "跌破EMA及局部支撑锚点，执行风险规避。", "#ef5350"
        voice_alert("趋势转弱，注意离场")
    else:
        status, detail, color = "💎 震荡蓄势", f"量能 {curr['vol_ratio']:.2f}x，EMA走平中。", "#1e90ff"

    st.markdown(f"<div class='status-card' style='border-left:8px solid {color};'><h1 style='color:{color};margin:0;'>{status}</h1><p style='color:#ccc;font-size:18px;'>{detail}</p></div>", unsafe_allow_html=True)

    # 绘图逻辑
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=df_plot['time'], open=df_plot['o'], high=df_plot['h'], low=df_plot['l'], close=df_plot['c'], name="K线"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_plot['time'], y=df_plot['ema20'], line=dict(color='#ffeb3b', width=1), name="EMA20"), row=1, col=1)

    # 局部极值锚点线
    fig.add_hline(y=upper, line_dash="dot", line_color="#ef5350", annotation_text="局部压板", row=1, col=1)
    fig.add_hline(y=lower, line_dash="dot", line_color="#26a69a", annotation_text="局部托盘", row=1, col=1)

    # 性能优化 (解决七)
    fig.update_layout(height=700, template="plotly_dark", uirevision=True, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# ==========================================
# 5. 主程序
# ==========================================
def main():
    st.sidebar.title("⚔️ Warrior V6.3")
    st.session_state.params = {
        "ma_len": st.sidebar.number_input("均量周期", 5, 100, 10),
        "expand_p": st.sidebar.slider("放量判定 (%)", 110, 500, 150),
        "body_r": st.sidebar.slider("突破实体比", 0.05, 0.90, 0.20),
        "rr_ratio": st.sidebar.slider("盈亏比", 1.0, 3.0, 1.5)
    }
    st.session_state.symbol = st.sidebar.text_input("代码", "ETH-USDT-SWAP")
    st.sidebar.success("✅ 金融级稳健逻辑已加载")
    dashboard_loop()

if __name__ == "__main__":
    main()
