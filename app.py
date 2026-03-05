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
# 1. 工业级底层架构
# ==========================================
if 'http_client' not in st.session_state:
    st.session_state.http_client = httpx.Client(timeout=10.0, http2=True)

if "last_voice" not in st.session_state:
    st.session_state.last_voice = ""

def voice_alert(text):
    if st.session_state.last_voice != text:
        try:
            components.html(
                f"<script>"
                f"var msg = new SpeechSynthesisUtterance('{text}');"
                f"msg.rate = 1.1; window.speechSynthesis.speak(msg);"
                f"</script>",
                height=0
            )
            st.session_state.last_voice = text
        except Exception:
            pass  # 声音失败不影响主界面

# ==========================================
# 2. 顶级视觉配置（兼容性增强）
# ==========================================
st.set_page_config(
    layout="wide",
    page_title="Warrior V6.2 | 巅峰实战版",
    page_icon="⚔️"
)

st.markdown("""
    <style>
    .block-container { padding: 1rem 1.5rem; }
    [data-testid="stMetric"] {
        background: #11141c;
        border: 1px solid #2d323e;
        padding: 15px;
        border-radius: 10px;
        min-height: 90px;
    }
    .status-card {
        background: #1a1c23;
        border-left: 8px solid #d4af37;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
    }
    .plan-card {
        background: #11141c;
        border: 1px solid #444;
        padding: 15px;
        border-radius: 10px;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. 核心引擎与局部锚点逻辑
# ==========================================
class WarriorEngine:
    def get_market_data(self, symbol):
        url = "https://www.okx.com/api/v5/market/candles"
        params = {"instId": symbol, "bar": "5m", "limit": "100"}
        try:
            resp = st.session_state.http_client.get(url, params=params)
            if resp.status_code != 200:
                return None

            data = resp.json().get('data', [])
            if not data:
                return None

            df = pd.DataFrame(
                data,
                columns=['ts','o','h','l','c','v','volCcy','volCcyQuote','confirm']
            )

            for col in ['o','h','l','c','v']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df = df[df['confirm'] == '1'].copy()
            df['time'] = pd.to_datetime(df['ts'].astype(int), unit='ms')

            return df.sort_values('time').reset_index(drop=True)

        except Exception:
            return None


def apply_warrior_logic(df, p):
    df = df.dropna().reset_index(drop=True)
    df['ma_v'] = df['v'].rolling(p['ma_len']).mean()

    # 防除零与异常
    df['total'] = (df['h'] - df['l']).replace(0, 0.001)
    df['body_ratio'] = abs(df['c'] - df['o']) / df['total']

    df['vol_ratio'] = df['v'] / df['ma_v'].replace(0, 1e-9)

    df['is_expand'] = df['v'] > df['ma_v'] * (p['expand_p'] / 100.0)
    df['buy_sig'] = df['is_expand'] & (df['c'] > df['o']) & (df['body_ratio'] > p['body_r'])
    df['sell_sig'] = df['is_expand'] & (df['c'] < df['o']) & (df['body_ratio'] > p['body_r'])

    window = df.tail(30)
    local_down = window[window['c'] < window['o']].nlargest(1, 'v')
    local_up = window[window['c'] > window['o']].nlargest(1, 'v')

    anchors = {
        'upper': local_down['h'].values[0] if not local_down.empty else window['h'].max(),
        'lower': local_up['l'].values[0] if not local_up.empty else window['l'].min()
    }
    return df, anchors

# ==========================================
# 4. 实时战报与主渲染（防崩保护）
# ==========================================
@st.fragment(run_every="10s")
def dashboard_loop():
    engine = WarriorEngine()
    df = engine.get_market_data(st.session_state.symbol)

    if df is None or df.empty:
        st.warning("📡 正在接入实时数据流...")
        return

    df, anchors = apply_warrior_logic(df, st.session_state.params)
    curr = df.iloc[-1]
    upper, lower = anchors['upper'], anchors['lower']

    # 战报逻辑
    try:
        if curr['buy_sig'] and curr['c'] > upper:
            status, detail, color = "🚀 核心突破", "局部最大量阴线压力已破，多头总攻！", "#26a69a"
            voice_alert("放量起涨，突破前高，直接开多")
        elif curr['sell_sig'] or curr['c'] < lower:
            status, detail, color = "❄️ 趋势转弱", "跌破局部最大量阳线支撑，空头占优。", "#ef5350"
            voice_alert("趋势转弱，注意离场或反手做空")
        else:
            status, detail, color = "💎 震荡蓄势", f"当前量能 {curr['vol_ratio']:.2f}x，观察局部锚点区间。", "#1e90ff"
            if curr['vol_ratio'] < 0.8:
                st.session_state.last_voice = ""

        st.markdown(
            f"<div class='status-card' style='border-left:8px solid {color};'>"
            f"<h1 style='color:{color};margin:0;'>{status}</h1>"
            f"<p style='color:#ccc;font-size:20px;'><b>逻辑分析：</b>{detail}</p>"
            f"</div>",
            unsafe_allow_html=True
        )

    except Exception:
        st.warning("⚠️ 战报解析异常，已自动保护界面")
        return

    # 进场计划模块
    if status != "💎 震荡蓄势":
        is_buy = status == "🚀 核心突破"
        sl = lower if is_buy else upper
        tp_dist = abs(curr['c'] - sl) * st.session_state.params['rr_ratio']
        tp = curr['c'] + tp_dist if is_buy else curr['c'] - tp_dist

        st.markdown(f"""
        <div class='plan-card'>
            <h3 style='color:{color};margin-top:0;'>📝 进场计划清单</h3>
            <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;'>
                <div><b>方向:</b> {'做多 (Long)' if is_buy else '做空 (Short)'}</div>
                <div><b>止损位:</b> ${sl:.2f}</div>
                <div><b>目标止盈:</b> ${tp:.2f}</div>
            </div>
            <p style='font-size:14px;color:#888;margin-top:10px;'>
                核心铁律：缩量是提醒，放量是信号。若回踩破位立即止损。
            </p>
        </div>
        """, unsafe_allow_html=True)

    # 核心指标
    c1, c2, c3 = st.columns(3)
    c1.metric("ETH 现价", f"${curr['c']:.2f}")
    c2.metric("当前量能比", f"{curr['vol_ratio']:.2f}x")
    c3.metric("刷新心跳", datetime.now().strftime('%H:%M:%S'))

    # 绘图（防崩与性能）
    try:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.75, 0.25],
            vertical_spacing=0.03
        )

        df_p = df.tail(80)

        fig.add_trace(
            go.Candlestick(
                x=df_p['time'],
                open=df_p['o'],
                high=df_p['h'],
                low=df_p['l'],
                close=df_p['c'],
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350',
                name="K线"
            ),
            row=1, col=1
        )

        buys = df_p[df_p['buy_sig']]
        fig.add_trace(
            go.Scatter(
                x=buys['time'],
                y=buys['l'] * 0.998,
                mode='markers',
                marker=dict(symbol='triangle-up', size=15),
                name='做多'
            ),
            row=1, col=1
        )

        fig.add_hline(
            y=upper,
            line_dash="dot",
            annotation_text="压力:局部阴高",
            row=1, col=1
        )
        fig.add_hline(
            y=lower,
            line_dash="dot",
            annotation_text="支撑:局部阳低",
            row=1, col=1
        )

        v_colors = [
            '#26a69a' if c >= o else '#ef5350'
            for c, o in zip(df_p['c'], df_p['o'])
        ]

        fig.add_trace(
            go.Bar(
                x=df_p['time'],
                y=df_p['v'],
                marker_color=v_colors,
                opacity=0.4
            ),
            row=2, col=1
        )

        fig.update_layout(
            height=700,
            template="plotly_dark",
            showlegend=False,
            xaxis_rangeslider_visible=False,
            margin=dict(t=10, b=10, l=10, r=50)
        )

        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    except Exception:
        st.warning("⚠️ 图表渲染异常，已跳过本轮更新")


# ==========================================
# 5. 控制中心
# ==========================================
def main():
    st.sidebar.title("⚔️ Warrior V6.2")

    with st.sidebar.expander("🏹 实战口诀锦囊", expanded=True):
        st.markdown("""
        **做多：** 缩量不破底，放量突破买。  
        **做空：** 缩量不过顶，放量跌破空。  
        ---
        **核心：** 缩量是提醒，放量是信号。
        """)

    with st.sidebar.expander("策略校准", expanded=True):
        ma_p = st.number_input("均量周期", 5, 100, 10)
        expand_p = st.slider("放量判定 (%)", 110, 500, 150)
        body_r = st.slider("突破实体比", 0.05, 0.90, 0.20)
        rr_ratio = st.slider("盈亏比 (1:X)", 1.0, 3.0, 1.5, step=0.1)

    st.session_state.params = {
        "ma_len": ma_p,
        "expand_p": expand_p,
        "body_r": body_r,
        "rr_ratio": rr_ratio
    }

    st.session_state.symbol = st.sidebar.text_input("合约代码", "ETH-USDT-SWAP")

    st.sidebar.info("局部极值模式：过去30根K线最大量为锚点")

    dashboard_loop()


if __name__ == "__main__":
    main()
