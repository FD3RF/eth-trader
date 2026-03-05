import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import httpx
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
    try:
        if st.session_state.last_voice != text:
            components.html(f"""
            <script>
                var msg = new SpeechSynthesisUtterance('{text}');
                msg.rate = 1.1;
                window.speechSynthesis.speak(msg);
            </script>
            """, height=0)
            st.session_state.last_voice = text
    except Exception:
        pass


# ==========================================
# 2. 顶级视觉配置
# ==========================================
st.set_page_config(layout="wide", page_title="Warrior V6.2 | 巅峰实战版", page_icon="⚔️")
st.markdown("""
    <style>
    .block-container { padding: 1rem 1.5rem; }
    [data-testid="stMetric"] {
        background: #11141c;
        border: 1px solid #2d323e;
        padding: 15px;
        border-radius: 10px;
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
            if resp.status_code == 200:
                data = resp.json().get('data', [])
                if not data:
                    return None

                df = pd.DataFrame(data, columns=['ts','o','h','l','c','v','volCcy','volCcyQuote','confirm'])
                for col in ['o','h','l','c','v']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                df = df[df['confirm'] == '1'].copy()
                df['time'] = pd.to_datetime(df['ts'].astype(int), unit='ms')
                return df.sort_values('time').reset_index(drop=True)
            return None
        except Exception:
            return None


def apply_warrior_logic(df, p):
    df = df.dropna().reset_index(drop=True)
    df['ma_v'] = df['v'].rolling(p['ma_len']).mean()

    df['body_ratio'] = abs(df['c'] - df['o']) / (df['h'] - df['l']).replace(0, 0.001)
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
# 4. 实时战报与主渲染
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
        st.warning("⚠️ 战报解析异常")
        return

    # 进场计划
    if status != "💎 震荡蓄势":
        is_buy = status == "🚀 核心突破"
        sl = lower if is_buy else upper
        tp_dist = abs(curr['c'] - sl) * st.session_state.params['rr_ratio']
        tp = curr['c'] + tp_dist if is_buy else curr['c'] - tp_dist

        st.markdown(f"""
        <div class='plan-card'>
            <h3 style='color:{color};margin-top:0;'>📝 进场计划</h3>
            <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;'>
                <div><b>方向:</b> {'做多 (Long)' if is_buy else '做空 (Short)'}</div>
                <div><b>止损:</b> ${sl:.2f}</div>
                <div><b>止盈:</b> ${tp:.2f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 核心指标
    c1, c2, c3 = st.columns(3)
    c1.metric("ETH 现价", f"${curr['c']:.2f}")
    c2.metric("量能比", f"{curr['vol_ratio']:.2f}x")
    c3.metric("心跳", datetime.now().strftime('%H:%M:%S'))

    # 绘图：买卖双三角 + 颜色区分
    try:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25])

        df_p = df.tail(80)

        fig.add_trace(go.Candlestick(
            x=df_p['time'],
            open=df_p['o'],
            high=df_p['h'],
            low=df_p['l'],
            close=df_p['c'],
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350',
            name="K线"
        ), row=1, col=1)

        # 多头买点（三角上 + 绿）
        buys = df_p[df_p['buy_sig']]
        fig.add_trace(go.Scatter(
            x=buys['time'],
            y=buys['l'] * 0.998,
            mode='markers',
            marker=dict(symbol='triangle-up', size=15),
            marker_color='#26a69a',
            name='做多'
        ), row=1, col=1)

        # 空头卖点（三角下 + 红）
        sells = df_p[df_p['sell_sig']]
        fig.add_trace(go.Scatter(
            x=sells['time'],
            y=sells['h'] * 1.002,
            mode='markers',
            marker=dict(symbol='triangle-down', size=15),
            marker_color='#ef5350',
            name='做空'
        ), row=1, col=1)

        # 锚点线
        fig.add_hline(y=upper, line_dash="dot", line_color="#ef5350", annotation_text="压力", row=1, col=1)
        fig.add_hline(y=lower, line_dash="dot", line_color="#26a69a", annotation_text="支撑", row=1, col=1)

        # 成交量
        v_colors = ['#26a69a' if c >= o else '#ef5350' for c, o in zip(df_p['c'], df_p['o'])]
        fig.add_trace(go.Bar(x=df_p['time'], y=df_p['v'], marker_color=v_colors, opacity=0.4), row=2, col=1)

        fig.update_layout(
            height=700,
            template="plotly_dark",
            showlegend=False,
            xaxis_rangeslider_visible=False,
            margin=dict(t=10, b=10, l=10, r=50)
        )

        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    except Exception:
        st.warning("图表渲染异常")


# ==========================================
# 5. 控制中心
# ==========================================
def main():
    st.sidebar.title("Warrior V6.2")

    with st.sidebar.expander("策略校准", expanded=True):
        ma_p = st.number_input("均量周期", 5, 100, 10)
        expand_p = st.slider("放量判定(%)", 110, 500, 150)
        body_r = st.slider("实体比", 0.05, 0.90, 0.20)
        rr_ratio = st.slider("盈亏比", 1.0, 3.0, 1.5, step=0.1)

    st.session_state.params = {
        "ma_len": ma_p,
        "expand_p": expand_p,
        "body_r": body_r,
        "rr_ratio": rr_ratio
    }

    st.session_state.symbol = st.sidebar.text_input("合约", "ETH-USDT-SWAP")
    dashboard_loop()


if __name__ == "__main__":
    main()
