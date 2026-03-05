import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import httpx
from datetime import datetime
import streamlit.components.v1 as components

# ==========================================
# 1. 底层与语音
# ==========================================
if 'http_client' not in st.session_state:
    st.session_state.http_client = httpx.Client(timeout=10.0)

if "last_voice" not in st.session_state:
    st.session_state.last_voice = ""


def voice_alert(text):
    try:
        if st.session_state.last_voice == text:
            return

        components.html(f"""
        <script>
            try {{
                var msg = new SpeechSynthesisUtterance('{text}');
                msg.rate = 1.05;
                window.speechSynthesis.speak(msg);
            }} catch(e) {{}}
        </script>
        """, height=0)

        st.session_state.last_voice = text

    except Exception:
        pass


def speak_status(status):
    if status == "放量突破做多":
        voice_alert("放量起涨，突破前高，直接开多")
    elif status == "放量跌破做空":
        voice_alert("放量跌破支撑，空头占优")
    elif status == "缩量回踩":
        voice_alert("缩量回踩，低点不破，观察")
    elif status == "缩量反弹":
        voice_alert("缩量反弹，高点不破，观察")


# ==========================================
# 2. 数据同步（真实K线）
# ==========================================
def get_market_data(symbol):
    url = "https://www.okx.com/api/v5/market/candles"
    params = {"instId": symbol, "bar": "5m", "limit": "100"}

    try:
        resp = st.session_state.http_client.get(url, params=params)
        if resp.status_code != 200:
            return None

        data = resp.json().get('data', [])
        if not data:
            return None

        df = pd.DataFrame(data, columns=['ts','o','h','l','c','v','volCcy','volCcyQuote','confirm'])
        for col in ['o','h','l','c','v']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df[df['confirm'] == '1']
        df['time'] = pd.to_datetime(df['ts'].astype(int), unit='ms')

        return df.sort_values('time').reset_index(drop=True)

    except Exception:
        return None


# ==========================================
# 3. 支撑/阻力与策略（口诀落地）
# ==========================================
def calc_support_resistance(df, window=30):
    df = df.dropna()
    tail = df.tail(window)
    support = tail['l'].min()
    resistance = tail['h'].max()
    return support, resistance


def apply_strategy(df, p):
    df = df.dropna().reset_index(drop=True)

    df['ma_v'] = df['v'].rolling(p['ma_len']).mean()
    df['body_ratio'] = abs(df['c'] - df['o']) / (df['h'] - df['l']).replace(0, 0.001)

    df['is_expand'] = df['v'] > df['ma_v'] * (p['expand_p'] / 100.0)
    df['is_contract'] = df['v'] < df['ma_v'] * 0.6

    df['buy_sig'] = df['is_expand'] & (df['c'] > df['o']) & (df['body_ratio'] > p['body_r'])
    df['sell_sig'] = df['is_expand'] & (df['c'] < df['o']) & (df['body_ratio'] > p['body_r'])

    return df


# ==========================================
# 4. 主循环（K线 + 三角 + 支撑压力）
# ==========================================
@st.fragment(run_every="10s")
def dashboard_loop():
    df = get_market_data(st.session_state.symbol)
    if df is None or df.empty:
        st.warning("数据接入中...")
        return

    df = apply_strategy(df, st.session_state.params)
    curr = df.iloc[-1]

    support, resistance = calc_support_resistance(df)

    # 状态判断
    status = "观察"
    color = "#1e90ff"

    if curr['is_contract'] and curr['l'] >= support:
        status = "缩量回踩"
        color = "#ffaa00"

    if curr['buy_sig'] and curr['c'] > resistance:
        status = "放量突破做多"
        color = "#26a69a"

    if curr['sell_sig'] and curr['c'] < support:
        status = "放量跌破做空"
        color = "#ef5350"

    speak_status(status)

    # 战报与进场计划
    st.markdown(f"""
    <div style='border-left:8px solid {color};padding:15px;background:#111'>
        <h2 style='color:{color};margin:0;'>{status}</h2>
        <p>现价：{curr['c']:.2f}</p>
        <p>支撑：{support:.2f} ｜ 阻力：{resistance:.2f}</p>
    </div>
    """, unsafe_allow_html=True)

    if status in ["放量突破做多", "放量跌破做空"]:
        is_buy = status == "放量突破做多"
        sl = support if is_buy else resistance
        tp = curr['c'] + (curr['c'] - sl) * st.session_state.params['rr_ratio']

        st.markdown(f"""
        <div style='padding:10px;border:1px solid #444'>
            <b>进场方向：</b>{'做多' if is_buy else '做空'}<br>
            <b>止损：</b>{sl:.2f}<br>
            <b>止盈：</b>{tp:.2f}
        </div>
        """, unsafe_allow_html=True)

    # 指标
    c1, c2 = st.columns(2)
    c1.metric("量能比", f"{curr['v'] / curr['ma_v']:.2f}x")
    c2.metric("心跳", datetime.now().strftime('%H:%M:%S'))

    # K线图（三角 + 支撑压力）
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)

    df_p = df.tail(80)
    fig.add_trace(go.Candlestick(
        x=df_p['time'],
        open=df_p['o'],
        high=df_p['h'],
        low=df_p['l'],
        close=df_p['c'],
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'
    ), row=1, col=1)

    # 买三角
    buys = df_p[df_p['buy_sig']]
    fig.add_trace(go.Scatter(
        x=buys['time'],
        y=buys['l'] * 0.998,
        mode='markers',
        marker=dict(symbol='triangle-up', size=14),
        marker_color='#26a69a',
        name='做多'
    ), row=1, col=1)

    # 卖三角
    sells = df_p[df_p['sell_sig']]
    fig.add_trace(go.Scatter(
        x=sells['time'],
        y=sells['h'] * 1.002,
        mode='markers',
        marker=dict(symbol='triangle-down', size=14),
        marker_color='#ef5350',
        name='做空'
    ), row=1, col=1)

    # 支撑压力线
    fig.add_hline(y=support, line_dash="dot", line_color="#26a69a", annotation_text="支撑")
    fig.add_hline(y=resistance, line_dash="dot", line_color="#ef5350", annotation_text="压力")

    fig.update_layout(height=650, template="plotly_dark", showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


# ==========================================
# 5. 控制中心
# ==========================================
def main():
    st.sidebar.title("工程交易版")

    with st.sidebar.expander("策略参数", expanded=True):
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
