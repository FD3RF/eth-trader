import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import httpx
from datetime import datetime
import streamlit.components.v1 as components

# ==========================================
# 1. 工业级底层架构 (修复: 兼容性优化)
# ==========================================
if 'http_client' not in st.session_state:
    # 彻底移除 http2 以确保在所有 Streamlit 节点无错启动
    st.session_state.http_client = httpx.Client(timeout=10.0, follow_redirects=True)

if "last_voice" not in st.session_state:
    st.session_state.last_voice = ""

def voice_alert(text):
    if st.session_state.last_voice != text:
        components.html(f"""<script>
            var msg = new SpeechSynthesisUtterance('{text}');
            msg.rate = 1.1; window.speechSynthesis.speak(msg);
        </script>""", height=0)
        st.session_state.last_voice = text

# ==========================================
# 2. 顶级视觉配置
# ==========================================
st.set_page_config(layout="wide", page_title="Warrior V6.2 Sniper", page_icon="⚔️")
st.markdown("""
    <style>
    .block-container { padding: 1rem 1.5rem; }
    [data-testid="stMetric"] { background: #0e1117; border: 1px solid #1f2937; padding: 15px; border-radius: 12px; }
    .status-card { background: #111827; border-left: 8px solid #fbbf24; padding: 20px; border-radius: 12px; margin-bottom: 20px; border: 1px solid #1f2937; }
    .plan-card { background: #0f172a; border: 2px solid #ef4444; padding: 20px; border-radius: 12px; margin-top: 15px; box-shadow: 0 4px 15px rgba(239, 68, 68, 0.2); }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. 核心引擎与对称审计逻辑
# ==========================================
def apply_warrior_logic(df, p):
    df = df.dropna().reset_index(drop=True)
    df['ma_v'] = df['v'].rolling(p['ma_len']).mean()
    df['body_ratio'] = abs(df['c'] - df['o']) / (df['h'] - df['l']).replace(0, 0.001)
    df['vol_ratio'] = df['v'] / df['ma_v'].replace(0, 1e-9)
    
    # 信号判定：量价指纹审计
    df['is_expand'] = df['v'] > df['ma_v'] * (p['expand_p'] / 100.0)
    df['buy_sig'] = df['is_expand'] & (df['c'] > df['o']) & (df['body_ratio'] > p['body_r'])
    df['sell_sig'] = df['is_expand'] & (df['c'] < df['o']) & (df['body_ratio'] > p['body_r'])
    
    # 局部锚点：30根K线动态回溯
    window = df.tail(30) 
    v_max_down = window[window['c'] < window['o']]
    v_max_up = window[window['c'] > window['o']]
    
    anchors = {
        'upper': v_max_down.nlargest(1, 'v')['h'].values[0] if not v_max_down.empty else window['h'].max(),
        'lower': v_max_up.nlargest(1, 'v')['l'].values[0] if not v_max_up.empty else window['l'].min()
    }
    return df, anchors

# ==========================================
# 4. 实时战场监控与全量渲染
# ==========================================
@st.fragment(run_every="10s")
def dashboard_loop():
    url = "https://www.okx.com/api/v5/market/candles"
    params = {"instId": st.session_state.symbol, "bar": "5m", "limit": "100"}
    
    try:
        resp = st.session_state.http_client.get(url, params=params)
        if resp.status_code != 200: 
            st.error(f"API 异常: {resp.status_code}")
            return
            
        data = resp.json().get('data', [])
        if not data: return

        df = pd.DataFrame(data, columns=['ts','o','h','l','c','v','volCcy','volCcyQuote','confirm'])
        for col in ['o','h','l','c','v']: df[col] = pd.to_numeric(df[col])
        df['time'] = pd.to_datetime(df['ts'].astype(int), unit='ms')
        df = df.sort_values('time').reset_index(drop=True)

        df, anchors = apply_warrior_logic(df, st.session_state.params)
        curr = df.iloc[-1]
        upper, lower = anchors['upper'], anchors['lower']
        
        # 1. 战报渲染与多空识别
        if curr['buy_sig'] or curr['c'] > upper:
            status, detail, color = "🚀 多头总攻", "压力锚点已被放量突破，主力意图向上。", "#10b981"
            voice_alert("放量突破压力位，进场做多")
        elif curr['sell_sig'] or curr['c'] < lower:
            status, detail, color = "❄️ 空头突袭", "支撑锚点失守，空头动能正在释放。", "#ef4444"
            voice_alert("跌破局部支撑，注意撤退或反手")
        else:
            status, detail, color = "💎 震荡蓄势", f"当前量能比 {curr['vol_ratio']:.2f}x，等待锚点区间破位。", "#3b82f6"

        st.markdown(f"<div class='status-card' style='border-left:8px solid {color};'><h1 style='color:{color};margin:0;'>{status}</h1><p style='color:#ccc;font-size:18px;'><b>核心逻辑：</b>{detail}</p></div>", unsafe_allow_html=True)

        # 2. 计划执行中心 (全量保本逻辑)
        if status != "💎 震荡蓄势":
            is_long = "多头" in status
            sl = lower if is_long else upper
            risk = abs(curr['c'] - sl)
            # 顶级铁律：盈利 1:1 后强制移动止损至开仓价
            be_trigger = curr['c'] + risk if is_long else curr['c'] - risk
            tp = curr['c'] + (risk * st.session_state.params['rr_ratio']) if is_long else curr['c'] - (risk * st.session_state.params['rr_ratio'])
            
            st.markdown(f"""<div class='plan-card'>
                <h3 style='color:#ef4444;margin-top:0;'>🎯 狙击手计划执行清单</h3>
                <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:20px;'>
                    <div><b>开仓/现价:</b><br><span style='font-size:20px;'>${curr['c']:.2f}</span></div>
                    <div><b>初始止损:</b><br><span style='font-size:20px;'>${sl:.2f}</span></div>
                    <div><b>保本触发点:</b><br><span style='font-size:20px;color:#fbbf24;'>${be_trigger:.2f}</span></div>
                </div>
                <p style='color:#fbbf24;margin-top:15px;font-weight:bold;'>⚠️ 顶级纪律：价格触及 ${be_trigger:.2f} 后，强制止损上移至开仓位，锁定 0 风险！</p>
            </div>""", unsafe_allow_html=True)

        # 3. 核心指标看板
        col1, col2, col3 = st.columns(3)
        col1.metric("ETH 现价", f"${curr['c']:.2f}")
        col2.metric("当前量能比", f"{curr['vol_ratio']:.2f}x")
        col3.metric("实时刷新", f"{datetime.now().strftime('%H:%M:%S')}")

        # 4. 高级绘图引擎 (修复: 全量信号标记)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
        df_p = df.tail(60) # 保持截图最清晰的 60 根 K 线
        
        fig.add_trace(go.Candlestick(x=df_p['time'], open=df_p['o'], high=df_p['h'], low=df_p['l'], close=df_p['c'], name="K线"), row=1, col=1)

        # [修复重点]：多空信号全量显示
        buys = df_p[df_p['buy_sig']]
        sells = df_p[df_p['sell_sig']]
        fig.add_trace(go.Scatter(x=buys['time'], y=buys['l']*0.998, mode='markers', marker=dict(symbol='triangle-up', size=18, color='#10b981'), name='B'), row=1, col=1)
        fig.add_trace(go.Scatter(x=sells['time'], y=sells['h']*1.002, mode='markers', marker=dict(symbol='triangle-down', size=18, color='#ef4444'), name='S'), row=1, col=1)
        
        # 支撑压力锚点线
        fig.add_hline(y=upper, line_dash="dash", line_color="#ef4444", annotation_text="压力锚点", row=1, col=1)
        fig.add_hline(y=lower, line_dash="dash", line_color="#10b981", annotation_text="支撑锚点", row=1, col=1)

        # 成交量
        v_colors = ['#10b981' if c >= o else '#ef4444' for c, o in zip(df_p['c'], df_p['o'])]
        fig.add_trace(go.Bar(x=df_p['time'], y=df_p['v'], marker_color=v_colors, opacity=0.4), row=2, col=1)

        fig.update_layout(height=650, template="plotly_dark", showlegend=False, xaxis_rangeslider_visible=False, margin=dict(t=10,b=10,l=10,r=50))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    except Exception as e:
        st.warning(f"数据接入引擎重载中... {e}")

# ==========================================
# 5. 控制中心
# ==========================================
def main():
    st.sidebar.title("⚔️ Warrior V6.2 Sniper")
    with st.sidebar.expander("🏹 实战口诀", expanded=True):
        st.markdown("**做多：** 缩量不破底，放量突破买。\n**做空：** 缩量不过顶，放量跌破空。")

    with st.sidebar.expander("策略校准", expanded=True):
        ma_p = st.number_input("均量周期", 5, 50, 10)
        expand_p = st.slider("放量判定 (%)", 110, 300, 150)
        body_r = st.slider("实体比率", 0.05, 0.90, 0.20)
        rr_ratio = st.slider("盈亏比 (1:X)", 1.0, 5.0, 1.5, step=0.1)
    
    st.session_state.params = {"ma_len": ma_p, "expand_p": expand_p, "body_r": body_r, "rr_ratio": rr_ratio}
    st.session_state.symbol = st.sidebar.text_input("合约代码", "ETH-USDT-SWAP")
    st.sidebar.divider()
    st.sidebar.success("🏹 指纹审计模式：已激活\n\n以过去30根K线最大成交量为核心锚点。")
    
    dashboard_loop()

if __name__ == "__main__":
    main()
