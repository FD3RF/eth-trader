import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import httpx
import streamlit.components.v1 as components
from datetime import datetime
import pytz

# ==========================================
# 1. 终极 UI 渲染引擎 (硬核边框与毛玻璃)
# ==========================================
st.set_page_config(layout="wide", page_title="Warrior Sniper V6.3 Pro", page_icon="⚔️")

st.markdown("""
    <style>
    .stApp { background-color: #05070a; color: #e6edf3; }
    .sticky-header {
        position: sticky; top: 0; z-index: 100;
        background: rgba(13, 17, 23, 0.95);
        border: 1px solid #30363d; border-radius: 12px;
        padding: 15px; margin-bottom: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
    }
    .bull-alert { color: #10b981; border: 2px solid #10b981; padding: 12px; border-radius: 8px; animation: blink 0.8s infinite; background: rgba(16, 185, 129, 0.1); font-weight: bold; font-size: 1.5rem; }
    .bear-alert { color: #ef4444; border: 2px solid #ef4444; padding: 12px; border-radius: 8px; animation: blink 0.8s infinite; background: rgba(239, 68, 68, 0.1); font-weight: bold; font-size: 1.5rem; }
    @keyframes blink { 0% { opacity: 1; } 50% { opacity: 0.4; } 100% { opacity: 1; } }
    
    .history-card {
        background: #0d1117; border-left: 5px solid #30363d;
        padding: 12px; margin-bottom: 8px; border-radius: 4px;
        border-top: 1px solid #161b22; border-right: 1px solid #161b22;
    }
    .bj-time-tag { color: #f0ad4e; font-family: monospace; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 核心交易引擎 (严格保留原始算法逻辑，零删减)
# ==========================================
def warrior_engine(data, p):
    df = pd.DataFrame(data, columns=['ts','o','h','l','c','v','volCcy','volCcyQuote','confirm'])
    for col in ['o','h','l','c','v']: df[col] = pd.to_numeric(df[col])
    
    # 强制 UTC 转换北京时间
    df['time'] = pd.to_datetime(df['ts'].astype(np.int64), unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')
    df = df.sort_values('time').reset_index(drop=True)
    
    # ATR 与 动态锚点计算
    df['tr'] = np.maximum(df['h'] - df['l'], np.maximum(abs(df['h'] - df['c'].shift(1)), abs(df['l'] - df['c'].shift(1))))
    df['atr'] = df['tr'].rolling(14).mean()
    ma_v = df['v'].rolling(p['ma_len']).mean()
    df['vol_r'] = df['v'] / ma_v.replace(0, 1e-9)
    
    lookback = df.tail(30)
    v_dn = lookback[lookback['c'] < lookback['o']]
    v_up = lookback[lookback['c'] > lookback['o']]
    press = v_dn.nlargest(1, 'v')['h'].values[0] if not v_dn.empty else lookback['h'].max()
    supp = v_up.nlargest(1, 'v')['l'].values[0] if not v_up.empty else lookback['l'].min()

    df['buy_tri'] = (df['vol_r'] > (p['exp']/100)) & (df['c'] > df['o']) & (df['c'] > press)
    df['sell_tri'] = (df['vol_r'] > (p['exp']/100)) & (df['c'] < df['o']) & (df['c'] < supp)
    
    return df, {'press': press, 'supp': supp, 'vol_r': df['vol_r'].iloc[-1]}

# ==========================================
# 3. 实时进程 (绝对原子锁 & 幽灵净化)
# ==========================================
def main():
    # --- 幽灵净化协议：首次运行强制洗掉所有旧缓存 ---
    if "v6_3_purged" not in st.session_state:
        st.session_state.sig_history = []
        st.session_state.triggered_signals = set()
        st.session_state.v6_3_purged = True

    if "sig_history" not in st.session_state: st.session_state.sig_history = []
    if "triggered_signals" not in st.session_state: st.session_state.triggered_signals = set()

    with st.sidebar:
        st.title("⚔️ Sniper Pro V6.3")
        voice_on = st.toggle("语音咆哮提醒", True) 
        ma_len = st.number_input("均量周期", 5, 60, 10)
        exp = st.slider("放量触发阈值%", 100, 300, 150)
        sym = st.text_input("合约代码", "ETH-USDT-SWAP")
        st.success("时区校准：Asia/Shanghai (UTC+8)")
        if st.button("清空战报历史"): 
            st.session_state.sig_history = []
            st.session_state.triggered_signals = set()
            st.rerun()

    report_slot = st.empty()
    chart_slot = st.empty()
    st.markdown("### 📜 信号复盘历史 (北京时间同步)")
    history_container = st.container()
    voice_slot = st.empty()

    @st.fragment(run_every="2s")
    def tick():
        try:
            with httpx.Client(http2=True, timeout=2.0) as client:
                r = client.get("https://www.okx.com/api/v5/market/candles", params={"instId": sym, "bar": "5m", "limit": "80"})
                df, res = warrior_engine(r.json()['data'], {"ma_len": ma_len, "exp": exp})
            
            curr = df.iloc[-1]
            curr_ts_id = str(curr['ts']) # 提取 OKX 原始时间戳作为唯一ID
            bj_now_full = curr['time'].strftime('%Y-%m-%d %H:%M:%S')
            bj_now_hms = curr['time'].strftime('%H:%M:%S')
            
            if curr['buy_tri']:
                status_html = f"<div class='bull-alert'>🚀 多头全军突击 | ${curr['c']:.2f}</div>"
                say_cmd = "放量起涨，多头全军突击，直接入场！" 
                h_color = "#10b981"
            elif curr['sell_tri']:
                status_html = f"<div class='bear-alert'>❄️ 空头全面砸盘 | ${curr['c']:.2f}</div>"
                say_cmd = "放量破位，空头全面砸盘，果断撤退！" 
                h_color = "#ef4444"
            else:
                status_html = f"<div style='color:#3b82f6; font-size:1.5rem;'>💎 震荡蓄势中 | <span class='bj-time-tag'>{bj_now_full}</span></div>"
                say_cmd = ""
                h_color = "#30363d"

            report_slot.markdown(f"""
                <div class='sticky-header'>
                    {status_html}
                    <div style='display:flex; justify-content:space-between; margin-top:12px; color:#8b949e; font-family:monospace;'>
                        <span>实时量能比: <b style='color:white;'>{res['vol_r']:.2f}x</b></span>
                        <span>压力锚点: <b style='color:#ef4444;'>{res['press']}</b></span>
                        <span>支撑锚点: <b style='color:#10b981;'>{res['supp']}</b></span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # === 【终极物理锁】基于 K线时间戳 + 指令的唯一哈希锁 ===
            unique_sig_id = f"{curr_ts_id}_{say_cmd}"
            
            if say_cmd and unique_sig_id not in st.session_state.triggered_signals:
                # 记录该信号，永久免疫这根K线的重复触发
                st.session_state.triggered_signals.add(unique_sig_id)
                st.session_state.sig_history.insert(0, {
                    "t": str(bj_now_hms), 
                    "msg": str(say_cmd), 
                    "p": float(curr['c']), 
                    "c": str(h_color)
                })
                
                if voice_on:
                    with voice_slot:
                        components.html(f"""
                            <script>
                            window.speechSynthesis.cancel();
                            var msg = new SpeechSynthesisUtterance('{say_cmd}');
                            msg.lang = 'zh-CN'; msg.pitch = 1.6; msg.rate = 1.3;
                            window.speechSynthesis.speak(msg);
                            </script>
                        """, height=0)

            # --- 图表渲染 ---
            with chart_slot:
                fig = go.Figure(data=[go.Candlestick(x=df['time'], open=df['o'], high=df['h'], low=df['l'], close=df['c'], name="K线")])
                b_pts = df[df['buy_tri']]; s_pts = df[df['sell_tri']]
                fig.add_trace(go.Scatter(x=b_pts['time'], y=b_pts['l']*0.998, mode='markers', name="多", marker=dict(symbol='triangle-up', size=14, color='#10b981', line=dict(width=1, color='white'))))
                fig.add_trace(go.Scatter(x=s_pts['time'], y=s_pts['h']*1.002, mode='markers', name="空", marker=dict(symbol='triangle-down', size=14, color='#ef4444', line=dict(width=1, color='white'))))
                fig.update_layout(height=500, template="plotly_dark", margin=dict(t=0,b=0,l=0,r=0), xaxis_rangeslider_visible=False, showlegend=False)
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

            # --- 历史战报渲染 ---
            with history_container:
                for item in st.session_state.sig_history[:10]:
                    st.markdown(f"""
                        <div class='history-card' style='border-left-color:{item['c']};'>
                            <span class='bj-time-tag'>[{item['t']}]</span> 
                            <b style='color:{item['c']};'>{item['msg']}</b> | 价格: <code>{item['p']}</code>
                        </div>
                    """, unsafe_allow_html=True)

        except Exception as e: 
            pass # 保持UI整洁，静默重试

    tick()

if __name__ == "__main__": main()
