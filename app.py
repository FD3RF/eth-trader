import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import httpx
import streamlit.components.v1 as components
from datetime import datetime

# ==========================================
# 1. 页面与全局安全配置
# ==========================================
st.set_page_config(layout="wide", page_title="Warrior Sniper V6.5 Pro", page_icon="⚔️")

st.markdown("""
    <style>
    .stApp { background-color: #05070a; color: #e6edf3; }
    .sticky-header {
        position: sticky; top: 0; z-index: 100;
        background: rgba(13, 17, 23, 0.98);
        border: 1px solid #30363d; border-radius: 12px;
        padding: 15px; margin-bottom: 20px;
        backdrop-filter: blur(12px);
    }
    .status-box { padding: 10px; border-radius: 8px; font-weight: bold; border: 1px solid #444; }
    .bull-alert { color: #10b981; border-color: #10b981; background: rgba(16,185,129,0.1); animation: blink 1s infinite; }
    .bear-alert { color: #ef4444; border-color: #ef4444; background: rgba(239,68,68,0.1); animation: blink 1s infinite; }
    .low-vol { color: #8b949e; background: #161b22; border-color: #30363d; }
    @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.6; } }
    .history-card { background: #0d1117; border-left: 4px solid #30363d; padding: 8px; margin-bottom: 4px; font-family: monospace; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 战神引擎 V6.5 (工程强化逻辑)
# ==========================================
def warrior_engine_v65(data, p):
    if not data or not isinstance(data, list): return None, "API 数据格式错误"
    
    try:
        df = pd.DataFrame(data, columns=['ts','o','h','l','c','v','volCcy','volCcyQuote','confirm'])
        # 强制数值转换
        for col in ['o','h','l','c','v','ts']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 核心防御：剔除无效时间戳，确保绘图不崩溃
        df['time'] = pd.to_datetime(df['ts'], unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')
        df = df.dropna(subset=['time', 'c']).sort_values('time').reset_index(drop=True)
        
        # ATR 波动率门槛计算
        df['tr'] = np.maximum(df['h'] - df['l'], np.maximum(abs(df['h'] - df['c'].shift(1)), abs(df['l'] - df['c'].shift(1))))
        df['atr'] = df['tr'].rolling(14).mean()
        df['atr_ma'] = df['atr'].rolling(50).mean()
        # 只有波动率达到均值的 85% 才视为有效市场
        df['is_volatile'] = df['atr'] > (df['atr_ma'] * 0.85)
        
        # 稳健锚点 (Quantile 防插针)
        lookback = df.tail(50)
        press = lookback['h'].quantile(0.90)
        supp = lookback['l'].quantile(0.10)
        
        # 放量因子
        df['vol_r'] = df['v'] / df['v'].rolling(p['ma_len']).mean().replace(0, 1e-9)
        
        # 最终逻辑：放量 + 突破锚点 + 必须处于活跃波动期
        df['buy_tri'] = (df['vol_r'] > (p['exp']/100)) & (df['c'] > df['o']) & (df['c'] > press) & df['is_volatile']
        df['sell_tri'] = (df['vol_r'] > (p['exp']/100)) & (df['c'] < df['o']) & (df['c'] < supp) & df['is_volatile']
        
        res = {'press': press, 'supp': supp, 'vol_r': df['vol_r'].iloc[-1], 'atr': df['atr'].iloc[-1], 'vol_stat': df['is_volatile'].iloc[-1]}
        return df, res
    except Exception as e:
        return None, f"算法异常: {str(e)}"

# ==========================================
# 3. 实时监控系统
# ==========================================
def main():
    if "sig_history" not in st.session_state: st.session_state.sig_history = []
    if "triggered_ids" not in st.session_state: st.session_state.triggered_ids = set()

    with st.sidebar:
        st.title("⚔️ Sniper V6.5 Pro")
        voice_on = st.toggle("开启语音咆哮", True)
        ma_len = st.number_input("均量周期", 5, 60, 10)
        exp = st.slider("放量阈值%", 100, 300, 150)
        sym = st.text_input("代码", "ETH-USDT-SWAP")
        if st.button("重置系统状态"):
            st.session_state.clear(); st.rerun()

    report_slot = st.empty()
    chart_slot = st.empty()
    st.markdown("### 📜 历史战报 (防抖签名版)")
    history_container = st.container()

    # 工业级 HTTP 客户端配置 (带自动重试)
    transport = httpx.HTTPTransport(retries=3)
    client = httpx.Client(transport=transport, timeout=3.5)

    @st.fragment(run_every="2s")
    def tick():
        try:
            resp = client.get(f"https://www.okx.com/api/v5/market/candles?instId={sym}&bar=5m&limit=100")
            if resp.status_code != 200:
                report_slot.error(f"OKX 连接失败: {resp.status_code}")
                return
            
            df, res = warrior_engine_v65(resp.json().get('data', []), {"ma_len": ma_len, "exp": exp})
            
            if df is None:
                report_slot.error(res) # 显示具体的算法异常原因
                return

            curr = df.iloc[-1]
            # 复合原子签名：时间戳 + 价格(5位精度) + 合约名，彻底杜绝数据回滚导致的重复信号
            ts_key = f"{int(curr['ts'])}_{curr['c']:.5f}_{sym}"
            bj_now = curr['time'].strftime('%H:%M:%S')
            
            # UI 状态逻辑
            status_class, msg = "status-box", ""
            if not res['vol_stat']:
                status_class += " low-vol"
                msg = f"💤 波动率过低 (ATR:{res['atr']:.2f}) | 停止狙击"
            elif curr['buy_tri']:
                status_class += " bull-alert"
                msg = f"🚀 多头突击！价格: {curr['c']}"
            elif curr['sell_tri']:
                status_class += " bear-alert"
                msg = f"❄️ 空头砸盘！价格: {curr['c']}"
            else:
                msg = f"💎 监控中 | {curr['time'].strftime('%Y-%m-%d %H:%M:%S')}"

            report_slot.markdown(f"""
                <div class='sticky-header'>
                    <div class='{status_class}' style='font-size:1.3rem; margin-bottom:10px;'>{msg}</div>
                    <div style='display:flex; justify-content:space-between; font-family:monospace; font-size:0.85rem; color:#8b949e;'>
                        <span>量比: <b style='color:white;'>{res['vol_r']:.2f}x</b></span>
                        <span>ATR: <b style='color:white;'>{res['atr']:.2f}</b></span>
                        <span>压: <b style='color:#ef4444;'>{res['press']:.2f}</b></span>
                        <span>支: <b style='color:#10b981;'>{res['supp']:.2f}</b></span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # 信号持久化与语音触发
            if ("突击" in msg or "砸盘" in msg) and ts_key not in st.session_state.triggered_ids:
                st.session_state.triggered_ids.add(ts_key)
                st.session_state.sig_history.insert(0, {"t": bj_now, "m": msg, "p": curr['c'], "id": ts_key})
                if len(st.session_state.sig_history) > 100: st.session_state.sig_history.pop()
                
                if voice_on:
                    st.components.v1.html(f"<script>speechSynthesis.speak(new SpeechSynthesisUtterance('{msg}'));</script>", height=0)

            # 绘图渲染
            with chart_slot:
                fig = go.Figure(data=[go.Candlestick(x=df['time'], open=df['o'], high=df['h'], low=df['l'], close=df['c'])])
                fig.update_layout(height=450, template="plotly_dark", margin=dict(t=0,b=0,l=0,r=0), xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

            # 战报渲染
            with history_container:
                for item in st.session_state.sig_history:
                    st.markdown(f"<div class='history-card'><b>[{item['t']}]</b> {item['m']} <br><small style='color:#444;'>SIG: {item['id']}</small></div>", unsafe_allow_html=True)

        except Exception as e:
            report_slot.warning(f"系统运行波动: {str(e)}")

    tick()

if __name__ == "__main__": main()
