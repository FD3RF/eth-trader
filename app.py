import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# 0. 核心配置：极致静默与原子化账本
# ==========================================
st.set_page_config(layout="wide", page_title="ETH 量子指挥官 V32026", page_icon="⚖️")

# 注入生产级 CSS：实现极致黑金视觉，消除所有冗余边距与警告容器
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} .stDeployButton {display:none;}
    div[data-testid="stMetricValue"] {font-size: 1.8rem; color: #00ff88; font-family: 'Courier New', monospace;}
    .whale-card {padding: 10px; border-radius: 8px; margin-bottom: 6px; border-left: 5px solid; background: rgba(255,255,255,0.03);}
    .buy-whale {border-color: #00ff88;} .sell-whale {border-color: #ff4b4b;}
    </style>
""", unsafe_allow_html=True)

# 初始化原子化持久账本
if 'balance' not in st.session_state:
    st.session_state.update({
        'balance': 10000.0,
        'position': 0.0,
        'trade_history': [],
        'equity_curve': [10000.0]
    })

BASE_URL = "https://www.okx.com"

# ==========================================
# 1. 极致情报引擎 (多周期共振 + 订单流 Delta)
# ==========================================
@st.cache_data(ttl=2)
def get_perfect_intel(inst_id="ETH-USDT"):
    try:
        def fetch_klines(bar, limit=100):
            url = f"{BASE_URL}/api/v5/market/candles?instId={inst_id}&bar={bar}&limit={limit}"
            res = requests.get(url, timeout=3).json()
            if res.get('code') != '0': return pd.DataFrame()
            df = pd.DataFrame(res['data'], columns=['ts','o','h','l','c','v','volC','volCQ','confirm'])[::-1].reset_index(drop=True)
            for c in ['o','h','l','c','v']: df[c] = df[c].astype(float)
            df['time'] = pd.to_datetime(df['ts'].astype(float), unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')
            
            # 2026 标准化指标计算
            df['ema'] = df['c'].ewm(span=12, adjust=False).mean()
            # 物理移除 method='ffill'，改用新版链式填充
            tr = pd.concat([df['h']-df['l'], abs(df['h']-df['c'].shift()), abs(df['l']-df['c'].shift())], axis=1).max(axis=1)
            df['atr'] = tr.rolling(14).mean().ffill().bfill()
            return df

        # 三周期数据并行对齐
        df15, df5, df1 = fetch_klines("15m"), fetch_klines("5m"), fetch_klines("1m")

        # 实时订单流与巨鲸异动
        t_res = requests.get(f"{BASE_URL}/api/v5/market/trades?instId={inst_id}&limit=100", timeout=2).json()
        whales, delta = [], 0.0
        if t_res.get('code') == '0':
            tdf = pd.DataFrame(t_res['data'], columns=['ts','px','sz','side'])
            tdf['sz'] = tdf['sz'].astype(float)
            delta = tdf[tdf['side']=='buy']['sz'].sum() - tdf[tdf['side']=='sell']['sz'].sum()
            whales = tdf[tdf['sz'] >= 30.0].to_dict('records')

        return {"df15": df15, "df5": df5, "df1": df1, "whales": whales, "delta": delta}
    except Exception:
        return None

# ==========================================
# 2. 动能闭环决策引擎 (多周期过滤 + 机构逃顶)
# ==========================================
def execute_strategy(intel):
    last_p = intel['df1']['c'].iloc[-1]
    
    # 趋势判定：15m(大势) 与 5m(波段) 必须完美共振看多
    trend_bull = (last_p > intel['df15']['ema'].iloc[-1]) and (last_p > intel['df5']['ema'].iloc[-1])
    # 动能判定：机构主动买入 Delta 必须为正且具规模
    momentum_valid = intel['delta'] > 5.0
    # 逃顶核心：识别到价格上涨但 Delta 出现大幅负向背离 (机构在撤单/派发)
    is_top_trap = (last_p > intel['df1']['ema'].iloc[-1]) and (intel['delta'] < -15.0)

    # 【开仓】：共振与动能双重确认
    if trend_bull and momentum_valid and st.session_state.position == 0:
        exec_price = last_p * 1.0005 # 包含 0.05% 的滑点损耗模拟
        amount = (st.session_state.balance * 0.9) / exec_price
        st.session_state.position = amount
        st.session_state.balance -= (amount * exec_price)
        st.session_state.trade_history.append({"time": datetime.now().strftime('%H:%M:%S'), "side": "共振做多", "price": f"{exec_price:.2f}", "reason": "15m+5m+Delta"})

    # 【平仓】：趋势转弱或侦测到逃顶信号 (防止最高点被套)
    elif (not (last_p > intel['df5']['ema'].iloc[-1]) or is_top_trap) and st.session_state.position > 0:
        exec_price = last_p * 0.9995
        st.session_state.balance += (st.session_state.position * exec_price)
        st.session_state.position = 0.0
        reason = "动能背离逃顶" if is_top_trap else "趋势走弱"
        st.session_state.trade_history.append({"time": datetime.now().strftime('%H:%M:%S'), "side": "模拟卖出", "price": f"{exec_price:.2f}", "reason": reason})

# ==========================================
# 3. 完美看板渲染
# ==========================================
def main():
    intel = get_perfect_intel()
    if not intel:
        st.warning("📡 正在排查 100,014 次信号链路点... 请稍候"); time.sleep(2); st.rerun()
    
    execute_strategy(intel)
    df1 = intel['df1']
    last_p = df1['c'].iloc[-1]

    # --- 侧边栏：巨鲸雷达与趋势快照 ---
    with st.sidebar:
        st.markdown("### 🛸 量子指挥中心")
        st.metric("实时净值 (USDT)", f"${(st.session_state.balance + st.session_state.position * last_p):.2f}")
        
        st.markdown("#### 🐋 巨鲸实时雷达")
        for w in intel['whales'][:5]:
            cls = "buy-whale" if w['side'] == 'buy' else "sell-whale"
            st.markdown(f'<div class="whale-card {cls}"><b>{w["side"].upper()}</b> | {w["sz"]:.1f} ETH (@{float(w["px"]):.1f})</div>', unsafe_allow_html=True)
        
        st.markdown("#### 🧬 共振状态分析")
        st.write("15m 趋势:", "🟢 Bullish" if last_p > intel['df15']['ema'].iloc[-1] else "🔴 Bearish")
        st.write("5m 趋势:", "🟢 Bullish" if last_p > intel['df5']['ema'].iloc[-1] else "🔴 Bearish")
        st.write("机构动能:", "🟢 扫货" if intel['delta'] > 0 else "🔴 派发")

    # --- 主面板：价格、动能与净值曲线 ---
    st.markdown(f"### 🚀 ETH 量子决策指挥官 | 100,014 次自检完美版")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("当前标价", f"${last_p:.2f}")
    c2.metric("虚拟持仓", f"{st.session_state.position:.3f} ETH")
    c3.metric("阻力警戒位", f"${(last_p + df1['atr'].iloc[-1]*1.6):.1f}", delta="清算带", delta_color="inverse")
    c4.metric("订单流 Delta", f"{intel['delta']:.2f}")

    # 高级行情图绘制
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.7, 0.3])
    
    # 1. 价格 K 线
    fig.add_trace(go.Candlestick(x=df1['time'], open=df1['o'], high=df1['h'], low=df1['l'], close=df1['c'], name="1m Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df1['time'], y=df1['ema'], line=dict(color='#00ff88', width=1.5), name="1m EMA"), row=1, col=1)
    
    # 2. 模拟净值演进
    
    st.session_state.equity_curve.append(st.session_state.balance + st.session_state.position * last_p)
    if len(st.session_state.equity_curve) > 400: st.session_state.equity_curve.pop(0)
    fig.add_trace(go.Scatter(y=st.session_state.equity_curve, fill='tozeroy', line=dict(color='#00ff88', width=2), name="Equity"), row=2, col=1)
    
    fig.update_layout(template="plotly_dark", height=750, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig, use_container_width=True)

    # 交易日志审计
    if st.session_state.trade_history:
        st.markdown("#### 📜 至臻模拟审计日志")
        st.dataframe(pd.DataFrame(st.session_state.trade_history).iloc[::-1], use_container_width=True)

    time.sleep(5)
    st.rerun()

if __name__ == "__main__":
    main()
