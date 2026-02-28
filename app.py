import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import numpy as np
import plotly.io as pio

# ====================== 安全配置（必须使用secrets）======================
try:
    API_KEY = st.secrets["API_KEY"]
    API_SECRET = st.secrets["API_SECRET"]
    PASSPHRASE = st.secrets["PASSPHRASE"]
except Exception as e:
    st.error("❌ 请在 .streamlit/secrets.toml 中配置您的OKX API密钥")
    st.stop()

# ====================== 页面配置 ======================
st.set_page_config(layout="wide", page_title="专业量化决策引擎·完美版", page_icon="📊")

# 自定义主题模板
pio.templates['custom_dark'] = pio.templates['plotly_dark']
pio.templates['custom_light'] = pio.templates['plotly']

# ====================== 状态初始化 ======================
def init_state():
    defaults = {
        'ls_ratio': 1.0,
        'ls_history': pd.DataFrame(),
        'last_cleanup': time.time(),
        'theme': 'dark',
        'alarm_on': False,
        'logs': [],
        'tg_token': '',
        'tg_chat_id': '',
        'equity_curve': [10000.0],          # 模拟账户净值曲线
        'balance': 10000.0,                  # 模拟余额
        'position': 0.0,                      # 模拟持仓
        'trade_history': [],                  # 交易记录
        'last_signal_prob': 50.0               # 上次信号胜率，用于去重推送
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ====================== 工具函数 ======================
def light_cleanup():
    """轻量级内存清理（每24小时清缓存）"""
    if time.time() - st.session_state.last_cleanup > 86400:
        st.cache_data.clear()
        st.session_state.last_cleanup = time.time()

def send_telegram(msg):
    """发送Telegram消息"""
    if st.session_state.tg_token and st.session_state.tg_chat_id:
        try:
            url = f"https://api.telegram.org/bot{st.session_state.tg_token}/sendMessage"
            requests.post(url, json={"chat_id": st.session_state.tg_chat_id, "text": msg, "parse_mode": "HTML"})
        except Exception as e:
            st.warning(f"Telegram发送失败: {e}")

def bayesian_update(prior, evidence):
    """
    贝叶斯更新函数
    prior: 先验概率（0-1之间）
    evidence: 新证据（0-1之间）
    返回后验概率（百分比）
    """
    denominator = prior * evidence + (1 - prior) * (1 - evidence)
    if denominator == 0:
        return 50.0
    posterior = (prior * evidence) / denominator
    return posterior * 100

# ====================== 数据获取（带缓存和降级）======================
@st.cache_data(ttl=15, max_entries=50)
def get_ls_ratio():
    """获取最新多空比，失败时返回上次有效值"""
    for attempt in range(3):
        try:
            url = "https://www.okx.com/api/v5/rubik/stat/contracts/long-short-account-ratio?instId=ETH-USDT&period=5m"
            res = requests.get(url, timeout=4).json()
            if res.get('code') == '0':
                return float(res['data'][0][1])
            time.sleep(0.3)
        except:
            pass
    # 返回上次存储的值
    return st.session_state.get('ls_ratio', 1.0)

@st.cache_data(ttl=300, max_entries=20)
def get_ls_history(limit=24):
    """获取24小时多空比历史（用于情绪图）"""
    try:
        url = "https://www.okx.com/api/v5/rubik/stat/contracts/long-short-account-ratio"
        params = {"instId": "ETH-USDT", "period": "1H", "limit": limit}
        res = requests.get(url, params=params, timeout=5).json()
        if res.get('code') == '0':
            data = res['data']
            df = pd.DataFrame(data, columns=['ts', 'long', 'short', 'instId'])
            df['ts'] = pd.to_datetime(df['ts'].astype(float), unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')
            df['ratio'] = df['long'].astype(float) / df['short'].astype(float)
            return df[['ts', 'ratio']]
    except Exception as e:
        st.warning(f"多空比历史获取失败: {e}")
    return pd.DataFrame()

@st.cache_data(ttl=60, max_entries=50)
def get_candles(bar="15m", limit=100, f_ema=12, s_ema=26):
    """获取K线并计算所有技术指标"""
    try:
        # 获取K线
        url = f"https://www.okx.com/api/v5/market/candles?instId=ETH-USDT&bar={bar}&limit={limit}"
        res = requests.get(url, timeout=6).json()
        if res.get('code') != '0':
            raise ValueError(f"API返回错误: {res.get('msg')}")
        df = pd.DataFrame(res['data'], columns=['ts','o','h','l','c','v','volCcy','volCcyQuote','confirm'])[::-1].reset_index(drop=True)
        df['time'] = pd.to_datetime(df['ts'].astype(float), unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')
        for col in ['o','h','l','c','v']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)

        # ---------- 技术指标计算 ----------
        # EMA
        df['ema_f'] = df['c'].ewm(span=f_ema, adjust=False).mean()
        df['ema_s'] = df['c'].ewm(span=s_ema, adjust=False).mean()

        # RSI (14)
        delta = df['c'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD (12,26,9)
        ema12 = df['c'].ewm(span=12, adjust=False).mean()
        ema26 = df['c'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # 布林带 (20,2)
        df['bb_mid'] = df['c'].rolling(20).mean()
        df['bb_std'] = df['c'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']

        # 成交量移动平均
        df['vol_ma'] = df['v'].rolling(10).mean()

        # ATR (14)
        tr = pd.concat([df['h']-df['l'], abs(df['h']-df['c'].shift()), abs(df['l']-df['c'].shift())], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()

        # 资金净流入（基于最近100笔成交数据）
        trades_res = requests.get("https://www.okx.com/api/v5/market/trades?instId=ETH-USDT&limit=100", timeout=5).json()
        if trades_res.get('code') == '0':
            trades_df = pd.DataFrame(trades_res['data'], columns=['ts', 'px', 'sz', 'side'])
            trades_df['ts'] = pd.to_datetime(trades_df['ts'].astype(float), unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')
            trades_df['sz'] = trades_df['sz'].astype(float)
            # 按分钟聚合净流入（主动买 - 主动卖）
            trades_df['minute'] = trades_df['ts'].dt.floor('min')
            agg = trades_df.groupby('minute').apply(
                lambda x: x[x['side'] == 'buy']['sz'].sum() - x[x['side'] == 'sell']['sz'].sum()
            ).reset_index(name='net_flow')
            df = df.merge(agg, left_on='time', right_on='minute', how='left')
            df['net_flow'].fillna(0, inplace=True)
        else:
            df['net_flow'] = 0

        return df
    except Exception as e:
        st.error(f"K线获取失败: {e}")
        return None

# ====================== 信号生成与策略计划 ======================
def generate_signal(df, ls_ratio):
    """
    多因子模型计算胜率，并返回详细交易计划
    返回: (胜率, 方向(1多/-1空/0观望), 入场区字符串, 止损价, 止盈价, 理由字符串)
    """
    if df is None or len(df) < 30:
        return 50.0, 0, "数据不足", None, None, "数据不足"

    last = df.iloc[-1]
    score = 50.0
    reasons = []

    # 1. EMA金叉/死叉
    if last['ema_f'] > last['ema_s']:
        score += 20
        reasons.append("EMA金叉")
    else:
        score -= 18
        reasons.append("EMA死叉")

    # 2. RSI
    if 30 < last['rsi'] < 70:
        score += 10
        reasons.append(f"RSI中性({last['rsi']:.1f})")
    elif last['rsi'] > 75:
        score -= 10
        reasons.append(f"RSI超买({last['rsi']:.1f})")
    elif last['rsi'] < 25:
        score += 5
        reasons.append(f"RSI超卖({last['rsi']:.1f})")
    else:
        reasons.append(f"RSI={last['rsi']:.1f}")

    # 3. MACD柱状图
    if last['macd_hist'] > 0:
        score += 12
        reasons.append("MACD柱为正")
    else:
        score -= 12
        reasons.append("MACD柱为负")

    # 4. 布林带位置
    if last['c'] > last['bb_upper']:
        score -= 15
        reasons.append("价格突破上轨")
    elif last['c'] < last['bb_lower']:
        score += 10
        reasons.append("价格跌破下轨")
    else:
        reasons.append("价格在布林带内")

    # 5. 成交量确认
    if last['v'] > last['vol_ma'] * 1.3:
        score += 8
        reasons.append("放量")
    else:
        score -= 4
        reasons.append("缩量")

    # 6. 资金净流入
    if last['net_flow'] > 0:
        score += 15
        reasons.append("资金净流入")
    else:
        score -= 15
        reasons.append("资金净流出")

    # 7. 多空比（作为反向指标）
    if ls_ratio < 0.95:
        score += 8
        reasons.append("多空比<0.95(空头极端)")
    elif ls_ratio > 1.05:
        score -= 8
        reasons.append("多空比>1.05(多头极端)")
    else:
        reasons.append(f"多空比={ls_ratio:.2f}")

    prob = max(min(score, 95), 5)  # 限制在5~95之间

    # 判断方向
    if prob > 55:
        direction = 1
    elif prob < 45:
        direction = -1
    else:
        direction = 0

    # 基于ATR计算动态止损止盈
    atr = last['atr'] if not pd.isna(last['atr']) else df['atr'].mean() if not df['atr'].isna().all() else 10.0
    if direction == 1:
        entry_zone = f"{last['c']-atr*0.5:.1f} ~ {last['c']+atr*0.5:.1f}"
        sl = last['c'] - atr * 1.5
        tp = last['c'] + atr * 2.5
    elif direction == -1:
        entry_zone = f"{last['c']-atr*0.5:.1f} ~ {last['c']+atr*0.5:.1f}"
        sl = last['c'] + atr * 1.5
        tp = last['c'] - atr * 2.5
    else:
        entry_zone = "观望"
        sl = None
        tp = None

    reason_str = " | ".join(reasons)
    return prob, direction, entry_zone, sl, tp, reason_str

# ====================== 侧边栏UI ======================
def render_sidebar(df):
    with st.sidebar:
        st.title("📊 专业量化引擎·完美版")
        st.caption("基于OKX实时数据 | 多因子模型")

        # 刷新控制
        hb = st.slider("自动刷新间隔 (秒)", 5, 60, 15)
        pause = st.checkbox("暂停自动刷新", False)
        st.session_state.alarm_on = st.checkbox("声音报警 (胜率>70%或<30%)", st.session_state.alarm_on)

        # 交易参数
        symbol = st.selectbox("交易对", ["ETH-USDT", "BTC-USDT", "SOL-USDT"], index=0)
        tf = st.selectbox("时间框架", ["1m", "5m", "15m", "30m", "1H"], index=2)
        f_ema = st.number_input("快线EMA", 5, 30, 12)
        s_ema = st.number_input("慢线EMA", 20, 100, 26)

        # 主题
        st.session_state.theme = st.selectbox("主题", ['dark', 'light'])

        if st.button("🔄 立即刷新数据"):
            st.cache_data.clear()
            st.rerun()

        st.divider()

        # 仓位计算器（入场价默认使用最新价格）
        with st.expander("💰 仓位计算器 (固定风险)", expanded=True):
            risk_pct = st.slider("单笔风险 (%)", 0.1, 5.0, 1.0, 0.1)
            account_balance = st.number_input("账户余额 (USDT)", 1000.0, 1000000.0, 10000.0)
            # 默认入场价取最新价格，若无数据则用3000
            default_entry = float(df['c'].iloc[-1]) if df is not None and not df.empty else 3000.0
            entry_price = st.number_input("计划入场价", value=default_entry, format="%.2f")
            stop_price = st.number_input("止损价", value=entry_price * 0.98, format="%.2f")
            if abs(entry_price - stop_price) < 0.01:
                st.warning("止损价过近，请调整")
                position_size = 0
            else:
                position_size = (account_balance * risk_pct / 100) / abs(entry_price - stop_price)
            st.success(f"建议开仓量: **{position_size:.4f} {symbol.split('-')[0]}**")

        # Telegram配置
        with st.expander("📱 Telegram通知", expanded=False):
            st.session_state.tg_token = st.text_input("Bot Token", value=st.session_state.get('tg_token', ''), type="password")
            st.session_state.tg_chat_id = st.text_input("Chat ID", value=st.session_state.get('tg_chat_id', ''))
            st.caption("创建Bot后 /myid 获取Chat ID")

        return hb, pause, symbol, tf, f_ema, s_ema

# ====================== 主界面 ======================
def main():
    light_cleanup()

    # 获取多空比数据
    ls_ratio = get_ls_ratio()
    st.session_state.ls_ratio = ls_ratio
    ls_history = get_ls_history(24)
    if not ls_history.empty:
        st.session_state.ls_history = ls_history

    # 先获取初始K线数据（用于侧边栏默认价格）
    df_init = get_candles(bar="15m", limit=100, f_ema=12, s_ema=26)
    if df_init is None:
        st.error("无法获取K线数据，请检查网络或API")
        st.stop()

    # 侧边栏（传入df_init用于默认价格）
    hb, pause, symbol, tf, f_ema, s_ema = render_sidebar(df_init)

    # 根据用户选择的参数重新获取K线数据
    df = get_candles(bar=tf, limit=100, f_ema=f_ema, s_ema=s_ema)
    if df is None:
        st.error("无法获取K线数据，请检查网络或API")
        st.stop()

    # 生成信号
    prob, direction, entry_zone, sl, tp, reason = generate_signal(df, ls_ratio)

    # Telegram推送（仅当胜率变化超过阈值且与上次不同）
    if (prob > 70 or prob < 30) and abs(prob - st.session_state.last_signal_prob) > 5:
        if prob > 70:
            emoji = "🚀"
            side = "多头"
        else:
            emoji = "⚠️"
            side = "空头"
        rr = abs((tp - df['c'].iloc[-1]) / (df['c'].iloc[-1] - sl)) if sl and tp else 0
        msg = f"""{emoji} {side}信号！
交易对: {symbol} | {tf}
胜率: {prob:.1f}%
价格: ${df['c'].iloc[-1]:.2f}
入场区: {entry_zone}
止损: ${sl:.2f}
止盈: ${tp:.2f}
盈亏比: 1:{rr:.2f}
理由: {reason}"""
        send_telegram(msg)
        st.session_state.last_signal_prob = prob

    # 声音报警
    if st.session_state.alarm_on and (prob > 70 or prob < 30):
        st.audio("https://www.soundjay.com/misc/sounds/bell-ringing-05.mp3", format="audio/mp3")

    # 标题
    st.markdown(f"""
        <h1 style='text-align: center; color: #00ff88; font-family: "Courier New", monospace;'>
            专业量化决策引擎·完美版
        </h1>
        <h4 style='text-align: center; color: #aaa; margin-top: -10px;'>
            {symbol} | {tf} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </h4>
    """, unsafe_allow_html=True)

    # 核心指标卡片
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("实时价格", f"${df['c'].iloc[-1]:.2f}", f"{df['c'].pct_change().iloc[-1]*100:.2f}%")
    col2.metric("AI胜率", f"{prob:.1f}%")
    col3.metric("多空比", f"{ls_ratio:.2f}")
    col4.metric("资金净流", f"{df['net_flow'].iloc[-1]:.0f} ETH")
    col5.metric("ATR波幅", f"{df['atr'].iloc[-1]:.2f}")

    st.divider()

    # 详细策略计划卡片
    if direction == 1:
        box_color = "#00ff88"
        action = "🔥 多头策略"
    elif direction == -1:
        box_color = "#ff4b4b"
        action = "❄️ 空头策略"
    else:
        box_color = "#FFD700"
        action = "⚖️ 观望"

    rr_value = abs((tp - df['c'].iloc[-1]) / (df['c'].iloc[-1] - sl)) if sl and tp and abs(df['c'].iloc[-1] - sl) > 0 else 0
    st.markdown(f"""
    <div style="border:2px solid {box_color}; border-radius:15px; padding:20px; margin-bottom:20px; background:rgba(0,0,0,0.3);">
        <h2 style="color:{box_color}; margin:0;">{action}</h2>
        <p style="color:#ccc;">胜率 <b style="color:{box_color};">{prob:.1f}%</b> | 信号理由: {reason}</p>
        <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:15px; margin-top:15px;">
            <div><span style="color:#aaa;">🎯 最佳入场区</span><br><b style="color:#00ff88;">{entry_zone}</b></div>
            <div><span style="color:#aaa;">🛡️ 动态止损</span><br><b style="color:#ff4b4b;">{f'${sl:.2f}' if sl else '无'}</b></div>
            <div><span style="color:#aaa;">💰 动态止盈</span><br><b style="color:#00ff88;">{f'${tp:.2f}' if tp else '无'}</b></div>
        </div>
        <div style="margin-top:15px;">
            <span style="color:#aaa;">📊 盈亏比: </span><b>{rr_value:.2f}</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 主图表（K线 + 指标）
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=("价格 & 指标", "MACD", "资金净流")
    )

    # 主图
    fig.add_trace(go.Candlestick(
        x=df['time'], open=df['o'], high=df['h'], low=df['l'], close=df['c'],
        name="K线", showlegend=False
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['ema_f'], line=dict(color='#00ff88', width=2), name="EMA快线"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['ema_s'], line=dict(color='#ff4b4b', width=2), name="EMA慢线"), row=1, col=1)
    # 布林带填充
    fig.add_trace(go.Scatter(x=df['time'], y=df['bb_upper'], line=dict(color='rgba(255,255,255,0.2)', width=1), name="BB上轨"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['bb_lower'], line=dict(color='rgba(255,255,255,0.2)', width=1), name="BB下轨", fill='tonexty', fillcolor='rgba(255,255,255,0.05)'), row=1, col=1)

    # MACD
    fig.add_trace(go.Scatter(x=df['time'], y=df['macd'], line=dict(color='#00ff88', width=1.5), name="MACD"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['macd_signal'], line=dict(color='#ff4b4b', width=1.5), name="信号线"), row=2, col=1)
    colors_hist = ['#00ff88' if val > 0 else '#ff4b4b' for val in df['macd_hist']]
    fig.add_trace(go.Bar(x=df['time'], y=df['macd_hist'], marker_color=colors_hist, name="MACD柱"), row=2, col=1)

    # 净流
    flow_colors = ['#00ff88' if x > 0 else '#ff4b4b' for x in df['net_flow']]
    fig.add_trace(go.Bar(x=df['time'], y=df['net_flow'], marker_color=flow_colors, name="资金净流"), row=3, col=1)

    fig.update_layout(
        template=pio.templates['custom_dark'] if st.session_state.theme == 'dark' else pio.templates['custom_light'],
        height=750,  # 略降低高度以适配更多屏幕
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

    # 多空比情绪图
    if not st.session_state.ls_history.empty:
        st.subheader("🌡️ 多空情绪温度计 (过去24小时)")
        ls_df = st.session_state.ls_history
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=ls_df['ts'], y=ls_df['ratio'],
            mode='lines+markers',
            line=dict(color='cyan', width=2),
            fill='tozeroy',
            fillcolor='rgba(0,255,255,0.1)',
            name='多空比'
        ))
        fig2.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="多空平衡")
        fig2.update_layout(
            template=pio.templates['custom_dark'] if st.session_state.theme == 'dark' else pio.templates['custom_light'],
            height=250,
            margin=dict(l=0, r=0, t=10, b=0)
        )
        st.plotly_chart(fig2, use_container_width=True)

    # 刷新控制
    if not pause:
        time.sleep(hb)
        st.rerun()
    else:
        st.stop()

if __name__ == "__main__":
    main()
