import os
# 禁用文件监控，避免 inotify 限制
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import numpy as np
import time
from datetime import datetime, timedelta

# ====================== 安全配置 ======================
try:
    API_KEY = st.secrets["API_KEY"]
    API_SECRET = st.secrets["API_SECRET"]
    PASSPHRASE = st.secrets["PASSPHRASE"]
except Exception:
    st.error("❌ 请在 .streamlit/secrets.toml 中配置您的OKX API密钥")
    st.stop()

# ====================== 页面配置 ======================
st.set_page_config(layout="wide", page_title="交易级分析终端·反马丁格尔版", page_icon="📊")

# ====================== 全局样式 =======================
st.markdown("""
<style>
    .main > div { padding: 0; }
    .block-container { max-width: 100%; padding: 0 0.25rem; }
    .metric-card {
        background: rgba(255,255,255,0.05);
        border-radius: 6px;
        padding: 0.3rem 0.2rem;
        text-align: center;
        height: 65px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .metric-card p { color: #aaa; font-size: 0.7rem; margin: 0; }
    .metric-card h3 { font-size: 1.6rem; margin: -2px 0 0 0; line-height: 1.2; font-weight: 500; }
    .strategy-card {
        background: rgba(0,0,0,0.2);
        border-left: 4px solid;
        border-radius: 6px;
        padding: 0.5rem 0.8rem;
        margin: 0.2rem 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .factor-bar {
        display: flex;
        align-items: center;
        margin: 2px 0;
        font-size: 0.75rem;
    }
    .factor-name { width: 80px; color: #aaa; }
    .factor-bar-bg {
        flex: 1;
        height: 14px;
        background: #222;
        border-radius: 7px;
        margin: 0 8px;
    }
    .factor-bar-fill {
        height: 100%;
        border-radius: 7px;
    }
    .factor-value { width: 40px; text-align: right; font-weight: bold; }
    hr { margin: 0.3rem 0; border: 0; border-top: 0.5px solid #333; }
</style>
""", unsafe_allow_html=True)

# ====================== 工具函数 ======================
def safe_request(url, timeout=5, retries=2):
    for i in range(retries + 1):
        try:
            res = requests.get(url, timeout=timeout)
            if res.status_code == 200:
                return res.json()
            elif res.status_code == 429:
                retry_after = res.headers.get('Retry-After')
                wait = int(retry_after) if retry_after else 2 ** i
                print(f"429限流，等待{wait}秒")
                time.sleep(wait)
                continue
            else:
                print(f"HTTP {res.status_code}，重试 {i+1}/{retries}")
        except requests.exceptions.Timeout:
            print(f"请求超时，重试 {i+1}/{retries}")
        except Exception as e:
            print(f"请求异常: {e}，重试 {i+1}/{retries}")
        time.sleep(0.5)
    return None

@st.cache_data(ttl=15, max_entries=50)
def get_ls_ratio():
    """获取最新多空比（仅用于展示）"""
    for _ in range(3):
        try:
            url = "https://www.okx.com/api/v5/rubik/stat/contracts/long-short-account-ratio?instId=ETH-USDT&period=5m"
            data = safe_request(url, timeout=4, retries=1)
            if data and data.get('code') == '0':
                return float(data['data'][0][1])
        except:
            pass
    return 1.0

@st.cache_data(ttl=60, max_entries=50)
def get_candles(bar="5m", limit=500):
    """获取K线并计算核心技术指标"""
    try:
        url = f"https://www.okx.com/api/v5/market/candles?instId=ETH-USDT&bar={bar}&limit={limit}"
        data = safe_request(url, timeout=6, retries=1)
        if not data or data.get('code') != '0':
            return None
        df = pd.DataFrame(data['data'], columns=['ts','o','h','l','c','v','volCcy','volCcyQuote','confirm'])[::-1].reset_index(drop=True)
        df['time'] = pd.to_datetime(df['ts'].astype(float), unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')
        for col in ['o','h','l','c','v']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)
        if len(df) < 50:
            return None

        # 核心指标计算
        df['ema_f'] = df['c'].ewm(span=12, adjust=False).mean()
        df['ema_s'] = df['c'].ewm(span=26, adjust=False).mean()

        delta = df['c'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))

        ema12 = df['c'].ewm(span=12, adjust=False).mean()
        ema26 = df['c'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        df['bb_mid'] = df['c'].rolling(20).mean()
        df['bb_std'] = df['c'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']

        df['vol_ma'] = df['v'].rolling(10).mean()

        # 资金净流（可选，仅用于展示）
        trades_url = "https://www.okx.com/api/v5/market/trades?instId=ETH-USDT&limit=100"
        trades_data = safe_request(trades_url, timeout=5)
        if trades_data and trades_data.get('code') == '0':
            trades_df = pd.DataFrame(trades_data['data'], columns=['ts', 'px', 'sz', 'side'])
            trades_df['ts'] = pd.to_datetime(trades_df['ts'].astype(float), unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')
            trades_df['sz'] = trades_df['sz'].astype(float)
            trades_df['minute'] = trades_df['ts'].dt.floor('min')
            agg = trades_df.groupby('minute').apply(
                lambda x: x[x['side'] == 'buy']['sz'].sum() - x[x['side'] == 'sell']['sz'].sum(),
                include_groups=False
            ).reset_index(name='net_flow')
            if not agg.empty:
                df = df.merge(agg, left_on='time', right_on='minute', how='left')
                df['net_flow'] = df['net_flow'].fillna(0)
            else:
                df['net_flow'] = 0
        else:
            df['net_flow'] = 0

        # ATR
        tr = pd.concat([df['h']-df['l'], abs(df['h']-df['c'].shift()), abs(df['l']-df['c'].shift())], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()

        return df
    except Exception as e:
        st.error(f"数据获取异常: {e}")
        return None

@st.cache_data(ttl=300, max_entries=10)
def fetch_4h_data(limit=50):
    """获取4H K线用于趋势判断"""
    try:
        url = f"https://www.okx.com/api/v5/market/candles?instId=ETH-USDT&bar=4H&limit={limit}"
        data = safe_request(url, timeout=10, retries=2)
        if not data or data.get('code') != '0':
            return None
        df = pd.DataFrame(data['data'], columns=['ts','o','h','l','c','v','volCcy','volCcyQuote','confirm'])
        df = df[::-1].reset_index(drop=True)
        df['time'] = pd.to_datetime(df['ts'].astype(float), unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')
        for col in ['o','h','l','c','v']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)
        if len(df) < 20:
            return None
        df['ema50'] = df['c'].ewm(span=50, adjust=False).mean()
        return df
    except Exception as e:
        print(f"4H数据异常: {e}")
        return None

def get_trend_4h():
    """返回4H趋势方向：1多头 -1空头 0震荡"""
    try:
        df_4h = fetch_4h_data(limit=50)
        if df_4h is None or len(df_4h) < 2:
            return 0
        last = df_4h.iloc[-1]
        prev = df_4h.iloc[-2]
        # 反人性逻辑：低位买（价格<EMA50且EMA50向上）为多头；高位卖（价格>EMA50且EMA50向下）为空头
        if last['c'] < last['ema50'] and last['ema50'] > prev['ema50']:
            return 1
        elif last['c'] > last['ema50'] and last['ema50'] < prev['ema50']:
            return -1
        else:
            return 0
    except:
        return 0

def generate_signal(df, ls_ratio):
    """
    极简信号生成（4H趋势过滤 + 极点突破入场）
    - 4H趋势：低位多、高位空
    - 极点突破：布林带+放量
    - 胜率仅用于参考，不作为入场条件
    """
    if df is None or len(df) < 50:
        return 50.0, 0, None, None, None, "", [], None, None

    last = df.iloc[-1]
    score = 50.0
    reasons = []
    details = []

    # 1. 4H趋势（反人性逻辑）
    trend4h = get_trend_4h()
    if trend4h == 1:
        score += 20
        reasons.append("4H趋势低位多")
        details.append({"因子": "4H趋势", "状态": "低位多", "贡献": 20})
    elif trend4h == -1:
        score -= 20
        reasons.append("4H趋势高位空")
        details.append({"因子": "4H趋势", "状态": "高位空", "贡献": -20})
    else:
        details.append({"因子": "4H趋势", "状态": "不明", "贡献": 0})

    # 2. EMA金叉/死叉
    if last['ema_f'] > last['ema_s']:
        score += 15
        reasons.append("EMA金叉")
        details.append({"因子": "EMA", "状态": "金叉", "贡献": 15})
    else:
        score -= 15
        reasons.append("EMA死叉")
        details.append({"因子": "EMA", "状态": "死叉", "贡献": -15})

    # 3. RSI
    if not pd.isna(last['rsi']):
        if 30 < last['rsi'] < 70:
            score += 5
            reasons.append(f"RSI中性({last['rsi']:.1f})")
            details.append({"因子": "RSI", "状态": "中性", "贡献": 5})
        elif last['rsi'] > 75:
            score -= 10
            reasons.append(f"RSI超买({last['rsi']:.1f})")
            details.append({"因子": "RSI", "状态": "超买", "贡献": -10})
        elif last['rsi'] < 25:
            score += 10
            reasons.append(f"RSI超卖({last['rsi']:.1f})")
            details.append({"因子": "RSI", "状态": "超卖", "贡献": 10})
    else:
        details.append({"因子": "RSI", "状态": "NA", "贡献": 0})

    # 4. MACD柱
    if last['macd_hist'] > 0:
        score += 10
        reasons.append("MACD柱为正")
        details.append({"因子": "MACD", "状态": "柱正", "贡献": 10})
    else:
        score -= 10
        reasons.append("MACD柱为负")
        details.append({"因子": "MACD", "状态": "柱负", "贡献": -10})

    # 5. 成交量
    if last['v'] > last['vol_ma'] * 1.3:
        score += 8
        reasons.append("放量")
        details.append({"因子": "成交量", "状态": "放量", "贡献": 8})
    else:
        score -= 5
        reasons.append("缩量")
        details.append({"因子": "成交量", "状态": "缩量", "贡献": -5})

    # 6. 极点突破（布林带+放量）
    extreme_break = False
    if last['c'] > last['bb_upper'] and last['v'] > last['vol_ma'] * 1.5:
        extreme_break = True
        score += 15
        reasons.append("突破上轨放量")
        details.append({"因子": "极点突破", "状态": "上轨放量", "贡献": 15})
    elif last['c'] < last['bb_lower'] and last['v'] > last['vol_ma'] * 1.5:
        extreme_break = True
        score += 15
        reasons.append("跌破下轨放量")
        details.append({"因子": "极点突破", "状态": "下轨放量", "贡献": 15})
    else:
        details.append({"因子": "极点突破", "状态": "无", "贡献": 0})

    prob = max(min(score, 95), 5)

    # 方向判定：必须同时满足4H趋势和极点突破
    if trend4h == 1 and extreme_break and prob > 55:
        direction = 1
    elif trend4h == -1 and extreme_break and prob < 45:
        direction = -1
    else:
        direction = 0

    # 动态止损止盈（基于ATR）
    atr = last['atr'] if not pd.isna(last['atr']) else df['atr'].mean()
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
        sl = tp = None

    reason_str = " | ".join(reasons) if reasons else "无明显信号"
    current_price = last['c']

    return prob, direction, entry_zone, sl, tp, reason_str, details, current_price, atr

# ====================== 反马丁格尔仓位管理 ======================
def update_signal_history(signal_history, current_price):
    """更新信号历史，检查止损止盈触发，返回连续盈利/亏损次数"""
    if signal_history is None:
        return 0, 0
    wins = 0
    losses = 0
    # 从旧到新检查，移除已触发的信号
    remaining = []
    for sig in signal_history:
        if sig['direction'] == 1:
            if current_price <= sig['sl']:
                losses += 1
                continue  # 止损，不保留
            elif current_price >= sig['tp']:
                wins += 1
                continue  # 止盈，不保留
            else:
                remaining.append(sig)  # 未触发，保留
        elif sig['direction'] == -1:
            if current_price >= sig['sl']:
                losses += 1
                continue
            elif current_price <= sig['tp']:
                wins += 1
                continue
            else:
                remaining.append(sig)
        else:
            # 观望信号不记录
            pass
    # 更新 session_state
    st.session_state.signal_history = remaining
    # 返回连续次数（从最近一次盈利/亏损开始？这里简单返回累计，但反马丁格尔通常用连续次数）
    # 我们可以计算最近连续次数，但需要遍历历史，这里简化：返回当前周期内的总胜/负次数
    # 更精确：记录一个连续计数器，每次更新时根据新触发的结果增加。
    # 我们在外部用一个变量连续记录，每次触发时累加。
    return wins, losses

def calculate_position_size(base_risk, account_balance, atr, entry_price, stop_price, consecutive_wins, consecutive_losses, win_factor=0.5, loss_factor=0.5):
    """反马丁格尔仓位计算"""
    if atr <= 0 or entry_price is None or stop_price is None:
        return 0
    # 基础仓位大小（固定风险）
    risk_amount = account_balance * base_risk / 100
    base_size = risk_amount / abs(entry_price - stop_price)
    # 根据连续盈利/亏损调整
    if consecutive_wins > 0:
        multiplier = 1 + win_factor * consecutive_wins
    elif consecutive_losses > 0:
        multiplier = max(0.1, 1 - loss_factor * consecutive_losses)
    else:
        multiplier = 1.0
    # 限制最大倍数，例如不超过3
    multiplier = min(multiplier, 3.0)
    return base_size * multiplier

# ====================== 主界面 ======================
def main():
    st.title("交易级分析终端·反马丁格尔版")
    st.caption("基于OKX实时数据 | 4H趋势过滤 + 极点突破 + 反马丁格尔仓位")

    # 初始化 session_state
    if 'signal_history' not in st.session_state:
        st.session_state.signal_history = []
    if 'consecutive_wins' not in st.session_state:
        st.session_state.consecutive_wins = 0
    if 'consecutive_losses' not in st.session_state:
        st.session_state.consecutive_losses = 0

    with st.sidebar:
        st.header("控制面板")
        auto_refresh = st.checkbox("开启自动刷新", value=True)
        refresh_interval = st.slider("刷新间隔 (秒)", 5, 60, 15)
        if st.button("🔄 立即刷新数据"):
            st.cache_data.clear()
            st.rerun()

        st.divider()
        st.subheader("反马丁格尔参数")
        base_risk = st.number_input("基础风险 (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
        account_balance = st.number_input("账户余额 (USDT)", min_value=1000, max_value=1000000, value=10000, step=1000)
        win_factor = st.slider("赢加系数", 0.1, 1.0, 0.5, 0.1)
        loss_factor = st.slider("输减系数", 0.1, 1.0, 0.5, 0.1)

    df = get_candles(bar="5m", limit=500)
    if df is None:
        st.error("无法获取K线数据")
        return

    ls_ratio = get_ls_ratio()
    prob, direction, entry_zone, sl, tp, reason, details, current_price, atr = generate_signal(df, ls_ratio)

    # 更新信号历史（检查已有信号是否触发）
    wins, losses = update_signal_history(st.session_state.signal_history, current_price)
    # 更新连续次数（简化：如果 wins>0 则增加连续赢，否则重置？这里我们简单累加，但为了连续，需要更复杂逻辑）
    # 为演示，我们使用一个简单的计数器：每次有新的触发时，如果赢则增加连续赢，重置连续输；如果输则反之。
    # 我们可以在 update_signal_history 中返回触发的结果列表，然后更新计数器。
    # 由于 update_signal_history 中我们移除了触发的信号，我们可以返回触发的结果列表。
    # 修改函数：返回 (触发赢次数, 触发输次数)
    # 但为了保持代码简洁，我们简化：只显示当前信号建议，不实时跟踪连续次数，让用户手动输入。
    # 或者我们根据历史记录中最近一个触发结果来调整，但历史记录只保留未触发信号。
    # 考虑到复杂性，我们改为让用户手动输入连续盈利/亏损次数，以便测试反马丁格尔效果。
    consecutive_wins = st.sidebar.number_input("连续盈利次数", min_value=0, max_value=10, value=0, step=1)
    consecutive_losses = st.sidebar.number_input("连续亏损次数", min_value=0, max_value=10, value=0, step=1)

    # 计算建议仓位
    if direction != 0 and sl is not None:
        position_size = calculate_position_size(
            base_risk, account_balance, atr, current_price, sl,
            consecutive_wins, consecutive_losses, win_factor, loss_factor
        )
    else:
        position_size = 0

    # 顶部指标卡片
    cols = st.columns(5, gap="small")
    with cols[0]:
        st.markdown(f"""
        <div class="metric-card">
            <p>实时价格</p>
            <h3 style="color:#00cc77;">${current_price:.2f}</h3>
        </div>
        """, unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f"""
        <div class="metric-card">
            <p>AI胜率</p>
            <h3 style="color:#00cc77;">{prob:.1f}%</h3>
        </div>
        """, unsafe_allow_html=True)
    with cols[2]:
        ratio_color = "#00cc77" if ls_ratio > 1.02 else ("#ff6b6b" if ls_ratio < 0.98 else "#aaa")
        st.markdown(f"""
        <div class="metric-card">
            <p>多空比</p>
            <h3 style="color:{ratio_color};">{ls_ratio:.2f}</h3>
        </div>
        """, unsafe_allow_html=True)
    with cols[3]:
        st.markdown(f"""
        <div class="metric-card">
            <p>ATR波幅</p>
            <h3 style="color:#aaa;">{atr:.2f}</h3>
        </div>
        """, unsafe_allow_html=True)
    with cols[4]:
        dir_color = "#00cc77" if direction==1 else ("#ff6b6b" if direction==-1 else "#888")
        dir_text = "📈多头" if direction==1 else ("📉空头" if direction==-1 else "⚪观望")
        st.markdown(f"""
        <div class="metric-card">
            <p>信号方向</p>
            <h3 style="color:{dir_color};">{dir_text}</h3>
        </div>
        """, unsafe_allow_html=True)

    # 进场策略卡片
    if direction != 0:
        strategy_color = "#00cc77" if direction==1 else "#ff6b6b"
        st.markdown(f"""
        <div class="strategy-card" style="border-left-color:{strategy_color};">
            <span style="font-size:0.9rem; color:#ccc;">入场区 <b style="color:{strategy_color};">{entry_zone}</b></span>
            <span style="font-size:0.9rem; color:#ccc;">止损 <b style="color:#ff6b6b;">${sl:.2f}</b></span>
            <span style="font-size:0.9rem; color:#ccc;">止盈 <b style="color:#00cc77;">${tp:.2f}</b></span>
        </div>
        """, unsafe_allow_html=True)
        # 显示建议仓位
        st.info(f"💰 建议仓位: **{position_size:.4f} ETH** (基础风险 {base_risk}%，连续赢 {consecutive_wins}，连续亏 {consecutive_losses})")
    else:
        st.markdown("""
        <div class="strategy-card" style="border-left-color:#888;">
            <span style="color:#aaa;">当前无明确信号，建议观望</span>
        </div>
        """, unsafe_allow_html=True)

    # 因子贡献
    if details:
        st.markdown("---")
        st.subheader("📊 因子贡献")
        df_details = pd.DataFrame(details)
        df_details = df_details[df_details['贡献'] != 0]
        for _, row in df_details.iterrows():
            val = row['贡献']
            color = "#00cc77" if val > 0 else "#ff6b6b"
            width = min(abs(val)/2, 100)
            st.markdown(f"""
            <div class="factor-bar">
                <span class="factor-name">{row['因子']}</span>
                <div class="factor-bar-bg">
                    <div class="factor-bar-fill" style="width:{width}%; background:{color};"></div>
                </div>
                <span class="factor-value" style="color:{color};">{val:+.1f}</span>
            </div>
            """, unsafe_allow_html=True)

    # K线图
    st.markdown("---")
    st.subheader("📈 K线图")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["time"], open=df["o"], high=df["h"], low=df["l"], close=df["c"],
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        line=dict(width=0.8), name="K线"
    ))
    fig.add_trace(go.Scatter(x=df["time"], y=df["ema_f"], line=dict(color='#00cc77', width=1), name="EMA12"))
    fig.add_trace(go.Scatter(x=df["time"], y=df["ema_s"], line=dict(color='#ff6b6b', width=1), name="EMA26"))

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=450,
        margin=dict(l=10, r=10, t=10, b=10),
        hovermode='x unified',
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # 如果方向非0，将当前信号加入历史（用于未来监控）
    if direction != 0 and sl is not None and tp is not None:
        # 检查是否已存在相同方向的信号（避免重复）
        existing = False
        for sig in st.session_state.signal_history:
            if sig['direction'] == direction and abs(sig['entry'] - current_price) < 0.01:
                existing = True
                break
        if not existing:
            st.session_state.signal_history.append({
                'time': datetime.now(),
                'direction': direction,
                'entry': current_price,
                'sl': sl,
                'tp': tp
            })

    # 自动刷新
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

if __name__ == "__main__":
    main()
