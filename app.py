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
st.set_page_config(layout="wide", page_title="交易级分析终端", page_icon="📊")

# ====================== 全局样式（紧凑）======================
st.markdown("""
<style>
    /* 压缩整体间距 */
    .main > div { padding: 0; }
    .block-container { max-width: 100%; padding: 0 0.25rem; }
    /* 指标卡片 */
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
    /* 方向标签 */
    .dir-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: bold;
        text-align: center;
    }
    /* 进场策略卡片 */
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
    /* 因子贡献条 */
    .factor-bar {
        display: flex;
        align-items: center;
        margin: 2px 0;
        font-size: 0.75rem;
    }
    .factor-name { width: 70px; color: #aaa; }
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
    """获取最新多空比"""
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
    """获取K线并计算所需技术指标"""
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

        # 计算基础指标
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

        # 资金净流入
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

        # Stochastic
        low_14 = df['l'].rolling(14).min()
        high_14 = df['h'].rolling(14).max()
        df['stoch_k'] = 100 * (df['c'] - low_14) / (high_14 - low_14).replace(0, np.nan)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()

        # ADX
        tr = pd.concat([df['h']-df['l'], abs(df['h']-df['c'].shift()), abs(df['l']-df['c'].shift())], axis=1).max(axis=1)
        df['+dm'] = np.where((df['h'] - df['h'].shift()) > (df['l'].shift() - df['l']), 
                             np.maximum(df['h'] - df['h'].shift(), 0), 0)
        df['-dm'] = np.where((df['l'].shift() - df['l']) > (df['h'] - df['h'].shift()), 
                             np.maximum(df['l'].shift() - df['l'], 0), 0)
        df['+di'] = 100 * (df['+dm'].rolling(14).mean() / tr.rolling(14).mean())
        df['-di'] = 100 * (df['-dm'].rolling(14).mean() / tr.rolling(14).mean())
        df['dx'] = 100 * abs(df['+di'] - df['-di']) / (df['+di'] + df['-di']).replace(0, np.nan)
        df['adx'] = df['dx'].rolling(14).mean()

        # ATR
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
        if last['c'] > last['ema50'] and last['ema50'] > prev['ema50']:
            return 1
        elif last['c'] < last['ema50'] and last['ema50'] < prev['ema50']:
            return -1
        else:
            return 0
    except:
        return 0

def generate_signal(df, ls_ratio, mvrv_z=0, weights=None):
    """生成信号（精简版，返回必要数据）"""
    if df is None or len(df) < 50:
        return 50.0, 0, None, None, None, "", [], None, None

    default_weights = {
        'ema_cross': (20, -18),
        'rsi_mid': 10,
        'rsi_overbought': -10,
        'rsi_oversold': 5,
        'macd_hist_pos': 12,
        'macd_hist_neg': -12,
        'bb_upper': -15,
        'bb_lower': 10,
        'volume_surge': 8,
        'volume_shrink': -4,
        'net_flow_pos': 15,
        'net_flow_neg': -15,
        'ls_ratio_low': 8,
        'ls_ratio_high': -8,
        'trend4h': 18,
        'extreme_break': 20,
        'stoch_cross': 8,
        'adx_strong': 6,
        'mvrv_low': 12,
        'mvrv_high': -15,
    }
    if weights is None:
        weights = default_weights

    last = df.iloc[-1]
    score = 50.0
    reasons = []
    details = []

    # 4H趋势
    trend4h = get_trend_4h()
    if trend4h == 1:
        score += weights['trend4h']
        reasons.append("4H趋势多头")
        details.append({"因子": "4H趋势", "状态": "多头", "贡献": weights['trend4h']})
    elif trend4h == -1:
        score -= weights['trend4h']
        reasons.append("4H趋势空头")
        details.append({"因子": "4H趋势", "状态": "空头", "贡献": -weights['trend4h']})
    else:
        details.append({"因子": "4H趋势", "状态": "不明", "贡献": 0})

    # EMA
    ema_pos, ema_neg = weights['ema_cross']
    if last['ema_f'] > last['ema_s']:
        score += ema_pos
        reasons.append("EMA金叉")
        details.append({"因子": "EMA", "状态": "金叉", "贡献": ema_pos})
    else:
        score += ema_neg
        reasons.append("EMA死叉")
        details.append({"因子": "EMA", "状态": "死叉", "贡献": ema_neg})

    # RSI
    if not pd.isna(last['rsi']):
        if 30 < last['rsi'] < 70:
            score += weights['rsi_mid']
            reasons.append(f"RSI中性({last['rsi']:.1f})")
            details.append({"因子": "RSI", "状态": "中性", "贡献": weights['rsi_mid']})
        elif last['rsi'] > 75:
            score += weights['rsi_overbought']
            reasons.append(f"RSI超买({last['rsi']:.1f})")
            details.append({"因子": "RSI", "状态": "超买", "贡献": weights['rsi_overbought']})
        elif last['rsi'] < 25:
            score += weights['rsi_oversold']
            reasons.append(f"RSI超卖({last['rsi']:.1f})")
            details.append({"因子": "RSI", "状态": "超卖", "贡献": weights['rsi_oversold']})
    else:
        details.append({"因子": "RSI", "状态": "NA", "贡献": 0})

    # MACD
    if last['macd_hist'] > 0:
        score += weights['macd_hist_pos']
        reasons.append("MACD柱为正")
        details.append({"因子": "MACD", "状态": "柱正", "贡献": weights['macd_hist_pos']})
    else:
        score += weights['macd_hist_neg']
        reasons.append("MACD柱为负")
        details.append({"因子": "MACD", "状态": "柱负", "贡献": weights['macd_hist_neg']})

    # 极点突破
    extreme_break = False
    if last['c'] > last['bb_upper'] and last['v'] > last['vol_ma'] * 1.5:
        extreme_break = True
        score += weights['extreme_break']
        reasons.append("突破上轨放量")
        details.append({"因子": "突破", "状态": "上轨放量", "贡献": weights['extreme_break']})
    elif last['c'] < last['bb_lower'] and last['v'] > last['vol_ma'] * 1.5:
        extreme_break = True
        score += weights['extreme_break']
        reasons.append("跌破下轨放量")
        details.append({"因子": "突破", "状态": "下轨放量", "贡献": weights['extreme_break']})
    else:
        details.append({"因子": "突破", "状态": "非极点", "贡献": 0})

    # 成交量
    if last['v'] > last['vol_ma'] * 1.3:
        score += weights['volume_surge']
        details.append({"因子": "成交量", "状态": "放量", "贡献": weights['volume_surge']})
    else:
        score += weights['volume_shrink']
        details.append({"因子": "成交量", "状态": "缩量", "贡献": weights['volume_shrink']})

    # 资金净流
    if last['net_flow'] > 0:
        score += weights['net_flow_pos']
        details.append({"因子": "资金净流", "状态": "净流入", "贡献": weights['net_flow_pos']})
    else:
        score += weights['net_flow_neg']
        details.append({"因子": "资金净流", "状态": "净流出", "贡献": weights['net_flow_neg']})

    # 多空比
    if ls_ratio < 0.95:
        score += weights['ls_ratio_low']
        reasons.append("多空比极端空")
        details.append({"因子": "多空比", "状态": "极空", "贡献": weights['ls_ratio_low']})
    elif ls_ratio > 1.05:
        score += weights['ls_ratio_high']
        reasons.append("多空比极端多")
        details.append({"因子": "多空比", "状态": "极多", "贡献": weights['ls_ratio_high']})
    else:
        details.append({"因子": "多空比", "状态": "中性", "贡献": 0})

    # Stochastic
    if not pd.isna(last['stoch_k']) and not pd.isna(last['stoch_d']):
        if last['stoch_k'] > last['stoch_d'] and last['stoch_k'] < 20:
            score += weights['stoch_cross']
            reasons.append("Stoch超卖金叉")
            details.append({"因子": "Stoch", "状态": "超卖金叉", "贡献": weights['stoch_cross']})
        elif last['stoch_k'] < last['stoch_d'] and last['stoch_k'] > 80:
            score -= weights['stoch_cross']
            reasons.append("Stoch超买死叉")
            details.append({"因子": "Stoch", "状态": "超买死叉", "贡献": -weights['stoch_cross']})

    # ADX
    if not pd.isna(last['adx']) and last['adx'] > 25:
        score += weights['adx_strong']
        details.append({"因子": "ADX", "状态": "强趋势", "贡献": weights['adx_strong']})

    # MVRV
    if mvrv_z < 0:
        score += weights['mvrv_low']
        details.append({"因子": "MVRV", "状态": "低估", "贡献": weights['mvrv_low']})
    elif mvrv_z > 7:
        score += weights['mvrv_high']
        details.append({"因子": "MVRV", "状态": "泡沫", "贡献": weights['mvrv_high']})

    prob = max(min(score, 95), 5)

    # 方向判定
    if trend4h == 1 and extreme_break and prob > 60:
        direction = 1
    elif trend4h == -1 and extreme_break and prob < 40:
        direction = -1
    else:
        direction = 0

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

# ====================== 主界面 ======================
def main():
    st.title("交易级分析终端")
    st.caption("基于OKX实时数据 | 多因子模型 | 不自动交易")

    # 获取数据
    df = get_candles(bar="5m", limit=500)
    if df is None:
        st.error("无法获取K线数据")
        return

    ls_ratio = get_ls_ratio()
    mvrv_z = np.random.uniform(-1, 3)  # 模拟MVRV

    prob, direction, entry_zone, sl, tp, reason, details, current_price, atr = generate_signal(df, ls_ratio, mvrv_z)

    # ====================== 顶部指标卡片（5个） ======================
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

    # ====================== 进场策略卡片（简洁） ======================
    if direction != 0:
        strategy_color = "#00cc77" if direction==1 else "#ff6b6b"
        st.markdown(f"""
        <div class="strategy-card" style="border-left-color:{strategy_color};">
            <span style="font-size:0.9rem; color:#ccc;">入场区 <b style="color:{strategy_color};">{entry_zone}</b></span>
            <span style="font-size:0.9rem; color:#ccc;">止损 <b style="color:#ff6b6b;">${sl:.2f}</b></span>
            <span style="font-size:0.9rem; color:#ccc;">止盈 <b style="color:#00cc77;">${tp:.2f}</b></span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="strategy-card" style="border-left-color:#888;">
            <span style="color:#aaa;">当前无明确信号，建议观望</span>
        </div>
        """, unsafe_allow_html=True)

    # ====================== 多因子贡献（进度条形式） ======================
    if details:
        st.markdown("---")
        st.subheader("📊 因子贡献")
        # 只显示贡献绝对值较大的前8个因子，避免杂乱
        df_details = pd.DataFrame(details)
        df_details['abs'] = df_details['贡献'].abs()
        df_details = df_details.sort_values('abs', ascending=False).head(8)

        for _, row in df_details.iterrows():
            val = row['贡献']
            color = "#00cc77" if val > 0 else "#ff6b6b"
            width = min(abs(val)/2, 100)  # 缩放显示
            st.markdown(f"""
            <div class="factor-bar">
                <span class="factor-name">{row['因子']}</span>
                <div class="factor-bar-bg">
                    <div class="factor-bar-fill" style="width:{width}%; background:{color};"></div>
                </div>
                <span class="factor-value" style="color:{color};">{val:+.1f}</span>
            </div>
            """, unsafe_allow_html=True)

    # ====================== K线图（紧凑） ======================
    st.markdown("---")
    st.subheader("📈 K线图")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["time"], open=df["o"], high=df["h"], low=df["l"], close=df["c"],
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        line=dict(width=0.8)
    ))
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=450,  # 适当降低高度
        margin=dict(l=10, r=10, t=10, b=10),
        hovermode='x unified',
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

if __name__ == "__main__":
    main()
