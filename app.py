import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
import numpy as np
import plotly.io as pio

# ====================== 配置 ======================
API_KEY = "YOUR_API_KEY"
API_SECRET = "YOUR_API_SECRET"
PASSPHRASE = "YOUR_PASSPHRASE"

pio.templates['custom_dark'] = pio.templates['plotly_dark']
pio.templates['custom_light'] = pio.templates['plotly']

st.set_page_config(layout="wide", page_title="量化回测与决策引擎", page_icon="📊")


# ====================== 策略概率模型 ======================
def bayesian_update(prior, evidence):
    denom = (prior * evidence) + ((1 - prior) * (1 - evidence))
    return (prior * evidence / denom) * 100 if denom else 50.0


def calculate_prob(df):
    if len(df) < 50:
        return 45.0

    is_golden = df['ema_f'].iloc[-1] > df['ema_s'].iloc[-1]
    rsi = df['rsi'].iloc[-1]
    rsi_ok = 30 < rsi < 70

    macd_ok = df['macd'].iloc[-1] > df['macd_signal'].iloc[-1]
    net_flow = df['net_flow'].iloc[-1]

    bb_upper = df['c'].rolling(20).mean() + 2 * df['c'].rolling(20).std()
    bb_lower = df['c'].rolling(20).mean() - 2 * df['c'].rolling(20).std()
    bb_squeeze = (bb_upper.iloc[-1] - bb_lower.iloc[-1]) / df['c'].iloc[-1] < 0.02

    prob = 50
    prob += 20 if is_golden else -15
    prob += 12 if rsi_ok else -10
    prob += 15 if macd_ok else -12
    prob += 10 if net_flow > 0 else -10
    prob += -20 if bb_squeeze else 0

    return max(min(prob, 90), 10)


# ====================== 数据获取 ======================
@st.cache_data(ttl=15)
def get_ls_ratio():
    try:
        url = "https://www.okx.com/api/v5/rubik/stat/contracts/long-short-account-ratio"
        res = requests.get(url, params={"instId": "ETH-USDT", "period": "5m"}, timeout=4).json()
        if res.get('code') == '0' and res.get('data'):
            return float(res['data'][0][1])
    except:
        pass
    return 1.0


@st.cache_data(ttl=60)
def get_candles(f_ema, s_ema, bar="15m", limit=200):
    try:
        url = f"https://www.okx.com/api/v5/market/candles?instId=ETH-USDT&bar={bar}&limit={limit}"
        res = requests.get(url, timeout=6).json()
        if res.get('code') != '0':
            raise ValueError("API error")

        df = pd.DataFrame(res['data'],
                         columns=['ts','o','h','l','c','v','volCcy','volCcyQuote','confirm'])[::-1]
        df['time'] = pd.to_datetime(df['ts'].astype(float), unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')
        for c in ['o','h','l','c','v']:
            df[c] = df[c].astype(float)

        df['ema_f'] = df['c'].ewm(span=f_ema).mean()
        df['ema_s'] = df['c'].ewm(span=s_ema).mean()

        diff = df['c'].diff()
        gain = diff.clip(lower=0).rolling(14).mean()
        loss = -diff.clip(upper=0).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))

        ema12 = df['c'].ewm(span=12).mean()
        ema26 = df['c'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()

        try:
            trades = requests.get("https://www.okx.com/api/v5/market/trades?instId=ETH-USDT&limit=100", timeout=6).json()
            if trades.get('code') == '0':
                tdf = pd.DataFrame(trades['data'], columns=['ts','px','sz','side'])
                tdf['sz'] = tdf['sz'].astype(float)
                df['net_flow'] = tdf[tdf['side']=='buy']['sz'].sum() - tdf[tdf['side']=='sell']['sz'].sum()
            else:
                df['net_flow'] = 0
        except:
            df['net_flow'] = 0

        tr = pd.concat([
            df['h']-df['l'],
            abs(df['h']-df['c'].shift()),
            abs(df['l']-df['c'].shift())
        ], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()

        return df

    except:
        dummy = pd.DataFrame({
            'time': pd.date_range('now', periods=200, freq='T'),
            'o': 1900, 'h': 1910, 'l': 1890, 'c': 1900, 'v': 1000
        })
        dummy['ema_f'] = dummy['c']
        dummy['ema_s'] = dummy['c']
        dummy['rsi'] = 50
        dummy['macd'] = 0
        dummy['macd_signal'] = 0
        dummy['net_flow'] = 0
        dummy['atr'] = 5
        return dummy


# ====================== 回测模块 ======================
def backtest(df, initial_balance=10000, risk=0.01, fee=0.0005):
    balance = initial_balance
    equity = []
    position = None

    for i in range(50, len(df)):
        sub = df.iloc[:i]
        prob = calculate_prob(sub)
        price = sub['c'].iloc[-1]
        atr = sub['atr'].iloc[-1] or 1

        # 开仓
        if position is None and prob > 65:
            size = (balance * risk) / atr
            position = {
                "entry": price,
                "size": size,
                "stop": price - atr * 1.5,
                "take": price + atr * 3
            }

        # 持仓管理
        if position:
            pnl = (price - position["entry"]) * position["size"]

            if price <= position["stop"]:
                pnl = (position["stop"] - position["entry"]) * position["size"]

            if price >= position["take"]:
                pnl = (position["take"] - position["entry"]) * position["size"]

            pnl -= abs(pnl) * fee
            balance += pnl
            position = None

        equity.append(balance)

    return {
        "final_balance": balance,
        "return_pct": (balance / initial_balance - 1) * 100,
        "equity_curve": equity
    }


# ====================== UI ======================
def main():
    st.title("📊 量化决策 + 回测引擎")

    df = get_candles(12, 26)
    ls = get_ls_ratio()

    prob = calculate_prob(df)
    bayes = bayesian_update(prob / 100, 0.6 if ls < 1 else 0.4)

    st.metric("胜率估计", f"{bayes:.1f}%")
    st.metric("多空比", f"{ls:.2f}")
    st.metric("当前价格", f"${df['c'].iloc[-1]:.2f}")

    # ===== 图表 =====
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=df['time'], open=df['o'], high=df['h'], low=df['l'], close=df['c']), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['ema_f'], name="EMA快线"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['ema_s'], name="EMA慢线"), row=1, col=1)

    flow = df['net_flow'] / (abs(df['net_flow']).max() or 1) * 800
    fig.add_trace(go.Bar(x=df['time'], y=flow, name="净流"), row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # ===== 回测 =====
    if st.button("🚀 执行回测"):
        with st.spinner("回测中..."):
            result = backtest(df)
        st.success(f"回测完成：收益 {result['return_pct']:.2f}%")
        st.line_chart(result['equity_curve'])


if __name__ == "__main__":
    main()
