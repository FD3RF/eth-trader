import streamlit as st
import requests
import pandas as pd
from config import API_URL, INSTRUMENT, BAR, LIMIT

st.title("AI智能量价自动分析系统（真实行情版）")

st.write("""
系统功能：
- 实时获取5分钟K线
- 只用K线 + 成交量分析
- 自动多空建议
- 语音播报文案
""")

def fetch_klines():
    params = {
        "instId": INSTRUMENT,
        "bar": BAR,
        "limit": LIMIT
    }
    resp = requests.get(API_URL, params=params)
    data = resp.json()

    if data.get("code") != "0":
        return None

    rows = []
    for item in data["data"]:
        # OKX返回：ts, o, h, l, c, vol
        rows.append({
            "timestamp": item[0],
            "open": float(item[1]),
            "high": float(item[2]),
            "low": float(item[3]),
            "close": float(item[4]),
            "vol": float(item[5])
        })

    df = pd.DataFrame(rows)
    df = df.iloc[::-1].reset_index(drop=True)  # 时间正序
    return df

def analyze(df):
    avg_vol = df["vol"].rolling(5).mean()
    last = df.iloc[-1]
    prev_high = df["high"].max()
    prev_low = df["low"].min()

    is_shrink = last["vol"] < (avg_vol.iloc[-1] * 0.6)
    is_expand = last["vol"] > (avg_vol.iloc[-1] * 1.5)

    break_up = last["close"] > prev_high
    break_down = last["close"] < prev_low

    if is_expand and break_up:
        direction = "做多"
        reason = "放量突破前高，多头资金进场"
        sl = prev_low
        tp1 = last["close"] * 1.01
        tp2 = last["close"] * 1.02
        match = "放量起涨，突破前高，直接开多"
    elif is_expand and break_down:
        direction = "做空"
        reason = "放量跌破前低，空头力量占优"
        sl = prev_high
        tp1 = last["close"] * 0.99
        tp2 = last["close"] * 0.98
        match = "放量杀跌，跌破前低，直接开空"
    elif is_shrink:
        direction = "观望"
        reason = "缩量动能衰竭"
        sl = tp1 = tp2 = None
        match = "缩量提醒，只看不动"
    else:
        direction = "观望"
        reason = "量能不明"
        sl = tp1 = tp2 = None
        match = "等待放量信号"

    return {
        "direction": direction,
        "reason": reason,
        "match": match,
        "last": last,
        "prev_high": prev_high,
        "prev_low": prev_low,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "tts": (
            f"当前{match}，建议{direction}，止损{sl:.2f}，目标{tp1:.2f}"
            if direction != "观望"
            else "缩量或量能不明，保持观望"
        )
    }

df = fetch_klines()

if df is not None:
    st.subheader("最近K线数据")
    st.dataframe(df)

    result = analyze(df)

    st.subheader("AI自动分析")
    st.write(f"方向建议：**{result['direction']}**")
    st.write(f"匹配口诀：{result['match']}")
    st.write(f"分析理由：{result['reason']}")
    st.write(f"当前收盘：{result['last']['close']:.2f}")
    st.write(f"前高：{result['prev_high']:.2f} / 前低：{result['prev_low']:.2f}")

    if result["direction"] != "观望":
        st.write(f"止损：{result['sl']:.2f}")
        st.write(f"止盈：TP1 {result['tp1']:.2f} / TP2 {result['tp2']:.2f}")

    st.subheader("语音播报")
    st.write(result["tts"])

    st.info("风险提示：5分钟波动大，严格止损")
else:
    st.error("行情获取失败，请检查网络或API")
