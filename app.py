import streamlit as st
import pandas as pd

st.title("AI智能量价自动分析系统（5分钟合约）")

st.write("""
输入5分钟K线数据：  
格式：开,高,低,收,量  
每行一根K线  
系统自动分析方向（多/空/观望）
""")

data = st.text_area(
    "K线数据输入",
    """100,105,98,102,5000
102,108,101,107,8000
107,110,105,109,12000
109,112,108,111,6000
111,115,110,114,14000"""
)

def analyze(df):
    # 计算平均量（最近5根）
    df["avg_vol"] = df["vol"].rolling(5).mean()

    last = df.iloc[-1]
    avg_vol = df["avg_vol"].iloc[-1]

    # 缩量 / 放量判断
    is_shrink = last["vol"] < (avg_vol * 0.6)
    is_expand = last["vol"] > (avg_vol * 1.5)

    # 前高 / 前低
    prev_high = df["high"].max()
    prev_low = df["low"].min()

    # 突破判断
    break_up = last["close"] > prev_high
    break_down = last["close"] < prev_low

    # AI决策逻辑
    if is_expand and break_up:
        direction = "做多"
        reason = "放量起涨突破前高，多头资金进场"
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
        reason = "缩量动能衰竭，等待放量信号"
        sl = tp1 = tp2 = None
        match = "缩量提醒，只看不动"
    else:
        direction = "观望"
        reason = "量能不明，趋势未确认"
        sl = tp1 = tp2 = None
        match = "量能不明，等待信号"

    return {
        "direction": direction,
        "reason": reason,
        "match": match,
        "last_price": last["close"],
        "prev_high": prev_high,
        "prev_low": prev_low,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "tts": (
            f"当前{match}，建议{direction}，止损{sl:.2f}，目标{tp1:.2f}"
            if direction != "观望"
            else "当前缩量或量能不明，保持观望"
        )
    }


if data:
    rows = []
    for line in data.strip().splitlines():
        parts = line.split(",")
        if len(parts) == 5:
            o, h, l, c, v = map(float, parts)
            rows.append({"open": o, "high": h, "low": l, "close": c, "vol": v})

    df = pd.DataFrame(rows)

    if len(df) >= 2:
        result = analyze(df)

        st.subheader("AI自动分析结果")

        st.write(f"方向建议：**{result['direction']}**")
        st.write(f"匹配口诀：{result['match']}")
        st.write(f"分析理由：{result['reason']}")
        st.write(f"当前价格：{result['last_price']}")
        st.write(f"前高：{result['prev_high']} / 前低：{result['prev_low']}")

        if result["direction"] != "观望":
            st.write(f"止损位：{result['sl']:.2f}")
            st.write(f"止盈目标：TP1 {result['tp1']:.2f} / TP2 {result['tp2']:.2f}")

        st.subheader("语音播报文案")
        st.write(result["tts"])

        st.info("风险提示：5分钟波动大，严格止损，连续止损两单请休息。")
