import streamlit as st
import pandas as pd

st.title("AI 智能量价交易决策系统（5分钟合约）")

st.write("输入最近5-10根5分钟K线数据（开高低收 + 量），系统基于量价口诀给出建议。")

# 数据输入
data = st.text_area(
    "请输入K线数据，每行一根K线：开,高,低,收,量（用逗号分隔）",
    """100,105,98,102,5000
102,108,101,107,8000
107,110,105,109,12000"""
)

if data:
    rows = []
    for line in data.strip().splitlines():
        parts = line.split(",")
        if len(parts) == 5:
            o, h, l, c, v = map(float, parts)
            rows.append({"open": o, "high": h, "low": l, "close": c, "vol": v})

    df = pd.DataFrame(rows)

    if len(df) >= 2:
        avg_vol = df["vol"].rolling(5).mean().iloc[-1]
        last = df.iloc[-1]
        prev_high = df["high"].max()
        prev_low = df["low"].min()

        # 缩量 / 放量判断
        is_shrink = last["vol"] < (avg_vol * 0.6)
        is_expand = last["vol"] > (avg_vol * 1.5)

        # 前高前低突破
        break_up = last["close"] > prev_high
        break_down = last["close"] < prev_low

        # 决策逻辑
        if is_expand and break_up:
            direction = "做多"
            match = "放量起涨，突破前高，直接开多"
            tp1 = last["close"] * 1.01
            tp2 = last["close"] * 1.02
            sl = prev_low
        elif is_expand and break_down:
            direction = "做空"
            match = "放量杀跌，跌破前低，直接开空"
            tp1 = last["close"] * 0.99
            tp2 = last["close"] * 0.98
            sl = prev_high
        elif is_shrink:
            direction = "观望"
            match = "缩量提醒，动能衰竭，只看不动"
            tp1 = tp2 = sl = None
        else:
            direction = "观望"
            match = "量能不明，等待放量信号"
            tp1 = tp2 = sl = None

        st.subheader("交易建议")
        st.write(f"方向建议：**{direction}**")
        st.write(f"匹配口诀：{match}")
        st.write(f"当前价格：{last['close']}")
        st.write(f"前高：{prev_high} / 前低：{prev_low}")

        if tp1:
            st.write(f"止盈目标：TP1 {tp1:.2f} / TP2 {tp2:.2f}")
            st.write(f"止损位：{sl:.2f}")

        # 语音文案
        if direction != "观望":
            tts = f"当前{match}，建议在{last['close']:.2f}附近{direction}，止损{sl:.2f}，目标{tp1:.2f}。"
        else:
            tts = "当前缩量或量能不明，保持观望，只看不动。"

        st.subheader("语音播报文案")
        st.write(tts)

        st.info("风险提示：5分钟波动大，严格止损，连续止损两单请休息。")
