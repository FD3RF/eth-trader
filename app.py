import streamlit as st
import pandas as pd
import joblib

st.title("AI智能量价自动分析系统")

# 加载模型
try:
    model = joblib.load("eth_ai_model.pkl")
except Exception as e:
    st.error(f"模型加载失败：{e}")
    st.stop()

st.write("""
输入5分钟K线：
格式：开,高,低,收,量
每行一根
""")

data = st.text_area(
    "K线数据",
    """100,105,98,102,5000
102,108,101,107,8000
107,110,105,109,12000"""
)

def analyze(df):
    df["return"] = df["close"].pct_change()
    df["vol_change"] = df["vol"].pct_change()

    features = df[["return", "vol_change"]].fillna(0).iloc[-1:].values

    try:
        pred = model.predict(features)[0]
    except:
        return "观望", "模型预测失败"

    if pred == 1:
        return "做多", "模型预测多头概率高"
    elif pred == -1:
        return "做空", "模型预测空头概率高"
    else:
        return "观望", "多空分歧"

if data:
    rows = []
    for line in data.strip().splitlines():
        try:
            o,h,l,c,v = map(float, line.split(","))
            rows.append({"open":o,"high":h,"low":l,"close":c,"vol":v})
        except:
            st.warning(f"无法解析：{line}")

    if rows:
        df = pd.DataFrame(rows)
        direction, reason = analyze(df)

        st.subheader("AI分析")
        st.write(f"方向：{direction}")
        st.write(f"理由：{reason}")
        st.write(f"收盘：{df.iloc[-1]['close']:.2f}")

        st.info("风险提示：仅供参考，严格止损")
