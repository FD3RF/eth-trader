import streamlit as st
import pandas as pd

# 设置页面配置
st.set_page_config(page_title="AI 智能决策量价机器人", layout="wide")

st.title("⚡ AI 智能决策系统 (5分钟合约量价版)")
st.caption("基于口诀：缩量是提醒，放量是信号")

# --- 侧边栏：参数设置 ---
with st.sidebar:
    st.header("⚙️ 策略参数")
    vol_ma_period = st.slider("均量参考周期", 5, 20, 5)
    risk_reward_ratio = st.number_input("目标盈亏比", value=1.5)
    st.divider()
    st.info("提示：请根据当前5分钟K线实时输入下方数据")

# --- 第一部分：数据输入 ---
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📍 关键位置")
    prev_high = st.number_input("最近前高价格", value=65000.0)
    prev_low = st.number_input("最近前低价格", value=64000.0)
    current_price = st.number_input("当前收盘价", value=64200.0)

with col2:
    st.subheader("📊 量能数据")
    avg_vol = st.number_input("5周期平均成交量", value=100.0)
    curr_vol = st.number_input("当前成交量", value=50.0)
    
    vol_ratio = curr_vol / avg_vol if avg_vol != 0 else 0
    if vol_ratio >= 1.5:
        st.error(f"量能状态：放量 ({vol_ratio:.2f}x)")
    elif vol_ratio <= 0.6:
        st.success(f"量能状态：缩量 ({vol_ratio:.2f}x)")
    else:
        st.warning(f"量能状态：平量 ({vol_ratio:.2f}x)")

with col3:
    st.subheader("🕯️ K线形态")
    k_type = st.selectbox("当前K线特征", ["十字星/小实体", "长下影线", "长上影线", "大阳线突破", "大阴线跌破"])
    shadow_break = st.checkbox("影线破但收盘未破 (插针)")

st.divider()

# --- 第二部分：口诀匹配与决策引擎 ---
st.subheader("🤖 智能决策分析")

decision = "等待信号"
motto = "量能不明（等待信号）"
action_color = "white"

# 做多逻辑判断
if current_price >= prev_low and current_price <= prev_low * 1.002 and vol_ratio <= 0.6:
    decision = "进入多头观察区"
    motto = "缩量回踩，低点不破，只看不动"
    action_color = "#1E90FF"
elif vol_ratio >= 1.5 and k_type == "大阳线突破" and current_price > prev_high:
    decision = "立刻做多"
    motto = "放量起涨，突破前高，直接开多"
    action_color = "#32CD32"
elif vol_ratio >= 2.0 and shadow_break and current_price > prev_low:
    decision = "激进试多"
    motto = "放量急跌，底部不破，这是机会"
    action_color = "#00FF7F"

# 做空逻辑判断
elif current_price <= prev_high and current_price >= prev_high * 0.998 and vol_ratio <= 0.6:
    decision = "进入空头观察区"
    motto = "缩量反弹，高点不破，只看不动"
    action_color = "#FFA500"
elif vol_ratio >= 1.5 and k_type == "大阴线跌破" and current_price < prev_low:
    decision = "立刻做空"
    motto = "放量下跌，跌破前低，直接开空"
    action_color = "#FF4500"
elif vol_ratio >= 2.0 and shadow_break and current_price < prev_high:
    decision = "激进试空"
    motto = "放量急涨，顶部不破，这是机会"
    action_color = "#FF6347"

# 展示决策卡片
st.markdown(f"""
    <div style="background-color:{action_color}; padding:20px; border-radius:10px; text-align:center;">
        <h2 style="color:white;">{decision}</h2>
        <p style="color:white; font-size:20px;">“{motto}”</p>
    </div>
""", unsafe_allow_html=True)

# --- 第三部分：风控自动计算 ---
if "直接" in motto or "立刻" in motto or "试" in motto:
    st.divider()
    st.subheader("📝 交易执行计划")
    
    if "多" in decision:
        sl = prev_low - (current_price * 0.001)
        tp = current_price + (current_price - sl) * risk_reward_ratio
    else:
        sl = prev_high + (current_price * 0.001)
        tp = current_price - (sl - current_price) * risk_reward_ratio
        
    c1, c2, c3 = st.columns(3)
    c1.metric("入场参考价", f"{current_price:.2f}")
    c2.metric("止损位置", f"{sl:.2f}", delta="-风控", delta_color="inverse")
    c3.metric("止盈目标", f"{tp:.2f}", delta="+盈利")
