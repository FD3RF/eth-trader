import streamlit as st
import pandas as pd
import ta
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import os

# ==========================================
# 1. 系统配置与页面初始化
# ==========================================
st.set_page_config(
    page_title="ETH/USDT 智能交易终端",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 注入赛博朋克风格 CSS
st.markdown("""
<style>
    /* 全局深色背景 */
    .stApp { background-color: #0E1117; }
    
    /* 侧边栏样式 */
    section[data-testid="stSidebar"] { background-color: #161920; border-right: 1px solid #333; }
    
    /* 核心指标卡片 */
    div[data-testid="stMetric"] {
        background-color: #1E1F2A;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    /* AI 信号显示框 */
    .ai-box {
        background: linear-gradient(145deg, #1a1c24 0%, #111217 100%);
        border-left: 4px solid #00D4FF;
        border-radius: 8px;
        padding: 20px;
        color: #e0e0e0;
        font-family: 'Consolas', monospace;
        line-height: 1.6;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.1);
        margin-top: 10px;
    }
    
    /* 强调文字 */
    .highlight { color: #00D4FF; font-weight: bold; }
    .bull { color: #00ffcc; font-weight: bold; }
    .bear { color: #ff3366; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 密钥加载 (优先读取Secrets，其次环境变量)
# ==========================================
try:
    AINFT_KEY = st.secrets["AINFT_KEY"]
except:
    AINFT_KEY = os.getenv("AINFT_KEY")

# 侧边栏配置
with st.sidebar:
    st.header("⚙️ 终端设置")
    if not AINFT_KEY:
        st.warning("⚠️ 未配置密钥")
        AINFT_KEY = st.text_input("输入 AINFT_KEY", type="password")
    
    st.markdown("---")
    st.markdown("**参数设置**")
    rsi_period = st.slider("RSI 周期", 7, 21, 14)
    ma_fast = st.number_input("快线 MA", value=20)
    ma_slow = st.number_input("慢线 MA", value=60)
    
    st.markdown("---")
    st.info("💡 数据源: Binance Spot\n🤖 模型: GPT-5.2 (AINFT)")

# ==========================================
# 3. 核心功能函数
# ==========================================
@st.cache_data(ttl=30, show_spinner=False)
def fetch_market_data(symbol="ETHUSDT", interval="15m", limit=150):
    """获取币安K线数据"""
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        # 设置超时，防止网络卡死
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        df = pd.DataFrame(data, columns=["time", "o", "h", "l", "c", "v", "ct", "qv", "n", "tb", "tq", "i"])
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        for col in ["o", "h", "l", "c", "v"]:
            df[col] = df[col].astype(float)
        return df
    except Exception as e:
        st.error(f"❌ 数据获取失败 ({symbol}): {e}")
        return pd.DataFrame()

def calculate_indicators(df, rsi_n, ma_s, ma_l):
    """计算技术指标"""
    if df.empty: return df
    
    # 移动平均线
    df[f"ma{ma_s}"] = df["c"].rolling(ma_s).mean()
    df[f"ma{ma_l}"] = df["c"].rolling(ma_l).mean()
    
    # RSI
    df["rsi"] = ta.momentum.RSIIndicator(df["c"], window=rsi_n).rsi()
    
    # 斐波那契 (基于最近100根K线)
    recent = df.tail(100)
    high = recent["h"].max()
    low = recent["l"].min()
    diff = high - low
    df["fib_0.618"] = high - diff * 0.618
    df["fib_0.5"] = high - diff * 0.5
    df["fib_0.382"] = high - diff * 0.382
    
    return df

def get_ai_analysis(eth_df, btc_df):
    """调用 AI 生成策略"""
    if not AINFT_KEY:
        return "⚠️ 请先配置 API Key"
    
    e = eth_df.iloc[-1]
    b = btc_df.iloc[-1]
    
    # 构造极简且精确的 Prompt
    prompt = f"""
    分析 ETH/USDT 15分钟级别交易机会。
    【ETH 数据】现价:{e['c']:.2f}, RSI:{e['rsi']:.1f}, MA{ma_fast}:{e[f'ma{ma_fast}']:.2f}, MA{ma_slow}:{e[f'ma{ma_slow}']:.2f}
    【BTC 数据】现价:{b['c']:.2f}, RSI:{b['rsi']:.1f}, 趋势:{'看涨' if b['c'] > b[f'ma{ma_slow}'] else '看跌'}
    
    请输出严格的 HTML 格式报告（不要 Markdown 代码块）：
    1. <b>方向</b>：[做多/做空/观望] (加粗颜色)
    2. <b>信号逻辑</b>：一句话概括 (例如：RSI超卖配合均线支撑)
    3. <b>进场点位</b>：具体价格区间
    4. <b>止损位</b>：具体价格
    5. <b>止盈目标</b>：TP1, TP2
    6. <b>胜率预估</b>：0-100%
    """
    
    url = "https://chat.ainft.com/webapi/chat/openai"
    headers = {"Authorization": f"Bearer {AINFT_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "gpt-5.2",
        "temperature": 0.3,
        "messages": [
            {"role": "system", "content": "你是华尔街顶级日内交易员，风格激进但风控严格。"},
            {"role": "user", "content": prompt}
        ]
    }
    
    try:
        res = requests.post(url, json=payload, headers=headers, timeout=20)
        if res.status_code == 200:
            return res.json()["choices"][0]["message"]["content"]
        else:
            return f"API 错误: {res.text}"
    except Exception as e:
        return f"请求异常: {str(e)}"

# ==========================================
# 4. 主界面逻辑
# ==========================================
st.title("🚀 ETH 15m 精准交易系统")

# 布局：左图表，右信号
col1, col2 = st.columns([2.5, 1])

with col1:
    # 顶部控制栏
    c1, c2 = st.columns([1, 5])
    with c1:
        refresh = st.button("🔄 立即扫描", type="primary", use_container_width=True)
    with c2:
        st.caption(f"上次更新: {datetime.now().strftime('%H:%M:%S')} | 周期: 15m")

    # 数据获取与处理
    if refresh or "eth_data" not in st.session_state:
        with st.spinner("正在连接交易所并计算指标..."):
            raw_eth = fetch_market_data("ETHUSDT")
            raw_btc = fetch_market_data("BTCUSDT")
            
            if not raw_eth.empty and not raw_btc.empty:
                st.session_state.eth_data = calculate_indicators(raw_eth, rsi_period, ma_fast, ma_slow)
                st.session_state.btc_data = calculate_indicators(raw_btc, rsi_period, ma_fast, ma_slow)
                # 触发 AI 分析
                st.session_state.ai_signal = get_ai_analysis(st.session_state.eth_data, st.session_state.btc_data)
            else:
                st.error("无法获取数据，请检查网络（Binance API 需要特定网络环境）。")

    # 绘图逻辑
    if "eth_data" in st.session_state:
        df = st.session_state.eth_data.tail(80) # 只显示最近80根
        
        # 创建交互式图表
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
        
        # K线
        fig.add_trace(go.Candlestick(
            x=df["time"], open=df["o"], high=df["h"], low=df["l"], close=df["c"], name="K线",
            increasing_line_color='#00ffcc', decreasing_line_color='#ff3366'
        ), row=1, col=1)
        
        # 均线
        fig.add_trace(go.Scatter(x=df["time"], y=df[f"ma{ma_fast}"], name=f"MA{ma_fast}", line=dict(color="#FFD700", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["time"], y=df[f"ma{ma_slow}"], name=f"MA{ma_slow}", line=dict(color="#00D4FF", width=1)), row=1, col=1)
        
        # RSI
        fig.add_trace(go.Scatter(x=df["time"], y=df["rsi"], name="RSI", line=dict(color="#9b59b6")), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="rgba(255,255,255,0.3)", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="rgba(255,255,255,0.3)", row=2, col=1)
        
        fig.update_layout(
            template="plotly_dark", 
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_rangeslider_visible=False,
            height=550,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("🤖 智能分析报告")
    if "eth_data" in st.session_state:
        # 实时价格看板
        cur = st.session_state.eth_data.iloc[-1]
        col_m1, col_m2 = st.columns(2)
        
        # 价格变动颜色
        delta_color = "normal"
        if cur['c'] > cur['o']: delta_color = "normal" # Streamlit自动处理涨跌颜色
        
        col_m1.metric("ETH 现价", f"{cur['c']:.2f}", f"{cur['rsi']:.1f} RSI")
        col_m2.metric("BTC 联动", f"{st.session_state.btc_data.iloc[-1]['c']:.0f}")
        
        st.markdown("---")
        
        # AI 结果展示
        if "ai_signal" in st.session_state:
            # 使用 HTML 渲染增加可读性
            st.markdown(f"""
            <div class="ai-box">
                {st.session_state.ai_signal.replace(chr(10), "<br>")}
            </div>
            """, unsafe_allow_html=True)
            
            # 底部风险提示
            st.warning("⚠️ 风险提示：本策略仅供参考，合约交易请严格带好止损。")
    else:
        st.info("👈 请点击左侧「立即扫描」启动系统")
