import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import plotly.graph_objects as go
import time
import joblib
from sklearn.ensemble import RandomForestClassifier
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="ETH AI终极盯盘", layout="wide")

st_autorefresh(interval=5000,key="refresh")

MODEL_FILE="ai_model.pkl"

INSTRUMENT="ETH/USDT:USDT"
TIMEFRAME="5m"
LIMIT=200

# ---------------------
# 状态记忆
# ---------------------

if "last_signal" not in st.session_state:
    st.session_state.last_signal=None

if "last_voice_time" not in st.session_state:
    st.session_state.last_voice_time=0

# ---------------------
# 浏览器语音
# ---------------------

def speak(text):

    js=f"""
    <script>
    var msg=new SpeechSynthesisUtterance("{text}");
    msg.lang='zh-CN';
    msg.rate=1.1;
    speechSynthesis.speak(msg);
    </script>
    """

    st.components.v1.html(js,height=0)

# ---------------------
# 交易所
# ---------------------

@st.cache_resource
def get_exchange():
    return ccxt.okx({
        "enableRateLimit":True,
        "options":{"defaultType":"swap"}
    })

# ---------------------
# 获取K线
# ---------------------

def fetch_data():

    ex=get_exchange()

    for _ in range(3):

        try:

            bars=ex.fetch_ohlcv(INSTRUMENT,timeframe=TIMEFRAME,limit=LIMIT)

            df=pd.DataFrame(
                bars,
                columns=["ts","open","high","low","close","vol"]
            )

            df["ts_dt"]=pd.to_datetime(df["ts"],unit="ms")

            df=df.dropna()

            if len(df)>60:
                return df

        except Exception:
            time.sleep(1)

    return None

# ---------------------
# 趋势
# ---------------------

def trend(df):

    ma20=df["close"].rolling(20).mean()
    ma50=df["close"].rolling(50).mean()

    if ma20.iloc[-1]>ma50.iloc[-1]:
        return "up"

    if ma20.iloc[-1]<ma50.iloc[-1]:
        return "down"

    return "side"

# ---------------------
# 趋势强度
# ---------------------

def trend_strength(df):

    ma20=df["close"].rolling(20).mean()
    ma50=df["close"].rolling(50).mean()

    slope=ma20.diff().iloc[-5:].mean()

    distance=abs(ma20.iloc[-1]-ma50.iloc[-1])

    vol=df["close"].pct_change().rolling(20).std().iloc[-1]

    if pd.isna(vol) or vol==0:
        vol=0.0001

    score=(distance/vol)*abs(slope)

    return round(score,2)

# ---------------------
# 支撑阻力
# ---------------------

def levels(df):

    res=df["high"].iloc[-40:-1].quantile(0.95)

    sup=df["low"].iloc[-40:-1].quantile(0.05)

    return res,sup

# ---------------------
# 假突破
# ---------------------

def fake_break(df,res,sup):

    curr=df.iloc[-1]
    prev=df.iloc[-2]

    avg=df["vol"].iloc[-50:-1].median()

    if curr["close"]>res and curr["vol"]<avg*0.8 and prev["close"]<res:
        return "fake_up"

    if curr["close"]<sup and curr["vol"]<avg*0.8 and prev["close"]>sup:
        return "fake_down"

    return None

# ---------------------
# 主力吸筹
# ---------------------

def accumulation(df):

    rng=df["high"].iloc[-30:].max()-df["low"].iloc[-30:].min()

    price=df["close"].iloc[-1]

    flat=rng/price<0.02

    avg=df["vol"].iloc[-50:-1].median()

    recent=df["vol"].iloc[-10:].mean()

    vol_expand=recent>avg*1.3

    return flat and vol_expand

# ---------------------
# 庄家拉升
# ---------------------

def whale_pump(df):

    vol=df["vol"].iloc[-3:].mean()

    base=df["vol"].iloc[-50:-3].mean()

    price_up=df["close"].iloc[-1]>df["close"].iloc[-5]

    return vol>base*2 and price_up

# ---------------------
# 砸盘
# ---------------------

def dump(df):

    avg=df["vol"].iloc[-50:-1].mean()

    sell=df["vol"].iloc[-1]>avg*2

    drop=df["close"].iloc[-1]<df["close"].iloc[-2]*0.99

    return sell and drop

# ---------------------
# 多周期共振
# ---------------------

def multi_tf():

    ex=get_exchange()

    df1=pd.DataFrame(
        ex.fetch_ohlcv(INSTRUMENT,"1m",limit=120),
        columns=["ts","open","high","low","close","vol"]
    )

    df5=pd.DataFrame(
        ex.fetch_ohlcv(INSTRUMENT,"5m",limit=120),
        columns=["ts","open","high","low","close","vol"]
    )

    df15=pd.DataFrame(
        ex.fetch_ohlcv(INSTRUMENT,"15m",limit=120),
        columns=["ts","open","high","low","close","vol"]
    )

    t1=trend(df1)
    t5=trend(df5)
    t15=trend(df15)

    if t1==t5==t15=="up":
        return "bull"

    if t1==t5==t15=="down":
        return "bear"

    return "mixed"

# ---------------------
# 回测胜率
# ---------------------

def backtest(df):

    wins=0
    trades=0

    for i in range(60,len(df)-5):

        vol=df["vol"].iloc[i]

        avg=df["vol"].iloc[i-50:i].mean()

        if vol>avg*1.5:

            trades+=1

            if df["close"].iloc[i+5]>df["close"].iloc[i]:
                wins+=1

    if trades==0:
        return 0

    return round(wins/trades*100,2)

# ---------------------
# AI训练
# ---------------------

def train_ai(df):

    X=[]
    y=[]

    for i in range(60,len(df)-5):

        vol=df["vol"].iloc[i]/df["vol"].iloc[i-50:i].mean()

        mom=df["close"].diff().iloc[i-3:i].sum()

        X.append([vol,mom])

        y.append(1 if df["close"].iloc[i+5]>df["close"].iloc[i] else 0)

    model=RandomForestClassifier()

    model.fit(X,y)

    joblib.dump(model,MODEL_FILE)

    return model

# ---------------------
# AI加载
# ---------------------

def load_ai(df):

    try:
        return joblib.load(MODEL_FILE)

    except:
        return train_ai(df)

# ---------------------
# AI预测
# ---------------------

def ai_predict(df):

    model=load_ai(df)

    vol=df["vol"].iloc[-1]/df["vol"].iloc[-50:-1].median()

    mom=df["close"].diff().iloc[-3:].sum()

    pred=model.predict([[vol,mom]])[0]

    prob=model.predict_proba([[vol,mom]])[0][1]

    return pred,prob

# ---------------------
# AI核心
# ---------------------

def ai_engine(df):

    res,sup=levels(df)

    strength=trend_strength(df)

    resonance=multi_tf()

    winrate=backtest(df)

    pred,prob=ai_predict(df)

    whale=whale_pump(df)

    sell=dump(df)

    fake=fake_break(df,res,sup)

    price=df["close"].iloc[-1]

    vol=df["vol"].iloc[-1]/df["vol"].iloc[-50:-1].median()

    status={"action":"AI扫描","motto":"静观其变","voice":None,"color":"#333"}

    if whale:
        status={"action":"庄家拉升","motto":"资金异动","voice":"检测庄家拉升","color":"green"}

    if sell:
        status={"action":"砸盘预警","motto":"注意风险","voice":"砸盘预警","color":"red"}

    if fake=="fake_up":
        status={"action":"假突破","motto":"突破无量","voice":"假突破警告","color":"purple"}

    if fake=="fake_down":
        status={"action":"假跌破","motto":"跌破无量","voice":"假跌破警告","color":"purple"}

    if vol>1.6 and price>res:
        status={"action":"直接开多","motto":"放量突破","voice":"放量起涨","color":"green"}

    if vol>1.6 and price<sup:
        status={"action":"直接开空","motto":"放量跌破","voice":"放量跌破","color":"red"}

    long_prob=round(prob*100,2)

    short_prob=100-long_prob

    score=min(int((vol>1.5)*25+(strength>3)*25+(long_prob>60)*25+(resonance=="bull")*25),100)

    return status,res,sup,strength,resonance,winrate,long_prob,short_prob,score

# ---------------------
# UI
# ---------------------

def render():

    df=fetch_data()

    if df is None:
        st.warning("数据异常")
        return

    status,res,sup,strength,resonance,winrate,long_prob,short_prob,score=ai_engine(df)

    now=time.time()

    if status["voice"]:

        if status["action"]!=st.session_state.last_signal and now-st.session_state.last_voice_time>15:

            speak(status["voice"])

            st.session_state.last_signal=status["action"]

            st.session_state.last_voice_time=now

    st.markdown(f"""
    <div style="background:{status['color']};padding:25px;border-radius:10px;text-align:center">
    <h1 style="color:white">{status['action']}</h1>
    <h3 style="color:yellow">{status['motto']}</h3>
    </div>
    """,unsafe_allow_html=True)

    st.write("多周期共振:",resonance)
    st.write("趋势强度:",strength)
    st.write("历史胜率:",winrate,"%")
    st.write("多头概率:",long_prob,"%")
    st.write("空头概率:",short_prob,"%")
    st.write("AI评分:",score,"/100")

    fig=go.Figure(data=[go.Candlestick(
        x=df["ts_dt"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"]
    )])

    fig.add_hline(y=res,line_dash="dash")
    fig.add_hline(y=sup,line_dash="dash")

    fig.update_layout(template="plotly_dark",height=500)

    st.plotly_chart(fig,use_container_width=True)

render()
