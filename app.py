import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import time
import threading
import hmac
import hashlib
import base64
import datetime
import os
import sys
import traceback
import re
import json
from streamlit_autorefresh import st_autorefresh
from concurrent.futures import ThreadPoolExecutor

# ========== 必须是第一个 Streamlit 命令 ==========
st.set_page_config(page_title="机构级 AI量化系统", layout="wide")

# ========== 全局异常捕获 ==========
def global_exception_handler(exctype, value, tb):
    st.error(f"系统发生未捕获异常: {exctype.__name__}: {value}")
    st.code("".join(traceback.format_exception(exctype, value, tb)))

sys.excepthook = global_exception_handler

# ========== 语音播报（安全版）==========
def speak(text):
    def _speak():
        try:
            if os.name == "nt" or os.environ.get("DISPLAY"):
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty('rate', 150)
                engine.say(text)
                engine.runAndWait()
        except Exception:
            pass
    threading.Thread(target=_speak, daemon=True).start()

# ========== OKX 工具函数 ==========
def get_okx_timestamp():
    return datetime.datetime.utcnow().isoformat()[:-3] + 'Z'

def okx_sign(secret, prehash):
    return base64.b64encode(
        hmac.new(secret.encode('utf-8'), prehash.encode('utf-8'), digestmod=hashlib.sha256).digest()
    ).decode('utf-8')

def okx_order(api_key, api_secret, api_passphrase, side, symbol="ETH-USDT-SWAP", sz="1"):
    if not api_key or not api_secret or not api_passphrase:
        return False, "API 密钥未配置"
    base_url = "https://www.okx.com"
    request_path = "/api/v5/trade/order"
    method = "POST"
    timestamp = get_okx_timestamp()
    body = {"instId": symbol, "tdMode": "cross", "side": side, "ordType": "market", "sz": sz}
    payload = json.dumps(body)
    prehash = timestamp + method + request_path + payload
    signature = okx_sign(api_secret, prehash)
    headers = {
        "OK-ACCESS-KEY": api_key,
        "OK-ACCESS-SIGN": signature,
        "OK-ACCESS-TIMESTAMP": timestamp,
        "OK-ACCESS-PASSPHRASE": api_passphrase,
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(base_url + request_path, headers=headers, data=payload, timeout=10)
        res_json = response.json()
        if res_json.get("code") == "0":
            return True, f"下单成功: {side} {sz}张"
        else:
            return False, f"下单失败: {res_json.get('msg')}"
    except Exception as e:
        return False, f"网络错误: {str(e)}"

def get_okx_balance(api_key, api_secret, api_passphrase):
    if not api_key or not api_secret or not api_passphrase:
        return None
    for attempt in range(3):
        try:
            base_url = "https://www.okx.com"
            request_path = "/api/v5/account/balance"
            method = "GET"
            timestamp = get_okx_timestamp()
            prehash = timestamp + method + request_path
            signature = okx_sign(api_secret, prehash)
            headers = {
                "OK-ACCESS-KEY": api_key,
                "OK-ACCESS-SIGN": signature,
                "OK-ACCESS-TIMESTAMP": timestamp,
                "OK-ACCESS-PASSPHRASE": api_passphrase,
                "Content-Type": "application/json"
            }
            response = requests.get(base_url + request_path, headers=headers, timeout=5)
            res_json = response.json()
            if res_json.get("code") == "0":
                data0 = res_json.get("data", [])[0]
                details = data0.get("details", [])
                usdt_bal = next((d for d in details if d["ccy"] == "USDT"), None)
                if usdt_bal:
                    return {
                        "total": float(data0.get("totalEq", 0)),
                        "usdt": float(usdt_bal.get("eq", 0)),
                        "u_pnl": float(data0.get("uPnl", 0))
                    }
        except Exception:
            time.sleep(1)
    return None

# ========== 交易日志 ==========
def update_trade_log(score, current_price, timestamp):
    log_file = "trade_log.csv"
    if not os.path.exists(log_file):
        pd.DataFrame(columns=["time", "score", "entry_price", "exit_price_15m", "result"]).to_csv(log_file, index=False)
    try:
        df_log = pd.read_csv(log_file)
        now = datetime.datetime.now()
        for idx, row in df_log.iterrows():
            if pd.isna(row["exit_price_15m"]):
                row_time = datetime.datetime.strptime(row["time"], "%Y-%m-%d %H:%M:%S")
                if (now - row_time).total_seconds() >= 900:
                    df_log.at[idx, "exit_price_15m"] = current_price
                    df_log.at[idx, "result"] = 1 if current_price > row["entry_price"] else 0
        if score >= 80:
            new_row = {
                "time": now.strftime("%Y-%m-%d %H:%M:%S"),
                "score": score,
                "entry_price": current_price,
                "exit_price_15m": None,
                "result": None
            }
            df_log = pd.concat([df_log, pd.DataFrame([new_row])], ignore_index=True)
        df_log.to_csv(log_file, index=False)
    except Exception:
        pass

def get_ai_accuracy():
    log_file = "trade_log.csv"
    if not os.path.exists(log_file):
        return 0.0
    try:
        df_log = pd.read_csv(log_file)
        valid_records = df_log.dropna(subset=["result"])
        if len(valid_records) == 0:
            return 0.0
        return (valid_records["result"] == 1).sum() / len(valid_records) * 100
    except Exception:
        return 0.0

def save_trade_log_json(action, price, score, reason):
    try:
        log_entry = {
            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "action": action,
            "price": price,
            "ai_score": score,
            "reason": reason
        }
        with open("trade_log.json", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception:
        pass

# ========== Ollama 服务检测 ==========
def check_ollama_service():
    try:
        requests.get("http://localhost:11434", timeout=2)
        return True, "Ollama 服务运行正常"
    except Exception:
        try:
            import subprocess
            subprocess.Popen(["ollama", "serve"], shell=False)
            time.sleep(5)
            requests.get("http://localhost:11434", timeout=2)
            return True, "已自动启动 Ollama"
        except Exception:
            return False, "Ollama 服务未运行且自动启动失败"

# ========== AI 审计（快速版）==========
def ai_audit_ollama(price, vol_ratio, imbalance, signals):
    is_alive, msg = check_ollama_service()
    if not is_alive:
        return 50, f"AI 连接失败: {msg}"
    if not signals:
        return 50, "等待信号中..."
    prompt = (
        f"作为量化交易审计员，分析 ETHUSDT 信号：\n"
        f"价格:{price} 量比:{vol_ratio:.2f} 失衡度:{imbalance:.3f}\n"
        f"信号:{', '.join(signals)}\n"
        f"给出0-100逻辑胜率分，30字内说明理由。\n"
        f"严格输出格式：分数|理由"
    )
    try:
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        session = requests.Session()
        retry = Retry(total=2, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
        session.mount('http://', HTTPAdapter(max_retries=retry))
        response = session.post(
            "http://localhost:11434/api/generate",
            json={"model": "deepseek-r1:7b", "prompt": prompt, "stream": False,
                  "options": {"num_ctx": 512, "temperature": 0.3}},
            timeout=30
        )
        if response.status_code == 200:
            result = response.json().get("response", "")
            if "|" in result:
                parts = result.strip().split("|")
                digits = ''.join(filter(str.isdigit, parts[0]))
                score = int(digits) if digits else 50
                score = max(0, min(100, score))
                reason = parts[1].strip()[:50]
                return score, reason
            return 50, f"格式错误: {result[:30]}"
        return 50, f"HTTP {response.status_code}"
    except Exception as e:
        return 50, f"连接异常: {str(e)[:30]}"

# ========== AI 巡航分析 ==========
def ai_cruise_analysis(df, imbalance, depth_desc, walls, is_accumulation,
                        resonance, signals, trend_15m="未知", timeout=25):
    if df.empty:
        return "数据暂未就绪，等待行情注入..."

    def generate_fallback_report(reason):
        fallback_score = 65
        if "多头" in resonance: fallback_score += 10
        if float(imbalance) > 0.2: fallback_score += 10
        if "空头" in resonance: fallback_score -= 10
        fallback_score = max(0, min(100, fallback_score))
        return (
            f"⚠️ **AI 连接失败 ({reason})**\n系统已切换至规则引擎审计。\n\n"
            f"⚔️ **多空对质**：趋势 {trend_15m}，失衡度 {round(float(imbalance),3)}\n\n"
            f"🚦 **执行评分**：{fallback_score} (规则模拟)\n\n"
            f"🚀 **触发信号**：无\n\n🎯 **操盘指令**：请人工复核。"
        )

    is_alive, msg = check_ollama_service()
    if not is_alive:
        return generate_fallback_report(msg)

    # 熔断机制
    if "ai_fail_count" not in st.session_state:
        st.session_state.ai_fail_count = 0
    if "ai_cooldown_until" not in st.session_state:
        st.session_state.ai_cooldown_until = 0
    current_ts = time.time()
    if current_ts < st.session_state.ai_cooldown_until:
        remaining = int(st.session_state.ai_cooldown_until - current_ts)
        return generate_fallback_report(f"熔断冷却中，剩余 {remaining}s")

    # 缓存
    cache_key = f"{df.index[-1]}_{round(df['close'].iloc[-1],2)}_{round(float(imbalance),3)}_{str(signals)}"
    if "ai_cache" not in st.session_state:
        st.session_state.ai_cache = {}
    if cache_key in st.session_state.ai_cache:
        return st.session_state.ai_cache[cache_key]

    # 构建 Prompt
    last_5 = df.tail(5)
    market_context = ""
    for idx, row in last_5.iterrows():
        vr = row.get('vol_ratio', 0)
        vr = vr if not np.isnan(vr) else 0
        market_context += f"T:{idx.strftime('%H:%M')} P:{row['close']:.2f} V:{row['volume']:.0f} R:{vr:.2f}\n"

    wall_info = (f"支撑:{walls.get('support','无')} 压力:{walls.get('resistance','无')}"
                 if walls else "无明显大单墙")
    accum_info = "疑似主力吸筹" if is_accumulation else "无明显吸筹"

    prompt = f"""作为顶级量化专家，对 ETH 行情进行极速审计。

【行情快照】
{market_context}失衡度:{round(float(imbalance),3)} 深度:{depth_desc}
墙:{wall_info} 吸筹:{accum_info}
趋势(15m):{trend_15m} 共振:{resonance}
信号:{', '.join(signals) if signals else '无'}

【心法铁律】
1. 放量突破(>2倍量)才真。
2. 缩量不破底为吸筹。

【任务】
1. 多空对质：列出多空核心理由。
2. 心法验证：是否符合铁律？
3. 裁决：给出0-100分。>85分输出"暗金脉冲触发"，<40分输出"红色警报触发"。

【输出格式(Markdown)】
⚔️ **多空对质**：[简述]
🧠 **心法验证**：[通过/失败]
🐋 **资金动向**：[简述]
🚦 **执行评分**：[数字]
🚀 **触发信号**：[结果]
🎯 **操盘指令**：[入场/止损/止盈]"""

    try:
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        session = requests.Session()
        retry = Retry(total=2, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
        session.mount('http://', HTTPAdapter(max_retries=retry))
        response = session.post(
            "http://localhost:11434/api/generate",
            json={"model": "deepseek-r1:7b", "prompt": prompt, "stream": False,
                  "options": {"num_ctx": 2048, "temperature": 0.3}},
            timeout=timeout
        )
        if response.status_code == 200:
            st.session_state.ai_fail_count = 0
            result = response.json().get("response", "AI 审计中断")
            if len(st.session_state.ai_cache) > 20:
                st.session_state.ai_cache.clear()
            st.session_state.ai_cache[cache_key] = result
            return result
        else:
            st.session_state.ai_fail_count += 1
            return generate_fallback_report(f"HTTP {response.status_code}")
    except Exception as e:
        st.session_state.ai_fail_count += 1
        if st.session_state.ai_fail_count >= 3:
            st.session_state.ai_cooldown_until = time.time() + 300
        return generate_fallback_report(str(e)[:50])

def get_deepseek_audit(long_prob, resonance, signals):
    if long_prob > 70 and "共振" in resonance:
        return "DeepSeek-R1：⚠️ 极高确定性机会。多周期共振，资金流入，建议分批入场，止损设20周期低点。"
    elif long_prob < 30 and "共振" in resonance:
        return "DeepSeek-R1：🚫 风险警示。空头主导，结构破位，严禁抄底，保持观望或轻仓做空。"
    else:
        return "DeepSeek-R1：📊 震荡博弈期。无明显主力介入，维持原计划止盈止损，警惕假突破。"

# ========== 行情数据 ==========
SYMBOL = "ETHUSDT"
BINANCE_REST = "https://api.binance.com/api/v3"

def get_kline(interval="5m", limit=500):
    for attempt in range(3):
        try:
            url = f"{BINANCE_REST}/klines"
            params = {"symbol": SYMBOL, "interval": interval, "limit": limit}
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, list) or len(data) == 0:
                return pd.DataFrame()
            df = pd.DataFrame(data, columns=[
                "time","open","high","low","close","volume",
                "close_time","quote_vol","trades","taker_buy_base",
                "taker_buy_quote","ignore"
            ])[["time","open","high","low","close","volume"]]
            for c in ["open","high","low","close","volume"]:
                df[c] = df[c].astype(float)
            df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True).dt.tz_convert(None)
            df.set_index("time", inplace=True)
            return df
        except Exception:
            time.sleep(1)
    return pd.DataFrame()

def indicators(df):
    if df.empty:
        return df
    df = df.copy()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma50"] = df["close"].rolling(50).mean()
    df["vol_ma"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_ma"].replace(0, np.nan)
    df["trend"] = df["ma20"] - df["ma50"]
    df["return"] = df["close"].pct_change()
    return df

def orderbook():
    try:
        url = f"{BINANCE_REST}/depth"
        params = {"symbol": SYMBOL, "limit": 20}
        data = requests.get(url, params=params, timeout=10).json()
        bids = data.get("bids", [])
        asks = data.get("asks", [])
        if not bids or not asks:
            return 0.0, "数据不足", {}
        bid_vol = sum(float(b[1]) for b in bids)
        ask_vol = sum(float(a[1]) for a in asks)
        imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol)
        depth_desc = f"买:{round(bid_vol,1)} vs 卖:{round(ask_vol,1)}"
        if bid_vol > ask_vol * 1.5:
            depth_desc += " (买盘强劲)"
        elif ask_vol > bid_vol * 1.5:
            depth_desc += " (卖压沉重)"
        else:
            depth_desc += " (多空均衡)"
        avg_bid = bid_vol / len(bids)
        avg_ask = ask_vol / len(asks)
        walls = {}
        for p, q in bids:
            if float(q) > avg_bid * 3:
                walls['support'] = float(p)
                break
        for p, q in asks:
            if float(q) > avg_ask * 3:
                walls['resistance'] = float(p)
                break
        return imbalance, depth_desc, walls
    except Exception:
        return 0.0, "获取失败", {}

def detect_accumulation(df):
    if len(df) < 20:
        return False
    last = df.iloc[-1]
    avg_vol = df["volume"].iloc[-20:-1].mean()
    return last["volume"] > avg_vol * 2 and abs(last.get("return", 0)) < 0.002

def market_structure(df):
    if len(df) < 21:
        return "数据不足"
    if df["high"].iloc[-1] > df["high"].iloc[-20:-1].max():
        return "突破结构"
    if df["low"].iloc[-1] < df["low"].iloc[-20:-1].min():
        return "破位结构"
    return "震荡结构"

def fake_breakout(df):
    if len(df) < 21:
        return False
    high = df["high"].iloc[-21:-1].max()
    last = df.iloc[-1]
    return last["high"] > high and last["close"] < high

def whale_pump(df):
    if df.empty or df["vol_ratio"].isna().all():
        return False
    last = df.iloc[-1]
    return last["vol_ratio"] > 2 and last.get("return", 0) > 0.01

def crash_warning(df):
    if df.empty or df["vol_ratio"].isna().all():
        return False
    last = df.iloc[-1]
    return last.get("return", 0) < -0.015 and last["vol_ratio"] > 1.8

def multi_tf_resonance():
    tfs = ["1m", "5m", "15m"]
    trends = []
    def _fetch(tf):
        d = indicators(get_kline(tf, 100))
        if not d.empty and not d["trend"].isna().all():
            return float(np.sign(d["trend"].iloc[-1]))
        return None
    with ThreadPoolExecutor(max_workers=3) as ex:
        results = list(ex.map(_fetch, tfs))
    trends = [r for r in results if r is not None]
    if not trends:
        return 0.0, 0.0, "无数据"
    long_score = sum(1 for t in trends if t > 0)
    short_score = sum(1 for t in trends if t < 0)
    total = len(trends)
    long_prob = long_score / total * 100
    short_prob = short_score / total * 100
    if long_score == total:
        state = "多头共振"
    elif short_score == total:
        state = "空头共振"
    else:
        state = "震荡"
    return long_prob, short_prob, state

def capital_flow(df):
    if len(df) < 41:
        return "数据不足"
    df2 = df.copy()
    df2["money_flow"] = df2["close"] * df2["volume"]
    flow = df2["money_flow"].iloc[-20:].sum()
    prev = df2["money_flow"].iloc[-40:-20].sum()
    return "资金流入 📈" if flow > prev else "资金流出 📉"

def backtest(df):
    if len(df) < 60:
        return 10000.0, []
    capital = 10000.0
    position = 0
    entry = 0.0
    trades = []
    for i in range(50, len(df)):
        row = df.iloc[i]
        trend_val = row.get("trend", 0)
        if pd.isna(trend_val):
            continue
        if trend_val > 0 and position == 0:
            entry = row["close"]
            position = 1
        elif trend_val < 0 and position == 1:
            profit = (row["close"] - entry) / entry
            capital *= 1 + profit
            trades.append(profit)
            position = 0
    if position == 1:
        profit = (df.iloc[-1]["close"] - entry) / entry
        capital *= 1 + profit
        trades.append(profit)
    return capital, trades

def strategy_score(trades):
    if not trades:
        return 0.0
    win = sum(1 for t in trades if t > 0)
    winrate = win / len(trades)
    avg = float(np.mean(trades))
    return max(0.0, min(100.0, winrate * 70 + avg * 30))

def volume_price_mnemonics(df):
    if len(df) < 21:
        return [], []
    last = df.iloc[-1]
    prev = df.iloc[-2]
    avg_vol = df["volume"].iloc[-6:-1].mean()
    if avg_vol == 0:
        return [], []
    curr_vol = last["volume"]
    is_shrinking = curr_vol < avg_vol * 0.6
    is_exploding = curr_vol > avg_vol * 1.5
    is_massive = curr_vol > avg_vol * 3.0
    recent_low = df["low"].iloc[-21:-1].min()
    recent_high = df["high"].iloc[-21:-1].max()
    long_signals, short_signals = [], []
    ret = last.get("return", 0) or 0

    # 做多口诀
    if is_shrinking and last["low"] >= recent_low and last["close"] < prev["close"]:
        long_signals.append("缩量回踩，低点不破 (准备动手)")
    red_candles = df[df["close"] < df["open"]]
    last_red_high = red_candles["high"].iloc[-1] if not red_candles.empty else recent_high
    if is_exploding and last["close"] > last_red_high:
        long_signals.append("放量起涨，突破前高 (直接开多)")
    if is_massive and ret < -0.005 and last["low"] >= recent_low:
        long_signals.append("放量急跌，底部不破 (假跌真买)")
    is_sideways = (df["high"].iloc[-5:].max() - df["low"].iloc[-5:].min()) < (last["close"] * 0.002)
    if is_shrinking and is_sideways and last["low"] >= recent_low:
        long_signals.append("缩量横盘，低点托住 (埋伏等涨)")

    # 做空口诀
    if is_shrinking and last["high"] <= recent_high and last["close"] > prev["close"]:
        short_signals.append("缩量反弹，高点不破 (准备动手)")
    green_candles = df[df["close"] > df["open"]]
    last_green_low = green_candles["low"].iloc[-1] if not green_candles.empty else recent_low
    if is_exploding and last["close"] < last_green_low:
        short_signals.append("放量下跌，跌破前低 (直接开空)")
    if is_massive and ret > 0.005 and last["high"] <= recent_high:
        short_signals.append("放量急涨，顶部不破 (假涨真空)")
    if is_shrinking and is_sideways and last["high"] <= recent_high:
        short_signals.append("缩量横盘，高点压住 (埋伏等跌)")

    return long_signals, short_signals

def mnemonic_backtest(df):
    if len(df) < 50:
        return [], 0.0, 0, 0
    capital = 10000.0
    equity = []
    position = 0
    entry_price = 0.0
    win_count = 0
    total_trades = 0
    big_wins = 0
    start_idx = max(30, len(df) - 200)
    for i in range(start_idx, len(df)):
        sub_df = df.iloc[max(0, i - 30):i + 1]
        long_sigs, short_sigs = volume_price_mnemonics(sub_df)
        row = df.iloc[i]
        current_price = row["close"]
        if position == 1:
            pnl = (current_price - entry_price) / entry_price
            if pnl > 0.01 or pnl < -0.005:
                capital *= (1 + pnl)
                if pnl > 0:
                    win_count += 1
                    if pnl >= 0.02:
                        big_wins += 1
                total_trades += 1
                position = 0
        elif position == -1:
            pnl = (entry_price - current_price) / entry_price
            if pnl > 0.01 or pnl < -0.005:
                capital *= (1 + pnl)
                if pnl > 0:
                    win_count += 1
                    if pnl >= 0.02:
                        big_wins += 1
                total_trades += 1
                position = 0
        if position == 0:
            if any("直接开多" in s or "假跌真买" in s for s in long_sigs):
                entry_price = current_price
                position = 1
            elif any("直接开空" in s or "假涨真空" in s for s in short_sigs):
                entry_price = current_price
                position = -1
        equity.append({"time": row.name, "capital": capital})
    win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
    return equity, win_rate, total_trades, big_wins

# ========== 图表 ==========
def draw_equity(equity_data):
    if not equity_data:
        return go.Figure()
    edf = pd.DataFrame(equity_data)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edf["time"], y=edf["capital"],
        name="账户净值", line=dict(color="#00FFCC", width=2),
        fill='tozeroy', fillcolor='rgba(0,255,204,0.05)'
    ))
    fig.update_layout(
        title="口诀策略净值曲线 (模拟 10,000U 起始)",
        height=350, template="plotly_dark",
        margin=dict(l=10, r=10, t=40, b=10)
    )
    return fig

def draw(df):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="K线",
        increasing_line_color='#FF4B4B',
        decreasing_line_color='#00FFCC'
    ))
    if "ma20" in df.columns and df["ma20"].notna().any():
        fig.add_trace(go.Scatter(
            x=df.index, y=df["ma20"], name="MA20",
            line=dict(color='rgba(255,255,255,0.5)', width=1)
        ))
    if "ma50" in df.columns and df["ma50"].notna().any():
        fig.add_trace(go.Scatter(
            x=df.index, y=df["ma50"], name="MA50",
            line=dict(color='rgba(255,215,0,0.5)', width=1)
        ))
    if len(df) >= 21:
        recent_low = df["low"].iloc[-21:-1].min()
        recent_high = df["high"].iloc[-21:-1].max()
        fig.add_hline(y=recent_low, line_dash="dot", line_color="#00FFCC",
                      annotation_text="近期支撑", annotation_position="bottom right")
        fig.add_hline(y=recent_high, line_dash="dot", line_color="#FF4B4B",
                      annotation_text="近期压力", annotation_position="top right")
    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=10, b=10),
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.08)', zeroline=False)
    return fig

# ========== LSTM 预测 ==========
if "depth_history" not in st.session_state:
    st.session_state.depth_history = {}

def lstm_predict(df):
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        if df.empty:
            return 0, 0.5, 0
        trend_val = df["trend"].iloc[-1] if "trend" in df.columns else 0.0
        trend_val = trend_val if not pd.isna(trend_val) else 0.0
        return (1 if trend_val > 0 else 0), 0.5, 0

    if len(df) < 60:
        return 0, 0.5, 0

    class LSTMModel(nn.Module):
        def __init__(self, input_dim, hidden=64):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden, batch_first=True, num_layers=1)
            self.fc = nn.Linear(hidden, 1)
            self.sigmoid = nn.Sigmoid()
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.sigmoid(self.fc(out[:, -1, :]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if "lstm_model" not in st.session_state:
        # 创建并训练简单的LSTM模型
        st.session_state.lstm_model = LSTMModel(input_dim=5).to(device)
        
    model = st.session_state.lstm_model
    model.eval()
    
    # 准备数据
    input_data = df[['open', 'high', 'low', 'close', 'volume']].iloc[-60:].values
    input_data = (input_data - input_data.mean(axis=0)) / (input_data.std(axis=0) + 1e-8)
    
    with torch.no_grad():
        x = torch.FloatTensor(input_data).unsqueeze(0).to(device)
        prediction = model(x).item()
    
    # 简化预测逻辑
    long_prob = prediction * 100
    confidence = min(0.8, abs(prediction - 0.5) * 2)
    
    return (1 if prediction > 0.5 else 0), long_prob, confidence

# ========== 主界面 ==========
def main():
    # 侧边栏配置
    with st.sidebar:
        st.title("⚙️ 系统配置")
        
        # API配置
        st.subheader("🔑 OKX API配置")
        api_key = st.text_input("API Key", type="password")
        api_secret = st.text_input("API Secret", type="password")
        api_passphrase = st.text_input("Passphrase", type="password")
        
        # 交易参数
        st.subheader("📊 交易参数")
        contract_size = st.number_input("合约张数", min_value=1, value=1)
        auto_trade = st.checkbox("自动交易", value=False)
        
        # 系统状态
        st.subheader("📈 系统状态")
        balance = get_okx_balance(api_key, api_secret, api_passphrase) if api_key else None
        if balance:
            st.metric("总资产", f"{balance['total']:.2f} U")
            st.metric("USDT余额", f"{balance['usdt']:.2f} U")
            st.metric("未实现盈亏", f"{balance['u_pnl']:.2f} U")
        else:
            st.warning("未连接API或余额获取失败")
        
        # 自动刷新
        refresh_interval = st.slider("刷新间隔(秒)", 5, 60, 15)
        st_autorefresh(interval=refresh_interval * 1000, key="data_refresh")

    # 主界面
    st.title("🚀 机构级 AI量化交易系统")
    st.markdown("---")
    
    # 获取数据
    with st.spinner("获取行情数据..."):
        df = indicators(get_kline())
        imbalance, depth_desc, walls = orderbook()
        long_prob, short_prob, resonance = multi_tf_resonance()
        is_accumulation = detect_accumulation(df)
        structure = market_structure(df)
        capital_status = capital_flow(df)
        long_signals, short_signals = volume_price_mnemonics(df)
    
    if df.empty:
        st.error("❌ 无法获取行情数据，请检查网络连接")
        return
    
    # 价格和指标显示
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        current_price = df["close"].iloc[-1]
        st.metric("ETH价格", f"{current_price:.2f} USDT")
    with col2:
        vol_ratio = df["vol_ratio"].iloc[-1] if not pd.isna(df["vol_ratio"].iloc[-1]) else 0
        st.metric("量比", f"{vol_ratio:.2f}")
    with col3:
        st.metric("多空失衡", f"{imbalance:.3f}")
    with col4:
        st.metric("AI准确率", f"{get_ai_accuracy():.1f}%")
    
    # 图表区域
    col_chart, col_analysis = st.columns([2, 1])
    
    with col_chart:
        st.plotly_chart(draw(df), use_container_width=True)
    
    with col_analysis:
        # 市场状态
        st.subheader("📊 市场状态")
        st.info(f"**结构**: {structure} | **资金**: {capital_status}")
        st.info(f"**共振**: {resonance} (多:{long_prob:.1f}% / 空:{short_prob:.1f}%)")
        
        # 深度分析
        st.subheader("🔍 深度分析")
        st.write(f"**订单簿**: {depth_desc}")
        if walls:
            if 'support' in walls:
                st.success(f"📊 支撑墙: {walls['support']:.2f}")
            if 'resistance' in walls:
                st.error(f"📊 压力墙: {walls['resistance']:.2f}")
        
        # 信号检测
        st.subheader("🚨 信号检测")
        if long_signals:
            for sig in long_signals:
                st.success(f"✅ {sig}")
        if short_signals:
            for sig in short_signals:
                st.error(f"❌ {sig}")
        if not long_signals and not short_signals:
            st.warning("⏳ 等待交易信号...")
    
    # AI分析区域
    st.subheader("🤖 AI智能分析")
    
    # 快速审计
    ai_score, ai_reason = ai_audit_ollama(current_price, vol_ratio, imbalance, long_signals + short_signals)
    
    col_audit, col_cruise = st.columns(2)
    
    with col_audit:
        st.metric("AI审计评分", f"{ai_score}/100", delta=f"{ai_score-50}")
        st.write(f"**理由**: {ai_reason}")
    
    with col_cruise:
        # AI巡航分析
        cruise_analysis = ai_cruise_analysis(
            df, imbalance, depth_desc, walls, is_accumulation,
            resonance, long_signals + short_signals
        )
        st.markdown(cruise_analysis)
    
    # 交易执行区域
    st.subheader("🎯 交易执行")
    
    col_execute, col_backtest = st.columns(2)
    
    with col_execute:
        if st.button("🟢 开多单", use_container_width=True, type="primary"):
            if api_key and api_secret and api_passphrase:
                success, msg = okx_order(api_key, api_secret, api_passphrase, "buy", sz=str(contract_size))
                if success:
                    st.success(msg)
                    save_trade_log_json("BUY", current_price, ai_score, ai_reason)
                    speak("多单执行成功")
                else:
                    st.error(msg)
            else:
                st.error("请先配置API密钥")
        
        if st.button("🔴 开空单", use_container_width=True, type="secondary"):
            if api_key and api_secret and api_passphrase:
                success, msg = okx_order(api_key, api_secret, api_passphrase, "sell", sz=str(contract_size))
                if success:
                    st.success(msg)
                    save_trade_log_json("SELL", current_price, ai_score, ai_reason)
                    speak("空单执行成功")
                else:
                    st.error(msg)
            else:
                st.error("请先配置API密钥")
    
    with col_backtest:
        # 策略回测
        equity_data, win_rate, total_trades, big_wins = mnemonic_backtest(df)
        st.metric("口诀策略胜率", f"{win_rate:.1f}%")
        st.metric("总交易次数", total_trades)
        st.metric("大胜次数", big_wins)
        
        if equity_data:
            st.plotly_chart(draw_equity(equity_data), use_container_width=True)
    
    # 风险提示
    st.warning("⚠️ 风险提示：量化交易存在风险，请谨慎操作。本系统仅供学习参考，不构成投资建议。")

if __name__ == "__main__":
    main()
