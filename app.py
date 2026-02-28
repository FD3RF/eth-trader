# ... (保持 import 和 fetch 逻辑不变)

def master_engine_v69(df):
    # --- 基础计算保持 V68 精度 ---
    df['ema20'] = df['c'].ewm(span=20, adjust=False).mean()
    df['ema60'] = df['c'].ewm(span=60, adjust=False).mean()
    df['tr'] = np.maximum(df['h'] - df['l'], np.maximum(abs(df['h'] - df['c'].shift(1)), abs(df['l'] - df['c'].shift(1))))
    df['atr'] = df['tr'].rolling(14).mean()
    atr = df['atr'].iloc[-1]
    
    # 偏离度与动能
    df['deviation'] = (df['c'] - df['ema20']).abs() / atr
    delta = df['c'].diff()
    df['rsi'] = 100 - (100 / (1 + (delta.where(delta > 0, 0)).rolling(14).mean() / (-delta.where(delta < 0, 0)).rolling(14).mean()))
    
    # --- 升级：更灵敏的模式识别 ---
    curr, prev = df.iloc[-1], df.iloc[-2]
    vol_mean = df['v'].rolling(20).mean().iloc[-1]
    
    # 结构确认
    is_uptrend = curr['ema20'] > curr['ema60'] and curr['c'] > prev['c']
    is_downtrend = curr['ema20'] < curr['ema60'] and curr['c'] < prev['c']
    
    score = 0
    signal = {"type": None, "dir": None, "score": 0}

    # 1. 趋势回踩逻辑 (优化灵敏度)
    if is_uptrend:
        if curr['l'] <= curr['ema20'] * 1.001:
            score += 4 # 结构分
            if curr['v'] > vol_mean * 0.9: score += 2 # 动能分
            if score >= 5: signal = {"type": "趋势回踩", "dir": "LONG", "score": score}
    elif is_downtrend:
        if curr['h'] >= curr['ema20'] * 0.999:
            score += 4
            if curr['v'] > vol_mean * 0.9: score += 2
            if score >= 5: signal = {"type": "趋势回踩", "dir": "SHORT", "score": score}

    # 2. 反转逻辑 (下调偏离度门槛至 1.5)
    if signal['type'] is None and curr['deviation'] > 1.5:
        if curr['rsi'] < 32:
            signal = {"type": "乖离反转", "dir": "LONG", "score": 6}
        elif curr['rsi'] > 68:
            signal = {"type": "乖离反转", "dir": "SHORT", "score": 6}

    return signal, atr, curr['deviation']

# ... (UI 逻辑部分参考 V68，重点是 generate_plan 里的 SL/TP 动态调整)
