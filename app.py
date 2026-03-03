# 在 compute_indicators 函数中添加以下逻辑

def compute_indicators(df):
    # 原有指标计算（现价、净流入、买压等）保留
    current_price = df['close'].iloc[-1]
    # ... 其他基础指标保持不变 ...

    # ---------- 新增：判断市场状态 ----------
    # 使用 ADX 或 EMA 斜率判断趋势强度
    high = df['high']
    low = df['low']
    close = df['close']
    
    # 计算 ADX（简化版，实际可使用 talib 或手动计算）
    period = 14
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(1)
    atr = tr.rolling(period).mean()
    plus_dm = (high - high.shift()).clip(lower=0)
    minus_dm = (low.shift() - low).clip(lower=0)
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9))
    adx = dx.rolling(period).mean().iloc[-1]
    
    # 判断趋势/震荡
    is_trend = adx > 25   # ADX > 25 认为有趋势
    is_oscillation = not is_trend
    
    # ---------- 方案A：回调顺势 ----------
    # 确定趋势方向
    ema_fast = close.ewm(span=8).mean()
    ema_slow = close.ewm(span=21).mean()
    trend_up = ema_fast.iloc[-1] > ema_slow.iloc[-1]
    trend_down = ema_fast.iloc[-1] < ema_slow.iloc[-1]
    
    # 寻找回调（以多头为例）
    if trend_up and is_trend:
        # 价格回踩EMA8但不破EMA21
        if close.iloc[-1] < ema_fast.iloc[-1] * 1.002 and close.iloc[-1] > ema_slow.iloc[-1]:
            # 前一根K线确认（如收阳）
            if close.iloc[-2] > open.iloc[-2]:
                signal = "回调做多"
                entry = current_price
                stop_loss = min(ema_slow.iloc[-1], close.iloc[-2] * 0.998)  # EMA21下方或前低
                take_profit = entry + (entry - stop_loss) * 1.5  # 盈亏比1.5:1
            else:
                signal = "观望中"
        else:
            signal = "观望中"
    elif trend_down and is_trend:
        # 空头回调
        if close.iloc[-1] > ema_fast.iloc[-1] * 0.998 and close.iloc[-1] < ema_slow.iloc[-1]:
            if close.iloc[-2] < open.iloc[-2]:
                signal = "回调做空"
                entry = current_price
                stop_loss = max(ema_slow.iloc[-1], close.iloc[-2] * 1.002)
                take_profit = entry - (stop_loss - entry) * 1.5
            else:
                signal = "观望中"
        else:
            signal = "观望中"
    
    # ---------- 方案B：均值回归 ----------
    elif is_oscillation:
        rsi = df['rsi'].iloc[-1]  # 假设已有RSI列
        if rsi < 25:
            signal = "超卖做多"
            entry = current_price
            stop_loss = current_price * 0.995
            take_profit = current_price * 1.005  # 小止盈
        elif rsi > 75:
            signal = "超买做空"
            entry = current_price
            stop_loss = current_price * 1.005
            take_profit = current_price * 0.995
        else:
            signal = "观望中"
    else:
        signal = "观望中"
    
    # 如果信号未触发，使用默认值
    if signal == "观望中":
        entry = current_price
        stop_loss = current_price * 0.998
        take_profit = current_price * 1.002
        risk_reward = 1.0
    else:
        risk = abs(entry - stop_loss)
        reward = abs(take_profit - entry)
        risk_reward = reward / risk if risk > 0 else 0
    
    return {
        'current_price': current_price,
        'net_inflow': net_inflow,
        'buy_pressure': buy_pressure,
        'mode': "趋势模式" if is_trend else "震荡模式",
        'suggestion': "顺势回调" if is_trend else "高抛低吸",
        'signal': signal,
        'entry': entry,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'risk_reward': risk_reward
    }
