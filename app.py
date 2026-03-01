def generate_signal(df, ls_ratio, mvrv_z=0, weights=None):
    """
    工程级信号生成：
    - 多周期过滤（4H趋势）
    - 极点突破入场
    - 多因子评分
    - 动态止损止盈
    返回：
    (胜率, 方向, 入场区, 止损, 止盈, 理由, 详细因子)
    """

    if df is None or len(df) < 50:
        return 50.0, 0, "数据不足", None, None, "数据不足", []

    # ===== 默认权重 =====
    default_weights = {
        'ema_cross': (20, -18),
        'rsi_mid': 10,
        'rsi_overbought': -10,
        'rsi_oversold': 5,
        'macd_hist_pos': 12,
        'macd_hist_neg': -12,
        'bb_upper': -15,
        'bb_lower': 10,
        'volume_surge': 8,
        'volume_shrink': -4,
        'net_flow_pos': 15,
        'net_flow_neg': -15,
        'ls_ratio_low': 8,
        'ls_ratio_high': -8,

        # 新增
        'trend4h': 18,
        'extreme_break': 20,
        'stoch_cross': 8,
        'adx_strong': 6,
        'mvrv_low': 12,
        'mvrv_high': -15,
    }
    if weights is None:
        weights = default_weights

    last = df.iloc[-1]
    score = 50.0
    reasons = []
    details = []

    # ================================
    # 4H趋势过滤
    # ================================
    trend4h = get_trend_4h()  # 1多 -1空 0不明
    if trend4h == 1:
        score += weights['trend4h']
        reasons.append("4H趋势多头")
        details.append({"因子": "4H趋势", "状态": "多头", "贡献": f"+{weights['trend4h']}"})
    elif trend4h == -1:
        score -= weights['trend4h']
        reasons.append("4H趋势空头")
        details.append({"因子": "4H趋势", "状态": "空头", "贡献": f"-{weights['trend4h']}"})
    else:
        details.append({"因子": "4H趋势", "状态": "不明", "贡献": "0"})

    # ================================
    # EMA
    # ================================
    ema_pos, ema_neg = weights['ema_cross']
    if last['ema_f'] > last['ema_s']:
        score += ema_pos
        reasons.append("EMA金叉")
        details.append({"因子": "EMA", "状态": "金叉", "贡献": f"+{ema_pos}"})
    else:
        score += ema_neg
        reasons.append("EMA死叉")
        details.append({"因子": "EMA", "状态": "死叉", "贡献": f"{ema_neg}"})

    # ================================
    # RSI
    # ================================
    if not pd.isna(last['rsi']):
        if 30 < last['rsi'] < 70:
            score += weights['rsi_mid']
            reasons.append(f"RSI中性({last['rsi']:.1f})")
            details.append({"因子": "RSI", "状态": "中性", "贡献": f"+{weights['rsi_mid']}"})
        elif last['rsi'] > 75:
            score += weights['rsi_overbought']
            reasons.append(f"RSI超买({last['rsi']:.1f})")
            details.append({"因子": "RSI", "状态": "超买", "贡献": f"{weights['rsi_overbought']}"})
        elif last['rsi'] < 25:
            score += weights['rsi_oversold']
            reasons.append(f"RSI超卖({last['rsi']:.1f})")
            details.append({"因子": "RSI", "状态": "超卖", "贡献": f"+{weights['rsi_oversold']}"})
    else:
        details.append({"因子": "RSI", "状态": "NA", "贡献": "0"})

    # ================================
    # MACD
    # ================================
    if last['macd_hist'] > 0:
        score += weights['macd_hist_pos']
        reasons.append("MACD柱为正")
        details.append({"因子": "MACD", "状态": "柱正", "贡献": f"+{weights['macd_hist_pos']}"})
    else:
        score += weights['macd_hist_neg']
        reasons.append("MACD柱为负")
        details.append({"因子": "MACD", "状态": "柱负", "贡献": f"{weights['macd_hist_neg']}"})

    # ================================
    # 布林带极点突破
    # ================================
    extreme_break = False
    if last['c'] > last['bb_upper'] and last['v'] > last['vol_ma'] * 1.5:
        extreme_break = True
        score += weights['extreme_break']
        reasons.append("突破上轨放量")
        details.append({"因子": "突破", "状态": "上轨放量", "贡献": f"+{weights['extreme_break']}"})
    elif last['c'] < last['bb_lower'] and last['v'] > last['vol_ma'] * 1.5:
        extreme_break = True
        score += weights['extreme_break']
        reasons.append("跌破下轨放量")
        details.append({"因子": "突破", "状态": "下轨放量", "贡献": f"+{weights['extreme_break']}"})
    else:
        details.append({"因子": "突破", "状态": "非极点", "贡献": "0"})

    # ================================
    # 成交量
    # ================================
    if last['v'] > last['vol_ma'] * 1.3:
        score += weights['volume_surge']
        details.append({"因子": "成交量", "状态": "放量", "贡献": f"+{weights['volume_surge']}"})
    else:
        score += weights['volume_shrink']
        details.append({"因子": "成交量", "状态": "缩量", "贡献": f"{weights['volume_shrink']}"})

    # ================================
    # 资金净流
    # ================================
    if last['net_flow'] > 0:
        score += weights['net_flow_pos']
        details.append({"因子": "资金净流", "状态": "净流入", "贡献": f"+{weights['net_flow_pos']}"})
    else:
        score += weights['net_flow_neg']
        details.append({"因子": "资金净流", "状态": "净流出", "贡献": f"{weights['net_flow_neg']}"})

    # ================================
    # 多空比极端
    # ================================
    if ls_ratio < 0.95:
        score += weights['ls_ratio_low']
        reasons.append("多空比极端空")
        details.append({"因子": "多空比", "状态": "极空", "贡献": f"+{weights['ls_ratio_low']}"})
    elif ls_ratio > 1.05:
        score += weights['ls_ratio_high']
        reasons.append("多空比极端多")
        details.append({"因子": "多空比", "状态": "极多", "贡献": f"{weights['ls_ratio_high']}"})
    else:
        details.append({"因子": "多空比", "状态": "中性", "贡献": "0"})

    # ================================
    # Stochastic（短周期超买超卖）
    # ================================
    if not pd.isna(last['stoch_k']) and not pd.isna(last['stoch_d']):
        if last['stoch_k'] > last['stoch_d'] and last['stoch_k'] < 20:
            score += weights['stoch_cross']
            reasons.append("Stoch超卖金叉")
            details.append({"因子": "Stoch", "状态": "超卖金叉", "贡献": f"+{weights['stoch_cross']}"})
        elif last['stoch_k'] < last['stoch_d'] and last['stoch_k'] > 80:
            score -= weights['stoch_cross']
            reasons.append("Stoch超买死叉")
            details.append({"因子": "Stoch", "状态": "超买死叉", "贡献": f"-{weights['stoch_cross']}"})

    # ================================
    # ADX趋势强度
    # ================================
    if not pd.isna(last['adx']) and last['adx'] > 25:
        score += weights['adx_strong']
        details.append({"因子": "ADX", "状态": "强趋势", "贡献": f"+{weights['adx_strong']}"})

    # ================================
    # MVRV（加密周期估值）
    # ================================
    if mvrv_z < 0:
        score += weights['mvrv_low']
        details.append({"因子": "MVRV", "状态": "低估", "贡献": f"+{weights['mvrv_low']}"})
    elif mvrv_z > 7:
        score += weights['mvrv_high']
        details.append({"因子": "MVRV", "状态": "泡沫", "贡献": f"{weights['mvrv_high']}"})

    # ================================
    # 概率与方向
    # ================================
    prob = max(min(score, 95), 5)
    direction = 1 if (trend4h == 1 and extreme_break and prob > 60) else \
                -1 if (trend4h == -1 and extreme_break and prob < 40) else 0

    # ================================
    # 动态止损止盈
    # ================================
    atr = last['atr'] if not pd.isna(last['atr']) else df['atr'].mean()
    if direction == 1:
        entry_zone = f"{last['c']-atr*0.5:.1f} ~ {last['c']+atr*0.5:.1f}"
        sl = last['c'] - atr * 1.5
        tp = last['c'] + atr * 2.5
    elif direction == -1:
        entry_zone = f"{last['c']-atr*0.5:.1f} ~ {last['c']+atr*0.5:.1f}"
        sl = last['c'] + atr * 1.5
        tp = last['c'] - atr * 2.5
    else:
        entry_zone = "观望"
        sl = tp = None

    reason_str = " | ".join(reasons) if reasons else "无明显信号"

    return prob, direction, entry_zone, sl, tp, reason_str, details 
