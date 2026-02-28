# ... (前面的数据获取部分保持不变)

def generate_smart_plan(curr_p, sup, res, win_rate, net_f, buy_ratio):
    # 策略常量：胜率低于 35% 说明抄底策略已经失效，必须反着干
    PANIC_ZONE = 35.0 
    
    # --- 1. AI 的心里话 (通俗提醒) ---
    tips = []
    if net_f < -15:
        tips.append("🔴 庄家正在撤退：看到没？大单一直在砸，这时候千万别去接飞刀！")
    elif net_f > 15:
        tips.append("🟢 有人在悄悄买：虽然价格没动，但主力资金在进场，可以关注。")
        
    if buy_ratio < 40:
        tips.append("💀 阵地快守不住了：买的人太少，空头太猛，下方可能还有个大坑。")

    # --- 2. 核心大白话决策 ---
    if win_rate < PANIC_ZONE:
        # 【阴跌/杀多模式】
        p = {
            "title": "🔴 AI 建议：反手做空",
            "tag": "别抄底，越抄越死",
            "color": "#ff4b4b",
            "why": f"最近抄底的 10 个人里有 8 个都亏了（胜率才 {win_rate:.1f}%），说明市场在走下坡路。咱不跟趋势作对。",
            "how_in": f"等价格稍微反弹到 ${res:.2f} 左右（也就是之前的‘假支撑’变‘真压力’的地方）再进场做空。",
            "how_out": f"万一价格冲回 ${res+10:.2f} 就要认怂，说明多头又杀回来了。",
            "money": f"目标看到 ${sup:.2f}，庄家大概率要砸到这里才停手。",
            "entry": res, "sl": res+10, "tp": sup
        }
    elif net_f < -20:
        # 【避险/拦截模式】
        p = {
            "title": "🟡 AI 建议：空仓看戏",
            "tag": "现在入场就是送人头",
            "color": "#ffd700",
            "why": "虽然价格跌了不少，但主力资金逃跑的速度太快了。现在买，就像在高速路上拦货车。",
            "how_in": "系统已经帮你锁死下单按钮了。等这波砸盘风头过去再说。",
            "how_out": "保住本金就是赢。",
            "money": "无",
            "entry": 0, "sl": 0, "tp": 0
        }
    else:
        # 【震荡/捡钱模式】
        p = {
            "title": "🟢 AI 建议：低吸波段",
            "tag": "行情稳住了，小仓位玩玩",
            "color": "#00ffcc",
            "why": "现在的行情是在‘跳舞’，没大跌风险，胜率也靠谱，适合在底部捞一把就走。",
            "how_in": f"在 ${sup:.2f} 附近挂买单，这里有很多大佬在护盘，比较安全。",
            "how_out": f"跌破 ${sup-10:.2f} 就撤，说明护盘的大佬跑路了。",
            "money": f"涨到 ${res:.2f} 附近赶紧卖，别贪心，落袋为安。",
            "entry": sup, "sl": sup-10, "tp": res
        }
    return p, tips

# --- UI 渲染改进 ---
with st.sidebar:
    st.header("🧠 AI 交易老大哥")
    
    plan, tip_list = generate_smart_plan(curr_p, sup_p, res_p, wr, net_f, (bids[1].sum()/(asks[1].sum()+bids[1].sum())*100))
    
    # 打印大白话提醒
    for tip in tip_list:
        st.warning(tip) if "🔴" in tip or "💀" in tip else st.success(tip)

    # 渲染带人性化解释的指令卡
    st.markdown(f"""
        <div style="border:4px solid {plan['color']}; padding:15px; border-radius:12px; background:rgba(255,255,255,0.05)">
            <h1 style="margin:0; color:{plan['color']}; font-size:24px">{plan['title']}</h1>
            <p style="margin:5px 0; color:{plan['color']}; font-weight:bold">「{plan['tag']}」</p>
            <p style="font-size:15px; margin-top:10px; line-height:1.5">💬 <b>大白话理由：</b>{plan['why']}</p>
            <hr style="border:0.5px solid #666">
            <p style="margin:8px 0">📍 <b>进场位置：</b>{plan['how_in']}</p>
            <p style="margin:8px 0">❌ <b>认输位置：</b>{plan['how_out']}</p>
            <p style="margin:8px 0; color:#00ffcc">💰 <b>收钱位置：</b>{plan['money']}</p>
        </div>
    """, unsafe_allow_html=True)
