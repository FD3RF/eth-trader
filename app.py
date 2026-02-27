# ==================== 1. 计算器核心逻辑 ====================
def calculate_position_size(account_balance, risk_pct, entry_price, stop_loss):
    """
    根据风险百分比计算头寸大小
    """
    if entry_price == stop_loss: return 0, 0, 0
    
    # 每笔风险金额 (Risk Per Trade)
    risk_amount = account_balance * (risk_pct / 100)
    # 点位风险距离
    risk_per_eth = abs(entry_price - stop_loss)
    # 应开仓数量 (ETH)
    pos_size = risk_amount / risk_per_eth
    # 名义价值
    notional_value = pos_size * entry_price
    # 杠杆倍数参考 (假设账户全额作为保证金)
    leverage = notional_value / account_balance
    
    return pos_size, risk_amount, leverage

# ==================== 2. UI 渲染：交互式风控侧边栏 ====================
with st.sidebar:
    st.header("🧮 盈亏比与仓位助手")
    
    with st.form("calc_form"):
        acc_bal = st.number_input("账户总余额 ($)", value=1000.0, step=100.0)
        risk_p = st.slider("每笔愿意亏损总资金的 (%)", 0.5, 5.0, 1.0, 0.5)
        
        st.divider()
        # 默认从拦截位或现价填入
        entry_p = st.number_input("拟买入位", value=curr_p)
        sl_p = st.number_input("拟止损位", value=levels["support"] if levels["support"] else curr_p * 0.99)
        tp_p = st.number_input("拟止盈位", value=levels["resistance"] if levels["resistance"] else curr_p * 1.02)
        
        submitted = st.form_submit_button("🔥 计算执行方案")

    if submitted:
        # 计算盈亏比
        reward = abs(tp_p - entry_p)
        risk = abs(entry_p - sl_p)
        rr_ratio = reward / risk if risk > 0 else 0
        
        pos_size, r_amt, lev = calculate_position_size(acc_bal, risk_p, entry_p, sl_p)
        
        # 结果展示
        st.subheader("📊 执行建议")
        if rr_ratio < 1.5:
            st.error(f"盈亏比太低: {rr_ratio:.2f} (不值得博)")
        elif rr_ratio >= 3.0:
            st.success(f"极佳盈亏比: {rr_ratio:.2f}")
        else:
            st.warning(f"中等盈亏比: {rr_ratio:.2f}")
            
        st.write(f"🔹 **风险金额**: ${r_amt:.2f}")
        st.write(f"🔹 **建议开仓**: {pos_size:.3f} ETH")
        st.write(f"🔹 **参考杠杆**: {lev:.2f} x")
        
        # 存入执行笔记
        if st.button("📝 将此方案写入快照"):
            st.session_state.saved_notes.insert(0, {
                'time': datetime.now().strftime("%H:%M:%S"),
                'note': f"计划：在{entry_p}入场，RR比{rr_ratio:.2f}，仓位{pos_size:.3f}ETH"
            })
