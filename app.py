if enable_scan and run_scan:
    # 解析用户输入的候选值
    body_list = [float(x.strip()) for x in body_thresholds_scan.split(',')]
    vol_list = [int(x.strip()) for x in vol_ma_periods_scan.split(',')]
    break_list = [float(x.strip()) for x in break_thresholds_scan.split(',')]

    # 生成所有参数组合
    param_combinations = list(product(body_list, vol_list, break_list))
    total_combos = len(param_combinations)
    st.subheader(f"🔍 参数扫描结果（共 {total_combos} 组）")

    # 用于存储结果的列表
    scan_results = []

    # 进度条
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, (b_th, v_period, br_th) in enumerate(param_combinations):
        status_text.text(f"正在测试组合 {idx+1}/{total_combos}：实体={b_th}, 成交量周期={v_period}, 突破阈值={br_th}")

        try:
            # 构建特征（注意：lookback 使用侧边栏固定值）
            df_feat_scan = build_features(df, lookback, v_period)
            if len(df_feat_scan) == 0:
                continue

            if enable_oos:
                split_point = int(len(df_feat_scan) * train_ratio)
                test_df = df_feat_scan.iloc[split_point:]
                test_records, test_equity = simulate(
                    test_df, 0, len(test_df),
                    fee_rate, slippage, apply_costs,
                    rr_ratio, min_hold, br_th, b_th
                )
                stats = compute_stats(test_records, test_equity, "测试集")
                # 提取测试集指标
                scan_results.append({
                    'body_threshold': b_th,
                    'vol_ma_period': v_period,
                    'break_threshold': br_th,
                    '交易数': stats['测试集_交易数'],
                    '胜率': stats['测试集_胜率'],
                    '总盈利': stats['测试集_总盈利'],
                    '最大回撤': stats['测试集_最大回撤 (%)'],
                    '夏普比率': stats['测试集_夏普比率'],
                    '盈亏比': stats['测试集_盈亏比'],
                })
            else:
                # 全样本扫描
                records, equity = simulate(
                    df_feat_scan, 0, len(df_feat_scan),
                    fee_rate, slippage, apply_costs,
                    rr_ratio, min_hold, br_th, b_th
                )
                stats = compute_stats(records, equity, "全样本")
                scan_results.append({
                    'body_threshold': b_th,
                    'vol_ma_period': v_period,
                    'break_threshold': br_th,
                    '交易数': stats['全样本_交易数'],
                    '胜率': stats['全样本_胜率'],
                    '总盈利': stats['全样本_总盈利'],
                    '最大回撤': stats['全样本_最大回撤 (%)'],
                    '夏普比率': stats['全样本_夏普比率'],
                    '盈亏比': stats['全样本_盈亏比'],
                })
        except Exception as e:
            st.warning(f"组合 {b_th}, {v_period}, {br_th} 运行出错：{e}")
            # 记录为0或跳过
            scan_results.append({
                'body_threshold': b_th,
                'vol_ma_period': v_period,
                'break_threshold': br_th,
                '交易数': 0,
                '胜率': 0,
                '总盈利': 0,
                '最大回撤': 0,
                '夏普比率': 0,
                '盈亏比': 0,
            })

        progress_bar.progress((idx + 1) / total_combos)

    status_text.text("扫描完成！")

    # 显示结果表格
    if scan_results:
        result_df = pd.DataFrame(scan_results)
        st.dataframe(result_df)

        # 可选的排序和筛选
        st.subheader("📊 按交易数排序")
        st.dataframe(result_df.sort_values('交易数', ascending=False))

        st.subheader("📊 按夏普比率排序")
        st.dataframe(result_df.sort_values('夏普比率', ascending=False))

        # 添加下载按钮
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 下载扫描结果 CSV",
            data=csv,
            file_name="parameter_scan_results.csv",
            mime="text/csv"
        )
    else:
        st.warning("没有有效的扫描结果。")
