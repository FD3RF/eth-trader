# ==========================================
# 语音与战报
# ==========================================
def build_report(curr, upper, lower):

    if curr["c"] > upper:
        return {
            "status": "🚀 多头总攻",
            "voice": "放量突破前高，多头总攻，考虑做多"
        }

    if curr["c"] < lower:
        return {
            "status": "❄️ 空头突袭",
            "voice": "跌破低点，空头突袭，观望或反手"
        }

    if curr.get("buy_sig", False):
        return {
            "status": "🟢 缩量回踩企稳",
            "voice": "缩量回踩低点不破，下影企稳，准备动手做多"
        }

    if curr.get("sell_sig", False):
        return {
            "status": "🔴 放量下跌",
            "voice": "放量下跌，空头占优，暂不做多"
        }

    return {
        "status": "💎 窄幅震荡",
        "voice": "窄幅震荡，无方向，等待缩量回踩"
    }


def speak_voice(voice):

    # 浏览器 TTS：必须用户交互触发（Streamlit 按钮可触发）
    js = f"""
    <script>
    try {{
        if ('speechSynthesis' in window) {{
            window.speechSynthesis.cancel();
            const msg = new SpeechSynthesisUtterance("{voice}");
            msg.lang = "zh-CN";
            msg.rate = 1.0;
            msg.pitch = 1.0;
            window.speechSynthesis.speak(msg);
        }}
    }} catch (e) {{
        console.error(e);
    }}
    </script>
    """
    st.components.v1.html(js, height=0)


# ==========================================
# 主界面：更新部分
# ==========================================
def update_dashboard():

    try:
        df = fetch_data(symbol)
        if df is None or df.empty:
            return

        df, anchors = apply_warrior_logic(df, params)
        curr = df.iloc[-1]

        upper = anchors["upper"]
        lower = anchors["lower"]

        report = build_report(curr, upper, lower)
        status = report["status"]
        voice = report["voice"]

        # 战报头
        with header_area.container():
            color = "#10b981" if "多头" in status else "#ef4444" if "空头" in status else "#3b82f6"

            st.markdown(f"""
            <div class="status-card"
            style="border-left:8px solid {color};">
            <h2 style="color:{color};margin:0;">
            {status} | 现价: ${curr["c"]:.2f}
            </h2>
            </div>
            """, unsafe_allow_html=True)

        # 语音战报区
        with voice_area.container():
            st.info(f"语音战报：{voice}")

            # 手动播报按钮（调试与用户交互触发）
            if st.button("🔊 手动播报语音"):
                speak_voice(voice)

            # 自动播报：仅在内容变化时触发
            if "last_voice" not in st.session_state:
                st.session_state.last_voice = ""

            if voice != st.session_state.last_voice:
                st.session_state.last_voice = voice
                speak_voice(voice)

        # 指标卡
        with metric_area.container():
            c1, c2, c3, c4 = st.columns(4)

            c1.metric("当前现价", f"${curr['c']:.2f}")
            c2.metric("放量系数", f"{(curr['v']/curr['ma_v'] if curr['ma_v']>0 else 0):.2f}x")
            c3.metric("多头锚点", f"${lower:.2f}")
            c4.metric("空头锚点", f"${upper:.2f}")

        # 图表（保持不变）
        with chart_area.container():

            df_p = df.tail(60)

            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                row_heights=[0.75, 0.25],
                vertical_spacing=0.02
            )

            fig.add_trace(
                go.Candlestick(
                    x=df_p["time"],
                    open=df_p["o"],
                    high=df_p["h"],
                    low=df_p["l"],
                    close=df_p["c"]
                ),
                row=1, col=1
            )

            buys = df_p[df_p["buy_sig"]]
            sells = df_p[df_p["sell_sig"]]

            fig.add_trace(
                go.Scatter(
                    x=buys["time"],
                    y=buys["l"] * 0.998,
                    mode="markers",
                    marker=dict(symbol="triangle-up", size=14)
                ),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=sells["time"],
                    y=sells["h"] * 1.002,
                    mode="markers",
                    marker=dict(symbol="triangle-down", size=14)
                ),
                row=1, col=1
            )

            fig.add_hline(y=upper, line_dash="dash")
            fig.add_hline(y=lower, line_dash="dash")

            fig.add_trace(
                go.Bar(
                    x=df_p["time"],
                    y=df_p["v"],
                    opacity=0.6
                ),
                row=2, col=1
            )

            fig.update_layout(
                height=620,
                template="plotly_dark",
                showlegend=False,
                xaxis_rangeslider_visible=False,
                margin=dict(t=10, b=10, l=10, r=30),
                hovermode="x unified",
                uirevision="chart-lock"
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False}
            )

    except Exception as e:
        st.error(f"数据更新异常: {e}")
