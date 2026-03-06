<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Streamlit 一键部署教程（专为你的 eth-trader 项目）</title>
    <style>
        body { font-family: 'Microsoft YaHei', sans-serif; background: #0e0e0e; color: #fff; padding: 30px; line-height: 1.8; }
        pre { background: #1f1f1f; padding: 20px; border-radius: 12px; overflow-x: auto; border: 1px solid #00ff9d; }
        h1, h2 { color: #00ff9d; }
        .step { background: #1a2a1a; padding: 20px; border-radius: 12px; margin: 20px 0; border-left: 6px solid #00ff9d; }
        .warning { background: #2a1a1a; padding: 15px; border-left: 6px solid #ff0066; }
        ul { padding-left: 25px; }
    </style>
</head>
<body>
    <h1>✅ Streamlit 部署教程（3分钟上线你的AI智能决策系统）</h1>
    <p>你的仓库 <strong>FD3RF/eth-trader</strong> 已经完美准备好（app.py + requirements.txt + 语音播报 + ccxt真实数据），我给你最简单、最稳定的部署方式：</p>
    <strong>推荐方式：Streamlit Community Cloud（免费、官方、一键部署）</strong>

    <div class="step">
        <h2>第1步：本地先测试（强烈建议）</h2>
        <ol>
            <li>确保已安装依赖：<code>pip install -r requirements.txt</code></li>
            <li>在项目根目录运行：<code>streamlit run app.py</code></li>
            <li>浏览器打开 http://localhost:8501，点击绿色按钮测试语音和实时K线是否正常</li>
        </ol>
        <p>✅ 全部OK后再部署</p>
    </div>

    <div class="step">
        <h2>第2步：把代码推送到 GitHub（你已经做好了）</h2>
        <p>确认你的仓库里有以下文件：</p>
        <ul>
            <li><code>app.py</code>（我们升级后的完整版）</li>
            <li><code>requirements.txt</code>（必须包含：streamlit, pandas, ccxt, plotly）</li>
            <li>（可选）.streamlit/config.toml（后面会教你加）</li>
        </ul>
        <p>如果还没推：</p>
        <pre><code>git add .
git commit -m "升级AI决策系统"
git push origin main</code></pre>
    </div>

    <div class="step">
        <h2>第3步：一键部署到 Streamlit Cloud（最重要）</h2>
        <ol>
            <li>打开浏览器进入 <a href="https://share.streamlit.io" target="_blank" style="color:#00ff9d">https://share.streamlit.io</a></li>
            <li>用 GitHub 账号登录（点右上角 Sign in with GitHub）</li>
            <li>点击 <strong>New app</strong> 按钮</li>
            <li>填写以下内容：
                <ul>
                    <li><strong>Repository</strong>：选择你的 <code>FD3RF/eth-trader</code></li>
                    <li><strong>Branch</strong>：main</li>
                    <li><strong>Main file path</strong>：app.py</li>
                </ul>
            </li>
            <li>点击 <strong>Deploy</strong></li>
        </ol>
        <p>部署过程大约 30~90 秒，完成后会自动给你一个链接，例如：<br>
        <strong>https://eth-trader.streamlit.app</strong></p>
    </div>

    <div class="step">
        <h2>第4步：优化上线体验（推荐）</h2>
        <p>在仓库根目录新建文件夹 <code>.streamlit</code>，里面新建文件 <code>config.toml</code>，内容粘贴下面代码：</p>
        <pre><code>[theme]
primaryColor = "#00ff9d"
backgroundColor = "#0e0e0e"
secondaryBackgroundColor = "#1f1f1f"
textColor = "#ffffff"

[server]
enableCORS = false
enableXsrfProtection = false

[client]
showSidebarNavigation = false</code></pre>
        <p>提交推送到 GitHub 后，Streamlit Cloud 会自动重启，界面就和你的第一张截图一模一样（深色+绿按钮）。</p>
    </div>

    <div class="warning">
        <strong>注意事项（必看）：</strong>
        <ul>
            <li>ccxt 调用的是币安公共接口，无需API Key，完全免费</li>
            <li>语音播报使用浏览器 Web Speech API（手机/电脑都支持）</li>
            <li>每天免费额度足够（每小时自动刷新数据）</li>
            <li>如果想自定义域名：部署成功后点 Settings → Custom domain</li>
            <li>连续两单止损自动暂停逻辑已在代码里，不用担心</li>
        </ul>
    </div>

    <h2>部署完成后的使用</h2>
    <p>打开你的 app 链接 → 直接点击右下角绿色大按钮「🚀 立即播报决策」</p>
    <p>AI机器人就会实时拉取以太坊5分钟永续合约数据，自动匹配口诀并语音播报，完全符合你第一张图片的界面！</p>

    <h2>其他高级部署方式（可选）</h2>
    <ul>
        <li><strong>Docker + Railway / Render</strong>：适合想要更多控制</li>
        <li><strong>阿里云 / 腾讯云服务器</strong>：用 <code>nohup streamlit run app.py --server.port 8501</code></li>
    </ul>
    <p>但 99% 的人用 Streamlit Cloud 就够了。</p>

    <p><strong>现在就去部署吧！</strong> 部署完把你的线上链接发给我，我帮你最后检查一次语音和策略是否完美。</p>
    <p>有任何卡住的地方（比如部署报错），直接截图告诉我，30秒帮你解决！</p>

    <hr>
    <p style="text-align:center; color:#666; font-size:14px;">已为你量身定制 · 基于 FD3RF/eth-trader 仓库 · 2026最新教程</p>
</body>
</html>
