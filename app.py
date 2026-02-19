import streamlit as st
import asyncio
import numpy as np
import ccxt.async_support as ccxt
from arch import arch_model  # 成功加载
from collections import deque

# --- 核心大脑：风险预判 ---
class QuantumBrain:
    @staticmethod
    def predict_vol(returns):
        """预测波动率，防止针尖爆仓"""
        if len(returns) < 20: return np.std(returns)
        try:
            # 缩放数据以适应 GARCH 拟合
            am = arch_model(returns * 100, vol='Garch', p=1, q=1, dist='t', show_batch=False)
            res = am.fit(disp="off")
            return np.sqrt(res.forecast(horizon=1).variance.values[-1, -1]) / 100
        except: return np.std(returns)

# --- 引擎：数据采集与同步 ---
class GodModeEngine:
    def __init__(self, symbols):
        self.symbols = symbols
        # 强制开启异步多交易所连接
        self.exchanges = {
            'binance': ccxt.binance({'enableRateLimit': True}),
            'okx': ccxt.okx({'enableRateLimit': True})
        }
        self.history = {s: deque(maxlen=60) for s in symbols}

    async def fetch_data(self):
        """并发抓取价格"""
        tasks = [ex.fetch_ticker(s) for ex in self.exchanges.values() for s in self.symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        latest_prices = {s: [] for s in self.symbols}
        for i, (ex_id, _) in enumerate(self.exchanges.items()):
            for j, s in enumerate(self.symbols):
                res = results[i * len(self.symbols) + j]
                if not isinstance(res, Exception) and res:
                    latest_prices[s].append(res['last'])
                    if ex_id == 'binance': self.history[s].append(res['last'])
        return latest_prices

# --- UI 渲染逻辑 ---
st.set_page_config(page_title="GOD-MODE QUANTUM", layout="wide")

@st.cache_resource
def get_engine():
    return GodModeEngine(["BTC/USDT", "ETH/USDT"])

engine = get_engine()
st.title("👁️ QUANTUM V100: 上帝视角实时内核")

# 侧边栏控制
run = st.sidebar.toggle("启动实时上帝视角")

placeholder = st.empty()

async def main():
    while run:
        try:
            # 1. 抓取数据
            prices_map = await engine.fetch_data()
            
            # 2. 防御性检查：彻底解决截图 的索引越界报错
            # 必须等待历史序列至少累积 3 个点
            ready = all(len(engine.history[s]) >= 3 for s in engine.symbols)
            
            if not ready:
                with placeholder.container():
                    st.info(f"🛰️ 引擎预热中... 数据同步进度: {[len(engine.history[s]) for s in engine.symbols]}/3")
                await asyncio.sleep(1.5)
                continue

            # 3. 渲染面板
            with placeholder.container():
                cols = st.columns(len(engine.symbols))
                for i, s in enumerate(engine.symbols):
                    # 安全计算收益率
                    h_list = list(engine.history[s])
                    rets = np.diff(np.log(h_list))
                    vol = QuantumBrain.predict_vol(rets)
                    
                    # 跨交易所价差 (上帝视角)
                    spread = np.std(prices_map[s]) / np.mean(prices_map[s]) if prices_map[s] else 0
                    
                    with cols[i]:
                        st.metric(s, f"${h_list[-1]:,.2f}", f"全网价差: {spread*100:.4f}%")
                        st.write(f"预测波动率: {vol:.5f}")
                        st.progress(min(max(1.0 - vol*30, 0.0), 1.0), text="运行环境安全度")
            
            await asyncio.sleep(1)
        except Exception as e:
            st.error(f"⚠️ 系统中断: {e}")
            break

if run:
    asyncio.run(main())
else:
    st.info("👈 请在侧边栏开启内核以启动上帝视角")
