import streamlit as st
import asyncio
import numpy as np
import ccxt.async_support as ccxt
from arch import arch_model  # 解决截图中的 arch 缺失
from scipy.optimize import minimize
from collections import deque

# --- 核心大脑：GARCH 风险预判 ---
class QuantumBrain:
    @staticmethod
    def predict_vol(returns):
        """识别波动率聚集，提前规避‘针尖行情’"""
        if len(returns) < 30: return np.std(returns)
        try:
            # 缩放数据提高稳定性
            model = arch_model(returns * 100, vol='Garch', p=1, q=1, dist='t', show_batch=False)
            res = model.fit(disp='off')
            return np.sqrt(res.forecast(horizon=1).variance.values[-1, -1]) / 100
        except: return np.std(returns)

# --- 引擎：全网上帝视角并发监控 ---
class GodModeEngine:
    def __init__(self, symbols):
        self.symbols = symbols
        # 建立异步连接池
        self.exchanges = {
            'binance': ccxt.binance({'enableRateLimit': True}),
            'okx': ccxt.okx({'enableRateLimit': True}),
            'bybit': ccxt.bybit({'enableRateLimit': True})
        }
        self.history = {s: deque(maxlen=100) for s in symbols}
        self.weights = {s: 0.0 for s in symbols}

    async def fetch_data(self):
        """并发抓取三大交易所，建立全网共识价"""
        tasks = [ex.fetch_ticker(s) for ex in self.exchanges.values() for s in self.symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        arb_data = {s: [] for s in self.symbols}
        for i, (ex_id, _) in enumerate(self.exchanges.items()):
            for j, s in enumerate(self.symbols):
                res = results[i * len(self.symbols) + j]
                if not isinstance(res, Exception) and res:
                    arb_data[s].append(res['last'])
                    if ex_id == 'binance': self.history[s].append(res['last'])
        return arb_data

# --- UI 架构：解决变量丢失与冲突 ---
st.set_page_config(page_title="GOD-MODE QUANTUM", layout="wide")

# 使用缓存资源确保引擎在刷新时不被重置
@st.cache_resource
def init_engine():
    return GodModeEngine(["BTC/USDT", "ETH/USDT"])

engine = init_engine()
st.title("👁️ QUANTUM V100: 全网实时上帝视角")

placeholder = st.empty()

async def main_loop():
    while True:
        try:
            # 1. 抓取多交易所实时数据
            data = await engine.fetch_data()
            
            # 2. 防御性检查：解决截图 的索引越界报错
            # 必须等待历史数据累积到可以计算收益率的程度（至少 2 个点）
            if any(len(engine.history[s]) < 5 for s in engine.symbols):
                with placeholder.container():
                    st.info("🛰️ 正在同步全网交易所数据，请等待预热 (约 5-10 秒)...")
                await asyncio.sleep(2)
                continue

            # 3. 大脑决策与渲染
            with placeholder.container():
                cols = st.columns(len(engine.symbols))
                for i, s in enumerate(engine.symbols):
                    prices = list(engine.history[s])
                    rets = np.diff(np.log(prices))
                    
                    vol = QuantumBrain.predict_vol(rets)
                    # 计算跨交易所偏离度 (Spread)
                    spread = np.std(data[s]) / np.mean(data[s]) if data[s] else 0
                    
                    with cols[i]:
                        st.metric(s, f"${prices[-1]:,.2f}", f"Spread: {spread*100:.4f}%")
                        st.write(f"预测波动率 (GARCH): {vol:.5f}")
                        # 风险雷达展示
                        st.progress(min(max(1.0 - vol*30, 0.0), 1.0), text="安全系数等级")
            
            await asyncio.sleep(1)
        except Exception as e:
            st.error(f"引擎运行异常: {e}")
            break

# 启动内核按钮
if st.sidebar.toggle("启动上帝视角实时内核"):
    asyncio.run(main_loop())
