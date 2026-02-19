import streamlit as st
import asyncio
import numpy as np
import ccxt.async_support as ccxt
from arch import arch_model
from collections import deque
import time

# --- 1. 大脑：风险与波动率计算 (解决 arch 报错) ---
class QuantumBrain:
    @staticmethod
    def predict_vol(returns):
        """识别波动率聚集，防止高位接针"""
        if len(returns) < 20: return np.std(returns)
        try:
            # 缩放 100 倍提高拟合稳定性
            am = arch_model(returns * 100, vol='Garch', p=1, q=1, dist='t', show_batch=False)
            res = am.fit(disp="off")
            return np.sqrt(res.forecast(horizon=1).variance.values[-1, -1]) / 100
        except: return np.std(returns)

# --- 2. 引擎：暴力连接穿透 (解决预热卡死) ---
class GodModeEngine:
    def __init__(self, symbols):
        self.symbols = symbols
        # 增加超时容忍度，适配云端网络
        self.exchanges = {
            'binance': ccxt.binance({'timeout': 30000, 'enableRateLimit': True}),
            'okx': ccxt.okx({'timeout': 30000, 'enableRateLimit': True})
        }
        self.history = {s: deque(maxlen=60) for s in symbols}

    async def fetch_data(self):
        """并发抓取价格，建立上帝视角共识"""
        tasks = [ex.fetch_ticker(s) for ex in self.exchanges.values() for s in self.symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        latest_prices = {s: [] for s in self.symbols}
        for i, (ex_id, _) in enumerate(self.exchanges.items()):
            for j, s in enumerate(self.symbols):
                idx = i * len(self.symbols) + j
                res = results[idx]
                if not isinstance(res, Exception) and res and 'last' in res:
                    latest_prices[s].append(res['last'])
                    if ex_id == 'binance': self.history[s].append(res['last'])
        return latest_prices

# --- 3. UI 界面：100% 自动运行架构 ---
st.set_page_config(page_title="GOD-MODE QUANTUM", layout="wide")

# 强制缓存引擎实例，防止 Streamlit 重复初始化
@st.cache_resource
def get_engine():
    return GodModeEngine(["BTC/USDT", "ETH/USDT"])

engine = get_engine()
st.title("👁️ QUANTUM V100: 上帝视角实时内核")

# 实时显示容器
placeholder = st.empty()

async def main():
    while True:
        try:
            # 1. 执行穿透抓取
            prices_map = await engine.fetch_data()
            
            # 2. 极致预热逻辑：只要有 2 个点就立刻显示，不再等待 (解决进度 0/3)
            ready = all(len(engine.history[s]) >= 2 for s in engine.symbols)
            
            if not ready:
                with placeholder.container():
                    st.info(f"🛰️ 正在穿透网络连接交易所... 进度: {[len(engine.history[s]) for s in engine.symbols]}/2")
                await asyncio.sleep(1)
                continue

            # 3. 渲染上帝视角面板
            with placeholder.container():
                cols = st.columns(len(engine.symbols))
                for i, s in enumerate(engine.symbols):
                    h_list = list(engine.history[s])
                    rets = np.diff(np.log(h_list))
                    
                    # 风险决策计算
                    vol = QuantumBrain.predict_vol(rets)
                    # 跨交易所偏离度
                    spread = np.std(prices_map[s]) / np.mean(prices_map[s]) if prices_map[s] else 0
                    
                    with cols[i]:
                        st.metric(s, f"${h_list[-1]:,.2f}", f"Spread: {spread*100:.4f}%")
                        st.write(f"实时风险系数: {vol:.5f}")
                        # 进度条显示环境安全性
                        st.progress(min(max(1.0 - vol*30, 0.0), 1.0), text="运行环境安全等级")
            
            await asyncio.sleep(1)
        except Exception as e:
            # 自动重连机制
            st.warning(f"正在自动恢复连接... {e}")
            await asyncio.sleep(2)

# --- 4. 暴力启动入口 ---
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception:
        # 兼容部分 Streamlit 环境的异步冲突
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(main())
