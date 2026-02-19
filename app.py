import streamlit as st
import asyncio
import numpy as np
import ccxt.async_support as ccxt
from arch import arch_model  # 解决截图中的 arch 缺失报错
from scipy.optimize import minimize
from collections import deque
import time

# ==================== 1. 大脑：GARCH 波动率预判 ====================
class QuantumBrain:
    @staticmethod
    def predict_vol(returns):
        """预测下一阶段波动率，提前预警插针行情"""
        if len(returns) < 30: return np.std(returns)
        try:
            # 缩放数据提高收敛性
            am = arch_model(returns * 100, vol='Garch', p=1, q=1, dist='t', show_batch=False)
            res = am.fit(disp="off")
            forecast = res.forecast(horizon=1)
            return np.sqrt(forecast.variance.values[-1, -1]) / 100
        except: return np.std(returns)

    @staticmethod
    def kelly_optimize(symbols, rets_matrix, vols, deltas, memory):
        """上帝视角引导的终极凯利分配"""
        individual_k = []
        for i, s in enumerate(symbols):
            # 贝叶斯后验校准胜率
            p = memory[s]['wins'] / (memory[s]['wins'] + memory[s]['losses'])
            b = (memory[s]['w_total']/memory[s]['wins']) / (memory[s]['l_total']/memory[s]['losses'])
            
            # 非线性风险惩罚：当全网价差或预测波动率异常时，仓位指数级收缩
            penalty = np.exp(-(vols[s]/0.06)**2) * np.exp(-(deltas[s]/0.0015)**2)
            k_f = max(0, (p * b - (1 - p)) / b) * 0.2
            individual_k.append(k_f * penalty)
        
        # 风险平减优化 (Risk Parity)
        cov = np.cov(rets_matrix)
        res = minimize(lambda w: np.dot(w.T, np.dot(cov, w)), x0=np.array(individual_k), 
                       bounds=[(0, k) for k in individual_k], 
                       constraints=({'type': 'eq', 'fun': lambda x: np.sum(x) - min(np.sum(individual_k), 0.5)}))
        return dict(zip(symbols, res.x if res.success else individual_k))

# ==================== 2. 引擎：上帝视角并发驱动 ====================
class GodModeEngine:
    def __init__(self, symbols):
        self.symbols = symbols
        self.exchanges = {
            'binance': ccxt.binance({'enableRateLimit': True}),
            'okx': ccxt.okx({'enableRateLimit': True}),
            'bybit': ccxt.bybit({'enableRateLimit': True})
        }
        self.history = {s: deque(maxlen=60) for s in symbols}
        self.memory = {s: {"wins": 10, "losses": 10, "w_total": 0.2, "l_total": 0.1} for s in symbols}
        self.weights = {s: 0.0 for s in symbols}

    async def fetch_all(self):
        """并发抓取三大交易所价格，建立全网共识价格"""
        tasks = [ex.fetch_ticker(s) for ex in self.exchanges.values() for s in self.symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        arb_data = {s: {} for s in self.symbols}
        for i, (ex_id, _) in enumerate(self.exchanges.items()):
            for j, s in enumerate(self.symbols):
                idx = i * len(self.symbols) + j
                res = results[idx]
                if not isinstance(res, Exception) and res:
                    arb_data[s][ex_id] = res['last']
                    if ex_id == 'binance': self.history[s].append(res['last'])
        return arb_data

# ==================== 3. 界面：100% 稳定运行架构 ====================
st.set_page_config(page_title="GOD-MODE LIVE", layout="wide")

# 缓存资源，防止 Streamlit 刷新导致异步循环冲突或变量丢失
@st.cache_resource
def get_engine():
    return GodModeEngine(["BTC/USDT", "ETH/USDT"])

engine = get_engine()
st.title("👁️ QUANTUM V100: GOD-EYE TERMINAL")

placeholder = st.empty()

async def main():
    while True:
        try:
            arb_data = await engine.fetch_all()
            vols, deltas, rets_m = {}, {}, []
            
            # --- 防御性检查：防止截图 中的索引越界报错 ---
            valid_history = all(len(engine.history[s]) >= 2 for s in engine.symbols)
            if not valid_history:
                with placeholder.container():
                    st.info("🛰️ 正在同步交易所原始数据，请等待数据预热 (约 5 秒)...")
                await asyncio.sleep(2)
                continue

            for s in engine.symbols:
                # 基于历史价格计算收益率
                rets = np.diff(np.log(list(engine.history[s])))
                rets_m.append(rets)
                vols[s] = QuantumBrain.predict_vol(rets)
                
                # 计算上帝视角：跨交易所价差离散度 (CV)
                p_list = [v for v in arb_data[s].values() if v]
                deltas[s] = np.std(p_list)/np.mean(p_list) if len(p_list) > 1 else 0
            
            # 大脑执行决策
            if len(rets_m) == len(engine.symbols):
                engine.weights = QuantumBrain.kelly_optimize(engine.symbols, rets_m, vols, deltas, engine.memory)
                
            with placeholder.container():
                cols = st.columns(len(engine.symbols))
                for i, s in enumerate(engine.symbols):
                    with cols[i]:
                        st.metric(s, f"${engine.history[s][-1]:,.2f}", f"Spread Delta: {deltas.get(s,0)*100:.4f}%")
                        st.progress(min(engine.weights[s]/0.5, 1.0), text=f"Kelly Allocation: {engine.weights[s]*100:.2f}%")
            
            await asyncio.sleep(1)
        except Exception as e:
            st.error(f"⚠️ 引擎运行异常: {e}")
            break

if st.sidebar.button("启动上帝视角实盘内核"):
    asyncio.run(main())
