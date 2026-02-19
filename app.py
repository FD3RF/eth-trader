import streamlit as st
import asyncio
import numpy as np
import ccxt.async_support as ccxt
from arch import arch_model  # 核心避险模型
from scipy.optimize import minimize
from collections import deque

# ==================== 1. 大脑：GARCH 预测与风险控制 ====================
class QuantumBrain:
    @staticmethod
    def predict_vol(returns):
        """GARCH(1,1) 非线性波动率预测：提前感知插针风险"""
        if len(returns) < 30: return np.std(returns)
        try:
            # 数据缩放 100 倍以提高模型收敛性
            am = arch_model(returns * 100, vol='Garch', p=1, q=1, dist='t', show_batch=False)
            res = am.fit(disp="off")
            forecast = res.forecast(horizon=1)
            return np.sqrt(forecast.variance.values[-1, -1]) / 100
        except: return np.std(returns)

    @staticmethod
    def get_weights(symbols, rets_matrix, vols, deltas, memory):
        """上帝视角引导的动态凯利分配"""
        individual_k = []
        for i, s in enumerate(symbols):
            # 贝叶斯后验校准胜率
            p = memory[s]['wins'] / (memory[s]['wins'] + memory[s]['losses'])
            b = (memory[s]['w_total']/memory[s]['wins']) / (memory[s]['l_total']/memory[s]['losses'])
            
            # 动态风险惩罚：当价差或波动率激增时，仓位指数级收缩
            penalty = np.exp(-(vols[s]/0.06)**2) * np.exp(-(deltas[s]/0.0015)**2)
            k_f = max(0, (p * b - (1 - p)) / b) * 0.2
            individual_k.append(k_f * penalty)
        
        # 风险平减优化 (Risk Parity)
        cov = np.cov(rets_matrix)
        res = minimize(lambda w: np.dot(w.T, np.dot(cov, w)), x0=np.array(individual_k), 
                       bounds=[(0, k) for k in individual_k], 
                       constraints=({'type': 'eq', 'fun': lambda x: np.sum(x) - min(np.sum(individual_k), 0.5)}))
        return dict(zip(symbols, res.x if res.success else individual_k))

# ==================== 2. 引擎：上帝视角并发抓取 ====================
class LiveEngine:
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
        """并发抓取三大交易所价格，建立上帝视角"""
        tasks = [ex.fetch_ticker(s) for ex in self.exchanges.values() for s in self.symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        arb_data = {s: {} for s in self.symbols}
        for i, (ex_id, _) in enumerate(self.exchanges.items()):
            for j, s in enumerate(self.symbols):
                res = results[i * len(self.symbols) + j]
                if not isinstance(res, Exception) and res:
                    arb_data[s][ex_id] = res['last']
                    if ex_id == 'binance': self.history[s].append(res['last'])
        return arb_data

    async def close(self):
        for ex in self.exchanges.values(): await ex.close()

# ==================== 3. 界面：Streamlit 驱动逻辑 ====================
st.set_page_config(page_title="GOD-MODE LIVE", layout="wide")

# 解决变量作用域报错：将引擎持久化到 session_state
if 'engine' not in st.session_state:
    st.session_state.engine = LiveEngine(["BTC/USDT", "ETH/USDT"])

engine = st.session_state.engine
placeholder = st.empty()

async def main():
    while True:
        try:
            # 1. 抓取多交易所数据
            arb_data = await engine.fetch_all()
            
            # 2. 计算波动率与价差
            vols, deltas, rets_m = {}, {}, []
            for s in engine.symbols:
                if len(engine.history[s]) < 30: continue
                rets = np.diff(np.log(list(engine.history[s])))
                rets_m.append(rets)
                vols[s] = QuantumBrain.predict_vol(rets)
                # 计算上帝视角价差离散度
                p_list = list(arb_data[s].values())
                deltas[s] = np.std(p_list)/np.mean(p_list) if len(p_list)>1 else 0
            
            # 3. 大脑决策
            if len(rets_m) == len(engine.symbols):
                engine.weights = QuantumBrain.get_weights(engine.symbols, rets_m, vols, deltas, engine.memory)
                
            # 4. 极限 UI 渲染
            with placeholder.container():
                cols = st.columns(len(engine.symbols))
                for i, s in enumerate(engine.symbols):
                    with cols[i]:
                        st.metric(s, f"${engine.history[s][-1]:,.2f}", f"Delta: {deltas.get(s,0)*100:.4f}%")
                        st.write(f"建议仓位: {engine.weights[s]*100:.2f}%")
                        st.progress(min(engine.weights[s]/0.5, 1.0))
            
            await asyncio.sleep(1) # 实盘频控
        except Exception as e:
            st.error(f"引擎异常: {e}")
            break

if st.sidebar.button("启动上帝视角实盘内核"):
    asyncio.run(main()) # 注意：Streamlit Cloud 建议使用这种启动方式
