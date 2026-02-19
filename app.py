import streamlit as st
import asyncio
import numpy as np
import pandas as pd
import ccxt.async_support as ccxt
import plotly.graph_objects as go
from arch import arch_model  # 确保 requirements.txt 有 arch
from scipy.optimize import minimize
from collections import deque

# ==================== 1. 大脑：核心数学内核 ====================
class QuantumBrain:
    @staticmethod
    def predict_garch_vol(returns):
        """GARCH(1,1) 非线性波动率预判：提前感知‘插针’风险"""
        if len(returns) < 30: return np.std(returns)
        try:
            # 数据缩放以提高收敛稳定性
            model = arch_model(returns * 100, vol='Garch', p=1, q=1, dist='t', show_batch=False)
            res = model.fit(disp='off')
            forecast = res.forecast(horizon=1)
            return np.sqrt(forecast.variance.values[-1, -1]) / 100
        except:
            return np.std(returns)

    @staticmethod
    def optimize_portfolio(symbols, returns_matrix, vols, arb_deltas, memory):
        """上帝视角 + 动态凯利：计算终极仓位权重"""
        individual_k = []
        for i, s in enumerate(symbols):
            # 贝叶斯后验胜率校准
            p = memory[s]['wins'] / (memory[s]['wins'] + memory[s]['losses'])
            b = (memory[s]['w_total']/memory[s]['wins']) / (memory[s]['l_total']/memory[s]['losses'])
            
            # 基础凯利公式 (bp-q)/b
            k_f = max(0, (p * b - (1 - p)) / b) * 0.2
            
            # 非线性风险衰减惩罚
            # 当预测波动率 > 6% 或 交易所价差 > 0.15% 时，仓位指数级塌缩
            penalty = np.exp(-(vols[s]/0.06)**2) * np.exp(-(arb_deltas[s]/0.0015)**2)
            individual_k.append(k_f * penalty)
        
        # 风险平减优化 (Minimize Variance)
        cov_matrix = np.cov(returns_matrix)
        def port_var(w): return np.dot(w.T, np.dot(cov_matrix, w))
        
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - min(np.sum(individual_k), 0.5)})
        res = minimize(port_var, x0=np.array(individual_k), bounds=[(0, k) for k in individual_k], constraints=cons)
        return dict(zip(symbols, res.x if res.success else individual_k))

# ==================== 2. 引擎：全局容器 ====================
class TradingSystem:
    def __init__(self, symbols):
        self.symbols = symbols
        # 初始化异步交易所连接
        self.exchanges = {
            'binance': ccxt.binance({'enableRateLimit': True}),
            'okx': ccxt.okx({'enableRateLimit': True}),
            'bybit': ccxt.bybit({'enableRateLimit': True})
        }
        self.history = {s: deque(maxlen=60) for s in symbols}
        self.memory = {s: {"wins": 10, "losses": 10, "w_total": 0.2, "l_total": 0.1} for s in symbols}
        self.weights = {s: 0.0 for s in symbols}

    async def fetch_global_data(self):
        """并发抓取上帝视角数据：Binance vs OKX vs Bybit"""
        tasks = []
        for ex in self.exchanges.values():
            for s in self.symbols:
                tasks.append(ex.fetch_ticker(s))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        arb_data = {s: {} for s in self.symbols}
        for i, (ex_name, ex) in enumerate(self.exchanges.items()):
            for j, s in enumerate(self.symbols):
                idx = i * len(self.symbols) + j
                ticker = results[idx]
                if not isinstance(ticker, Exception) and ticker:
                    price = ticker['last']
                    arb_data[s][ex_name] = price
                    if ex_name == 'binance': self.history[s].append(price)
        return arb_data

    async def close(self):
        for ex in self.exchanges.values():
            await ex.close()

# ==================== 3. 界面：Streamlit 终极终端 ====================
st.set_page_config(page_title="GOD-EYE QUANTUM", layout="wide")

# 使用 session_state 持久化系统实例，防止变量定义错误 (NameError)
if 'sys' not in st.session_state:
    st.session_state.sys = TradingSystem(["BTC/USDT", "ETH/USDT"])

sys = st.session_state.sys
st.title("👁️ QUANTUM V100: GOD-EYE VIEW")

if st.sidebar.button("清理并重启系统"):
    asyncio.run(sys.close())
    del st.session_state.sys
    st.rerun()

placeholder = st.empty()

async def live_loop():
    """实时主循环：异步驱动"""
    while True:
        try:
            # 1. 获取上帝视角数据
            arb_data = await sys.fetch_global_data()
            
            # 2. 核心大脑计算
            vols, deltas, returns_matrix = {}, {}, []
            for s in sys.symbols:
                prices = list(sys.history[s])
                if len(prices) < 30: continue
                
                rets = np.diff(np.log(np.array(prices) + 1e-9))
                returns_matrix.append(rets)
                vols[s] = QuantumBrain.predict_garch_vol(rets)
                
                # 计算跨交易所偏离度 (Spread Delta)
                p_list = list(arb_data[s].values())
                deltas[s] = np.std(p_list) / np.mean(p_list) if len(p_list) > 1 else 0

            if len(returns_matrix) == len(sys.symbols):
                sys.weights = QuantumBrain.optimize_portfolio(sys.symbols, returns_matrix, vols, deltas, sys.memory)

            # 3. 渲染 UI
            with placeholder.container():
                cols = st.columns(len(sys.symbols))
                for i, s in enumerate(sys.symbols):
                    with cols[i]:
                        st.metric(s, f"${sys.history[s][-1]:,.2f}", f"Delta: {deltas.get(s,0)*100:.4f}%")
                        st.write(f"建议仓位: {sys.weights[s]*100:.2f}%")
                        st.progress(min(sys.weights[s]/0.5, 1.0))
            
            await asyncio.sleep(1) # 避开 API 限制
        except Exception as e:
            st.error(f"运行中发生异常: {e}")
            break

# 启动异步循环
if 'loop_started' not in st.session_state:
    st.session_state.loop_started = True
    asyncio.run(live_loop())
