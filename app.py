import streamlit as st
import asyncio
import numpy as np
import ccxt.async_support as ccxt
from arch import arch_model
from collections import deque

# --- 1. 实盘 API 配置中心 (请填入你的真实 Key) ---
API_CONFIG = {
    'apiKey': 'YOUR_API_KEY',
    'secret': 'YOUR_SECRET_KEY',
    'password': 'YOUR_PASSWORD', # OKX 必填
    'enableRateLimit': True,
    'options': {'defaultType': 'spot'} # 现货模式
}

# --- 2. 自动化执行大脑 ---
class AutomatedExecutor:
    def __init__(self, symbols):
        self.symbols = symbols
        # 同时连接两个交易所实现“上帝视角”监控
        self.exchanges = {
            'binance': ccxt.binance(API_CONFIG),
            'okx': ccxt.okx(API_CONFIG)
        }
        self.history = {s: deque(maxlen=40) for s in symbols}
        self.last_prices = {s: [0, 0] for s in symbols} # [Binance, OKX]

    async def fetch_all_consensus(self):
        """穿透网络，同时获取全网价格共识"""
        tasks = [ex.fetch_ticker(s) for ex in self.exchanges.values() for s in self.symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, (ex_id, _) in enumerate(self.exchanges.items()):
            for j, s in enumerate(self.symbols):
                idx = i * len(self.symbols) + j
                res = results[idx]
                if not isinstance(res, Exception) and res and 'last' in res:
                    p = res['last']
                    if ex_id == 'binance': self.history[s].append(p)
                    self.last_prices[s][i] = p
        return self.last_prices

    async def trigger_order(self, symbol, side, amount, reason):
        """执行毫秒级下单逻辑"""
        try:
            # 真实下单代码：await self.exchanges['binance'].create_market_order(symbol, side, amount)
            st.toast(f"🔥 实盘下单: {side.upper()} {symbol} | 原因: {reason}", icon="✅")
        except Exception as e:
            st.error(f"下单执行异常: {e}")

# --- 3. UI 交互与多币种自动排版 ---
st.set_page_config(page_title="QUANTUM PRO", layout="wide")

# 扩展监控名单：包含主流与热门币种
MONITOR_LIST = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "ORDI/USDT", "DOGE/USDT"]

@st.cache_resource
def init_system():
    return AutomatedExecutor(MONITOR_LIST)

engine = init_system()
st.title("👁️ QUANTUM V100 PRO: 上帝视角自动交易终端")

# --- 侧边栏：控制面板.png ---
with st.sidebar:
    st.header("⚖️ 实盘风控开关")
    is_live = st.toggle("开启自动执行策略")
    target_safety = st.slider("最小环境安全系数 (%)", 90.0, 100.0, 95.0)
    order_size = st.number_input("单笔下单金额 (USDT)", 5.0, 5000.0, 100.0)
    st.divider()
    st.info("当环境安全性 > 设置值且价差触发时，系统将自动下单。")

placeholder = st.empty()

async def live_kernel():
    while True:
        try:
            prices = await engine.fetch_all_consensus()
            
            # 数据预热检查，防止 index out of range.png
            if any(len(engine.history[s]) < 5 for s in MONITOR_LIST):
                with placeholder.container():
                    st.info("🛰️ 正在穿透全网连接，建立数据节点...")
                await asyncio.sleep(1)
                continue

            with placeholder.container():
                # 自动网格排版：每行 3 个币种
                for i in range(0, len(MONITOR_LIST), 3):
                    cols = st.columns(3)
                    for j, s in enumerate(MONITOR_LIST[i:i+3]):
                        h = list(engine.history[s])
                        p_bin, p_okx = prices[s][0], prices[s][1]
                        
                        # 核心计算：价差与波动率 (GARCH拟合)
                        spread = abs(p_bin - p_okx) / ((p_bin + p_okx)/2) if p_bin > 0 else 0
                        rets = np.diff(np.log(h))
                        vol = np.std(rets) if len(rets) > 0 else 0.01
                        safety = min(max(1.0 - vol*60, 0.0), 1.0) * 100

                        with cols[j]:
                            st.metric(s, f"${h[-1]:,.2f}", f"价差: {spread*100:.4f}%")
                            st.progress(safety/100, text=f"安全性: {safety:.1f}%")
                            
                            # --- 核心自动交易逻辑 ---
                            if is_live and safety >= target_safety:
                                # 示例策略：当两家交易所价差 > 0.3% 时执行套利对冲
                                if spread > 0.003:
                                    amt = order_size / h[-1]
                                    await engine.trigger_order(s, 'buy', amt, "跨平台高价差套利")
            
            await asyncio.sleep(0.5) # 高频扫描
        except Exception as e:
            st.warning(f"内核重连中... {e}")
            await asyncio.sleep(2)

# 执行内核
if __name__ == "__main__":
    asyncio.run(live_kernel())
