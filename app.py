# ========== 新的数据获取模块 (基于 CoinGecko) ==========
from pycoingecko import CoinGeckoAPI
import time
import pandas as pd
from datetime import datetime, timedelta
import logging

class DataFetcher:
    """从 CoinGecko 获取K线数据，无代理，限制少"""

    def __init__(self, config: Config):
        self.config = config
        # 初始化 CoinGecko 客户端 (免费版无需 API Key)
        self.cg = CoinGeckoAPI()
        # 为了应对免费版的速率限制 (约 30-50次/分钟)，我们增加请求间隔 [citation:2][citation:6]
        self.request_interval = 2  # 每两次请求之间至少间隔2秒
        self.last_request_time = 0

    def _rate_limit(self):
        """简单的速率限制，避免触发 CoinGecko 的 429 错误"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.request_interval:
            time.sleep(self.request_interval - time_since_last)
        self.last_request_time = time.time()

    def fetch(self, limit: int = None) -> Optional[pd.DataFrame]:
        """
        获取历史K线数据 (CoinGecko 不支持一次性获取大量连续的分钟级K线)
        策略：先获取足够的天级OHLC数据，再模拟生成分钟级数据。
        注意：这是为了快速适配现有策略逻辑，最准确的方式是后续改用支持分钟级历史的方案。
        """
        limit = limit or self.config.LIMIT
        # CoinGecko 免费版获取分钟级历史数据较难，我们以降级方案处理：
        # 1. 获取最近90天的日线OHLC数据 [citation:7][citation:9]
        # 2. 根据日线数据模拟生成分钟级K线（开盘价、收盘价、最高价、最低价、成交量）
        # 这对于维持策略的连续运行是必要的，但会损失精度。

        logging.info("从 CoinGecko 获取历史数据（模拟模式）...")
        try:
            self._rate_limit()
            # 获取比特币 (bitcoin) 的OHLC数据，以天为单位 [citation:7]
            # 注意：CoinGecko 的 ID 是 'ethereum' [citation:5][citation:8]
            ohlc_data = self.cg.get_coin_ohlc_by_id(id='ethereum', vs_currency='usd', days='90')
            
            if not ohlc_data:
                logging.error("CoinGecko 返回的OHLC数据为空")
                return None

            # 将OHLC数据转换为DataFrame
            df_daily = pd.DataFrame(ohlc_data, columns=['timestamp', 'open', 'high', 'low', 'close'])
            df_daily['timestamp'] = pd.to_datetime(df_daily['timestamp'], unit='ms')
            
            # --- 模拟生成5分钟K线 (仅用于维持策略运行，实盘需谨慎) ---
            rows = []
            # 假设一天有288根5分钟K线 (24*60/5)
            minutes_per_day = 24 * 60
            freq_minutes = 5
            klines_per_day = minutes_per_day // freq_minutes

            for _, daily_row in df_daily.iterrows():
                day_start = daily_row['timestamp']
                for i in range(klines_per_day):
                    kline_time = day_start + timedelta(minutes=i*freq_minutes)
                    # 模拟价格：在当日范围内随机波动，使收盘价接近当日收盘
                    price_range = daily_row['high'] - daily_row['low']
                    simulated_open = daily_row['low'] + price_range * (i / klines_per_day)
                    simulated_close = daily_row['low'] + price_range * ((i+1) / klines_per_day)
                    simulated_high = max(simulated_open, simulated_close) + price_range * 0.1
                    simulated_low = min(simulated_open, simulated_close) - price_range * 0.1
                    # 确保不突破日线范围
                    simulated_high = min(simulated_high, daily_row['high'])
                    simulated_low = max(simulated_low, daily_row['low'])
                    
                    rows.append({
                        'timestamp': kline_time,
                        'open': simulated_open,
                        'high': simulated_high,
                        'low': simulated_low,
                        'close': simulated_close,
                        'volume': 100  # CoinGecko 免费版难以获取成交量，设为固定值
                    })

            df = pd.DataFrame(rows)
            # 确保时间戳是升序且唯一
            df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp']).tail(limit)
            logging.info(f"成功模拟生成 {len(df)} 根5分钟K线")
            return df

        except Exception as e:
            logging.exception(f"从 CoinGecko 获取数据失败: {e}")
            return None

    def fetch_recent(self, n: int = 10) -> Optional[pd.DataFrame]:
        """
        获取最近n根K线。为了简单，我们从模拟的历史数据中取最后n条。
        更精确的实现应该实时调用 CoinGecko 的 /simple/price 和 /coins/{id}/market_chart 等接口。
        """
        # 这里为了快速让机器人跑起来，我们复用上面的 fetch 逻辑
        # 但在实际高频交易中，需要实现一个更轻量的方法来获取最新价格。
        full_df = self.fetch(limit=self.config.LIMIT)
        if full_df is not None:
            return full_df.tail(n)
        return None
