#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETHUSDT 1分钟高频交易机器人 (优化版)
策略：左侧入场，基于缩量回踩/反弹关键位
配置参数见下方 Config 类
"""

import time
import json
import logging
from logging.handlers import RotatingFileHandler
from collections import deque
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import pandas as pd
import numpy as np
import requests

# ========== 配置类 ==========
class Config:
    # 交易对及周期
    SYMBOL = "ETHUSDT"
    INTERVAL = "1m"
    LIMIT = 1000                     # 每次加载K线数量（足够计算指标）
    
    # 策略核心参数 (方向A高频)
    LOOKBACK = 20                    # 关键点回溯周期
    VOL_WINDOW = 5                    # 成交量均线周期
    EXPAND_THRESH = 1.0               # 放量阈值（左侧入场不使用，但保留）
    SHRINK_THRESH = 0.8               # 缩量阈值
    BODY_MIN_RATIO = 0.05              # 实体占比最小值
    TOUCH_TOLERANCE = 0.005            # 接近关键点容忍度 (0.5%)
    STOP_LOSS_PCT = 0.002              # 止损比例 (0.2%)
    RISK_REWARD = 3.0                  # 盈亏比
    
    # 信号过滤
    MIN_SAME_SIGNAL_INTERVAL = 3       # 同方向信号最小间隔（根K线数）
    ALLOW_MULTI_POSITIONS = False      # 是否允许同时持有多个方向仓位 (建议False)
    
    # 系统参数
    PROXY = None                       # 代理服务器，如 "http://127.0.0.1:7890"
    REFRESH_SEC = 10                    # 轮询间隔（秒）
    MAX_RETRIES = 3                     # 数据获取最大重试次数
    RETRY_DELAY = 2                     # 重试等待秒数
    
    # 日志
    LOG_FILE = "eth_bot_optimized.log"
    LOG_LEVEL = logging.INFO

# ========== 数据获取模块 ==========
class DataFetcher:
    """从币安获取K线数据，支持代理和重试"""
    
    ENDPOINTS = [
        "https://api.binance.com/api/v3/klines",
        "https://api1.binance.com/api/v3/klines",
        "https://api2.binance.com/api/v3/klines"
    ]
    
    def __init__(self, config: Config):
        self.config = config
        self.proxies = {"http": config.PROXY, "https": config.PROXY} if config.PROXY else None
        self.session = requests.Session()
        self.session.proxies.update(self.proxies) if self.proxies else None
    
    def fetch(self, limit: int = None) -> Optional[pd.DataFrame]:
        """获取K线数据，返回DataFrame"""
        limit = limit or self.config.LIMIT
        params = {
            "symbol": self.config.SYMBOL,
            "interval": self.config.INTERVAL,
            "limit": limit
        }
        
        for endpoint in self.ENDPOINTS:
            for attempt in range(self.config.MAX_RETRIES):
                try:
                    resp = self.session.get(endpoint, params=params, timeout=5)
                    if resp.status_code == 200:
                        data = resp.json()
                        df = pd.DataFrame(data, columns=[
                            'timestamp', 'open', 'high', 'low', 'close', 'volume',
                            'close_time', 'q_vol', 'trades', 't_buy_base', 't_buy_quote', 'ignore'
                        ])
                        # 类型转换
                        for col in ['open', 'high', 'low', 'close', 'volume']:
                            df[col] = pd.to_numeric(df[col])
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                        return df
                    else:
                        logging.warning(f"HTTP {resp.status_code} from {endpoint}")
                except Exception as e:
                    logging.warning(f"Attempt {attempt+1} failed for {endpoint}: {e}")
                time.sleep(self.config.RETRY_DELAY)
        logging.error("All endpoints failed after retries.")
        return None
    
    def fetch_recent(self, n: int = 10) -> Optional[pd.DataFrame]:
        """仅获取最近n根K线（用于快速更新）"""
        return self.fetch(limit=n)

# ========== 策略计算模块 ==========
class StrategyEngine:
    """左侧入场策略引擎，支持批量计算和增量更新"""
    
    def __init__(self, config: Config):
        self.config = config
        self.last_signal_time = None          # 上次信号时间戳
        self.last_signal_type = None           # 上次信号类型 (1或-1)
        self.cached_df = None                  # 缓存的历史DataFrame，用于增量更新
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有技术指标"""
        data = df.copy()
        
        # 关键点（前高/前低，shift 1避免使用当前K线）
        data['prev_high'] = data['high'].rolling(self.config.LOOKBACK, min_periods=1).max().shift(1)
        data['prev_low']  = data['low'].rolling(self.config.LOOKBACK, min_periods=1).min().shift(1)
        
        # 成交量均线
        data['vol_ma'] = data['volume'].rolling(self.config.VOL_WINDOW, min_periods=1).mean()
        
        # 实体占比
        data['body'] = abs(data['close'] - data['open'])
        data['range'] = data['high'] - data['low']
        data['body_ratio'] = data['body'] / data['range']
        data['solid'] = data['body_ratio'] >= self.config.BODY_MIN_RATIO
        
        return data
    
    def detect_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """在已计算指标的数据上检测左侧入场信号"""
        # 左侧做多条件
        long_cond = (
            (data['low'] <= data['prev_low'] * (1 + self.config.TOUCH_TOLERANCE)) &
            (data['close'] > data['prev_low'] * 0.999) &          # 未有效跌破
            (data['volume'] < data['vol_ma'] * self.config.SHRINK_THRESH) &
            data['solid']
        )
        
        # 左侧做空条件
        short_cond = (
            (data['high'] >= data['prev_high'] * (1 - self.config.TOUCH_TOLERANCE)) &
            (data['close'] < data['prev_high'] * 1.001) &          # 未有效突破
            (data['volume'] < data['vol_ma'] * self.config.SHRINK_THRESH) &
            data['solid']
        )
        
        # 初始化信号列
        data['signal'] = 0
        data.loc[long_cond, 'signal'] = 1
        data.loc[short_cond, 'signal'] = -1
        # 如果同时满足，取消信号（理论上不会发生）
        data.loc[long_cond & short_cond, 'signal'] = 0
        
        # 计算入场价、止损、止盈（使用收盘价作为入场价）
        data['entry'] = data['close']
        data['stop'] = np.nan
        data['target'] = np.nan
        
        # 为每个信号计算止损止盈
        for idx in data[data['signal'] != 0].index:
            row = data.loc[idx]
            if row['signal'] == 1:  # 做多
                stop = row['prev_low'] * (1 - self.config.STOP_LOSS_PCT)
                target = row['entry'] + (row['entry'] - stop) * self.config.RISK_REWARD
            else:  # 做空
                stop = row['prev_high'] * (1 + self.config.STOP_LOSS_PCT)
                target = row['entry'] - (stop - row['entry']) * self.config.RISK_REWARD
            data.loc[idx, 'stop'] = stop
            data.loc[idx, 'target'] = target
        
        return data
    
    def process_full(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理完整DataFrame，返回带信号的数据"""
        data = self.calculate_indicators(df)
        data = self.detect_signals(data)
        self.cached_df = data
        return data
    
    def update_latest(self, new_rows: pd.DataFrame) -> Tuple[bool, Optional[Dict]]:
        """
        增量更新：传入最新的几根K线，返回是否有新信号及信号详情
        new_rows: 包含最新K线的DataFrame，需按时间升序
        """
        if self.cached_df is None:
            # 首次调用，需先处理完整数据
            return False, None
        
        # 合并新旧数据
        combined = pd.concat([self.cached_df, new_rows]).drop_duplicates(subset=['timestamp']).sort_values('timestamp')
        combined = self.calculate_indicators(combined)
        combined = self.detect_signals(combined)
        
        # 找出新添加的行中有信号的
        old_len = len(self.cached_df)
        new_part = combined.iloc[old_len:]
        signal_rows = new_part[new_part['signal'] != 0]
        
        self.cached_df = combined
        
        if signal_rows.empty:
            return False, None
        
        # 取最新的一条信号（按时间）
        latest_signal = signal_rows.iloc[-1].to_dict()
        
        # 信号冷却检查
        if latest_signal['signal'] == self.last_signal_type:
            if self.last_signal_time is not None:
                # 计算与上次信号的时间间隔（以K线数为单位）
                last_idx = self.cached_df[self.cached_df['timestamp'] == self.last_signal_time].index
                curr_idx = self.cached_df[self.cached_df['timestamp'] == latest_signal['timestamp']].index
                if len(last_idx) > 0 and len(curr_idx) > 0:
                    interval = curr_idx[0] - last_idx[0]
                    if interval < self.config.MIN_SAME_SIGNAL_INTERVAL:
                        logging.debug(f"信号冷却中，跳过同方向信号 (间隔 {interval} < {self.config.MIN_SAME_SIGNAL_INTERVAL})")
                        return False, None
        
        self.last_signal_time = latest_signal['timestamp']
        self.last_signal_type = latest_signal['signal']
        return True, latest_signal

# ========== 交易模拟模块 ==========
@dataclass
class Position:
    type: str          # 'long' or 'short'
    entry: float
    stop: float
    target: float
    time: datetime
    status: str = 'open'
    exit: float = None
    pnl: float = None

class SimulatedTrader:
    """模拟交易记录与风控"""
    
    def __init__(self, config: Config):
        self.config = config
        self.positions: List[Position] = []
        self.closed: List[Position] = []
        self.daily_stats = deque(maxlen=90)  # 保存90天统计
    
    def can_open_new(self, signal_type: int) -> bool:
        """检查是否可以开新仓"""
        if not self.config.ALLOW_MULTI_POSITIONS and len(self.positions) > 0:
            return False
        # 可选：限制最大持仓数
        return True
    
    def open_position(self, signal: Dict) -> Optional[Position]:
        """根据信号开仓"""
        sig_type = signal['signal']
        if sig_type == 1:
            pos = Position(
                type='long',
                entry=signal['entry'],
                stop=signal['stop'],
                target=signal['target'],
                time=signal['timestamp']
            )
        elif sig_type == -1:
            pos = Position(
                type='short',
                entry=signal['entry'],
                stop=signal['stop'],
                target=signal['target'],
                time=signal['timestamp']
            )
        else:
            return None
        
        if self.can_open_new(sig_type):
            self.positions.append(pos)
            logging.info(f"开仓: {pos.type.upper()} @ {pos.entry:.2f} | 止损: {pos.stop:.2f} | 止盈: {pos.target:.2f} | 时间: {pos.time}")
            return pos
        else:
            logging.debug(f"无法开仓: 已有持仓或风控限制")
            return None
    
    def update_positions(self, current_kline: Dict):
        """根据最新K线的高低价检查止损止盈"""
        high = current_kline['high']
        low = current_kline['low']
        ts = current_kline['timestamp']
        
        for pos in self.positions[:]:
            if pos.status != 'open':
                continue
            
            if pos.type == 'long':
                if low <= pos.stop:
                    # 止损
                    pos.exit = pos.stop
                    pos.pnl = (pos.exit - pos.entry) / pos.entry * 100
                    pos.status = 'closed'
                    logging.info(f"多头止损 @ {pos.exit:.2f} | 盈亏: {pos.pnl:.2f}% | 时间: {ts}")
                    self.closed.append(pos)
                    self.positions.remove(pos)
                elif high >= pos.target:
                    # 止盈
                    pos.exit = pos.target
                    pos.pnl = (pos.exit - pos.entry) / pos.entry * 100
                    pos.status = 'closed'
                    logging.info(f"多头止盈 @ {pos.exit:.2f} | 盈亏: {pos.pnl:.2f}% | 时间: {ts}")
                    self.closed.append(pos)
                    self.positions.remove(pos)
            else:  # short
                if high >= pos.stop:
                    pos.exit = pos.stop
                    pos.pnl = (pos.entry - pos.exit) / pos.entry * 100
                    pos.status = 'closed'
                    logging.info(f"空头止损 @ {pos.exit:.2f} | 盈亏: {pos.pnl:.2f}% | 时间: {ts}")
                    self.closed.append(pos)
                    self.positions.remove(pos)
                elif low <= pos.target:
                    pos.exit = pos.target
                    pos.pnl = (pos.entry - pos.exit) / pos.entry * 100
                    pos.status = 'closed'
                    logging.info(f"空头止盈 @ {pos.exit:.2f} | 盈亏: {pos.pnl:.2f}% | 时间: {ts}")
                    self.closed.append(pos)
                    self.positions.remove(pos)
    
    def get_summary(self) -> Dict:
        """获取交易统计摘要"""
        if not self.closed:
            return {}
        df = pd.DataFrame([{'pnl': p.pnl} for p in self.closed])
        wins = df[df['pnl'] > 0]
        losses = df[df['pnl'] < 0]
        total = len(df)
        win_rate = len(wins) / total * 100 if total > 0 else 0
        total_pnl = df['pnl'].sum()
        avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
        avg_loss = losses['pnl'].mean() if len(losses) > 0 else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        return {
            'total_trades': total,
            'win_rate': win_rate,
            'total_pnl_pct': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor
        }
    
    def print_summary(self):
        """打印统计摘要到日志"""
        stats = self.get_summary()
        if not stats:
            return
        logging.info("="*40)
        logging.info(f"交易次数: {stats['total_trades']}")
        logging.info(f"胜率: {stats['win_rate']:.2f}%")
        logging.info(f"总盈亏%: {stats['total_pnl_pct']:.2f}%")
        logging.info(f"平均盈利%: {stats['avg_win']:.2f}%")
        logging.info(f"平均亏损%: {stats['avg_loss']:.2f}%")
        logging.info(f"盈亏比: {stats['profit_factor']:.2f}")
        logging.info("="*40)

# ========== 主控制类 ==========
class TradingBot:
    def __init__(self, config: Config):
        self.config = config
        self.fetcher = DataFetcher(config)
        self.strategy = StrategyEngine(config)
        self.trader = SimulatedTrader(config)
        self.setup_logging()
    
    def setup_logging(self):
        """配置日志"""
        logger = logging.getLogger()
        logger.setLevel(self.config.LOG_LEVEL)
        
        # 控制台输出
        console = logging.StreamHandler()
        console.setLevel(self.config.LOG_LEVEL)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console.setFormatter(formatter)
        logger.addHandler(console)
        
        # 文件输出（轮转）
        file_handler = RotatingFileHandler(
            self.config.LOG_FILE, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    def initialize(self):
        """初始化：获取完整历史数据并计算"""
        logging.info("正在初始化，获取历史数据...")
        df = self.fetcher.fetch()
        if df is None:
            logging.error("初始化失败，无法获取数据")
            return False
        self.strategy.process_full(df)
        logging.info(f"初始化完成，加载 {len(df)} 根K线")
        return True
    
    def run_once(self):
        """单次运行：获取最新数据，更新策略，处理信号和持仓"""
        # 获取最新几根K线（例如最近5根）
        new_df = self.fetcher.fetch(limit=5)
        if new_df is None:
            logging.warning("获取最新数据失败")
            return
        
        # 增量更新策略
        has_signal, signal = self.strategy.update_latest(new_df)
        if has_signal and signal:
            self.trader.open_position(signal)
        
        # 用最新的完整K线（即new_df中最后一行）更新持仓
        if len(new_df) > 0:
            latest = new_df.iloc[-1].to_dict()
            self.trader.update_positions(latest)
        
        # 可选：定期打印统计
        if len(self.trader.closed) % 10 == 0 and len(self.trader.closed) > 0:
            self.trader.print_summary()
    
    def run_forever(self):
        """主循环"""
        if not self.initialize():
            return
        
        logging.info("开始主循环...")
        while True:
            try:
                self.run_once()
            except Exception as e:
                logging.exception("运行异常")
            time.sleep(self.config.REFRESH_SEC)

# ========== 参数扫描工具（可选）==========
def parameter_scan(df: pd.DataFrame, param_grid: Dict) -> pd.DataFrame:
    """
    参数扫描：遍历参数组合，计算信号数量和期望收益（简化版）
    注意：此函数需要完整的策略计算逻辑，此处仅为示例
    """
    from itertools import product
    
    results = []
    keys = list(param_grid.keys())
    for values in product(*param_grid.values()):
        params = dict(zip(keys, values))
        # 临时创建一个配置对象
        class TempConfig:
            pass
        cfg = TempConfig()
        for k, v in params.items():
            setattr(cfg, k, v)
        # 补充必要的固定参数
        cfg.LOOKBACK = 20
        cfg.STOP_LOSS_PCT = 0.002
        cfg.RISK_REWARD = 3.0
        cfg.MIN_SAME_SIGNAL_INTERVAL = 3
        
        engine = StrategyEngine(cfg)
        data = engine.calculate_indicators(df)
        data = engine.detect_signals(data)
        signals = data[data['signal'] != 0]
        signal_count = len(signals)
        
        # 简单模拟收益（不考虑止损止盈实现，仅估算）
        # 假设胜率30%，盈亏比3:1，期望收益 per trade = 0.3*3 - 0.7*1 = 0.2
        # 此处仅作演示
        results.append({
            **params,
            'signal_count': signal_count,
            'est_profit_per_trade': 0.2,  # 需实际回测
        })
    
    return pd.DataFrame(results).sort_values('signal_count', ascending=False)

# ========== 主入口 ==========
if __name__ == "__main__":
    # 使用配置运行机器人
    bot = TradingBot(Config)
    bot.run_forever()
    
    # 如需参数扫描，请取消下面注释（需先加载历史数据df）
    # df = pd.read_csv("your_1min_data.csv", parse_dates=['timestamp'])
    # param_grid = {
    #     'VOL_WINDOW': [3,5,8],
    #     'SHRINK_THRESH': [0.7,0.8,0.9],
    #     'BODY_MIN_RATIO': [0.0,0.05,0.1],
    #     'TOUCH_TOLERANCE': [0.003,0.005,0.01]
    # }
    # scan_results = parameter_scan(df, param_grid)
    # print(scan_results.head(20))
