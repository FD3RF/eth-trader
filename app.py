"""
单元测试示例 - 使用 pytest 运行
"""
import pytest
import numpy as np
from datetime import datetime
from app import Position, TradingConfig, RiskManager

@pytest.fixture
def config():
    return TradingConfig()

@pytest.fixture
def position():
    return Position(
        symbol="ETH/USDT",
        direction=1,
        entry_price=2000,
        entry_time=datetime.now(),
        size=1.0,
        stop_loss=1950,
        take_profit=2100,
        initial_atr=50
    )

def test_position_pnl(position):
    assert position.pnl(2050) == 50.0
    assert position.pnl(1950) == -50.0

def test_position_should_close_stop_loss(position):
    should, reason, price, size = position.should_close(2010, 1940, datetime.now(), TradingConfig())
    assert should is True
    assert reason == "止损"
    assert price == 1950
    assert size == 1.0

def test_position_should_close_take_profit(position):
    should, reason, price, size = position.should_close(2110, 2000, datetime.now(), TradingConfig())
    assert should is True
    assert reason == "止盈"
    assert price == 2100
    assert size == 1.0

def test_risk_manager_calc_position_size(config):
    risk = RiskManager(config, None, None)
    size = risk.calc_position_size(
        balance=10000,
        prob=0.6,
        atr=100,
        price=2000,
        recent_returns=np.array([0.01]*20),
        is_aggressive=False,
        equity_curve=[],
        cov_matrix=None,
        positions={},
        current_symbols=['ETH/USDT'],
        symbol_current_prices={'ETH/USDT': 2000}
    )
    assert size > 0
    assert size < 10000 / 2000  # 不超过最大杠杆
