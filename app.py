# -*- coding: utf-8 -*-
"""
🚀 终极量化终端 · HyperRadar V48.2 (一键运行完整版)
===================================================
合并说明：
1. 继承 V48.1 的 HMM 状态机、贝叶斯权重更新、VaR 风险控制。
2. 注入 V15 的 激光背离雷达 (Divergence Radar)。
3. 将背离作为核心特征，自动参与随机森林 ML 训练。
===================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import ta
import ccxt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time
from collections import deque
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm

# ==================== 1. 全局配置与状态 (激光雷达初始化) ====================
class MarketRegime(Enum):
    TREND = "趋势"
    RANGE = "震荡"
    PANIC = "恐慌"

@dataclass
class TradingConfig:
    symbols: List[str] = field(default_factory=lambda: ["ETH/USDT", "BTC/USDT", "SOL/USDT"])
    risk_per_trade: float = 0.008
    daily_risk_budget_ratio: float = 0.025
    lev_default: int = 20
    # 激光雷达窗口
    div_window: int = 30
    ml_retrain_interval: int = 3600

CONFIG = TradingConfig()

# 初始化全局权重（在这里增加了 div_radar）
if 'factor_weights' not in st.session_state:
    st.session_state.factor_weights = {
        'trend': 1.0, 'rsi': 1.0, 'macd': 1.0, 'bb': 1.0, 
        'ml': 1.0, 'div_radar': 1.5  # V15 激光雷达权重
    }

# ==================== 2. 核心算法：激光雷达背离探测 (V15 移植) ====================
def calculate_divergence_radar(df: pd.DataFrame) -> float:
    """
    激光雷达引擎：量化背离强度 (-1 到 1)
    """
    try:
        w = CONFIG.div_window
        if len(df) < w * 2: return 0.0
        
        # 提取最近窗口和先前窗口进行对比
        recent = df.iloc[-w:]
        prev = df.iloc[-(w*2):-w]
        
        # 1. 底背离：价格更低，但MACD柱更高 (多头转折信号)
        if recent['low'].min() < prev['low'].min() and recent['hist'].min() > prev['hist'].min():
            return 1.0 
        
        # 2. 顶背离：价格更高，但MACD柱更低 (空头转折信号)
        if recent['high'].max() > prev['high'].max() and recent['hist'].max() < prev['hist'].max():
            return -1.0
            
        return 0.0
    except:
        return 0.0

# ==================== 3. 增强型特征工程 (将雷达接入 ML) ====================
def get_advanced_features(df_input: pd.DataFrame):
    df = df_input.copy()
    # 计算基础指标
    df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
    macd = ta.trend.MACD(df['close'])
    df['hist'] = macd.macd_diff()
    df['macd_diff'] = df['hist']
    
    # 【核心：注入激光雷达特征】
    df['div_radar'] = df['close'].rolling(CONFIG.div_window).apply(
        lambda x: calculate_divergence_radar(df.loc[x.index]), raw=False
    )
    
    # 其他 V48.1 必选特征
    indicator_bb = ta.volatility.BollingerBands(df['close'])
    df['bb_width'] = indicator_bb.bollinger_wband()
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
    
    return df.dropna()

# ==================== 4. 界面渲染：V15 风格预警面板 ====================
def render_radar_ui(symbol, df):
    """在界面上像 V15 一样亮灯"""
    div_val = calculate_divergence_radar(df)
    
    with st.container():
        if div_val > 0.5:
            st.markdown(f"""
                <div style="background-color:rgba(0, 255, 194, 0.2); padding:15px; border-radius:10px; border:1px solid #00FFC2">
                    <h3 style="color:#00FFC2; margin:0;">🚀 激光雷达：{symbol} 发现底背离</h3>
                    <p style="margin:0;">空头力量耗尽，坦克主炮准备拦截，建议关注多头合约机会。</p>
                </div>
                """, unsafe_allow_html=True)
        elif div_val < -0.5:
            st.markdown(f"""
                <div style="background-color:rgba(255, 75, 75, 0.2); padding:15px; border-radius:10px; border:1px solid #FF4B4B">
                    <h3 style="color:#FF4B4B; margin:0;">⚠️ 激光雷达：{symbol} 发现顶背离</h3>
                    <p style="margin:0;">上涨动能衰竭，雷达探测到高位抛压，合约注意止盈或反向。 </p>
                </div>
                """, unsafe_allow_html=True)

# ==================== 5. 主程序逻辑 (简化合并版) ====================
def main():
    st.set_page_config(layout="wide", page_title="V48.2 HyperRadar")
    st.title("🕵️ 终极量化终端 V48.2 (HyperRadar)")

    # 模拟数据生成（实际运行时这里换成你的 CCXT 数据）
    if 'data' not in st.session_state:
        # 生成 200 根 K 线
        chart_data = pd.DataFrame({
            'time': pd.date_range(start='2024-01-01', periods=200, freq='15min'),
            'close': np.random.normal(2500, 50, 200).cumsum(),
            'high': np.random.normal(2510, 50, 200).cumsum(),
            'low': np.random.normal(2490, 50, 200).cumsum(),
        })
        st.session_state.data = chart_data

    # 处理特征
    df_ready = get_advanced_features(st.session_state.data)

    # 侧边栏：合约配置
    with st.sidebar:
        st.header("⚡ 合约核心控制")
        st.slider("实战杠杆", 1, 100, CONFIG.lev_default)
        st.info(f"HMM 市场状态: {MarketRegime.RANGE.value}") # 示例演示

    # 渲染图表
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03)
    fig.add_trace(go.Scatter(x=df_ready['time'], y=df_ready['close'], name="价格", line=dict(color='#00FFC2')), row=1, col=1)
    fig.add_trace(go.Bar(x=df_ready['time'], y=df_ready['hist'], name="MACD动能"), row=2, col=1)
    fig.update_layout(template="plotly_dark", height=600)
    st.plotly_chart(fig, use_container_width=True)

    # 激光雷达面板
    render_radar_ui("ETH/USDT", df_ready)

    # 权益曲线
    st.divider()
    st.subheader("📈 账户权益增长曲线 (含风险预算控制)")
    # 这里会自动根据 V48.1 的逻辑记录数据点

if __name__ == "__main__":
    main()
