# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import ta
import time
from dataclasses import dataclass

# =============================
# CONFIG
# =============================

class CONFIG:
    base_risk = 0.01
    max_portfolio_risk = 0.05
    kelly_fraction = 0.5
    lookback_prob = 400
    cooldown_bars = 3
    timeframe = "15m"
    limit = 500


# =============================
# POSITION STRUCTURE (R system)
# =============================

@dataclass
class Position:
    entry: float
    stop: float
    size: float
    direction: int

    def risk_per_unit(self):
        return abs(self.entry - self.stop)

    def risk_amount(self):
        return self.risk_per_unit() * self.size

    def pnl(self, price):
        return (price - self.entry) * self.size * self.direction

    def r_multiple(self, price):
        return self.pnl(price) / self.risk_amount()


# =============================
# PROBABILITY MODEL
# =============================

class ProbabilityModel:

    def __init__(self):
        self.df = None

    def fit(self, df):
        self.df = df.copy()

    def predict(self):
        if len(self.df) < CONFIG.lookback_prob:
            return 0.5

        window = self.df.iloc[-CONFIG.lookback_prob:]
        state = window["vol"].iloc[-1]

        bucket = window[
            (window["vol"] > state * 0.9) &
            (window["vol"] < state * 1.1)
        ]

        if len(bucket) < 30:
            return 0.5

        future = bucket["return"].shift(-1)
        return (future > 0).mean()


# =============================
# RISK MANAGER
# =============================

class RiskManager:

    def __init__(self):
        self.r_history = []
        self.cooldown = 0

    def register(self, r):
        self.r_history.append(r)
        if len(self.r_history) > 200:
            self.r_history.pop(0)

    def kelly_size(self, prob, equity):

        if self.cooldown > 0:
            self.cooldown -= 1
            return 0

        if len(self.r_history) < 30:
            return equity * CONFIG.base_risk

        wins = [r for r in self.r_history if r > 0]
        losses = [abs(r) for r in self.r_history if r < 0]

        if not wins or not losses:
            return equity * CONFIG.base_risk

        win_rate = len(wins) / len(self.r_history)
        avg_win = np.mean(wins)
        avg_loss = np.mean(losses)

        b = avg_win / avg_loss
        kelly = win_rate - (1 - win_rate) / b
        kelly = max(0, kelly)

        size = equity * kelly * CONFIG.kelly_fraction

        if size <= 0:
            self.cooldown = CONFIG.cooldown_bars

        return size


# =============================
# DATA FETCH
# =============================

def fetch_data(exchange, symbol):

    ohlcv = exchange.fetch_ohlcv(
        symbol,
        timeframe=CONFIG.timeframe,
        limit=CONFIG.limit
    )

    df = pd.DataFrame(
        ohlcv,
        columns=["time", "open", "high", "low", "close", "volume"]
    )

    df["time"] = pd.to_datetime(df["time"], unit="ms")

    df["return"] = df["close"].pct_change()
    df["ema50"] = ta.trend.ema_indicator(df["close"], 50)
    df["ema200"] = ta.trend.ema_indicator(df["close"], 200)
    df["atr"] = ta.volatility.average_true_range(
        df["high"], df["low"], df["close"], 14
    )
    df["vol"] = df["return"].rolling(30).std()

    return df.dropna()


# =============================
# STREAMLIT APP
# =============================

st.set_page_config(layout="wide")
st.title("🚀 Extreme Institutional ETH Trader")

mode = st.sidebar.selectbox("Mode", ["SIMULATION", "LIVE"])

api_key = st.sidebar.text_input("API Key", type="password")
secret = st.sidebar.text_input("Secret", type="password")

symbol = st.sidebar.text_input("Symbol", "ETH/USDT")
exchange_name = st.sidebar.selectbox("Exchange", ["binance", "bybit"])

if st.sidebar.button("Start Trading"):

    if exchange_name == "binance":
        exchange = ccxt.binance({
            "apiKey": api_key,
            "secret": secret,
            "enableRateLimit": True
        })
    else:
        exchange = ccxt.bybit({
            "apiKey": api_key,
            "secret": secret,
            "enableRateLimit": True
        })

    prob_model = ProbabilityModel()
    risk = RiskManager()

    equity = 10000
    position = None

    st.success("Trading started...")

    while True:

        df = fetch_data(exchange, symbol)
        prob_model.fit(df)

        prob = prob_model.predict()
        last = df.iloc[-1]

        st.write(f"Probability: {round(prob,3)}")

        if position is None and prob > 0.55:

            stop = last["close"] - last["atr"] * 1.5
            risk_cap = risk.kelly_size(prob, equity)

            if risk_cap > 0:

                size = risk_cap / (last["close"] - stop)

                position = Position(
                    entry=last["close"],
                    stop=stop,
                    size=size,
                    direction=1
                )

                if mode == "LIVE":
                    exchange.create_market_buy_order(symbol, size)

                st.write("Position Opened")

        elif position:

            if last["low"] <= position.stop:

                exit_price = position.stop

                if mode == "LIVE":
                    exchange.create_market_sell_order(
                        symbol, position.size
                    )

                pnl = position.pnl(exit_price)
                r = position.r_multiple(exit_price)

                equity += pnl
                risk.register(r)

                st.write(f"Stopped Out. R: {round(r,2)}")

                position = None

        st.write(f"Equity: {round(equity,2)}")

        time.sleep(60)
