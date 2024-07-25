"""
	SimpleMACrossoverStrategy

	This module implements a basic yet powerful Moving Average Crossover trading strategy
	with integrated stop-loss functionality. The strategy is designed to work within the
	quantjourney backtesting framework, providing a foundation for more complex trading systems.

	Key Features:
	- Utilizes two Simple Moving Averages (SMA) for trend identification
	- Implements a configurable stop-loss mechanism for risk management
	- Generates buy and sell signals based on SMA crossovers and stop-loss triggers
	- Provides methods for signal generation, position management, and strategy visualization

	Strategy Overview:
	The strategy uses two SMAs - a short-term (default 50-day) and a long-term (default 200-day).
	1. Buy Signal: Generated when the short-term SMA crosses above the long-term SMA
	2. Sell Signal: Generated when the short-term SMA crosses below the long-term SMA
	3. Stop-Loss: Implemented to limit potential losses on each trade

	Key Components:
	1. Signal Generation: Based on SMA crossovers and stop-loss conditions
	2. Position Management: Determines entry and exit points for trades with Kelly Criterion sizing
	3. Risk Management: Incorporates a stop-loss mechanism to limit downside risk
	4. Visualization: Plots strategy performance and key indicators

	Strategy Parameters:
	- config_file_path: Path to the configuration file for strategy settings
	- stop_loss_pct: Stop-loss percentage (default: 5%)
	- instruments: List of financial instruments to trade
	- trading_range: Start and end dates for the backtest
	- indicators_config: Configuration for technical indicators (SMAs)
	- initial_capital: Starting capital for the backtest
	- min_trade_size: Minimum trade size (default: 1 share)
	- max_position_size: Maximum position size as a percentage of the portfolio (default: 10%)

	Usage:
	1. Initialization:
	strategy = SimpleMACrossoverStrategy(
		backtest_name="ma_crossover_backtest",
		config_file_path="path/to/config.json",
		instruments=['AAPL', 'MSFT', 'GOOGL'],
		trading_range={'start': '2010-01-01', 'end': '2023-12-31'},
		indicators_config=[
			{'function': 'SMA', 'price_cols': ['adj_close'], 'params': {'window': 50}},
			{'function': 'SMA', 'price_cols': ['adj_close'], 'params': {'window': 200}}
		],
		...
	)

	Implementation Details:
	- Inherits from BaseStrategy class in the quantjourney framework
	- Uses TechnicalIndicators for calculating SMAs (in BaseStrategy)
	- Generates signals based on SMA crossovers and stop-loss conditions
	- Calculates positions and weights for portfolio management

	Advanced Considerations:
	- Parameter Optimization: Consider optimizing SMA periods and stop-loss percentage
	- Additional Indicators: Integrate other technical indicators for signal confirmation
	- Dynamic Position Sizing: Implement more sophisticated position sizing based on volatility or other factors
	- Multiple Timeframes: Explore using multiple timeframes for improved signal quality
	- Market Regime Detection: Incorporate methods to identify and adapt to different market conditions

	Future Enhancements:
	- Implement trailing stop-loss for potentially improved risk management
	- Add support for limit orders and take-profit mechanisms
	- Develop more sophisticated entry and exit criteria
	- Incorporate fundamental data for a hybrid approach
	- Implement machine learning techniques for adaptive parameter tuning

	Notes:
	- This strategy is intended for educational and research purposes
	- Always perform thorough backtesting and forward testing before live trading
	- Past performance does not guarantee future results

	Author: jpolec
	Date: 2024-07-24
	Version: 1.0.0
"""

import pandas as pd
import numpy as np
import asyncio
import matplotlib.pyplot as plt

from quantjourney.backtesting.strategies.base_strategy import BaseStrategy
from quantjourney.indicators.technical_indicators import TechnicalIndicators
from quantjourney.portfolio.instrument_plots import InstrumentPlots
from quantjourney.data.utils.data_logs import data_logger

logger = data_logger()

# Simple Moving Average Crossover Strategy with Stop-Loss -----------------------------------------------------------
class SimpleMACrossoverStrategy(BaseStrategy):
    def __init__(self, config_file_path: str, **kwargs):
        super().__init__(config_file_path, **kwargs)
        self.ti = TechnicalIndicators()

        self.min_trade_size = kwargs.get('min_trade_size', 1)
        self.max_position_size = kwargs.get('max_position_size', 0.1)
        self.stop_loss_pct = kwargs.get('stop_loss_pct', 0.05)
        self.leverage = kwargs.get('leverage', 1.0)
        self.sma_short_window = kwargs.get('sma_short_window', 50)
        self.sma_long_window = kwargs.get('sma_long_window', 200)
        self.position_sizing_strategy = kwargs.get('position_sizing_strategy', 'BASIC')
        self.risk_strategies = kwargs.get('risk_strategies', ['stop_loss', 'position_sizing'])
        self.max_portfolio_risk = kwargs.get('max_portfolio_risk', 0.20)
        self.max_drawdown = kwargs.get('max_drawdown', 0.15)
        self.max_var = kwargs.get('max_var', 0.05)
        self.max_cvar = kwargs.get('max_cvar', 0.07)

        # Update RiskManager config
        self.rm.config.update({
            'stop_loss_percentage': self.stop_loss_pct,
            'max_position_size': self.max_position_size,
            'min_position_size': self.min_trade_size,
            'position_sizing_strategy': self.position_sizing_strategy,
            'risk_strategies': self.risk_strategies,
            'max_portfolio_risk': self.max_portfolio_risk,
            'max_drawdown': self.max_drawdown,
            'max_var': self.max_var,
            'max_cvar': self.max_cvar,
            'leverage': self.leverage
        })


    async def _preprocess_data(self):
        await super()._preprocess_data()

	# Main strategy logic ----------------------------------------------------------------------------------

    def _generate_signals(self) -> pd.DataFrame:
        close = self.instruments_data.get_feature('adj_close')
        sma_short = self.instruments_data.get_feature('SMA_50')
        sma_long = self.instruments_data.get_feature('SMA_200')

        signals = pd.DataFrame(index=close.index, columns=close.columns)
        signals[sma_short > sma_long] = 1  # Buy signals
        signals[sma_short < sma_long] = -1  # Sell signals
        return signals


    def _generate_positions(self, signals: pd.DataFrame) -> pd.DataFrame:
        # Initial position sizing
        positions = signals.copy()

        # Apply risk management
        adjusted_positions = self.rm.adjust_positions(positions, self.portfolio_data)
        return adjusted_positions


    def _generate_weights(self, positions: pd.DataFrame) -> pd.DataFrame:
        # Calculate weights based on positions
        weights = positions.abs().div(positions.abs().sum(axis=1), axis=0).fillna(0)
        weights = weights.clip(upper=self.max_position_size)
        weights = weights.div(weights.sum(axis=1), axis=0).fillna(0)
        return weights


    def _generate_trades(self, positions: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
        current_positions = self.portfolio_data.positions.iloc[-1] if self.portfolio_data.positions is not None else pd.Series(0, index=positions.columns)
        trades = positions.subtract(current_positions, fill_value=0)

        # Apply minimum trade size
        trades[trades.abs() < self.min_trade_size] = 0
        return trades


	# Plot strategy performance ----------------------------------------------------------------------------

    def plot_strategy(self):
        strategy_lines = [
            ('SMA_50', '--', 'blue'),
            ('SMA_200', '--', 'red')
        ]
        for instrument in self.instruments:
            InstrumentPlots.strategy_plot_instrument(
                instruments_data=self.instruments_data,
                price_col='adj_close',
                strategy_lines=strategy_lines,
                instrument=instrument,
                initial_capital=self.initial_capital
            )

if __name__ == '__main__':

	# Universe of instruments to trade
	instruments = [
		'C',
		'F',
		'AAPL',
		'TSLA',
		'XOM',
		'INTC',
		'ORCL',
		'IBM',
		'WMT',
		'DIS'
	]
	ma_strategy = SimpleMACrossoverStrategy(
		backtest_name="ma_crossover_strategy",
		config_file_path="quantjourney/backtesting/strategies/default_strategy.json",
		instruments=instruments,
		trading_range={'start': '2004-01-01', 'end': '2020-12-31'},		# Trading range for the backtest
		indicators_config=[
			{'function': 'SMA', 'price_cols': ['adj_close'], 'params': {'window': 50}},
			{'function': 'SMA', 'price_cols': ['adj_close'], 'params': {'window': 200}}
		],
		initial_capital=10000,								# Initial capital for the backtest
		min_trade_size=100,									# Minimum trade size (in number of shares)
		max_position_size=0.05,								# Maximum position size as a percentage of the portfolio
		stop_loss_pct=0.03,									# Stop-loss percentage for risk management
		leverage=1.5,										# Leverage factor for the strategy
		sma_short_window=40,								# Short-term SMA window
		sma_long_window=180,								# Long-term SMA window
		position_sizing_strategy='KELLY_CRITERION',			# Position sizing strategy (e.g., 'BASIC', 'KELLY_CRITERION')
		risk_strategies=['stop_loss', 'position_sizing'],	# Risk management strategies to apply
		max_portfolio_risk=0.15,							# Maximum portfolio risk as a percentage of capital
		max_drawdown=0.10,									# Maximum drawdown allowed for the strategy
		max_var=0.03,										# Maximum Value at Risk (VaR) for the portfolio
		max_cvar=0.05,										# Maximum Conditional Value at Risk (CVaR) for the

		generate_reports=True,
		generate_plots=False
	)
	asyncio.run(ma_strategy.run_simulation())
	#ma_strategy.plot_strategy()