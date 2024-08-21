"""
Example Usage of LearningBacktester

This example demonstrates how to create a simple moving average crossover strategy
using the LearningBacktester framework.

Steps:
1. Define a strategy class that inherits from LearningBacktester
2. Implement the required abstract methods
3. Create an instance of the strategy
4. Run the backtest simulation

"""
import pandas as pd
import numpy as np
import asyncio

from quantjourney.backtesting.learning_backtester import LearningBacktester
from quantjourney.indicators.technical_indicators import TechnicalIndicators
from quantjourney.data.utils.data_logs import data_logger
logger = data_logger()

class LearningSimpleMovingAverageCrossover(LearningBacktester):
	def __init__(self, config_file_path: str, **kwargs):
		super().__init__(config_file_path, **kwargs)
		self.ti = TechnicalIndicators()
		self.sma_short_window = kwargs.get('sma_short_window', 50)
		self.sma_long_window = kwargs.get('sma_long_window', 200)


	def _generate_signals(self) -> pd.DataFrame:
		sma_short = self.instruments_data.get_feature(f'SMA_{self.sma_short_window}')
		sma_long = self.instruments_data.get_feature(f'SMA_{self.sma_long_window}')

		# Create a mask for valid data points
		valid_data = sma_short.notna() & sma_long.notna()

		signals = pd.DataFrame(0, index=sma_short.index, columns=sma_short.columns)
		signals[valid_data & (sma_short > sma_long)] = 1  # Buy signals
		signals[valid_data & (sma_short < sma_long)] = -1  # Sell signals

		# Forward fill only after the first valid signal for each instrument
		first_valid = signals.abs().idxmax()
		for col in signals.columns:
			signals[col] = signals[col].loc[first_valid[col]:].fillna(method='ffill')

		signals = signals.fillna(0)  # Fill remaining NaNs with 0

		if signals.empty:
			logger.error("Signal generation failed.")
			return pd.DataFrame()

		logger.info(f"Generated signals for {len(signals.columns)} instruments.")
		return signals


	def _generate_positions(self, signals: pd.DataFrame) -> pd.DataFrame:
		# Simple implementation: directly use signals as positions
		return signals


	def _generate_weights(self, positions: pd.DataFrame) -> pd.DataFrame:
		# Equal weighting for all active positions
		weights = positions.astype(bool).astype(float)
		return weights.div(weights.sum(axis=1), axis=0).fillna(0)


async def run_backtest():
	config = {
		'backtest_name': 'Simple Moving Average Crossover',
		'instruments': ['AAPL', 'GOOGL', 'MSFT'],
		'initial_capital': 100000,
		'trading_range': {'start': '2020-01-01', 'end': '2021-12-31'},
		'indicators_config': [
			{'function': 'SMA', 'price_cols': ['close'], 'params': {'window': 50}},
			{'function': 'SMA', 'price_cols': ['close'], 'params': {'window': 200}}
		]
	}

	strategy = LearningSimpleMovingAverageCrossover('quantjourney/backtesting/default_strategy.json', **config)
	await strategy.run_simulation()

if __name__ == '__main__':
	asyncio.run(run_backtest())