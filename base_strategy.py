"""
	Backtesting Framework for Trading Strategies

	Base Strategy Class - Base class for all trading strategies.

	author: jpolec
	date: 2024-07-10
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple, Union
from enum import Enum
import json
import asyncio
from abc import ABC, abstractmethod
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))

from quantjourney.backtesting.engines import (
	MarketDataProvider, DataPreprocessor, EventsProcessor, ForecastPredictor, RiskManager,
	PortfolioOptimizer, TradingCostCalculator, PerformanceAnalyst,
	PerformanceReportGenerator, ProfitLossTracker, MarketRegimeAnalyzer, TradeExecutor
)
from quantjourney.portfolio.portfolio_data import PortfolioData
from quantjourney.portfolio.portfolio_calculations import PortfolioCalculations
from quantjourney.portfolio.portfolio_plots import PortfolioPlots
from quantjourney.portfolio.instrument_data import InstrumentData
from quantjourney.portfolio.instrument_calculations import InstrumentCalculations
from quantjourney.portfolio.instrument_plots import InstrumentPlots
from quantjourney.reporting.performance_metrics import PerformanceMetrics

from quantjourney.indicators.technical_indicators import TechnicalIndicators
from quantjourney.backtesting.strategies.alphas.alpha_calc import AlphaEval

from quantjourney.file.file import load_pickle, save_pickle, get_local_file_path, FileTypes
from quantjourney.data.utils.data_logs import data_logger
from quantjourney.other.decorators import timer, loggertext, error_logger
logger = data_logger()

import _dev.dev  # For better error messages


# Base Strategy Class ----------------------------------------------------------
class BaseStrategy(ABC):
	"""
	Base class for all trading strategies.

	This class provides a framework for implementing trading strategies,
	including data handling, signal generation, position management,
	and performance evaluation.
	"""

	@error_logger("Error during BaseStrategy initialization")
	def __init__(self, config_file_path: str, **kwargs) -> None:
		"""
		Initialize the BaseStrategy.

		Args:
			config_file_path (str): Path to the configuration file.
			**kwargs: Additional keyword arguments to override config file settings.

		Raises:
			FileNotFoundError: If the configuration file is not found.
			ValueError: If there's an error in decoding the JSON configuration file.
		"""
		try:
			self.config = self._load_config(config_file_path)
		except (FileNotFoundError, ValueError) as e:
			logger.error(f"Error loading configuration: {e}")
			raise

		self.backtest_name = kwargs.get('backtest_name', 'Backtest Default Strategy')
		self.generate_reports = kwargs.get('generate_reports', False)
		self.generate_plots = kwargs.get('generate_plots', False)
		self.load_data = kwargs.get('load_data', False)

		self._initialize_strategy(kwargs)
		self._validate_strategy_parameters()
		self._initialize_engines()

		self.ti = TechnicalIndicators()


	@error_logger("Error in loading config")
	def _load_config(self, config_file_path: str) -> Dict[str, Any]:
		"""
		Load the strategy configuration file.

		Args:
			config_file_path (str): Path to the configuration file.

		Returns:
			Dict[str, Any]: Loaded configuration.

		Raises:
			FileNotFoundError: If the configuration file is not found.
			ValueError: If there's an error in decoding the JSON configuration file.
		"""
		if not os.path.isfile(config_file_path):
			raise FileNotFoundError(f"Configuration file '{config_file_path}' not found.")
		try:
			with open(config_file_path, 'r') as config_file:
				return json.load(config_file)
		except json.JSONDecodeError as e:
			raise ValueError(f"Error decoding JSON configuration file: {e}")


	# Initialize strategy and eninges ----------------------------------------------------------

	@error_logger("Error in initialization strategy")
	def _initialize_strategy(self, kwargs: Dict[str, Any]) -> None:
		"""
		Initialize the strategy parameters.

		To handle dynamic instrument selection for backtests we have to specify, e.g.:
			"indices": ["SPY500", "RUT"],
			"instruments": ["GOOGL"],
			"filters": {
				"market_cap_min": 10000000000,
				"sector": "Technology"
			},
		It will fetch all stocks from the indices and filter them based on the filters.
		Or if instruments is empty, then it will fetch all instruments from the indices.
		Or if indices is empty, then it will fetch all instruments from the filters.
		If filters is empty, then nothing is filtered.

		Args:
			kwargs (Dict[str, Any]): Additional keyword arguments to override config settings.
				- instruments (List[str]): List of instruments to trade.
				- indices (List[str]): List of indices to track.
				- filters (List[str]): List of filters to apply.
				- initial_capital (float): Initial capital for the strategy.
				- trading_range (Dict[str, str]): Start and end dates for the trading range.
				- leverage (float): Leverage for the strategy.
				- frequency (Dict[str, str]): Frequency of the data.
				- rebalance_frequency (str): Frequency of rebalancing.
				- rebalance_strategy (str): Rebalancing strategy.
				- verbose (bool): Whether to print verbose logs.
		"""
		self.instruments = kwargs.get('instruments', self.config.get('instruments', []))
		self.indices = kwargs.get('indices', self.config.get('indices', []))
		self.filters = kwargs.get('filters', self.config.get('filters', []))

		self.initial_capital = kwargs.get('initial_capital', self.config.get('initial_capital', 10000.0))
		self.trading_range = kwargs.get('trading_range', self.config.get('trading_range', {'start': '2020-01-01', 'end': '2020-12-31'}))
		self.leverage = kwargs.get('leverage', self.config.get('leverage', 1.0))
		self.portfolio_value = self.initial_capital
		self.indicators_config = kwargs.get('indicators_config', self.config.get('indicators_config', False))

		freq_config = self.config.get('frequency', {})
		self.frequency = kwargs.get('frequency', freq_config.get('frequency', 'daily'))
		self.rebalance_frequency = kwargs.get('rebalance_frequency', freq_config.get('rebalance_frequency', 'monthly'))
		self.rebalance_strategy = kwargs.get('rebalance_strategy', freq_config.get('rebalance_strategy', 'periodic'))

		self.portfolio_data = None
		self.instruments_data = None
		self.market_data = None
		self.market_regime_data = None

		self.verbose = kwargs.get('verbose', True)


	def _validate_strategy_parameters(self) -> None:
		"""
		Validate the strategy parameters after initialization.

		Raises:
			ValueError: If any strategy parameters are invalid.
		"""
		if not self.instruments:
			raise ValueError("No instruments specified for the strategy")

		if not isinstance(self.initial_capital, (int, float)) or self.initial_capital <= 0:
			raise ValueError("Initial capital must be a positive number")

		if not isinstance(self.trading_range, dict) or \
			'start' not in self.trading_range or \
			'end' not in self.trading_range:
			raise ValueError("Trading range must be a dict with 'start' and 'end' keys")


	@error_logger("Error in initialization engines")
	def _initialize_engines(self) -> None:
		"""
		Initialize the engines for the strategy.
		Each engine is initialized with the corresponding configuration parameters.

		Available engines:
			- MarketDataProvider
			- DataPreprocessor
			- MarketRegimeAnalyzer
			- EventsProcessor
			- ForecastPredictor
			- RiskManager
			- ProfitLossTracker
			- TradingCostCalculator
			- PerformanceAnalyst
			- PerformanceReportGenerator
			- TradeExecutor
		"""
		engine_params = [
					('mdp', MarketDataProvider, 'market_data_provider'),
					('dp', DataPreprocessor, 'data_preprocessor'),
					('mra', MarketRegimeAnalyzer, 'market_regime_analyzer'),
					('ep', EventsProcessor, 'events_processor'),
					('fp', ForecastPredictor, 'forecast_predictor'),
					('rm', RiskManager, 'risk_manager'),
					('plt', ProfitLossTracker, 'profitloss_tracker'),
					('tcc', TradingCostCalculator, 'trading_cost_calculator'),
					('pa', PerformanceAnalyst, 'performance_analyst'),
					('prg', PerformanceReportGenerator, 'performance_report_generator'),
					('te', TradeExecutor, 'trade_executor'),
				]
		for attr, cls, param_key, *extra_args in engine_params:
			if cls == PerformanceReportGenerator:
				# Initialize PerformanceReportGenerator with None for portfolio_data and instruments_data
				setattr(self, attr, cls(self.config.get(param_key, {}), None, None))
			else:
				setattr(self, attr, cls(self.config.get(param_key, {}), *extra_args))

		rules = self.config.get('events_processor', {}).get('signal_events', {}).get('rules', {})
		self.strategies = list(rules.keys())


	# Get and Process data ----------------------------------------------------------

	@loggertext("Getting market data.")
	async def _get_market_data(self) -> None:
		"""
		Fetch market data using the MarketDataProvider.

		This method either loads pre-existing data or fetches new data
		based on the 'load_data' flag.

		Raises:
			Exception: If there's an error in fetching or loading the data.
		"""
		try:
			if self.load_data:
				from quantjourney.data.load_data import load_portfolio_data
				self.portfolio_data = load_portfolio_data()

				# Convert index to datetime
				self.portfolio_data.net_asset_value.index = pd.to_datetime(self.portfolio_data.net_asset_value.index, utc=True)
				self.portfolio_data.asset_weights.index = pd.to_datetime(self.portfolio_data.asset_weights.index, utc=True)

				# Convert instruments data index to datetime
				for column in self.portfolio_data.instruments.data.columns.levels[0]:
					self.portfolio_data.instruments.data[column].index = pd.to_datetime(self.portfolio_data.instruments.data[column].index, utc=True)

				self.instruments_data = self.portfolio_data.instruments
			else:
				self.market_data = await self.mdp.get_market_data(self.instruments, self.indices, self.filters, self.trading_range)
			logger.info("Market data successfully retrieved.")

		except Exception as e:
			logger.error(f"Error in getting market data: {e}")
			raise


	@loggertext("Preprocessing data.")
	async def _preprocess_data(self) -> PortfolioData:
		"""
		Use the existing DataPreprocessor to process the market data and create PortfolioData and InstrumentData objects.

		To create the PortfolioData object, the method:
			- Prepares the data using the DataPreprocessor.
			- Adds technical indicators to the InstrumentData object based on the configuration.

		How the Technical Indicators are added:
			- The method loops over each instrument and each indicator in the configuration.
			- It calls the corresponding indicator function from the TechnicalIndicators class.
			- It adds the indicator results to the InstrumentData object.

		Examples:
			- For a single-price indicator like SMA, it adds a new column to the InstrumentData.
			- For multi-price indicators like STOCH, it adds multiple columns to the InstrumentData with '_{params_valuea}'.

		indicators_config=[
			{'function': 'SMA', 'price_cols': ['adj_close'], 'params': {'window': 50}},
			{'function': 'STOCH', 'price_cols': ['high', 'low', 'close'], 'params': {'fastk_period': 14, 'slowk_period': 3, 'slowd_period': 3}},
			{'function': 'RSI', 'price_cols': ['close'], 'params': {'window': 14}},
			{'function': 'BB', 'price_cols': ['close'], 'params': {'window': 20, 'stddev': 2}},
		]

		How the Fundamental key ratios are added: (Note: not shared in this snippet)
			- The method loops over each instrument and each key ratio in the configuration.
			- It calls the corresponding key ratio function from the FundamentalData class.
			- It adds the key ratio results to the InstrumentData object.

		fundamental_config = [
			{'function': 'GROSS_PROFIT_MARGIN', 'params': {'period': 'annual'}},
			{'function': 'NET_PROFIT_MARGIN', 'params': {'period': 'quarterly'}},
			{'function': 'ROE', 'params': {'trailing': 4}},
			{'function': 'DEBT_TO_EQUITY', 'params': {}},
			{'function': 'PE_RATIO', 'params': {'trailing': True}},
			{'function': 'EV_TO_EBITDA', 'params': {'trailing': 4}},
			{'function': 'DIVIDEND_YIELD', 'params': {}},
			{'function': 'FREE_CASH_FLOW_YIELD', 'params': {'trailing': 4}},
			{'function': 'ALTMAN_Z_SCORE', 'params': {}},
			{'function': 'EARNINGS_GROWTH', 'params': {'years': 5}},
			{'function': 'OPERATING_MARGIN', 'params': {'trailing': 4}},
		]
		"""
		if self.load_data:
			return

		# Set PortfolioData and InstrumentData objects
		self.portfolio_data = await self.dp.prepare_data(self.market_data)
		self.instruments_data = self.portfolio_data.instruments

		# Ensure datetime index and check alignment
		self.instruments_data.data.index = self.instruments_data.data.index.tz_localize(None).floor('D')
		self.portfolio_data.asset_weights.index = self.portfolio_data.asset_weights.index.floor('D')

		# Add indicators to instruments data as per config
		if self.indicators_config:
			logger.info("Calculating technical indicators.")
			instruments = self.instruments_data.data.columns.levels[0]

			for instrument in instruments:
				try:
					logger.info(f"Processing instrument: {instrument}")

					# Loop over each indicator in the config
					for indicator_config in self.indicators_config:
						function_name = indicator_config['function']
						price_cols = indicator_config['price_cols']
						params = indicator_config['params']

						# Prepare price data
						price_data = {col: self.instruments_data.data[instrument, col] for col in price_cols}

						# Check if all required price columns are available
						if not all(col in self.instruments_data.data[instrument].columns for col in price_cols):
							logger.warning(f"Not all required price columns available for {function_name} on {instrument}. Skipping.")
							continue

						indicator_function = getattr(self.ti, function_name)

						# Call the indicator function based on the number of price columns e.g. STOCH (3), SMA (1)
						if len(price_cols) == 1:
							result = indicator_function(price_data[price_cols[0]], **params)
						else:
							result = indicator_function(**price_data, **params)

						# Create a new column(s) for each result
						if isinstance(result, pd.DataFrame):
							for col in result.columns:
								column_name = f"{col}"
								self.instruments_data.data[(instrument, column_name)] = result[col]
								logger.info(f"Added indicator {column_name}")
						elif isinstance(result, pd.Series):
							param_str = '_'.join(str(v) for v in params.values())
							column_name = f"{function_name}_{param_str}"
							self.instruments_data.data[(instrument, column_name)] = result
							logger.info(f"Added indicator {column_name}")
						else:
							logger.warning(f"Unexpected result type for {function_name} on {instrument}")

				except Exception as e:
					logger.error(f"Error processing {instrument}: {e}", exc_info=True)

			logger.info("All indicators added successfully")

		logger.info("All indicators added successfully")

		return self.portfolio_data


	# Generate Forecasts ----------------------------------------------------------

	@timer
	@loggertext("Generating forecasts.")
	async def _generate_forecasts(self) -> Dict[str, pd.DataFrame]:
		"""
		Generate forecasts for the portfolio and each instrument.

		Returns:
			Dict[str, pd.DataFrame]: A dictionary of forecasts for each instrument.

		Raises:
			Exception: If there's an error in generating forecasts.
		"""
		try:
			await self.fp.pre_train(self.portfolio_data)
			forecasts = await self.fp.generate_forecasts(self.portfolio_data)
			logger.info("Forecasts generated successfully.")
			return forecasts
		except Exception as e:
			logger.error(f"Error generating forecasts: {e}")
			raise


	# Generate Events ----------------------------------------------------------

	@timer
	@loggertext("Generate events per strategies.")
	async def _generate_events(self) -> Dict[str, Dict[str, pd.DataFrame]]:
		"""
		Generate events for each instrument.

		Returns:
			Dict[str, Dict[str, pd.DataFrame]]: A dictionary containing the generated events for each instrument.
		"""
		events = {}
		try:
			for instrument in self.instruments:
				logger.info(f"Generating events for {instrument}")

				market_events = await self.ep.generate_market_events(self.market_data[instrument])

				signal_events_df = pd.DataFrame()
				for strategy in self.strategies:
					signals = await self.ep.generate_events(self.instruments_data[instrument], strategy)
					signal_events_df = pd.concat([signal_events_df, signals], axis=1)

				order_events_df = await self.ep.generate_order_events(self.instruments_data[instrument], signal_events_df)

				events[instrument] = {
					'market_events': market_events,
					'signal_events': signal_events_df,
					'order_events': order_events_df
				}

			return events
		except Exception as e:
			logger.error(f"Error generating events: {e}")
			raise


	# Execute Trades ----------------------------------------------------------

	@loggertext("Executing trades.")
	async def _execute_trades(self, events: Dict[str, Dict[str, pd.DataFrame]]) -> None:
		"""
		Execute trades based on the generated events with Exchange Broker defined in TradeExecutor.

		Args:
			events (Dict[str, Dict[str, pd.DataFrame]]): Dictionary of events for each instrument.
		"""
		try:
			for instrument, instrument_events in events.items():
				order_events = instrument_events.get('order_events')
				if order_events is not None:
					for _, order in order_events.iterrows():
						await self.te.execute_order(instrument, order)
			logger.info("Trades executed successfully.")
		except Exception as e:
			logger.error(f"Error executing trades: {e}")
			raise


	# Signal and Positions Generation ------------------------------------------

	@abstractmethod
	def _generate_signals(self):
		"""
		Generate trading signals for the strategy.

		This method should be implemented by child classes to define the specific signal generation logic for the strategy.

		Returns:
			pd.DataFrame: DataFrame containing generated signals.

		To get Technical Indicators created in _preprocess_data, we can:
			- Use self.instruments_data.get_feature('indicator_name')
			- Use self.instruments_data.data[instrument, 'indicator_name']
		Examples:
			close = self.instruments_data.get_feature('adj_close')
			sma_short = self.instruments_data.get_feature('SMA_short')
			sma_long = self.instruments_data.get_feature('SMA_long')
			positions = self.instruments_data.data[instrument, 'position']

		To get Fundamental Data created in _preprocess_data, we can: (Note: not shared in this snippet)
			- Use self.instruments_data.get_fundamental('fundamental_name')
			- Use self.instruments_data.data[instrument, 'fundamental_name']

		Examples:
			GROSS_PROFIT_MARGIN = self.instruments_data.get_fundamental('GROSS_PROFIT_MARGIN')
			NET_PROFIT_MARGIN = self.instruments_data.get_fundamental('NET_PROFIT_MARGIN')
			ROE = self.instruments_data.get_fundamental('ROE')

		Raises:
			NotImplementedError: If the method is not implemented in a child class.
		"""


	@abstractmethod
	def _generate_positions(self, signals: pd.DataFrame):
		"""
		Generate positions based on the signals and strategy model.

		This method should be implemented by child classes to define
		the specific position generation logic for the strategy.

		Args:
			signals (pd.DataFrame): DataFrame containing generated signals.

		Returns:
			pd.DataFrame: DataFrame containing generated positions.

		Raises:
			NotImplementedError: If the method is not implemented in a child class.
		"""


	# Main Simulation ----------------------------------------------------------

	@timer
	async def run_simulation(self) -> None:
		"""
		Run the backtesting simulation.

		This method orchestrates the entire backtesting process, including
		data retrieval, preprocessing, signal generation, position calculation,
		and performance evaluation.

		Raises:
			Exception: If there's an error during the simulation process.
		"""

		logger.info(f"Running Strategy Simulation: {self.backtest_name}")

		await self._get_market_data()
		await self._preprocess_data()

		await self._generate_regime()
		await self._generate_forecasts()
		await self._generate_events()

		signals = self._generate_signals()
		logger.info("Signals generated successfully.")

		positions = self._generate_positions(signals)
		logger.info("Positions generated successfully.")

		returns_data = self.instruments_data.data.xs('returns', axis=1, level=1)

		# Ensure all DataFrames have datetime index and are aligned
		signals, positions, returns_data = map(self._ensure_datetime_index, [signals, positions, returns_data])
		common_index = signals.index.intersection(positions.index).intersection(returns_data.index)
		signals, positions, returns_data = [df.loc[common_index] for df in [signals, positions, returns_data]]

		# Normalize positions to get asset weights and remove any NaN values
		weights = positions.divide(positions.sum(axis=1), axis=0).fillna(0)

		# Calculate portfolio returns based on asset weights and returns data
		portfolio_returns = (weights * returns_data).sum(axis=1)

		self._update_portfolio_data(weights, positions, portfolio_returns)
		self._update_instrument_data(positions, weights, returns_data)

		if self.generate_reports:
			self.prg._update_data(self.portfolio_data, self.instruments_data)
			report = self.prg.generate_text_report()
			print(report)
			self.prg.generate_full_report('./_output')
		if self.generate_plots:
			self.prg.plot_strategy_results()


		logger.info("Simulation and performance reporting completed successfully.")


	# RunSimulation Additional Methods -----------------------------------------

	def _ensure_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
		"""
		Ensure the DataFrame has a datetime index.
		"""
		if not isinstance(df.index, pd.DatetimeIndex):
			logger.warning(f"Index is not DatetimeIndex. Current type: {type(df.index)}")
			try:
				df.index = pd.to_datetime(df.index)
				logger.info("Successfully converted index to DatetimeIndex.")
			except Exception as e:
				logger.error(f"Failed to convert index to DatetimeIndex: {str(e)}")
		return df


	def _update_portfolio_data(self, weights: pd.DataFrame, positions: pd.DataFrame, portfolio_returns: pd.Series) -> None:
		"""
		Update the portfolio data with the results of the simulation.

		Args:
			weights (pd.DataFrame): Asset weights.
			positions (pd.DataFrame): Asset positions.
			portfolio_returns (pd.Series): Portfolio returns.
		"""
		try:
			self.portfolio_data.asset_weights = weights
			self.portfolio_data.positions = positions
			self.portfolio_data.returns = portfolio_returns
			self.portfolio_data.cumulative_returns = (1 + portfolio_returns).cumprod() - 1
			self.portfolio_data.net_asset_value = self.portfolio_data.cumulative_returns * self.initial_capital + self.initial_capital
			self.portfolio_data.drawdown = (self.portfolio_data.net_asset_value.cummax() - self.portfolio_data.net_asset_value) / self.portfolio_data.net_asset_value.cummax()
		except Exception as e:
			logger.error(f"Error updating portfolio data: {e}")
			raise


	def _update_instrument_data(self, positions: pd.DataFrame, weights: pd.DataFrame, returns_data: pd.DataFrame) -> None:
		"""
		Update the instrument data with the results of the simulation.

		Args:
			positions (pd.DataFrame): Asset positions.
			weights (pd.DataFrame): Asset weights.
			returns_data (pd.DataFrame): Asset returns.
		"""
		try:
			instruments = self.instruments_data.data.columns.get_level_values(0).unique() if isinstance(self.instruments_data.data.columns, pd.MultiIndex) else [self.instruments_data.data.name]
			for instrument in instruments:
				if isinstance(self.instruments_data.data.columns, pd.MultiIndex):
					self.instruments_data.data.loc[:, (instrument, 'position')] = positions[instrument]
					self.instruments_data.data.loc[:, (instrument, 'weight')] = weights[instrument]
					self.instruments_data.data.loc[:, (instrument, 'returns')] = returns_data[instrument]
					self.instruments_data.data.loc[:, (instrument, 'cumulative_returns')] = (1 + returns_data[instrument]).cumprod() - 1
				else:
					self.instruments_data.data['position'] = positions
					self.instruments_data.data['weight'] = weights
					self.instruments_data.data['returns'] = returns_data
					self.instruments_data.data['cumulative_returns'] = (1 + returns_data).cumprod() - 1
		except Exception as e:
			logger.error(f"Error updating instrument data: {e}")
			raise


	# Rebalance Methods ----------------------------------------------------------

	@timer
	@loggertext("Error rebalancing portfolio.")
	async def _rebalance_portfolio(self, current_date: pd.Timestamp) -> None:
		"""
		Rebalance the portfolio based on the rebalance frequency and strategy.

		Args:
			current_date (pd.Timestamp): The current date in the simulation.
		"""
		try:
			# Check if it's time to rebalance
			if self._is_rebalance_time(current_date):
				logger.info(f"Rebalancing portfolio on {current_date}")

				if self.rebalance_strategy == 'periodic':
					# Implement periodic rebalancing logic
					target_weights = self._calculate_target_weights()
					current_weights = self.portfolio_data.asset_weights.iloc[-1]
					rebalance_orders = self._generate_rebalance_orders(current_weights, target_weights)
					await self._execute_rebalance_orders(rebalance_orders)

				elif self.rebalance_strategy == 'threshold':
					# Implement threshold-based rebalancing logic
					pass  # Add your threshold-based rebalancing logic here

				else:
					logger.warning(f"Unknown rebalance strategy: {self.rebalance_strategy}")

		except Exception as e:
			logger.error(f"Error during portfolio rebalancing: {e}")
			raise


	@loggertext("Error checking rebalance time.")
	def _is_rebalance_time(self, current_date: pd.Timestamp) -> bool:
		"""
		Check if it's time to rebalance based on the rebalance frequency.

		Args:
			current_date (pd.Timestamp): The current date in the simulation.

		Returns:
			bool: True if it's time to rebalance, False otherwise.
		"""
		if self.rebalance_frequency == 'daily':
			return True
		elif self.rebalance_frequency == 'weekly':
			return current_date.dayofweek == 0  # Rebalance on Mondays
		elif self.rebalance_frequency == 'monthly':
			return current_date.day == 1  # Rebalance on the first day of each month
		elif self.rebalance_frequency == 'quarterly':
			return current_date.month in [1, 4, 7, 10] and current_date.day == 1
		else:
			logger.warning(f"Unknown rebalance frequency: {self.rebalance_frequency}")
			return False


	@loggertext("Error calculating target weights.")
	def _calculate_target_weights(self) -> pd.Series:
		"""
		Calculate the target weights for rebalancing.

		Returns:
			pd.Series: The target weights for each asset.
		"""
		pass


	@loggertext("Error generating rebalance orders.")
	def _generate_rebalance_orders(self, current_weights: pd.Series, target_weights: pd.Series) -> pd.DataFrame:
		"""
		Generate orders to rebalance the portfolio.

		Args:
			current_weights (pd.Series): Current portfolio weights.
			target_weights (pd.Series): Target portfolio weights.

		Returns:
			pd.DataFrame: Orders to execute for rebalancing.
		"""
		pass


	@loggertext("Error executing rebalance orders.")
	async def _execute_rebalance_orders(self, rebalance_orders: pd.DataFrame) -> None:
		"""
		Execute the rebalancing orders.

		Args:
			rebalance_orders (pd.DataFrame): Orders to execute for rebalancing.
		"""
		pass


	# Load and Save Strategy State ----------------------------------------------------------

	def save_strategy_state(self, file_path: str) -> None:
		"""
		Save the current state of the strategy to a file.
		"""
		state = {
			'config': self.config,
			'backtest_name': self.backtest_name,
			'portfolio_data': self.portfolio_data,
			'instruments_data': self.instruments_data,
		}
		save_pickle(state, file_path)

		logger.info(f"Strategy state saved to {file_path}")


	def load_strategy_state(self, file_path: str) -> None:
		"""
		Load the strategy state from a file.
		"""
		state = load_pickle(file_path)
		self.config = state.get('config')
		self.backtest_name = state.get('backtest_name')
		self.portfolio_data = state.get('portfolio_data')
		self.instruments_data = state.get('instruments_data')

		logger.info(f"Strategy state loaded from {file_path}")


# UnitTests Class --------------------------------------------------------------
class UnitTests(Enum):

	RUN_STRATEGY = 1

async def run_unit_test(unit_test: UnitTests):

	if unit_test == UnitTests.RUN_STRATEGY:

		# As we have abstractmethods for _generate_signals and _generate_positions it will not work, so only to show the structure of strategy
		my_strategy = BaseStrategy(
			config_file_path="quantjourney/backtesting/strategies/default_strategy.json",
			instruments=[	'MSFT', 'ORCL', 'IBM', 'GOOGL', 'F', 'AMZN', 'TSLA', 'META', 'NFLX', 'NVDA',
							'INTC', 'AMD', 'CSCO', 'QCOM', 'ADBE', 'CRM', 'PYPL', 'SQ', 'UBER', 'LYFT'],
			trading_range={'start': '2004-01-01', 'end': '2023-12-31'},
			initial_capital=10000,
			backtest_name="my_strategy",
			load_data=False,
			generate_reports=True,
			generate_plots=False
		)
		await my_strategy.run_simulation()


async def main():
		is_run_all_tests = False
		unit_test = UnitTests.RUN_STRATEGY
		if is_run_all_tests:
			for unit_test in UnitTests:
				await run_unit_test(unit_test=unit_test)
		else:
			await run_unit_test(unit_test=unit_test)

if __name__ == '__main__':
	asyncio.run(main())
