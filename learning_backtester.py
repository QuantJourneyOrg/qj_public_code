
"""
	Learning Backtester - Framework for Trading Strategies

	Abstract base class for backtesting trading strategies with
	stripped-down functionality for learning purposes.

	Only:
		- Fetching market data
		- Preprocessing data
		- Generating signals
		- Generating positions
		- Calculating returns
		- Updating portfolio data
		- Generating report

	author: jpolec
	date: 2024-08-20
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List
import json
import os
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# QuantJourney imports ---------------------------------------------------------
from quantjourney.portfolio.portfolio_data import PortfolioData
from quantjourney.portfolio.portfolio_calculations import PortfolioCalculations
from quantjourney.indicators.technical_indicators import TechnicalIndicators
from quantjourney.backtesting.engines import (
	MarketDataProvider,
	DataPreprocessor,
	PerformanceReportGenerator
)
from quantjourney.file.file import load_pickle, save_pickle, get_local_file_path, FileTypes
from quantjourney.data.utils.data_logs import data_logger
from quantjourney.other.decorators import timer, loggertext, error_logger
logger = data_logger()


# LearningBacktester class ---------------------------------------------------------
class LearningBacktester(ABC):
	"""
	A simplified backtesting framework for learning purposes.
	This class provides the core structure for implementing and testing trading strategies.
	"""
	def __init__(self, config_file_path: str, **kwargs) -> None:
		"""
		Initialize the backtester with configuration parameters.

		Args:
			config_file_path (str): Path to configuration file
			**kwargs: Additional keyword arguments
		"""
		try:
			self.config = self._load_config(config_file_path)
		except (FileNotFoundError, ValueError) as e:
			logger.error(f"Error loading configuration: {e}")
			raise

		self.backtest_name = kwargs.get('backtest_name', 'Backtest Default Strategy')

		# Initialize strategy parameters
		self._initialize_strategy(kwargs)

		# Validate strategy parameters
		self._validate_strategy_parameters()

		# Initialize engines
		self._initialize_engines()

		# Initialize data
		self.market_data = None
		self.portfolio_data = None
		self.instruments_data = None


	def _load_config(self, config_file_path: str) -> Dict[str, Any]:
		"""
		Load configuration from a JSON file.
		This method reads strategy parameters and settings from an external file.

		Example configuration file:
		{
			"instruments": ["AAPL", "GOOGL", "MSFT"],
			"initial_capital": 10000.0,
			"trading_range": {"start": "2020-01-01", "end": "2020-12-31"},
			"indicators_config": [
				{"function": "SMA", "price_cols": ["close"], "params": {"window": 50}},
				{"function": "SMA", "price_cols": ["close"], "params": {"window": 200}}
			]
		}

		Args:
			config_file_path (str): Path to configuration file

		Returns:
			dict: Configuration parameters
		"""
		if not os.path.isfile(config_file_path):
			raise FileNotFoundError(f"Configuration file '{config_file_path}' not found.")
		try:
			with open(config_file_path, 'r') as config_file:
				return json.load(config_file)
		except json.JSONDecodeError as e:
			raise ValueError(f"Error decoding JSON configuration file: {e}")


	# Initialize strategy and eninges ----------------------------------------------------------

	def _initialize_strategy(self, kwargs: Dict[str, Any]) -> None:
		""""
		Initialize strategy parameters from configuration and keyword arguments.
		This method initializes key strategy components like instruments, capital, and trading range.

		Args:
			kwargs (Dict[str, Any]): Additional keyword arguments to override config settings.
				- instruments (List[str]): List of instruments to trade.
				- initial_capital (float): Initial capital for the strategy.
				- trading_range (Dict[str, str]): Start and end dates for the trading range.
				- indicators_config (List[Dict[str, Any]]): Configuration for technical indicators.
		"""
		# Instrument parameters
		self.instruments = kwargs.get('instruments', self.config.get('instruments', []))

		# Capital parameters
		self.initial_capital = kwargs.get('initial_capital', self.config.get('initial_capital', 10000.0))
		self.trading_range = kwargs.get('trading_range', self.config.get('trading_range', {'start': '2020-01-01', 'end': '2020-12-31'}))
		self.indicators_config = kwargs.get('indicators_config', self.config.get('indicators_config', []))


	def _validate_strategy_parameters(self) -> None:
		"""
		Validate strategy parameters after initialization.

		We have to check that the instruments, initial capital, and trading range are valid.

		Raises:
			ValueError: If any of the parameters are invalid.
		"""
		if not self.instruments:
			raise ValueError("No instruments specified for the strategy")

		if not isinstance(self.initial_capital, (int, float)) or self.initial_capital <= 0:
			raise ValueError("Initial capital must be a positive number")

		if not isinstance(self.trading_range, dict) or 'start' not in self.trading_range or 'end' not in self.trading_range:
			raise ValueError("Trading range must be a dict with 'start' and 'end' keys")


	def _initialize_engines(self) -> None:
		"""
		Initialize the essential engines for the learning backtester.

		This method sets up the core components needed for a basic backtesting experience:
		- MarketDataProvider: Provides historical market data.
		- DataPreprocessor: Processes market data and creates necessary data structures.
		- PerformanceReportGenerator: Generates basic performance reports.
		"""
		engine_params = [
			('mdp', MarketDataProvider, 'market_data_provider'),
			('dp', DataPreprocessor, 'data_preprocessor'),
			('prg', PerformanceReportGenerator, 'performance_report_generator'),
		]

		for attr, cls, param_key in engine_params:
			if cls == PerformanceReportGenerator:
				# Initialize PerformanceReportGenerator with None for portfolio_data and instruments_data
				setattr(self, attr, cls(self.config.get(param_key, {}), None, None))
			else:
				setattr(self, attr, cls(self.config.get(param_key, {})))

		# Set the data provider
		self.data_provider = self.mdp


	# Get and Process Off-Line data ----------------------------------------------------------

	async def _get_market_data(self) -> None:
		"""
		Fetch market data for the specified instruments and trading range.
		"""
		try:
			# We use self.data_provider which can be both for Off-Line and RealTime data (based on config)
			self.market_data = await self.data_provider.get_market_data(instruments=self.instruments,
																		trading_range=self.trading_range)

			if self.market_data is None or (isinstance(self.market_data, dict) and len(self.market_data) == 0):
				logger.error("No market data was retrieved.")
				raise

			logger.info("Market data successfully retrieved.")

		except Exception as e:
			logger.error(f"Error in getting market data: {e}")
			raise


	async def _preprocess_market_data(self) -> None:
		"""
		Process the raw market data into a format suitable for strategy implementation.
		This includes calculating necessary indicators and features.

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


	# Signal, Positions, Weights, Trades Generation Abstract Classes -------------------------

	@abstractmethod
	def _generate_signals(self) -> pd.DataFrame:
		"""
		Generate trading signals based on the processed data.
		This is where the core logic of your trading strategy is implemented.

		Generate raw signals (-1, 0, 1) for each instrument.

		Returns:
			pd.DataFrame: DataFrame containing generated signals.

		To get Technical Indicators created in _preprocess_market_data, we can:
			- Use self.instruments_data.get_feature('indicator_name')
			- Use self.instruments_data.data[instrument, 'indicator_name']
		Examples:
			close = self.instruments_data.get_feature('adj_close')
			sma_short = self.instruments_data.get_feature('SMA_short')
			sma_long = self.instruments_data.get_feature('SMA_long')
			positions = self.instruments_data.data[instrument, 'position']

		To get Fundamental Data created in _preprocess_market_data, we can: (Note: not shared in this snippet)
			- Use self.instruments_data.get_fundamental('fundamental_name')
			- Use self.instruments_data.data[instrument, 'fundamental_name']

		Examples:
			GROSS_PROFIT_MARGIN = self.instruments_data.get_fundamental('GROSS_PROFIT_MARGIN')
			NET_PROFIT_MARGIN = self.instruments_data.get_fundamental('NET_PROFIT_MARGIN')
			ROE = self.instruments_data.get_fundamental('ROE')

		Raises:
			NotImplementedError: If the method is not implemented in a child class.
		"""
		pass


	@abstractmethod
	def _generate_positions(self, signals: pd.DataFrame) -> pd.DataFrame:
		"""
		Convert trading signals into actual position sizes.
		This method determines how much to invest in each instrument based on the signals.

		Positions represent the amount of each asset you want to hold in your portfolio.
		They can be expressed as:
		- Number of shares/contracts
		- Monetary value of the position
		- Percentage of the portfolio

		Args:
			signals (pd.DataFrame): DataFrame containing generated signals.

		Returns:
			pd.DataFrame:   DataFrame containing generated positions.
							Index: DatetimeIndex - dates of the positions
							Columns: Instrument names
							Values: Position sizes (can be number of shares, monetary value, or percentage)

		Example:
			def _generate_positions(self, signals):
				# Simple example: convert signals directly to positions
				return signals * 100  # Each signal results in a position of 100 shares

		Notes:
			- Consider incorporating position sizing techniques (e.g., equal weight, Kelly criterion)
			- Implement risk management rules (e.g., maximum position size, stop-loss levels)
			- Ensure that the sum of absolute position values doesn't exceed your maximum leverage
		"""
		pass


	@abstractmethod
	def _generate_weights(self, positions: pd.DataFrame) -> pd.DataFrame:
		"""
		Calculate the weight of each instrument in the portfolio.
		This method determines the proportion of the portfolio allocated to each instrument.

		Weights represent the proportion of your total portfolio value allocated to each asset.
		They are typically expressed as percentages and should sum to 1 (or 100%).

		This method converts the positions into portfolio weights, ensuring proper
		allocation of capital across all instruments.

		Args:
			positions (pd.DataFrame): DataFrame containing generated positions.

		Returns:
			pd.DataFrame:   DataFrame containing generated weights.
							Index: DatetimeIndex - dates of the weights
							Columns: Instrument names
							Values: Weight of each instrument in the portfolio (typically 0 to 1)

		Example:
			def _generate_weights(self, positions):
				# Simple example: equal weighting for all non-zero positions
				weights = (positions != 0).astype(float)
				return weights.div(weights.sum(axis=1), axis=0).fillna(0)

		Notes:
			- Ensure that weights sum to 1 (or close to 1, accounting for floating-point precision)
			- Consider minimum and maximum weight constraints for diversification
			- You may implement more sophisticated weighting schemes based on your strategy
				(e.g., risk parity, minimum variance portfolio)
		"""
		pass


	# Main Simulation ----------------------------------------------------------

	async def run_simulation(self) -> None:
		"""
		Run the backtesting simulation, and orchestrate the entire backtesting process, including
		data retrieval, preprocessing, signal generation, position calculation, and performance evaluation.

		Raises:
			Exception: If there's an error during the simulation process.
		"""
		logger.info(f"Running Strategy Simulation: {self.backtest_name}")

		await self._get_market_data()					# Fetch market data
		await self._preprocess_market_data()			# Preprocess market data

		signals = self._generate_signals()				# Generate signals
		positions = self._generate_positions(signals)	# Generate positions based on signals
		weights = self._generate_weights(positions)		# Generate weights based on positions

		self._calculate_returns(positions)				# Calculate returns
		self._update_portfolio_data(weights, positions) # Update portfolio data

		self._generate_basic_report()					# Generate basic report
		self._generate_advanced_report()				# Generate advanced report using PerformanceReportGenerator

		logger.info("Simulation completed successfully.")


	def _calculate_returns(self, positions: pd.DataFrame) -> None:
		try:
			# Assuming InstrumentData has a method to get close prices
			if hasattr(self.instruments_data, 'get_close_prices'):
				close_prices = self.instruments_data.get_close_prices()
			elif hasattr(self.instruments_data, 'data'):
				# If it's storing data in a 'data' attribute
				close_prices = self.instruments_data.data.xs('close', axis=1, level=1)
			else:
				raise AttributeError("Unable to retrieve close prices from InstrumentData")

			# Calculate returns
			returns = close_prices.pct_change()

			# Ensure positions and returns have the same columns
			common_columns = positions.columns.intersection(returns.columns)
			positions = positions[common_columns]
			returns = returns[common_columns]

			self.portfolio_returns = (positions * returns).sum(axis=1)
			self.cumulative_returns = (1 + self.portfolio_returns).cumprod() - 1

		except Exception as e:
			logger.error(f"Error in calculating returns: {e}")
			# Set empty Series if calculation fails
			self.portfolio_returns = pd.Series()
			self.cumulative_returns = pd.Series()


	def _update_portfolio_data(self, weights: pd.DataFrame, positions: pd.DataFrame) -> None:
		"""
		Update the portfolio data with the latest weights, positions, and returns.
		"""
		self.portfolio_data.asset_weights = weights
		self.portfolio_data.positions = positions
		self.portfolio_data.returns = self.portfolio_returns
		self.portfolio_data.cumulative_returns = self.cumulative_returns
		self.portfolio_data.net_asset_value = self.initial_capital * (1 + self.cumulative_returns)

		# Update the PerformanceReportGenerator with the new data
		self.prg._update_data(self.portfolio_data, self.instruments_data)


	def _generate_basic_report(self) -> None:
		"""
		Generate a basic performance report for the backtest.
		"""
		total_return = self.cumulative_returns.iloc[-1]
		sharpe_ratio = self.portfolio_returns.mean() / self.portfolio_returns.std() * np.sqrt(252)  # Assuming daily returns

		print("Basic Performance Report")
		print("========================")
		print(f"Total Return: {total_return:.2%}")
		print(f"Sharpe Ratio: {sharpe_ratio:.2f}")


	def _generate_advanced_report(self) -> None:
		"""
		Generate an advanced performance report using the PerformanceReportGenerator.
		"""
		print("\nAdvanced Performance Report")
		print("============================")
		text_report = self.prg.generate_text_report()
		print(text_report)
		self.prg.plot_strategy_results()