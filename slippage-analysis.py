"""
	Slippage Analyzer with Enhanced Features and Machine Learning
	Part of QuantJourney Framework
	---------------------------------------------------------

	Last Updated: 2024-11-23

	Note:
	This module is part of a larger educational and prototyping framework and may lack
	advanced features or optimizations found in production-grade systems.

	Proprietary License - QuantJourney Framework

	This file is part of the QuantJourney Framework and is licensed for internal, non-commercial use only.
	Modifications are permitted solely for personal, non-commercial testing. Redistribution and commercial use are prohibited.

	For full terms, see the LICENSE file or contact Jakub Polec at jakub@quantjourney.pro.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from typing import Tuple, Dict, Any
import warnings

warnings.filterwarnings("ignore")


class SlippageAnalyzer:
	def __init__(
		self,
		lookback_period: int = 10,
		min_order_size: float = 100,
		max_order_pct: float = 0.03,
	):
		self.lookback_period = lookback_period
		self.min_order_size = min_order_size
		self.max_order_pct = max_order_pct
		self.scaler = RobustScaler()
		self.model = None

	def calculate_price_metrics(
		self,
		df: pd.DataFrame
	) -> pd.DataFrame:
		"""
		Calculate price-based features with reduced correlations
		"""
		# Basic price metrics
		df["returns"] = df["Close"].pct_change()
		df["log_returns"] = np.log1p(df["returns"])

		# Volatility with different timeframes
		df["vol_short"] = df["returns"].rolling(5).std() * np.sqrt(252)
		df["vol_mid"] = df["returns"].rolling(21).std() * np.sqrt(252)
		df["vol_long"] = df["returns"].rolling(63).std() * np.sqrt(252)

		# Ranges and spreads
		df["true_range"] = np.maximum(
			df["High"] - df["Low"],
			np.maximum(
				abs(df["High"] - df["Close"].shift(1)),
				abs(df["Low"] - df["Close"].shift(1)),
			),
		)
		df["rel_spread"] = (df["High"] - df["Low"]) / df["Close"]

		# Non-linear transformations
		df["price_acceleration"] = df["returns"].diff()
		df["range_intensity"] = (df["true_range"] / df["Close"]) * np.sqrt(df["Volume"])

		price_cols = df.filter(
			regex="^(vol_|returns|true_range|rel_spread|price|range)"
		).columns
		df[price_cols] = df[price_cols].fillna(method="ffill").fillna(0)

		return df

	def calculate_volume_metrics(
		self,
		df: pd.DataFrame
	) -> pd.DataFrame:
		"""
		Calculate volume-based features with non-linear relationships
		"""
		# Base volume metrics
		df["log_volume"] = np.log1p(df["Volume"])

		# Volume moving averages with different timeframes
		df["vol_ma_short"] = df["Volume"].rolling(5).mean()
		df["vol_ma_mid"] = df["Volume"].rolling(21).mean()
		df["vol_ma_long"] = df["Volume"].rolling(63).mean()

		# Volume ratios
		df["vol_ratio_short"] = df["Volume"] / df["vol_ma_short"]
		df["vol_ratio_mid"] = df["Volume"] / df["vol_ma_mid"]
		df["vol_ratio_long"] = df["Volume"] / df["vol_ma_long"]

		# Non-linear volume metrics
		df["vol_impact"] = np.power(df["Volume"] / df["vol_ma_mid"], 0.6)
		df["vol_surge"] = (df["Volume"] / df["vol_ma_long"]).clip(upper=5)

		volume_cols = df.filter(regex="^(vol_|log_volume)").columns
		df[volume_cols] = df[volume_cols].fillna(method="ffill").fillna(1)

		return df

	def calculate_market_impact(
		self,
		df: pd.DataFrame
	) -> pd.DataFrame:
		"""
		Calculate market impact with more complex relationships
		"""
		# Liquidity measures
		df["amihud_illiq"] = abs(df["returns"]) / (df["Volume"] * df["Close"])
		df["turnover"] = df["Volume"] * df["Close"]
		df["turnover_vol"] = (
			df["turnover"].rolling(21).std() / df["turnover"].rolling(21).mean()
		)

		# Market impact components
		df["base_impact"] = np.power(
			self.max_order_pct / df["vol_ratio_mid"].clip(lower=1e-8), 0.5
		)
		df["vol_adjusted_impact"] = df["base_impact"] * np.exp(df["vol_surge"] - 1)

		# Non-linear combinations
		df["impact_score"] = (
			df["base_impact"]
			* np.exp(df["vol_short"] * 2)
			* np.power(df["amihud_illiq"].clip(lower=1e-8), 0.3)
		)

		impact_cols = [
			"amihud_illiq",
			"turnover",
			"turnover_vol",
			"base_impact",
			"vol_adjusted_impact",
			"impact_score",
		]
		df[impact_cols] = df[impact_cols].fillna(method="ffill").fillna(0)

		return df

	def calculate_slippage(
		self,
		df: pd.DataFrame
	) -> pd.DataFrame:
		"""
		Calculate realistic slippage with more randomness and non-linear effects
		"""
		# Base components
		df["spread_cost"] = df["rel_spread"] * 0.5
		df["volatility_cost"] = df["vol_mid"] * np.exp(df["vol_surge"] - 1) * 0.2

		# Market impact with decay
		df["market_impact"] = (
			0.1
			* np.power(self.max_order_pct / df["vol_ratio_mid"].clip(lower=1e-8), 0.6)
			* df["vol_mid"]
			* np.exp(-df["turnover_vol"])
		)

		# Additional components
		df["momentum_cost"] = (
			0.05 * abs(df["returns"]) * np.sign(df["price_acceleration"])
		)
		df["liquidity_cost"] = 0.1 * np.power(df["amihud_illiq"].clip(lower=1e-8), 0.3)

		# Random market noise (increased variance)
		noise = np.random.normal(0, 0.0005, len(df))

		# Combine components with non-linear interactions
		df["slippage"] = (
			df["spread_cost"]
			+ df["market_impact"] * (1 + df["volatility_cost"])
			+ df["momentum_cost"] * df["liquidity_cost"]
			+ noise
		).clip(
			0, 0.05
		)  # Cap at 5%

		cost_cols = [
			"spread_cost",
			"volatility_cost",
			"market_impact",
			"momentum_cost",
			"liquidity_cost",
			"slippage",
		]
		df[cost_cols] = df[cost_cols].fillna(method="ffill").fillna(0)

		return df

	def prepare_features(
		self,
		df: pd.DataFrame
	) -> pd.DataFrame:
		"""
		Prepare feature set with reduced correlation
		"""
		df = self.calculate_price_metrics(df)
		df = self.calculate_volume_metrics(df)
		df = self.calculate_market_impact(df)
		df = self.calculate_slippage(df)

		# Select less correlated features
		feature_cols = [
			"vol_short",
			"vol_long",  # Different timeframe volatilities
			"rel_spread",
			"range_intensity",
			"vol_ratio_mid",
			"vol_surge",
			"turnover_vol",
			"amihud_illiq",
			"impact_score",
		]

		return df[feature_cols].fillna(method="ffill").fillna(0)

	def train_model(
		self,
		X: pd.DataFrame,
		y: pd.Series
	) -> Dict[str, Any]:
		"""
		Train model with increased regularization
		"""
		mask = ~(X.isna().any(axis=1) | y.isna())
		X = X[mask]
		y = y[mask]

		tscv = TimeSeriesSplit(n_splits=5, test_size=50)
		cv_scores = []

		# More regularized model parameters
		self.model = RandomForestRegressor(
			n_estimators=100,
			max_depth=4,  # Reduced depth
			min_samples_leaf=30,  # Increased min samples
			min_samples_split=30,  # Increased min split
			max_features="sqrt",  # Reduced feature subset
			random_state=42,
		)

		for train_idx, val_idx in tscv.split(X):
			X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
			y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

			self.model.fit(X_train, y_train)
			y_pred = self.model.predict(X_val)
			cv_scores.append(r2_score(y_val, y_pred))

		self.model.fit(X, y)
		y_pred = self.model.predict(X)

		return {
			"metrics": {
				"mse": mean_squared_error(y, y_pred),
				"mae": mean_absolute_error(y, y_pred),
				"r2": r2_score(y, y_pred),
				"cv_r2_mean": np.mean(cv_scores),
				"cv_r2_std": np.std(cv_scores),
			},
			"feature_importance": dict(zip(X.columns, self.model.feature_importances_)),
		}

	def analyze_stock(
		self,
		ticker: str,
		start_date: str,
		end_date: str
	) -> Dict[str, Any]:
		"""
		Analysis pipeline
		"""
		try:
			df = yf.download(ticker, start=start_date, end=end_date, progress=False)
			if df.empty:
				raise ValueError(f"No data found for {ticker}")

			df_features = self.prepare_features(df)

			mask = ~(df_features.isna().any(axis=1) | df["slippage"].isna())
			df_features = df_features[mask]
			df = df[mask]

			X = pd.DataFrame(
				self.scaler.fit_transform(df_features),
				columns=df_features.columns,
				index=df_features.index,
			)

			results = self.train_model(X, df["slippage"])
			df.loc[X.index, "predicted_slippage"] = self.model.predict(X)

			# Perform SHAP analysis
			self.explain_model_with_shap(X)

			self.plot_analysis(df, results)

			return results

		except Exception as e:
			print(f"Error in analysis: {str(e)}")
			return None

	def explain_model_with_shap(
		self,
		X: pd.DataFrame
	) -> None:
		"""
		Use SHAP to explain model predictions and visualize feature impacts.
		"""
		try:
			# Initialize SHAP explainer
			explainer = shap.Explainer(self.model, X)
			shap_values = explainer(X)

			# Generate SHAP summary plot
			print("\nGenerating SHAP Summary Plot...")
			shap.summary_plot(shap_values, X, plot_type="bar")

			# Generate SHAP dependence plot for a key feature
			print("\nGenerating SHAP Dependence Plot for 'vol_ratio_mid'...")
			shap.dependence_plot("vol_ratio_mid", shap_values.values, X)

		except Exception as e:
			print(f"Error in SHAP analysis: {str(e)}")

	def plot_analysis(
		self,
		df: pd.DataFrame,
		results: Dict[str, Any]
	) -> None:
		"""
		Create visualization
		"""
		fig, axes = plt.subplots(3, 2, figsize=(12, 10))

		valid_mask = df["slippage"].notna() & df["predicted_slippage"].notna()
		axes[0, 0].plot(
			df.index[valid_mask], df["slippage"][valid_mask], label="Actual", alpha=0.6
		)
		axes[0, 0].plot(
			df.index[valid_mask],
			df["predicted_slippage"][valid_mask],
			label="Predicted",
			alpha=0.6,
		)
		axes[0, 0].set_title("Actual vs Predicted Slippage")
		axes[0, 0].legend()

		importance_df = pd.DataFrame.from_dict(
			results["feature_importance"], orient="index", columns=["Importance"]
		).sort_values("Importance", ascending=True)

		importance_df.plot(kind="barh", ax=axes[0, 1])
		axes[0, 1].set_title("Feature Importance")

		sns.histplot(df["slippage"].dropna(), kde=True, ax=axes[1, 0])
		axes[1, 0].set_title("Slippage Distribution")

		error = (
			df.loc[valid_mask, "slippage"] - df.loc[valid_mask, "predicted_slippage"]
		)
		sns.histplot(error, kde=True, ax=axes[1, 1])
		axes[1, 1].set_title("Prediction Error Distribution")

		valid_mask = (
			df["vol_ratio_mid"].notna()
			& df["slippage"].notna()
			& df["vol_short"].notna()
		)
		scatter = axes[2, 0].scatter(
			df.loc[valid_mask, "vol_ratio_mid"],
			df.loc[valid_mask, "slippage"],
			c=df.loc[valid_mask, "vol_short"],
			cmap="viridis",
			alpha=0.5,
		)
		plt.colorbar(scatter, ax=axes[2, 0], label="Short-term Volatility")
		axes[2, 0].set_title("Slippage vs Volume Ratio")

		valid_mask = df["predicted_slippage"].notna() & error.notna()
		axes[2, 1].scatter(
			df.loc[valid_mask, "predicted_slippage"], error[valid_mask], alpha=0.5
		)
		axes[2, 1].axhline(y=0, color="r", linestyle="--")
		axes[2, 1].set_title("Residuals vs Predicted")

		plt.tight_layout()
		plt.show()


def main():
	analyzer = SlippageAnalyzer(
		lookback_period=10, min_order_size=100, max_order_pct=0.03
	)

	ticker = "AAPL"
	results = analyzer.analyze_stock(
		ticker=ticker,
		start_date="2020-01-01",
		end_date="2023-12-31"
	)

	if results:
		print("\nModel Performance Metrics:")
		print(f"MSE: {results['metrics']['mse']:.6f}")
		print(f"MAE: {results['metrics']['mae']:.6f}")
		print(f"R2: {results['metrics']['r2']:.6f}")
		print(
			f"CV R2: {results['metrics']['cv_r2_mean']:.6f} ± {results['metrics']['cv_r2_std']:.6f}"
		)

		print("\nTop Feature Importance:")
		for feature, importance in sorted(
			results["feature_importance"].items(), key=lambda x: x[1], reverse=True
		)[:5]:
			print(f"{feature}: {importance:.4f}")


if __name__ == "__main__":
	main()
