"""
    M2 Money Supply vs BTC Price Analysis Module
    ------------------------------------------

    Last Updated: 2024-11-28

    Note:
    This module is part of a larger educational and prototyping framework and may lack
    advanced features or optimizations found in production-grade systems.

    Proprietary License - QuantJourney Framework
    This file is part of the QuantJourney Framework and is licensed for internal, non-commercial use only.
    Modifications are permitted solely for personal, non-commercial testing. Redistribution and commercial use are prohibited.

    For full terms, see the LICENSE file or contact Jakub Polec at jakub@quantjourney.pro.
"""

from __future__ import annotations
import asyncio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
from quantjourney.data.data_connector import DataConnector
from quantjourney.logging import logger


@dataclass
class AnalysisConfig:

    start_date: str
    end_date: str
    m2_series_id: str = "WM2NS"
    btc_exchange: str = "binance"
    btc_pair: str = "BTC/USDT"
    timeframe: str = "1d"
    plot_style: str = "dark_background"
    figure_size: Tuple[int, int] = (12, 6)


class DataValidationError(Exception):

    pass


class M2BTCAnalyzer:

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.dc = DataConnector()
        self.m2_data: Optional[pd.Series] = None
        self.btc_data: Optional[pd.DataFrame] = None

    async def fetch_data(self) -> None:
        try:
            start_time = datetime.now()

            # Fetch data concurrently
            m2_future = asyncio.create_task(self._fetch_m2_data())
            btc_future = asyncio.create_task(self._fetch_btc_data())

            self.m2_data, self.btc_data = await asyncio.gather(m2_future, btc_future)

            # Log performance metrics
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Data fetch completed in {execution_time:.2f} seconds")

            # Validate data
            self._validate_data()

        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            raise

    async def _fetch_m2_data(self) -> pd.Series:
        try:
            return self.dc.fred.get_series(
                self.config.m2_series_id,
                observation_start=self.config.start_date,
                observation_end=self.config.end_date,
            )
        except Exception as e:
            logger.error(f"Error fetching M2 data: {str(e)}")
            raise

    async def _fetch_btc_data(self) -> pd.DataFrame:
        try:
            return await self.dc.ccxt.async_get_ohlcv(
                exchange=self.config.btc_exchange,
                ticker=self.config.btc_pair,
                timeframe=self.config.timeframe,
                start_date=self.config.start_date,
                end_date=self.config.end_date,
                read_from_db=False,
                write_to_db=False,
            )
        except Exception as e:
            logger.error(f"Error fetching BTC data: {str(e)}")
            raise

    def _validate_data(self) -> None:
        if self.m2_data is None or self.btc_data is None:
            raise DataValidationError("Missing required data")

        if len(self.m2_data) < 2 or len(self.btc_data) < 2:
            raise DataValidationError("Insufficient data points for analysis")

        if self.m2_data.isnull().any() or self.btc_data["close"].isnull().any():
            raise DataValidationError("Data contains null values")

    def calculate_metrics(self) -> Dict[str, float]:
        metrics = {
            "correlation": self._calculate_correlation(),
            "m2_growth": self._calculate_m2_growth(),
            "btc_return": self._calculate_btc_return(),
            "lag_weeks": self._estimate_lag(),
        }
        return metrics

    def _calculate_correlation(self) -> float:
        # Resample data to common frequency
        m2_resampled = self.m2_data.resample("D").interpolate()
        common_dates = m2_resampled.index.intersection(self.btc_data.index)
        return m2_resampled[common_dates].corr(self.btc_data["close"][common_dates])

    def _calculate_m2_growth(self) -> float:
        return (self.m2_data.iloc[-1] / self.m2_data.iloc[0] - 1) * 100

    def _calculate_btc_return(self) -> float:
        return (
            self.btc_data["close"].iloc[-1] / self.btc_data["close"].iloc[0] - 1
        ) * 100

    def _estimate_lag(self) -> int:
        # Implementation for lag calculation
        # This would involve cross-correlation analysis
        return 10  # Placeholder

    def create_plot(self) -> None:
        try:
            plt.style.use(self.config.plot_style)
            fig, ax1 = plt.subplots(figsize=self.config.figure_size)

            # Plot M2 data
            ax1.set_xlabel("Date")
            ax1.set_ylabel("M2 Money Supply (Trillions USD)", color="white")
            m2_line = ax1.plot(
                self.m2_data.index,
                self.m2_data.values / 1000,
                "white",
                linewidth=1,
                label="Global Money Supply US",
            )
            ax1.tick_params(axis="y", labelcolor="white")

            # Plot BTC data
            ax2 = ax1.twinx()
            ax2.set_ylabel("BTC-USD", color="#ff9800")
            btc_line = ax2.plot(
                self.btc_data.index,
                self.btc_data["close"],
                color="#ff9800",
                linewidth=1,
                label="XBT-USD Cross Rate",
            )
            ax2.tick_params(axis="y", labelcolor="#ff9800")

            # Styling
            ax1.grid(True, linestyle="--", alpha=0.3)
            plt.title(
                f"BTC lags global M2 money supply by ~{self._estimate_lag()} weeks",
                pad=20,
            )

            # Legend
            lines = m2_line + btc_line
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc="upper left")

            # Last values
            last_m2 = f"{self.m2_data.iloc[-1]/1000:.3f}T"
            last_btc = f"{self.btc_data['close'].iloc[-1]:.2f}"
            plt.figtext(0.15, 0.95, f"Last Price\n{last_m2}", color="white")
            plt.figtext(0.85, 0.95, f"{last_btc}", color="#ff9800")

            plt.tight_layout()
            plt.show()

        except Exception as e:
            logger.error(f"Error creating plot: {str(e)}")
            raise


async def main():
    try:
        config = AnalysisConfig(
            start_date="2023-09-01", end_date=datetime.now().strftime("%Y-%m-%d")
        )

        analyzer = M2BTCAnalyzer(config)
        await analyzer.fetch_data()

        metrics = analyzer.calculate_metrics()
        logger.info("Analysis Metrics:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.2f}")

        analyzer.create_plot()

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
