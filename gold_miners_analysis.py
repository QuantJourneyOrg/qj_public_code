"""
    Gold vs Miners Analysis
    ------------------------------------------

    Last Updated: 2024-11-29

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
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass

from quantjourney.data.data_connector_new import DataConnector


@dataclass
class AnalysisConfig:
    start_date: str
    end_date: str
    gold_symbol: str = "GLD"
    miner_symbols: List[str] = None
    plot_style: str = "dark_background"
    figure_size: tuple = (14, 10)
    background_color: str = "#222435"


class DataValidationError(Exception):
    pass


class GoldMinersAnalyzer:
    def __init__(
        self,
        config: AnalysisConfig,
        data_connector: DataConnector
    ) -> None:
        self.config = config
        self.dc = data_connector
        self.gold_data: Optional[pd.Series] = None
        self.miners_data: Dict[str, pd.Series] = {}
        self.failed_symbols: List[str] = []

    async def fetch_data(self) -> None:
        """
        Fetch Gold and Miners data using EOD Historical Data.
        """
        try:
            # Create lists for EOD batch request
            all_symbols = [self.config.gold_symbol] + self.config.miner_symbols
            exchanges = ["US"] * len(all_symbols)
            period_starts = [self.config.start_date] * len(all_symbols)
            period_ends = [self.config.end_date] * len(all_symbols)

            # Fetch all data at once using DataConnector's EOD interface
            dfs = await self.dc.eod.async_get_ohlcv(
                tickers=all_symbols,
                exchanges=exchanges,
                granularity="d",
                period_starts=period_starts,
                period_ends=period_ends,
            )

            # Process Gold data
            if not dfs[0].empty:
                self.gold_data = self._calculate_returns(dfs[0])
            else:
                raise DataValidationError("Failed to fetch gold data")

            # Process Miners data
            for symbol, df in zip(self.config.miner_symbols, dfs[1:]):
                if not df.empty:
                    self.miners_data[symbol] = self._calculate_returns(df)
                else:
                    print(f"Failed to fetch data for {symbol}")
                    self.failed_symbols.append(symbol)

            if not self.miners_data:
                raise DataValidationError("No miner data could be fetched")

        except Exception as e:
            raise RuntimeError(f"Error in data fetching: {str(e)}")

    def _calculate_returns(
        self,
        df: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate percentage returns from price data.
        """
        returns = (df["adj_close"] / df["adj_close"].iloc[0] - 1) * 100
        returns.index = pd.to_datetime(df.index)
        return returns

    def create_plot(self) -> None:
        """
        Create the performance plots.
        """
        if self.failed_symbols:
            print(f"Note: The following symbols failed: {', '.join(self.failed_symbols)}")

        # Set the style and colors
        plt.style.use(self.config.plot_style)

        # Create figure with dark background
        fig, axs = plt.subplots(
            2, 1, figsize=self.config.figure_size, gridspec_kw={"height_ratios": [2, 1]}
        )
        fig.patch.set_facecolor(self.config.background_color)
        axs[0].set_facecolor(self.config.background_color)
        axs[1].set_facecolor(self.config.background_color)

        # Colors for miners
        miner_colors = ["#FF69B4", "#4EC9B0", "#569CD6", "#CE9178", "#B5CEA8"]

        # Top Plot: Gold and Average Miners
        avg_miners = pd.DataFrame(self.miners_data).mean(axis=1)
        axs[0].plot(
            self.gold_data.index,
            self.gold_data.values,
            label="Gold (GLD)",
            color="#FFD700",
            linewidth=1,
        )
        axs[0].plot(
            avg_miners.index,
            avg_miners.values,
            label="Avg Miners",
            color="#FF69B4",
            linewidth=1,
        )
        axs[0].set_title("Gold vs Average Miners Performance", fontsize=14, pad=20, color="white")
        axs[0].set_ylabel("Performance (%)", color="white")
        axs[0].legend(loc="upper left", facecolor=self.config.background_color, labelcolor="white")
        axs[0].grid(True, linestyle="--", alpha=0.2)
        axs[0].tick_params(colors="white")
        for spine in axs[0].spines.values():
            spine.set_color("#666666")

        # Bottom Plot: Individual Miners
        for (miner, data), color in zip(self.miners_data.items(), miner_colors):
            axs[1].plot(data.index, data.values, label=miner, color=color, linewidth=1)
        axs[1].set_title(
            "Individual Miners Performance", fontsize=14, pad=20, color="white"
        )
        axs[1].set_ylabel("Performance (%)", color="white")
        axs[1].set_xlabel("Date", color="white")
        axs[1].legend(
            loc="upper left",
            fontsize=10,
            facecolor=self.config.background_color,
            labelcolor="white",
        )
        axs[1].grid(True, linestyle="--", alpha=0.2)
        axs[1].tick_params(colors="white")
        for spine in axs[1].spines.values():
            spine.set_color("#666666")

        # Add QuantJourney credit
        fig.text(
            0.99,
            0.01,
            "Made with QuantJourney",
            fontsize=8,
            color="gray",
            alpha=0.7,
            ha="right",
            va="bottom",
        )

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)
        plt.show()


async def main():

    dc = DataConnector()

    config = AnalysisConfig(
        start_date="2010-01-01",
        end_date=datetime.now().strftime("%Y-%m-%d"),
        miner_symbols=["NEM", "GOLD", "AEM", "KGC"],
    )

    analyzer = GoldMinersAnalyzer(config, dc)
    try:
        await analyzer.fetch_data()
        analyzer.create_plot()
    except Exception as e:
        print(f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
