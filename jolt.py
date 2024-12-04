"""
    JOLT BEA (Bureau of Economic Analysis) for QuantJourney Framework
    ---------------------------------------------------------

    Last Updated: 2024-12-04

    Proprietary License - QuantJourney Framework

    This file is part of the QuantJourney Framework and is licensed for internal,
    non-commercial use only. Modifications are permitted solely for personal, non-commercial testing.
    Redistribution and commercial use are prohibited.

    For full terms, see the LICENSE file or contact Jakub Polec at jakub@quantjourney.pro.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import asyncio
from scipy import stats
from matplotlib import colors
from datetime import datetime

# QuantJourney modules
from quantjourney.data.data_connector import DataConnector
from quantjourney.logging import logger

async def fetch_market_data(eod):
    """
    Fetch market data using the EodConnector for key indices and ETFs.
    """
    # Define market instruments and their sources
    market_series = {
        "SPY": {"ticker": "SPY", "exchange": "US"},   # S&P 500 ETF
        "VIX": {"ticker": "VXX", "exchange": "US"},
        "TLT": {"ticker": "TLT", "exchange": "US"},   # Treasury Bond ETF
        "XLF": {"ticker": "XLF", "exchange": "US"},   # Financial Sector ETF
        "QQQ": {"ticker": "QQQ", "exchange": "US"}    # Nasdaq 100 ETF
    }

    data = {}

    for name, info in market_series.items():
        print(f"Fetching {name} data...")
        try:
            # Fetch OHLCV data for each ticker
            df_list = await eod.async_get_ohlcv(
                tickers=[info["ticker"]],
                exchanges=[info["exchange"]],
                granularity="d",
                period_starts=["2020-01-01"],
                period_ends=["2024-12-01"]
            )

            if df_list and len(df_list) > 0:
                df = df_list[0]

                if not df.empty:
                    # Ensure datetime index from the 'datetime' column
                    if 'datetime' in df.columns:
                        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
                        df.set_index('datetime', inplace=True)
                    else:
                        print(f"'datetime' column missing for {name}. Skipping.")
                        continue

                    # Add the adjusted close price to the data dictionary
                    data[name] = df['adj_close']
                else:
                    print(f"No data returned for {name}.")
            else:
                print(f"No data returned for {name}.")
        except AttributeError as e:
            print(f"Error fetching {name}: {e}. Please check the connector or the method name.")
        except Exception as e:
            print(f"Unexpected error fetching {name}: {e}")

    # Convert the collected data to a DataFrame with datetime index
    if data:
        df = pd.DataFrame(data)
        print("Final Market DataFrame sample:\n", df.head())  # Debug: final DataFrame
        return df
    else:
        print("No market data was fetched.")
        return pd.DataFrame()

async def fetch_jolts_data(fred):
    """
    Fetch JOLTS data for different sectors
    """
    # Updated list of FRED series for different sectors
    sector_series = {
        "Construction": "JTS2300JOR",
        "Total Nonfarm": "JTSJOR",
        "Manufacturing": "JTU3000JOR",
        "Professional and Business Services": "JTS540099JOR",
        "Health Care and Social Assistance": "JTS6200JOL",
        "Leisure and Hospitality": "JTS7000JOR",
        "Transportation, Warehousing, and Utilities": "JTU480099JOR",
        "Retail Trade": "JTS4400JOR",
        "Accommodation and Food Services": "JTS7200JOR",
        "Government": "JTS9000JOR",
        "Private Education and Health Services": "JTS6000JOR",
        "Durable Goods Manufacturing": "JTS3200JOR",
        "Financial Activities": "JTU510099JOR",
        "State and Local": "JTS9200JOR",
        "Trade, Transportation, and Utilities": "JTU4000JOR",
        "Arts, Entertainment, and Recreation": "JTS7100JOR",
        "Mining and Logging": "JTU110099JOR",
        "Real Estate and Rental and Leasing": "JTU5300JOR",
        "Nondurable Goods Manufacturing": "JTS3400JOR",
        "Private Educational Services": "JTU6100JOR",
        "Wholesale Trade": "JTU4200JOR",
        "Other Services": "JTU8100JOR",
        "Federal": "JTU9100JOR",
    }

    start_date = "2020-01-01"
    end_date = "2024-12-01"

    # Fetch data for each sector
    data = {}
    for sector, series_id in sector_series.items():
        print(f"Fetching data for {sector}...")
        sector_data = await fred.async_get_series(
            search_id=series_id, observation_start=start_date, observation_end=end_date
        )
        data[sector] = sector_data

    # Combine data into a DataFrame
    df = pd.DataFrame(data)
    df.index = pd.to_datetime(df.index)

    # Calculate monthly net changes
    df_changes = df.diff().dropna()

    return df_changes

def create_correlation_analysis(jolts_data, market_data):
    """
    Enhanced correlation analysis with market factors.
    """
    # Convert market_data to timezone-naive for alignment
    market_data.index = market_data.index.tz_localize(None)

    # Resample market_data to the first trading day of each month
    market_monthly = market_data.resample('MS').first()

    # Align market_monthly to match the jolts_data index
    market_monthly = market_monthly.reindex(jolts_data.index, method='ffill')  # Forward-fill to align dates

    if jolts_data.empty or market_monthly.empty:
        print("No overlapping data after alignment. Correlation matrix is empty.")
        return

    # Initialize correlations DataFrame
    correlations = pd.DataFrame(index=jolts_data.columns)

    # Calculate correlations
    for indicator in market_monthly.columns:
        corr_values = []
        for sector in jolts_data.columns:
            x = jolts_data[sector].dropna()
            y = market_monthly[indicator].dropna()
            aligned_x, aligned_y = x.align(y, join='inner')

            if len(aligned_x) >= 2:
                corr_values.append(stats.pearsonr(aligned_x, aligned_y)[0])
            else:
                corr_values.append(np.nan)
        correlations[indicator] = corr_values

    if correlations.empty or correlations.isna().all().all():
        print("Correlation matrix is empty or contains only NaN values.")
        return

    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlations, annot=True, cmap='RdYlBu', center=0, fmt='.2f')
    plt.title('JOLTS Sectors vs Market Indicators Correlation Matrix')
    plt.tight_layout(pad=2, rect=[0.2, 0, 1, 1])  # Adjust `rect` to move the left margin
    plt.show()

def create_sector_relative_strength(jolts_data, market_data):
    """
    Analyze sector performance relative to market benchmarks
    """
    # Calculate relative strength vs SPY
    spy_monthly = market_data['SPY'].resample('ME').last().pct_change()  # Changed from 'M' to 'ME'
    sector_rs = pd.DataFrame()

    for sector in jolts_data.columns:
        sector_changes = jolts_data[sector]
        # Calculate rolling correlation with market
        rolling_corr = sector_changes.rolling(window=6).corr(spy_monthly)
        # Calculate relative strength
        rolling_beta = (sector_changes.rolling(window=6).cov(spy_monthly) / 
                       spy_monthly.rolling(window=6).var())
        sector_rs[f"{sector}_correlation"] = rolling_corr
        sector_rs[f"{sector}_beta"] = rolling_beta

    return sector_rs

def create_main_plots(data):
    if data.empty:
        print("No data available for plotting")
        return

    # Separate healthcare data
    healthcare_data = data[["Health Care and Social Assistance"]]
    other_data = data.drop(columns=["Health Care and Social Assistance"])

    # Prepare the data for plotting
    monthly_totals = other_data.sum(axis=1)
    data_with_total = other_data.copy()
    data_with_total["Total"] = monthly_totals

    # Create the figure with three subplots
    fig, ax = plt.subplots(3, 1, figsize=(14, 16), gridspec_kw={"height_ratios": [2, 2, 2]})

    # Plot for other sectors
    ax1 = ax[0]
    other_data.plot(kind="bar", stacked=True, ax=ax1, width=1, colormap="tab20")
    ax1.plot(monthly_totals, color="white", linewidth=2, label="Total Net Change")
    ax1.set_title("Breakdown of US JOLTS Job Openings by Sector (Excl. Healthcare) - 1M Net Change")
    ax1.set_ylabel("Thousands")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.7)
    ax1.set_xticks(range(0, len(other_data.index), 3))
    ax1.set_xticklabels(other_data.index.strftime("%b %Y")[::3], rotation=45)

    # Move legend below the first plot
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize="small")

    # Plot for Healthcare
    ax2 = ax[1]
    healthcare_data.plot(kind="bar", stacked=True, ax=ax2, width=1, color="tab:red")
    ax2.set_title("Breakdown of US JOLTS Job Openings for Healthcare - 1M Net Change")
    ax2.set_ylabel("Thousands")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.7)
    ax2.set_xticks(range(0, len(healthcare_data.index), 3))
    ax2.set_xticklabels(healthcare_data.index.strftime("%b %Y")[::3], rotation=45)

    # Heatmap for other sectors
    ax3 = ax[2]
    sns.heatmap(
        other_data.T, annot=False, fmt=".0f", cmap="RdYlGn", linewidths=0.5, ax=ax3
    )
    ax3.set_title("Net Changes Heatmap (Excl. Healthcare)")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Sectors")
    ax3.set_xticks(range(0, len(other_data.index), 3))
    ax3.set_xticklabels(other_data.index.strftime("%b %Y")[::3], rotation=45)

    # Adjust layout
    plt.tight_layout()
    plt.show()

def create_table_plot(data):
    if data.empty:
        print("No data available for plotting")
        return

    # Drop Healthcare column
    data = data.drop(columns=["Health Care and Social Assistance"])

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(20, 8))

    # Normalize values for color mapping
    norm = colors.Normalize(vmin=data.min().min(), vmax=data.max().max())
    cmap = plt.cm.RdYlGn

    # Plot cell colors
    n_rows, n_cols = data.shape
    for row_index, row_label in enumerate(data.index):
        for col_index, col_label in enumerate(data.columns):
            value = data.iloc[row_index, col_index]
            color = cmap(norm(value))
            ax.add_patch(
                plt.Rectangle((col_index, n_rows - row_index - 1), 1, 1, color=color)
            )
            # Determine text color for better contrast
            text_color = "white" if norm(value) > 0.5 else "black"
            ax.text(
                col_index + 0.5,
                n_rows - row_index - 0.5,
                f"{value:,.0f}",
                ha="center",
                va="center",
                fontsize=9,
                color=text_color,
            )

    # Set up the axes
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(data.columns, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(data.index, fontsize=10)
    ax.tick_params(left=False, bottom=False)

    # Remove grid and frame
    ax.grid(False)
    ax.axis("off")

    # Title
    plt.title("JOLTS Month-by-Month Added/Subtracted Values", fontsize=14, pad=20)

    # Show the plot
    plt.tight_layout()
    plt.show()

async def main():

    # Getting DataConnector to get access to connectors (FRED, EOD)
    dc = DataConnector()
    fred = dc.fred
    eod = dc.eod

    jolts_data = await fetch_jolts_data(fred)
    market_data = await fetch_market_data(eod)

    if not market_data.empty and not jolts_data.empty:
        create_main_plots(jolts_data)
        create_table_plot(jolts_data)
        create_correlation_analysis(jolts_data, market_data)
        sector_rs = create_sector_relative_strength(jolts_data, market_data)
        print(sector_rs)
    else:
        print("Unable to create visualizations due to missing data")

if __name__ == "__main__":
    asyncio.run(main())
