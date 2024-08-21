# Backtester Directory and Name Convention

## Introduction
As we develop a suite of backtesting tools, it’s essential to maintain a clear and organized structure. This document outlines the different types of backtesters we are building, ranging from beginner-friendly versions to advanced, specialized systems. Each backtester serves a distinct purpose, allowing users to progressively enhance their skills and tackle more complex scenarios.

### We have following structure of directories:
```python
backtesting/
├── strategies/
│   └── (All strategy files using the naming convention)
├── engines/
├── hypothesis/
├── learning_backtester.py
├── backtester.py
└── advanced_backtester.py
```

## LearningBacktester: (for learning, begineers)

The LearningBacktester is designed as an entry point for those new to backtesting. It focuses on educational clarity, with simplified data processing and performance calculations. The key features include:

* The most basic version, focused on educational purposes
* Simplified data handling and performance calculations
* Basic signal generation and position sizing
* Minimal configuration options
* Clear, well-commented code for easy understanding

## Backtester: (standard version)
The standard Backtester is a more comprehensive tool designed for general-purpose backtesting. It is suitable for users who have mastered the basics and are ready to explore more robust features:

* A more complete, general-purpose backtesting system
* Robust data handling for multiple assets
* Support for various types of signals and position sizing
* Basic performance metrics and reporting
* Configurable parameters for different strategies

## AdvancedBacktester:
The AdvancedBacktester builds upon the standard Backtester by incorporating sophisticated features aimed at more experienced users:

Multiple simultaneous strategy testing
Portfolio Managmeent
Support for optimization and parameter tuning

## IntradayBacktester:
The IntradayBacktester is a specialized tool designed specifically for intraday trading strategies, offering high-frequency data processing and real-time simulation capabilities:

Connects with real-time data providers to simulate live trading environments.
Specialized for intraday trading strategies
 Specific tools for managing risks inherent to intraday trading.
Tick-by-tick simulation capabilities
Performance metrics relevant to intraday trading

## Additional Considerations

This structure allows for a natural progression in complexity and specialization. Begineers can start with the LearningBacktester to understand the basics, then move on to the standard Backtester for more realistic simulations. The AdvancedBacktester and IntradayBacktester can be used for more specific or complex needs.