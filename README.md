Algorithmic Trading Bot

This project is an algorithmic trading bot designed to interact with Binance Futures, using multiple technical indicators for signal generation, backtesting, and live trading execution.

Features

EMA, RSI, MACD, ATR, and Bollinger Bands for signal generation

Backtesting framework with performance metrics (Sharpe ratio, Max Drawdown, Profit Factor, etc.)

Dynamic stop-loss and take-profit levels

Live trading on Binance Futures

Installation

Clone the repository:

git clone https://github.com/yourusername/algorithmic-trading-bot.git
cd algorithmic-trading-bot

Install required dependencies:

pip install ccxt numpy pandas matplotlib mplfinance

Configuration

Update main.py to customize:

Trading pair

Timeframe

Risk parameters (TP/SL levels, ATR multipliers)

Usage

Backtesting

python main.py

Live Trading

Uncomment the relevant sections in main.py to execute live trades.

Files Overview

binance_client.py: Manages Binance Futures API interactions

strategy.py: Contains signal generation logic and backtesting

trade_manager.py: Handles trade execution and position management

main.py: The main entry point for running the bot

Example

# Run the bot
if __name__ == "__main__":
    main()

Performance Metrics

Final Balance

Win Rate

Risk/Reward Ratio

Profit Factor

Maximum Drawdown

Sharpe Ratio

Kelly Criterion

Disclaimer

This trading bot is for educational purposes only. Trading cryptocurrencies is risky and can lead to significant financial loss. Use it at your own risk.

