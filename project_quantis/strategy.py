import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf

class SignalGenerator:
    def __init__(self, client, ema_weight=0.4, rsi_weight=0.4, macd_weight=0.4, bb_weight=0.8, atr_weight=0.2,
                 signal_threshold=0.5):
        self.client = client
        self.ema_weight = ema_weight
        self.rsi_weight = rsi_weight
        self.macd_weight = macd_weight
        self.atr_weight = atr_weight
        self.bb_weight = bb_weight
        self.signal_threshold = signal_threshold

    def fetch_candles(self, historical=False):
        if historical:
            # Fetching larger historical data for backtesting
            candles = self.client.fetch_latest_candles(limit=1000)
        else:
            candles = self.client.fetch_latest_candles()

        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms') + pd.Timedelta(hours=3)  # Adjusting to GMT+3
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        return df

    def ema_strategy(self, df, short=8, medium=21, long=55):
        # Calculate EMAs
        df['ema_short'] = df['close'].ewm(span=short, adjust=False).mean()
        df['ema_medium'] = df['close'].ewm(span=medium, adjust=False).mean()
        df['ema_long'] = df['close'].ewm(span=long, adjust=False).mean()

        # Calculate slopes
        df['ema_short_slope'] = df['ema_short'].diff(1)
        df['ema_medium_slope'] = df['ema_medium'].diff(1)

        # Vectorized conditions
        curr_buy_cond = (df['ema_short'] > df['ema_medium']) & \
                        (df['ema_medium'] > df['ema_long']) & \
                        (df['ema_short_slope'] > 0) & \
                        (df['close'] > df['ema_short'])

        curr_sell_cond = (df['ema_short'] < df['ema_medium']) & \
                         (df['ema_medium'] < df['ema_long']) & \
                         (df['ema_short_slope'] < 0) & \
                         (df['close'] < df['ema_short'])

        # Generate signals
        df['ema_signal'] = 0
        df.loc[curr_buy_cond, 'ema_signal'] = 1
        df.loc[curr_sell_cond, 'ema_signal'] = -1

        return df

    def rsi_strategy(self, df, period=14, divergence_lookback=5):
        # Calculate RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # Dynamic Thresholds
        def calculate_dynamic_levels(rsi):
            volatility = rsi.rolling(14).std()
            ob_scale = np.clip(volatility * 1.5, 5, 10)
            os_scale = np.clip(volatility * 1.5, 5, 10)
            return 70 - ob_scale, 30 + os_scale

        df['ob_level'], df['os_level'] = calculate_dynamic_levels(df['rsi'])

        # Reintroduce Divergence Detection
        def detect_divergence(price, rsi, lookback):
            # Detect local price peaks/troughs without shifting
            peaks = price.rolling(lookback).max()
            troughs = price.rolling(lookback).min()

            # Simplified divergence conditions
            bull_div = (price <= troughs) & (rsi > rsi.rolling(lookback).min())
            bear_div = (price >= peaks) & (rsi < rsi.rolling(lookback).max())

            return np.select([bull_div, bear_div], [1, -1], default=0)

        df['divergence'] = detect_divergence(df['close'], df['rsi'], divergence_lookback)

        # Faster trend filter
        df['trend_ema'] = df['close'].ewm(span=20).mean()

        # Vectorized trend validation
        df['above_trend'] = df['close'] > df['trend_ema']
        df['below_trend'] = df['close'] < df['trend_ema']
        df['valid_trend'] = np.select(
            [
                df['above_trend'].rolling(2).apply(lambda x: x.all(), raw=True).fillna(False).astype(bool),
                df['below_trend'].rolling(2).apply(lambda x: x.all(), raw=True).fillna(False).astype(bool)
            ],
            [1, -1],
            default=0
        )

        # Initialize signal score
        df['rsi_score'] = 0

        # Add weights for RSI, Divergence, and Trend
        df['rsi_score'] += np.where(df['rsi'] < df['os_level'], 1, 0)  # RSI oversold adds weight
        df['rsi_score'] += np.where(df['divergence'] == 1, 1, 0)  # Bullish divergence adds weight
        df['rsi_score'] += np.where(df['valid_trend'] == 1, 1, 0)  # Bullish trend adds weight

        df['rsi_score'] -= np.where(df['rsi'] > df['ob_level'], 1, 0)  # RSI overbought subtracts weight
        df['rsi_score'] -= np.where(df['divergence'] == -1, 1, 0)  # Bearish divergence subtracts weight
        df['rsi_score'] -= np.where(df['valid_trend'] == -1, 1, 0)  # Bearish trend subtracts weight

        # Generate final signals based on combined score
        df['rsi_signal'] = df['rsi_score'].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)

        return df

    def macd_strategy(self, df, short_window=12, long_window=26, signal_window=9, cw=0.3, hw=0.7, threshold=0.5):
        # Calculate core MACD components
        df['macd_ema_short'] = df['close'].ewm(span=short_window, adjust=False).mean()
        df['macd_ema_long'] = df['close'].ewm(span=long_window, adjust=False).mean()
        df['macd'] = df['macd_ema_short'] - df['macd_ema_long']
        df['macd_sgn_line'] = df['macd'].ewm(span=signal_window, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_sgn_line']

        # Generate crossover signal
        df['crossover_signal'] = 0
        crossover_buy = (df['macd'] > df['macd_sgn_line']) & (df['macd'].shift(1) <= df['macd_sgn_line'].shift(1))
        crossover_sell = (df['macd'] < df['macd_sgn_line']) & (df['macd'].shift(1) >= df['macd_sgn_line'].shift(1))
        df.loc[crossover_buy, 'crossover_signal'] = 1
        df.loc[crossover_sell, 'crossover_signal'] = -1

        # Calculate histogram components
        df['hist_diff'] = df['macd_hist'] - df['macd_hist'].shift(1)
        df['hist_slope'] = df['macd_hist'].diff().rolling(3).mean()
        hist_buy = (df['hist_diff'] > 0) & (df['macd_hist'] > 0)
        hist_sell = (df['hist_diff'] < 0) & (df['macd_hist'] < 0)

        # Strengthen signals with steeper slopes
        hist_buy = hist_buy & (df['hist_slope'] > df['hist_slope'].rolling(5).mean())
        hist_sell = hist_sell & (df['hist_slope'] < df['hist_slope'].rolling(5).mean())

        # Generate histogram reversal signal
        df['hist_signal'] = 0
        df.loc[hist_buy, 'hist_signal'] = 1
        df.loc[hist_sell, 'hist_signal'] = -1

        # Combine signals with weights
        df['combined_macd'] = (df['crossover_signal'] * cw) + (df['hist_signal'] * hw)
        df['macd_signal'] = df['combined_macd'].apply(lambda x: 1 if x >= threshold else -1 if x <= -threshold else 0)

        return df

    def bollinger_bands_strategy(self, df, window=20, num_std=2):
        # Calculate Bollinger Bands
        df['middle_band'] = df['close'].rolling(window=window).mean()
        rolling_std = df['close'].rolling(window=window).std()
        df['upper_band'] = df['middle_band'] + (rolling_std * num_std)
        df['lower_band'] = df['middle_band'] - (rolling_std * num_std)

        # Calculate Bollinger Band Width (BBW)
        df['band_width'] = (df['upper_band'] - df['lower_band']) / df['middle_band']

        # Generate mean reversion signals
        df['bb_signal'] = 0

        # Buy when price touches or breaks the lower band
        buy_condition = (df['close'] < df['lower_band']) & (df['close'].shift(1) >= df['lower_band'])
        df.loc[buy_condition, 'bb_signal'] = 1

        # Sell when price touches or breaks the upper band
        sell_condition = (df['close'] > df['upper_band']) & (df['close'].shift(1) <= df['upper_band'])
        df.loc[sell_condition, 'bb_signal'] = -1

        return df

    def atr_strategy(self, df, period=14):
        df['high_low'] = df['high'] - df['low']
        df['high_close'] = np.abs(df['high'] - df['close'].shift(1))
        df['low_close'] = np.abs(df['low'] - df['close'].shift(1))
        df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        df['atr'] = df['true_range'].rolling(window=period).mean()

        # Generate ATR-based signals (example: volatility filter)
        df['atr_signal'] = 0
        df.loc[df['atr'] > df['atr'].rolling(50).mean(), 'atr_signal'] = 1  # High volatility: valid signals
        df.loc[df['atr'] < df['atr'].rolling(50).mean(), 'atr_signal'] = -1  # Low volatility: ignore signals

        return df

    def generate_signal(self, historical=False):
        df = self.fetch_candles(historical=historical)
        df = self.ema_strategy(df)
        df = self.rsi_strategy(df)
        df = self.macd_strategy(df)
        df = self.bollinger_bands_strategy(df)
        df = self.atr_strategy(df)

        def print_signal_counts(df, signal_column='final_signal'):
            buys = (df[signal_column] == 1).sum()
            sells = (df[signal_column] == -1).sum()
            print(f"{signal_column}: Buy signals: {buys}, Sell signals: {sells}")

        df['combined_signal'] = (
                df['ema_signal'] * self.ema_weight +
                df['rsi_signal'] * self.rsi_weight +
                df['macd_signal'] * self.macd_weight +
                df['bb_signal'] * self.bb_weight +
                df['atr_signal'] * self.atr_weight)

        df['final_signal'] = df['combined_signal'].apply(
            lambda x: 1 if x > self.signal_threshold else (-1 if x < -self.signal_threshold else 0))

        print_signal_counts(df, signal_column='ema_signal')
        print_signal_counts(df, signal_column='rsi_signal')
        print_signal_counts(df, signal_column='macd_signal')
        print_signal_counts(df, signal_column='bb_signal')
        print_signal_counts(df, signal_column='atr_signal')
        print_signal_counts(df, signal_column='final_signal')

        if historical:
            df.to_csv('../data/backtest_signals.csv')  # Save backtest signals for analysis
        else:
            df.to_csv('../data/signals.csv')

        return df

class Backtester:
    def __init__(self, signal_generator, initial_balance=1000, fee_rate=0.0005,
                 mode=1, take_profit=0.02, stop_loss=0.01,
                 dynamic_levels=False, atr_sl_mult=1.5, atr_tp_mult=3):
        """
        Initialize the Backtester class.

        :param signal_generator: Instance of the SignalGenerator class.
        :param initial_balance: Starting balance for the backtest.
        :param stop_loss: Stop loss level as a percentage.
        :param take_profit: Take profit level as a percentage.
        :param fee_rate: Trading fee rate (e.g., 0.0005 for Binance).
        :param mode: Trading mode (1: Close on TP/SL, 2: Close on opposite signal).
        :param dynamic_levels: Use ATR-based TP/SL for mode=1
        """
        self.signal_generator = signal_generator
        self.initial_balance = initial_balance
        self.balance = self.initial_balance
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.fee_rate = fee_rate
        self.mode = mode
        self.entry_price = 0
        self.position_size = 0
        self.position_type = None
        self.trades = []
        self.mode = mode
        self.dynamic_levels = dynamic_levels
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult

    def execute_trade(self, action, price, timestamp):
        """
        Execute a trade and log the details.

        :param action: 'BUY', 'SELL', or 'CLOSE'
        :param price: Execution price
        :param timestamp: Execution time
        """
        fee = self.position_size * price * self.fee_rate
        profit = 0

        if action == 'BUY':
            fee = self.balance * self.fee_rate
            self.position_size = (self.balance - fee) / price
            self.balance -= (self.position_size * price + fee)
            self.entry_price = price
            self.position_type = 'LONG'
        elif action == 'SELL':
            fee = self.balance * self.fee_rate
            self.position_size = -(self.balance - fee) / price
            self.balance -= (self.position_size * price + fee)
            self.entry_price = price
            self.position_type = 'SHORT'
        elif action == 'CLOSE':
            if self.position_type == 'LONG':
                fee = self.position_size * price * self.fee_rate
                self.balance += (self.position_size * price) - fee
                profit = (price - self.entry_price) * self.position_size - fee
            elif self.position_type == 'SHORT':
                fee = -self.position_size * price * self.fee_rate
                self.balance += (self.position_size * price) - fee
                profit = (self.entry_price - price) * -self.position_size - fee
            self.position_size = 0
            self.position_type = None

        net_equity = self.balance + (self.position_size * price)  # if self.position_type == 'LONG' else 0)

        self.trades.append({
            'timestamp': timestamp,
            'type': action,
            'price': price,
            'position': round(self.position_size, 4),
            'fee': round(fee, 2),
            'balance': round(self.balance, 2),
            'net_equity': round(net_equity, 2),
            'profit': round(profit, 2)
        })

    def run_backtest(self):
        """
        Run the backtest simulation.
        """
        df = self.signal_generator.generate_signal(historical=True)

        # Initial balance log
        self.trades.append({
            'timestamp': df['timestamp'].iloc[0],
            'type': 'INITIAL_BALANCE',
            'price': df['close'].iloc[0],
            'position': 0,
            'fee': 0,
            'balance': self.initial_balance,
            'net_equity': self.initial_balance,
            'profit': 0
        })

        for i, row in df.iterrows():
            signal = row['final_signal']
            price = row['close']
            timestamp = row['timestamp']

            if self.position_type is None:
                if signal == 1:
                    self.execute_trade('BUY', price, timestamp)
                elif signal == -1:
                    self.execute_trade('SELL', price, timestamp)
            else:
                if self.mode == 1:  # TP/SL exit mode
                    # Calculate SL/TP levels
                    if self.dynamic_levels:
                        current_atr = row['atr']
                        sl = current_atr * self.atr_sl_mult
                        tp = current_atr * self.atr_tp_mult
                    else:
                        sl = self.entry_price * self.stop_loss
                        tp = self.entry_price * self.take_profit

                    # Determine exit prices
                    if self.position_type == 'LONG':
                        sl_price = self.entry_price - sl if self.dynamic_levels else self.entry_price * (
                                    1 - self.stop_loss)
                        tp_price = self.entry_price + tp if self.dynamic_levels else self.entry_price * (
                                    1 + self.take_profit)
                        exit_cond = price <= sl_price or price >= tp_price
                    else:  # SHORT
                        sl_price = self.entry_price + sl if self.dynamic_levels else self.entry_price * (
                                    1 + self.stop_loss)
                        tp_price = self.entry_price - tp if self.dynamic_levels else self.entry_price * (
                                    1 - self.take_profit)
                        exit_cond = price >= sl_price or price <= tp_price

                    if exit_cond:
                        self.execute_trade('CLOSE', price, timestamp)


                elif self.mode == 2:  # Close on opposite signal
                    if (signal == -1 and self.position_type == 'LONG') or (
                            signal == 1 and self.position_type == 'SHORT'):
                        self.execute_trade('CLOSE', price, timestamp)
                        if signal == 1:
                            self.execute_trade('BUY', price, timestamp)
                        elif signal == -1:
                            self.execute_trade('SELL', price, timestamp)

        # Close any remaining position at the end of backtest
        if self.position_type is not None:
            self.execute_trade('CLOSE', df['close'].iloc[-1], df['timestamp'].iloc[-1])

        # Save trade log to CSV
        trades_df = pd.DataFrame(self.trades)
        trades_df.to_csv('../data/trades.csv')

        return trades_df

    def calculate_performance_metrics(self):
        """
        Calculate performance metrics from the trade log.
        """
        trades_df = pd.DataFrame(self.trades)
        closed_trades = trades_df[trades_df['type'] == 'CLOSE']

        if closed_trades.empty:
            return {
                'final_balance': self.initial_balance,
                'total_trades': 0,
                'win_rate': 0,
                'risk_reward_ratio': 0,
                'profit_factor': 0,
                'profit_per_trade': 0,
                'expectancy': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'kelly_criterion': 0,
            }

        profits = closed_trades['profit']
        wins = profits[profits > 0]
        losses = profits[profits <= 0]

        total_trades = len(closed_trades)
        win_rate = len(wins) / total_trades
        average_win = wins.mean() if not wins.empty else 0
        average_loss = losses.mean() if not losses.empty else 0

        risk_reward_ratio = average_win / abs(average_loss) if average_loss != 0 else 0
        profit_factor = wins.sum() / abs(losses.sum()) if not losses.empty else float('inf')
        profit_per_trade = profits.mean()
        expectancy = (average_win * win_rate) - (average_loss * (1 - win_rate))

        trades_df['net_equity'] = trades_df['net_equity'].cummax()
        trades_df['drawdown'] = trades_df['net_equity'] - trades_df['net_equity'].cummax()
        max_drawdown = trades_df['drawdown'].min()

        sharpe_ratio = profit_per_trade / profits.std() if profits.std() != 0 else 0
        kelly_criterion = win_rate - ((1 - win_rate) / risk_reward_ratio) if risk_reward_ratio != 0 else 0

        final_balance = trades_df['balance'].iloc[-1]

        metrics = {
            'final_balance': round(final_balance, 2),
            'total_trades': total_trades,
            'win_rate': round(win_rate * 100, 2),
            'risk_reward_ratio': round(risk_reward_ratio, 2),
            'profit_factor': round(profit_factor, 2),
            'profit_per_trade': round(profit_per_trade, 2),
            'expectancy': round(expectancy, 2),
            'max_drawdown': round(max_drawdown, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'kelly_criterion': round(kelly_criterion * 100, 2)
        }

        return metrics

    def plot_trades(self):
        """
        Plot candlestick chart with buy and sell signals.
        """
        df = self.signal_generator.fetch_candles(historical=True)
        trades_df = pd.DataFrame(self.trades)

        # Convert timestamps to datetime and align with OHLC data
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])

        # Create DatetimeIndex for OHLC data
        df.set_index('timestamp', inplace=True)
        ohlc = df[['open', 'high', 'low', 'close', 'volume']]

        # Align trade signals with OHLC index
        buy_signals = trades_df[trades_df['type'] == 'BUY'].set_index('timestamp')
        sell_signals = trades_df[trades_df['type'] == 'SELL'].set_index('timestamp')

        # Create price series aligned with OHLC index
        buy_prices = pd.Series(index=ohlc.index, dtype='float64')
        sell_prices = pd.Series(index=ohlc.index, dtype='float64')

        # Populate signals where timestamps match exactly
        buy_prices[buy_prices.index.intersection(buy_signals.index)] = buy_signals['price']
        sell_prices[sell_prices.index.intersection(sell_signals.index)] = sell_signals['price']

        # Create addplots with aligned data
        apds = [
            mpf.make_addplot(buy_prices, type='scatter', markersize=100, marker='^', color='g'),
            mpf.make_addplot(sell_prices, type='scatter', markersize=100, marker='v', color='r')
        ]

        # Plot with formatted dates
        mpf.plot(ohlc,
                 type='candle',
                 style='charles',
                 addplot=apds,
                 volume=True,
                 title='Backtest Candlestick Chart with Signals',
                 ylabel='Price',
                 datetime_format='%Y-%m-%d %H:%M',
                 warn_too_much_data=2000)