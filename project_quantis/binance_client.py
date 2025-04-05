import ccxt
import config
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Starting Binance Futures Client...")

class BinanceFuturesClient:
    def __init__(self, symbol, timeframe='1m', candlestick_limit=500):
        self.symbol = symbol
        self.timeframe = timeframe
        self.candlestick_limit = candlestick_limit

        self.exchange = ccxt.binance({
            'apiKey': config.BINANCE_API_KEY,
            'secret': config.BINANCE_API_SECRET,
            'options': {
                'defaultType': 'future',  # Ensures we are using Binance Futures
            },
            'enableRateLimit': True,
        })
        logging.info("Binance Futures client initialized.")

    def safe_api_call(self, func, *args, **kwargs):
        """Handles API calls safely with error handling and retries."""
        for attempt in range(3):
            try:
                return func(*args, **kwargs)
            except ccxt.NetworkError as e:
                logging.warning(f"Network error: {e}. Retrying {attempt + 1}/3...")
                time.sleep(2)
            except ccxt.ExchangeError as e:
                logging.error(f"Exchange error: {e}")
                break
            except Exception as e:
                logging.error(f"Unexpected error: {e}")
                break
        return None

    def fetch_market_data(self):
        """Fetch latest market ticker data"""
        return self.safe_api_call(self.exchange.fetch_ticker, self.symbol)

    def fetch_balance(self):
        """Fetch account balance on Binance Futures"""
        return self.safe_api_call(self.exchange.fetch_balance)

    def fetch_open_positions(self):
        """Fetch open positions on Binance Futures"""
        return self.safe_api_call(self.exchange.fetch_positions)

    def fetch_historical_trades(self, limit=100):
        """Fetch recent trade history"""
        return self.safe_api_call(self.exchange.fetch_my_trades, self.symbol, limit=limit)

    def fetch_latest_candles(self, limit = None):
        """Fetch latest candlestick data for the given symbol"""
        if limit is None:
            limit = self.candlestick_limit
        return self.safe_api_call(self.exchange.fetch_ohlcv, self.symbol, self.timeframe, limit=limit)

    def set_leverage(self, leverage=20):
        """Set leverage for the given symbol"""
        params = {'symbol': self.symbol.replace("/", ""), 'leverage': leverage}
        return self.safe_api_call(self.exchange.fapiPrivatePostLeverage, params)

    def fetch_funding_rate(self):
        """Fetch the latest funding rate for the given symbol"""
        return self.safe_api_call(self.exchange.fetch_funding_rate, self.symbol)

    def fetch_order_book(self, limit=50):
        """Fetch the order book for the given symbol"""
        return self.safe_api_call(self.exchange.fetch_order_book, self.symbol, limit)

    def fetch_open_orders(self):
        """Fetch open orders for the given symbol"""
        return self.safe_api_call(self.exchange.fetch_open_orders, self.symbol)

    def fetch_order_status(self, order_id):
        """Fetch details of a specific order"""
        return self.safe_api_call(self.exchange.fetch_order, order_id, self.symbol)

    def place_order(self, side, type, amount = None, price = None, close_position = False):
        """Place an order for given symbol.
        :param side: 'buy' or 'sell'
        :param type: 'limit', 'market'
        :param price: order price, None for market orders
        :param amount: Position size in contract quantity
        """
        order = self.exchange.create_order(
            symbol=self.symbol,
            type = type,
            side = side,
            amount = amount,
            price = price,
            params = {'closePosition': close_position}
        )
        return order

    def place_stop_order(self, side, type, amount = None, price = None,
                         close_position = True, stop_price = None, time_in_force = 'GTE_GTC', working_type = 'mark_price'):
        """Place an order for given symbol.
        :param side: 'buy' or 'sell'
        :param type: 'stop_market' or 'take_profit_market'
        :param price: order price, None for market orders
        :param amount: Position size in contract quantity
        :param close_position: True for tp/sl
        :param stop_price: Stop market price
        :param time_in_force: 'GTE_GTC' for tp/sl
        """
        order = self.exchange.create_order(
            symbol=self.symbol,
            type = type,
            side = side,
            amount = amount,
            price = price,
            params = {'closePosition': close_position,
                      'stopPrice': stop_price,
                      'timeInForce': time_in_force,
                      'workingType': working_type}
        )
        return order

    def cancel_order(self, order_id):
        """Cancel an existing order"""
        return self.safe_api_call(self.exchange.cancel_order, order_id, self.symbol)