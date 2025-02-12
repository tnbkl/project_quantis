import ccxt
import config
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Starting Binance Futures Client...")

class BinanceFuturesClient:
    def __init__(self, symbol, timeframe='1m', take_profit=0.02, stop_loss=0.01, balance_allocation=0.1, candlestick_limit=10):
        self.symbol = symbol
        self.timeframe = timeframe
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.balance_allocation = balance_allocation
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
        #self.ensure_margin_mode()

    def ensure_margin_mode(self, margin_mode="ISOLATED"):
        """Ensure margin mode is set to ISOLATED only if it's not already set"""
        try:
            balance = self.safe_api_call(self.exchange.fetch_balance)
            if not balance:
                logging.error("Failed to fetch account balance.")
                return

            positions = self.safe_api_call(self.exchange.fetch_positions)
            if not positions:
                logging.error("Failed to fetch account positions.")
                return

            for position in positions:
                if position['symbol'] == self.symbol.replace("/", "") and position.get(
                        'marginType') != margin_mode.upper():
                    params = {"symbol": self.symbol.replace("/", ""), "marginType": margin_mode.upper()}
                    response = self.safe_api_call(self.exchange.fapiPrivatePostMarginType, params)
                    if response:
                        logging.info(f"Margin mode set to {margin_mode.upper()} for {self.symbol}.")
                    else:
                        logging.warning(f"Failed to set margin mode for {self.symbol}.")
                    return
            logging.info(f"Margin mode for {self.symbol} is already set to {margin_mode.upper()}.")
        except Exception as e:
            logging.error(f"Error ensuring margin mode: {e}")

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

    def fetch_balance(self):
        """Fetch account balance on Binance Futures"""
        return self.safe_api_call(self.exchange.fetch_balance)

    def fetch_market_data(self):
        """Fetch latest market ticker data"""
        return self.safe_api_call(self.exchange.fetch_ticker, self.symbol)

    def fetch_open_positions(self):
        """Fetch open positions on Binance Futures"""
        return self.safe_api_call(self.exchange.fetch_positions)

    def fetch_latest_candles(self):
        """Fetch latest candlestick data for the given symbol"""
        return self.safe_api_call(self.exchange.fetch_ohlcv, self.symbol, self.timeframe, limit=self.candlestick_limit)

    def place_order(self, order_type, side, amount, price=None):
        """Place an order on Binance Futures"""
        params = {'type': order_type}
        if price:
            params['price'] = price
        return self.safe_api_call(self.exchange.create_order, self.symbol, order_type, side, amount, price, params)

    def fetch_order_book(self, limit=50):
        """Fetch the order book for the given symbol"""
        return self.safe_api_call(self.exchange.fetch_order_book, self.symbol, limit)

    def fetch_open_orders(self):
        """Fetch open orders for the given symbol"""
        return self.safe_api_call(self.exchange.fetch_open_orders, self.symbol)

    def fetch_order_status(self, order_id):
        """Fetch details of a specific order"""
        return self.safe_api_call(self.exchange.fetch_order, order_id, self.symbol)

    def cancel_order(self, order_id):
        """Cancel an existing order"""
        return self.safe_api_call(self.exchange.cancel_order, order_id, self.symbol)

    def set_leverage(self, leverage=20):
        """Set leverage for the given symbol"""
        params = {'symbol': self.symbol.replace("/", ""), 'leverage': leverage}
        return self.safe_api_call(self.exchange.fapiPrivatePostLeverage, params)

    def fetch_funding_rate(self):
        """Fetch the latest funding rate for the given symbol"""
        return self.safe_api_call(self.exchange.fetch_funding_rate, self.symbol)

    def fetch_historical_trades(self, limit=50):
        """Fetch recent trade history"""
        return self.safe_api_call(self.exchange.fetch_my_trades, self.symbol, limit=limit)