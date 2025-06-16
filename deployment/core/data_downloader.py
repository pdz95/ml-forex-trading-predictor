# data_downloader.py

# Core imports - lightweight
import pandas as pd
import time
import random
import logging

# LOGGERA conf
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        #logging.FileHandler('data_download.log')  # File output (opcjonalne)
    ]
)
# Setup logger
logger = logging.getLogger(__name__)


class DataDownloader:
    """
    Downloads and preprocesses financial data from Yahoo Finance.

    Handles EUR/USD and multi-asset data retrieval with error logging
    and data validation for the trading prediction pipeline.

    Attributes:
        symbol (str): Financial instrument symbol (e.g., 'EURUSD=X')
        logs (list): Collection of download operation logs

    Example:
        downloader = DataDownloader("EURUSD=X")
        data = downloader.download_data(period="1y")
    """
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self.logs = []

    def _log(self, level, message):
        """Custom logging method"""
        log_entry = f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {level} - {message}"
        self.logs.append(log_entry)
        logger.info(message)  # Nadal loguj normalnie

    def get_logs(self):
        """Zwróć zebrane logi"""
        return '\n'.join(self.logs)

    def download_data(self, interval='1d', period='20y') -> pd.DataFrame:
        """Download stock data from Yahoo Finance"""
        try:
            # Lazy import - only when downloading
            import yfinance as yf

            # Add delay to avoid rate limiting
            time.sleep(random.uniform(1, 2))

            commodity = yf.Ticker(self.symbol)
            df = commodity.history(period=period, interval=interval)

            if df.empty:
                self._log("WARNING", f"No data found for symbol: {self.symbol}")
                return pd.DataFrame()

            df = df.reset_index()
            df['date'] = pd.to_datetime(df['Date']).dt.strftime('%d-%m-%Y')
            df['date'] = pd.to_datetime(df['date'], dayfirst=True)
            df = df.drop('Date', axis=1)

            self._log("INFO", f"Downloaded {len(df)} rows for {self.symbol}")
            return df

        except Exception as e:
            self._log("ERROR", f"Error downloading data for {self.symbol}: {e}")
            return pd.DataFrame()