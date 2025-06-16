# multi_asset_loader.py

# Core imports - lightweight
import pandas as pd
import logging

# Setup logger
logger = logging.getLogger(__name__)


class MultiAssetDataLoader:
    """
    Loads and processes data from multiple asset classes for analysis.

    Combines data from 13 global markets (stocks, bonds, commodities)
    to provide multi-asset context for EUR/USD predictions.

    Args:
        threshold_percentile (int): Percentile threshold for significant moves
    """
    def __init__(self, threshold_percentile=67):
        self.threshold_percentile = threshold_percentile
        self.asset_configs = {
            'gold': {
                'symbol': 'GC=F',
                'features': ['date', 'MA_20', 'volatility_20', 'MACD', 'RSI'],
                'rename_suffix': '_gold'
            },
            'sp500': {
                'symbol': '^GSPC',
                'features': ['date', 'MA_20', 'volatility_10', 'MACD', 'returns_5'],
                'rename_suffix': '_sp500'
            },
            'dollar': {
                'symbol': 'DX-Y.NYB',
                'features': ['date', 'MA_10', 'volatility_5', 'MACD', 'returns_3'],
                'rename_suffix': '_dollar'
            },
            'vix': {
                'symbol': '^VIX',
                'features': ['date', 'MA_5', 'volatility_5', 'returns_1'],
                'rename_suffix': '_vix'
            },
            'tnx': {
                'symbol': '^TNX',
                'features': ['date', 'MA_20', 'returns_5', 'volatility_10'],
                'rename_suffix': '_tnx'
            },
            'chf': {
                'symbol': 'USDCHF=X',
                'features': ['date', 'MA_20', 'RSI', 'volatility_10', 'returns_3'],
                'rename_suffix': '_chf'
            },
            'oil': {
                'symbol': 'CL=F',
                'features': ['date', 'MA_20', 'volatility_10', 'momentum_10', 'RSI'],
                'rename_suffix': '_oil'
            },
            'gbp': {
                'symbol': 'GBPUSD=X',
                'features': ['date', 'MA_20', 'volatility_10', 'RSI', 'returns_5'],
                'rename_suffix': '_gbp'
            },
            'jpy': {
                'symbol': 'USDJPY=X',
                'features': ['date', 'MA_20', 'volatility_5', 'MACD', 'momentum_10'],
                'rename_suffix': '_jpy'
            },
            'eurgbp': {
                'symbol': 'EURGBP=X',
                'features': ['date', 'MA_10', 'RSI', 'volatility_10', 'returns_3'],
                'rename_suffix': '_eurgbp'
            },
            'eurjpy': {
                'symbol': 'EURJPY=X',
                'features': ['date', 'MA_20', 'volatility_10', 'momentum_5', 'RSI'],
                'rename_suffix': '_eurjpy'
            },
            'aud': {
                'symbol': 'AUDUSD=X',
                'features': ['date', 'MA_10', 'volatility_5', 'RSI', 'returns_3'],
                'rename_suffix': '_aud'
            },
            'cad': {
                'symbol': 'USDCAD=X',
                'features': ['date', 'MA_20', 'volatility_10', 'returns_5'],
                'rename_suffix': '_cad'
            }
        }

    def load_single_asset(self, asset_name: str) -> pd.DataFrame:
        """Load and process single asset data"""
        if asset_name not in self.asset_configs:
            available_assets = list(self.asset_configs.keys())
            raise ValueError(f"Asset '{asset_name}' not configured. Available assets: {available_assets}")

        # Lazy imports - only when loading data
        from .data_downloader import DataDownloader
        from .feature_engineer import FeatureEngineer

        config = self.asset_configs[asset_name]
        logger.info(f"Loading asset: {asset_name} ({config['symbol']})")

        try:
            # Download data
            data = DataDownloader(config['symbol']).download_data()

            if data.empty:
                logger.warning(f"No data received for {asset_name}")
                return pd.DataFrame()

            # Feature engineering
            features, _ = FeatureEngineer(
                data,
                threshold_percentile=self.threshold_percentile,
                apply_scaling=False  # No scaling in multi-asset loader
            ).create_features()

            # Select specific features
            available_features = [col for col in config['features'] if col in features.columns]
            if len(available_features) != len(config['features']):
                missing = set(config['features']) - set(available_features)
                logger.warning(f"Missing features for {asset_name}: {missing}")

            features = features[available_features]

            # Rename columns (except date)
            if config['rename_suffix']:
                rename_dict = {col: f"{col}{config['rename_suffix']}"
                               for col in features.columns if col != 'date'}
                features = features.rename(columns=rename_dict)

            logger.info(f"Asset {asset_name} loaded: {features.shape}")
            return features

        except Exception as e:
            logger.error(f"Error loading asset {asset_name}: {e}")
            return pd.DataFrame()

    def load_all_assets(self) -> pd.DataFrame:
        """Load all assets and merge into single DataFrame"""
        logger.info("Loading all assets...")

        # Start with gold as base
        features_combined = self.load_single_asset('gold')

        if features_combined.empty:
            logger.error("Failed to load base asset (gold)")
            return pd.DataFrame()

        # Merge all other assets
        for asset_name in self.asset_configs.keys():
            if asset_name == 'gold':  # Skip gold as it's already loaded
                continue

            asset_features = self.load_single_asset(asset_name)

            if not asset_features.empty:
                features_combined = features_combined.merge(
                    asset_features,
                    on='date',
                    how='left'
                )
            else:
                logger.warning(f"Skipping empty asset: {asset_name}")

        logger.info(f"Combined features shape: {features_combined.shape}")
        return features_combined

    def get_available_assets(self) -> list[str]:
        """Return list of available asset names"""
        return list(self.asset_configs.keys())

    def get_asset_config(self, asset_name: str):
        """Get configuration for specific asset"""
        if asset_name not in self.asset_configs:
            raise ValueError(f"Asset '{asset_name}' not configured")
        return self.asset_configs[asset_name]