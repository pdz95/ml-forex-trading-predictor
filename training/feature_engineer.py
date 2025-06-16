# feature_engineer.py

# Core imports - lightweight
import pandas as pd
import numpy as np
import logging

# Setup logger
logger = logging.getLogger(__name__)


class FeatureEngineer:
    def __init__(self, df: pd.DataFrame, price_col: str = 'Close', date_col: str = 'date',
                 threshold_percentile=67, apply_scaling=True) -> None:
        self.df = df.copy()
        self.price_col = price_col
        self.date_col = date_col
        self.threshold_percentile = threshold_percentile
        self.apply_scaling = apply_scaling
        self.scaler = None

    def create_features(self):
        """Create technical and time-based features"""
        df = self.df.copy()

        df[self.date_col] = pd.to_datetime(df[self.date_col])

        # NAJPIERW oblicz target na DZISIEJSZYCH cenach (bez shift)
        df['today_return'] = df[self.price_col].pct_change()

        # Threshold based on returns
        valid_returns = df['today_return'].dropna()
        threshold = np.percentile(np.abs(valid_returns), self.threshold_percentile)
        df['Target_signal'] = np.where(df['today_return'] > threshold, 2,
                                       np.where(df['today_return'] < -threshold, 0, 1))

        # Shift for Close for features
        df[self.price_col] = df[self.price_col].shift(1)

        df = self._add_lag_features(df)
        df = self._add_return_features(df)
        df = self._add_moving_averages(df)
        df = self._add_volatility_features(df)
        df = self._add_momentum_features(df)
        df = self._add_technical_indicators(df)
        df = self._add_time_features(df)
        df = self._add_rolling_statistics(df)

        # Clean up
        df, scaler = self._finalize_dataframe(df)
        if 'today_return' in df.columns:
            df = df.drop('today_return', axis=1)

        df = df.dropna()
        logger.info(f"Features created: {df.shape[1]} columns, {df.shape[0]} rows")
        return df, scaler

    def _add_lag_features(self, df):
        """Add price lag features"""
        for i in [5, 10]:
            df[f'price_lag_{i}'] = df[self.price_col].shift(i)

        for lag in [1, 7, 12, 24]:
            df[f'lag_{lag}'] = df[self.price_col].shift(lag)

        return df

    def _add_return_features(self, df):
        """Add return-based features"""
        for i in [1, 2, 3, 5, 10, 20]:
            df[f'returns_{i}'] = df[self.price_col].pct_change(i)

        df['diff_1'] = df[self.price_col].diff()
        df['pct_change'] = df[self.price_col].pct_change()

        return df

    def _add_moving_averages(self, df):
        """Add moving average features"""
        for window in [5, 10, 20]:
            df[f'MA_{window}'] = df[self.price_col].rolling(window=window).mean()

        for span in [5, 10, 20]:
            df[f'EWM_{span}'] = df[self.price_col].ewm(span=span, adjust=False).mean()

        return df

    def _add_volatility_features(self, df):
        """Add volatility-based features"""
        for window in [5, 10, 20]:
            df[f'volatility_{window}'] = df['returns_1'].rolling(window=window).std()

        return df

    def _add_momentum_features(self, df):
        """Add momentum indicators"""
        for window in [5, 10, 20]:
            df[f'momentum_{window}'] = df[self.price_col] - df[self.price_col].shift(window)

        return df

    def _add_technical_indicators(self, df):
        """Add advanced technical indicators"""
        # RSI
        delta = df[self.price_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = df[self.price_col].ewm(span=12, adjust=False).mean()
        ema_26 = df[self.price_col].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # Bollinger Bands
        sma_20 = df[self.price_col].rolling(window=20).mean()
        std_20 = df[self.price_col].rolling(window=20).std()
        df['Bollinger_Upper'] = sma_20 + (2 * std_20)
        df['Bollinger_Lower'] = sma_20 - (2 * std_20)

        # Stochastic Oscillator
        low_14 = df[self.price_col].rolling(window=14).min()
        high_14 = df[self.price_col].rolling(window=14).max()
        df['Stochastic'] = ((df[self.price_col] - low_14) / (high_14 - low_14)) * 100

        # ATR
        df['ATR'] = df[self.price_col].diff().abs().rolling(window=14).mean()

        return df

    def _add_time_features(self, df):
        """Add time-based features"""
        df['day_of_week'] = df[self.date_col].dt.dayofweek
        df['month'] = df[self.date_col].dt.month
        df['year'] = df[self.date_col].dt.year
        df['quarter'] = df[self.date_col].dt.quarter
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

        # Cyclical encoding
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        return df

    def _add_rolling_statistics(self, df):
        """Add rolling window statistics"""
        for window in [7, 24]:
            roll = df[self.price_col].rolling(window=window)
            df[f'rolling_mean_{window}'] = roll.mean()
            df[f'rolling_std_{window}'] = roll.std()
            df[f'rolling_min_{window}'] = roll.min()
            df[f'rolling_max_{window}'] = roll.max()

        return df

    def _finalize_dataframe(self, df):
        """Final cleanup, scaling and type conversion"""
        # Lazy import sklearn only when scaling needed
        if self.apply_scaling:
            from sklearn.preprocessing import StandardScaler

        # Identyfikuj feature columns (bez target i date)
        feature_cols = [col for col in df.columns
                        if col not in [self.date_col, 'Target_signal', 'today_return']]

        # Skalowanie features
        scaler = None
        if self.apply_scaling:
            scaler = StandardScaler()
            df[feature_cols] = scaler.fit_transform(df[feature_cols])
            self.scaler = scaler

        # Konwersja na float64
        numeric_cols = df.select_dtypes(include=['float', 'int']).columns
        df[numeric_cols] = df[numeric_cols].astype('float64')

        df = df.dropna()
        return df, scaler