# pipeline.py
import joblib
import os
import pandas as pd
import logging

from .data_downloader import DataDownloader
from .feature_engineer import FeatureEngineer
from .multi_asset_loader import MultiAssetDataLoader
from sklearn.preprocessing import StandardScaler
from .dataset_splitter import DatasetSplitCategorical
from .model_trainer import TrainModel
from sklearn.ensemble import VotingClassifier

from .dataset_splitter import DatasetSplitCategorical

# Setup logger
logger = logging.getLogger(__name__)


class TradingModelPipeline:

    def __init__(self, symbol="EURUSD=X", threshold_percentile=80, test_size=500, validation_size=200):
        self.symbol = symbol
        self.threshold_percentile = threshold_percentile
        self.test_size = test_size
        self.validation_size = validation_size
        self.model_list = ["logistic_regression", "xgboost", "catboost", "lightgbm"]

        # Will be populated during pipeline
        self.df_ready = None
        self.X = None
        self.y = None
        self.scaler = None
        self.feature_names = None
        self.trained_models = None
        self.ensemble_model = None

    def prepare_data(self):
        """Step 1: Data preparation and feature engineering"""

        logger.info("Preparing data...")

        # EUR/USD features (NO SCALING) - DOKŁADNIE JAK W JUPYTER
        eurusd_data = DataDownloader(self.symbol).download_data()
        features_eurusd, _ = FeatureEngineer(
            eurusd_data,
            threshold_percentile=self.threshold_percentile,
            apply_scaling=False  # ← KLUCZOWE: bez skalowania
        ).create_features()

        # Additional market indicators - DOKŁADNIE JAK W JUPYTER
        loader = MultiAssetDataLoader(threshold_percentile=self.threshold_percentile)
        features_additional = loader.load_all_assets()

        # Merge raw data - DOKŁADNIE JAK W JUPYTER
        self.df_ready = features_eurusd.merge(features_additional, on='date', how='left')
        self.df_ready = self.df_ready.dropna()

        logger.info(f"Data prepared: {self.df_ready.shape}")
        return self.df_ready

    def create_features(self):
        """Step 2: Feature selection and scaling"""
        if self.df_ready is None:
            raise ValueError("Data not prepared! Call prepare_data() first.")

        logger.info("Creating features...")

        # Define feature columns
        feature_columns = [col for col in self.df_ready.columns
                           if col not in ['date', 'Target_signal', 'Open', 'High', 'Low', 'Close', 'Volume']]

        # Raw features
        X_raw = self.df_ready[feature_columns]
        self.y = self.df_ready['Target_signal']

        # TERAZ skalowanie kompletnego zestawu cech
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_raw)

        # Convert back to DataFram
        self.X = pd.DataFrame(X_scaled, columns=feature_columns, index=X_raw.index)

        # Save ALL feature names
        self.feature_names = list(self.X.columns)

        logger.info(f"Features created: {len(self.feature_names)}")
        return self.X, self.y

    def train_models(self):
        """Step 3: Model training"""
        if self.X is None or self.y is None:
            raise ValueError("Features not created! Call create_features() first.")


        logger.info("Training models...")

        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = DatasetSplitCategorical(
            X=self.X, y=self.y,
            test_size=self.test_size,
            validation_size=self.validation_size
        ).split_dataset()

        # Train individual models
        self.trained_models = TrainModel(
            model_list=self.model_list,
            X_train=X_train,
            y_train=y_train
        ).train_models()

        # Create ensemble
        lr_model = self.trained_models[0]["model"]
        catboost_model = self.trained_models[2]["model"]
        lightgbm_model = self.trained_models[3]["model"]

        self.ensemble_model = VotingClassifier(
            estimators=[
                ('logistic', lr_model),
                ('catboost', catboost_model),
                ('lightgbm', lightgbm_model)
            ],
            voting='soft'
        )

        # Fit ensemble
        self.ensemble_model.fit(X_train, y_train)

        logger.info("Models trained successfully!")
        return self.trained_models, self.ensemble_model

    def save_model(self, filename='models/models.pkl'):
        """Step 4: Save complete model """
        if self.ensemble_model is None:
            raise ValueError("Models not trained! Call train_models() first.")

        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)

        X_train, X_val, X_test, y_train, y_val, y_test = DatasetSplitCategorical(
            X=self.X, y=self.y,
            test_size=self.test_size,
            validation_size=self.validation_size
        ).split_dataset()

        # Create ensemble_models structure
        lr_model = self.trained_models[0]["model"]
        catboost_model = self.trained_models[2]["model"]
        lightgbm_model = self.trained_models[3]["model"]

        ensemble_models = [
            {'name': 'logistic_regression', 'model': lr_model},
            {'name': 'catboost', 'model': catboost_model},
            {'name': 'lightgbm', 'model': lightgbm_model},
            {'name': 'ensemble', 'model': self.ensemble_model}
        ]

        jupyter_data = {
            'X_test': X_val,  # Test data
            'y_test': y_val,  # Test labels
            'trained_models': ensemble_models,  # All models + ensemble
            'feature_names': self.feature_names,  # Feature names
            'scaler': self.scaler  # Scaler
        }

        joblib.dump(jupyter_data, filename)
        logger.info(f"Model saved to {filename}")
        return filename

    def run_complete_pipeline(self):
        """Run entire pipeline"""
        self.prepare_data()
        self.create_features()
        self.train_models()
        filename = self.save_model()

        logger.info("Pipeline completed successfully!")
        logger.info(f"Target distribution: {self.df_ready['Target_signal'].value_counts().to_dict()}")
        logger.info(f"Model saved as: {filename}")

        return filename