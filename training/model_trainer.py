# model_trainer.py

# Core imports - lightweight
import logging

# Setup logger
logger = logging.getLogger(__name__)


class TrainModel:
    def __init__(self, model_list: list[str], X_train, y_train) -> None:
        self.model_list = model_list
        self.X_train = X_train
        self.y_train = y_train

        self.model_map = {
            "logistic_regression": self._logistic_regression,
            "xgboost": self._xgboost,
            "svc": self._svc,
            "knn": self._knn,
            "random_forest": self._random_forest,
            "catboost": self._catboost,
            "lightgbm": self._lightgbm,
            "naive_bayes": self._naive_bayes,
            "adaboost": self._adaboost
        }

    def train_models(self):
        """Train all specified models"""
        trained_models = []
        for model_name in self.model_list:
            if model_name in self.model_map:
                logger.info(f"Training {model_name}...")
                trained_model = self.model_map[model_name]()
                trained_models.append({
                    "name": model_name,
                    "model": trained_model
                })
                logger.info(f"{model_name} trained successfully")
            else:
                logger.warning(
                    f"Model '{model_name}' is not available. Available models: {list(self.model_map.keys())}")

        logger.info(f"Training completed. {len(trained_models)} models trained.")
        return trained_models

    def _logistic_regression(self):
        """Train Logistic Regression model"""
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=5000, random_state=1)
        model.fit(self.X_train, self.y_train)
        return model

    def _xgboost(self):
        """Train XGBoost model"""
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=400,
            max_depth=2,
            learning_rate=0.1,
            random_state=1,
            n_jobs=-1,
            eval_metric='logloss'
        )
        model.fit(self.X_train, self.y_train)
        return model

    def _random_forest(self):
        """Train Random Forest model"""
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=25,
            class_weight='balanced',
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=1,
            n_jobs=-1,
        )
        model.fit(self.X_train, self.y_train)
        return model

    def _svc(self):
        """Train Support Vector Classifier"""
        from sklearn.svm import SVC
        model = SVC(kernel='sigmoid', probability=True, random_state=1)
        model.fit(self.X_train, self.y_train)
        return model

    def _knn(self):
        """Train K-Nearest Neighbors model"""
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(self.X_train, self.y_train)
        return model

    def _catboost(self):
        """Train CatBoost model"""
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.1,
            random_seed=1,
            verbose=False,
        )
        model.fit(self.X_train, self.y_train)
        return model

    def _lightgbm(self):
        """Train LightGBM model"""
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            random_state=1,
            n_jobs=-1,
            class_weight='balanced',
            verbose=-1
        )
        model.fit(self.X_train, self.y_train)
        return model

    def _naive_bayes(self):
        """Train Naive Bayes model"""
        from sklearn.naive_bayes import GaussianNB
        model = GaussianNB()
        model.fit(self.X_train, self.y_train)
        return model

    def _adaboost(self):
        """Train AdaBoost model"""
        from sklearn.ensemble import AdaBoostClassifier
        model = AdaBoostClassifier(
            n_estimators=100,
            learning_rate=1.0,
            random_state=1
        )
        model.fit(self.X_train, self.y_train)
        return model