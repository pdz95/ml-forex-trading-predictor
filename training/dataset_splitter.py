# dataset_splitter.py

# Core imports - lightweight
import logging

# Setup logger
logger = logging.getLogger(__name__)

class DatasetSplitCategorical:
   def __init__(self, X, y, test_size: int = 400, validation_size: int = 50) -> None:
       self.test_size = test_size
       self.validation_size = validation_size
       self.total_val_test = validation_size + test_size
       self.X = X  # numpy array or pandas DataFrame
       self.y = y  # numpy array or pandas Series

   def split_dataset(self):
       """Split dataset into train/validation/test sets using time series approach"""
       # Test set – ostatnie `test_size` próbek
       X_test = self.X[-self.test_size:]
       y_test = self.y[-self.test_size:]

       # Validation set – tuż przed testowym
       X_val = self.X[-self.total_val_test:-self.test_size]
       y_val = self.y[-self.total_val_test:-self.test_size]

       # Train set – wszystko przed validation + test
       X_train = self.X[:-self.total_val_test]
       y_train = self.y[:-self.total_val_test]

       # Logowanie
       logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
       logger.info(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
       logger.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

       return X_train, X_val, X_test, y_train, y_val, y_test