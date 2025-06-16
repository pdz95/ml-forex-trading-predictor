# prediction.py

# Core imports - lightweight
import logging

# Setup logger
logger = logging.getLogger(__name__)

class Prediction:
    def __init__(self, model, X, y) -> None:
        self.trained_model = model[0]['model']
        self.X = X
        self.y = y

    def predict(self):
        # Lazy import sklearn metrics
        from sklearn.metrics import confusion_matrix, classification_report

        # Accuracy/score
        accuracy = self.trained_model.score(self.X, self.y)

        # Predictions
        predictions = self.trained_model.predict(self.X)

        # Confusion Matrix
        cm = confusion_matrix(self.y, predictions)

        # Classification Report
        classif_report = classification_report(self.y, predictions)

        return predictions, accuracy, cm, classif_report