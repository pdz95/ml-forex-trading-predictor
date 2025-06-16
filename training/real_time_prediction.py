import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class RealTimePrediction:
    def __init__(self, trained_model, scaler, expected_features):
        self.model = trained_model
        self.scaler = scaler
        self.expected_features = expected_features

    def predict_tomorrow(self, prepared_data):
        """Make prediction using already prepared data"""

        # Get latest row
        latest_row = prepared_data.iloc[-1:].copy()
        #latest_row.to_csv("latest_raw.csv")
        date_value = latest_row['date'].iloc[0]

        # Extract features (same logic as training)
        feature_columns = [col for col in prepared_data.columns
                           if col not in ['date', 'Target_signal', 'Open', 'High', 'Low', 'Close', 'Volume']]

        # Scale features
        X_raw = latest_row[feature_columns]

        X_scaled = self.scaler.transform(X_raw)
        X_scaled_df = pd.DataFrame(X_scaled, columns=feature_columns, index=latest_row.index)

        # Select expected features
        X_final = X_scaled_df[self.expected_features]

        # Predict
        prediction_proba = self.model.predict_proba(X_final)[0]
        predicted_class = int(np.argmax(prediction_proba))

        return {
            'bearish_prob': float(prediction_proba[0]),
            'neutral_prob': float(prediction_proba[1]),
            'bullish_prob': float(prediction_proba[2]),
            'predicted_class': predicted_class,
            'confidence': float(np.max(prediction_proba)),
            'timestamp': pd.Timestamp.now(),
            'date': date_value,
            'prediction_date': latest_row['date'].iloc[0]

        }