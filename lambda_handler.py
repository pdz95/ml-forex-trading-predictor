import json
import os
import joblib
from datetime import datetime

# Basic environment setup
os.environ['NUMBA_CACHE_DIR'] = '/tmp'
os.environ['NUMBA_DISABLE_JIT'] = '1'
os.environ['MPLBACKEND'] = 'Agg'  # Non-interactive backend
os.environ['MPLCONFIGDIR'] = '/tmp'

def lambda_handler(event, context):
    """
    AWS Lambda handler for EUR/USD prediction
    """
    try:
        # Load models and data from pickle
        data = joblib.load('models/models.pkl')
        trained_models = data['trained_models']
        scaler = data['scaler']
        feature_names = data['feature_names']
        ensemble = trained_models[3]['model']  # Get ensemble model

        # Import classes (assuming they're in deployment folder)
        from deployment.core.data_downloader import DataDownloader
        from deployment.core.multi_asset_loader import MultiAssetDataLoader
        from deployment.core.real_time_prediction import RealTimePrediction

        # Download fresh EUR/USD data
        eurusd_data = DataDownloader("EURUSD=X").download_data(period='1y')

        # Create feature engineer (you'll need to import this)
        from training.feature_engineer import FeatureEngineer
        features_eurusd, _ = FeatureEngineer(eurusd_data, threshold_percentile=75,
                                             apply_scaling=False).create_features()

        # Load multi-asset data
        loader = MultiAssetDataLoader(threshold_percentile=75)
        features_additional = loader.load_all_assets()

        # Merge features
        final_features = features_eurusd.merge(features_additional, on='date', how='left').dropna()

        # Create predictor and make prediction
        predictor = RealTimePrediction(
            trained_model=ensemble,
            scaler=scaler,
            expected_features=feature_names
        )

        result = predictor.predict_tomorrow(final_features)

        prediction_result = {
            'prediction': ['bearish', 'neutral', 'bullish'][result['predicted_class']],
            'probabilities': {
                'bearish': result['bearish_prob'],
                'neutral': result['neutral_prob'],
                'bullish': result['bullish_prob']
            },
            'confidence': result['confidence'],
            'timestamp': datetime.now().isoformat()
        }


        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(prediction_result)
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'error': str(e)})
        }