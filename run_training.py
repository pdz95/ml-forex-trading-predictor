#!/usr/bin/env python3
"""
EUR/USD Trading Model Training Script

Usage:
    python run_training.py

This will train the complete ML pipeline and save models to models/ directory.
"""

import sys
import os
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

def main():
    print("Starting EUR/USD Trading Model Training...")
    print("=" * 50)
    
    try:
        # Import pipeline
        from training.pipeline import TradingModelPipeline
        
        # Initialize pipeline with your settings
        pipeline = TradingModelPipeline(
            symbol="EURUSD=X",
            threshold_percentile=75,  # Your 75th percentile setting
            test_size=500,           # Adjust as needed
            validation_size=200       # Adjust as needed
        )
        
        print("Step 1: Preparing data...")
        pipeline.prepare_data()
        print(f"Data shape: {pipeline.df_ready.shape}")
        
        print("Step 2: Creating features...")
        pipeline.create_features()
        print(f"Features: {len(pipeline.feature_names)}")
        
        print("Step 3: Training models...")
        trained_models, ensemble = pipeline.train_models()
        print(f"rained {len(trained_models)} models + ensemble")
        
        print("Step 4: Saving models...")
        model_file = pipeline.save_model('models/models.pkl')
        print(f"Models saved to: {model_file}")
        
        print("=" * 50)
        print("Training completed successfully!")
        print(f"Model file: {model_file}")
        print("Ready for deployment!")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()