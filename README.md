# ML-Powered EUR/USD Trading Predictor

Advanced EUR/USD movement prediction system using ensemble machine learning and multi-asset analysis.

## Live Demo
🚀 **[Try it live on AWS](http://3.94.54.203:8081)**

This is a production-ready system deployed on AWS infrastructure.

## Key Features

- **Next-day prediction** - Bullish/Bearish/Neutral signals
- **Multi-asset analysis** - 13 global markets (stocks, bonds, commodities)
- **Ensemble ML** - CatBoost + LightGBM + Logistic Regression
- **Serverless deployment** - AWS Lambda + Streamlit
- **Real-time data** - Yahoo Finance API

## Architecture

```
├── Frontend (Streamlit + plotly)
├── Backend (Python + AWS Lambda)
├── ML Pipeline (scikit-learn + CatBoost + LightGBM)
├── Data Sources (yfinance - 13 assets)
└── Infrastructure (AWS EC2 + Ubuntu + systemd + Docker)
```

##  Quick Start

### Local Setup

```bash
# Clone repository
git clone https://github.com/pdz95/eur-usd-ml-predictor
cd eur-usd-ml-predictor

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py --server.port=8081
```

## Performance Metrics

- **Bullish AUC**: 0.935 (excellent upward movement prediction)
- **Bearish AUC**: 0.889 (strong downward movement detection)
- **Neutral Precision**: 87.9% (outstanding sideways market recognition)

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **Frontend** | Streamlit, Plotly |
| **Backend** | Python 3.11, pandas, numpy |
| **ML** | scikit-learn, CatBoost, LightGBM |
| **Cloud** | AWS Lambda, ECR, EC2 |
| **Infrastructure** | Ubuntu 24.04, systemd, Docker |
| **Data** | yfinance API |

## How It Works

1. **Data collection** - 13 financial instruments (EUR/USD, S&P500, Gold, VIX, etc.)
2. **Feature engineering** - 90+ technical indicators
3. **Ensemble prediction** - 3 ML models vote on outcome
4. **Signal filtering** - Only movements above 75th percentile
5. **Real-time serving** - AWS Lambda + caching

## Example Prediction

```json
{
  "prediction": "bullish",
  "confidence": 0.847,
  "probabilities": {
    "bearish": 0.123,
    "neutral": 0.030,
    "bullish": 0.847
  },
  "timestamp": "2025-06-16T10:30:00Z"
}
```

## Project Structure

```
eur-usd-ml-predictor/
├── app.py                 # Main Streamlit application
├── lambda_handler.py      # AWS Lambda handler
├── requirements.txt       # Python dependencies
├── dockerfile            # Docker for Lambda
├── setup.sh              # Deployment script
├── deployment/
│   ├── core/
│   │   ├── data_downloader.py      # Data downloading
│   │   ├── feature_engineer.py     # Feature engineering
│   │   ├── multi_asset_loader.py   # Multi-asset data
│   │   └── real_time_prediction.py # Real-time prediction
│   └── ui/
│       └── sidebar.py     # UI components
├── training/
│   └── feature_engineer.py
└── models/
    ├── models.pkl         # Trained models
    └── performance_plots.json
```

## Disclaimer

This system is an **educational and demonstration project**. It does not constitute investment advice. Do not use for actual trading decisions. Past performance does not guarantee future results.

## Executive Summary

This project demonstrates:

- **Full-stack development** (Frontend + Backend + ML + Cloud)
- **Production-ready deployment** (Docker + AWS + monitoring)
- **Clean code practices** (modularity, error handling, logging)
- **Financial domain knowledge** (technical analysis, risk management)
- **ML expertise** (ensemble methods, evaluation metrics, feature engineering)

## Contact

**Author**: Paweł Działak
**Email**: pdzialak55@gmail.com  
**Year**: 2025
