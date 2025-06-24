import asyncio
import sys
import streamlit as st
import joblib
import json
import boto3
import plotly.io as pio
import plotly.graph_objects as go
from deployment.core.data_downloader import DataDownloader
from deployment.ui.sidebar import create_sidebar

# Fix asyncio event loop error
if sys.platform.startswith('linux'):
    import asyncio
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except:
        pass



st.set_page_config(
    page_title="ML-Powered EUR/USD Trading Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded")

create_sidebar()

st.title("📈 ML-Powered EUR/USD Trading Predictor")
st.info("""

This is a **portfolio demonstration project** showcasing machine learning in financial markets. The system analyzes market patterns and multi-asset correlations to predict significant EUR/USD moves.

**Key Features:**
- **"Golden Arrow" signals** - targets moves beyond 75th percentile only
- **Multi-asset analysis** - incorporates 13 global markets (stocks, bonds, commodities)
- **Educational purpose** - demonstrates ML ensemble techniques in finance

⚠️ **IMPORTANT DISCLAIMER:** This system is for educational and demonstration purposes only. Not financial advice. Do not use for actual trading decisions. Past performance does not guarantee future results.
""")

@st.cache_data
def get_stock_data_with_logs(symbol, period):
    downloader = DataDownloader(symbol)
    data = downloader.download_data(period=period)
    logs = downloader.get_logs()
    return data, logs

@st.cache_data(ttl=43200)  # Cache for 12h
def call_lambda_prediction():
    """Call lambda and return prediction"""

    # Lambda AWS client
    lambda_client = boto3.client('lambda', region_name='us-east-1')

    # Call lambda
    response = lambda_client.invoke(
        FunctionName='trading-lambda', #Lambda function name
        InvocationType='RequestResponse',
        Payload=json.dumps({})
    )

    payload = response['Payload'].read()
    result = json.loads(payload)
    return json.loads(result['body'])




# Header
st.title("EUR/USD ML Trading System")



tab1, tab2, tab3 = st.tabs(["Live Trading", "Model Performance", "Tech stack"])

with tab1:
    # Main content area
    col1, col2, col3 = st.columns([2, 2, 3])
    with col1:
        st.subheader("EUR/USD Price Chart")

        data, download_logs = get_stock_data_with_logs("EURUSD=X", "50d")

        # Data check ups + log
        if data.empty or 'date' not in data.columns:
            st.error("Cannot load EUR/USD data. Please try again later.")
            st.subheader("Download Logs")
            st.code(download_logs)

        else:
            # Create plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data['date'],
                y=data['Close'],
                mode='lines',
                name='EUR/USD'
            ))
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Price"
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Next Day Prediction")

        if st.button("Get Prediction"):
            with st.spinner("Analyzing markets..."):
                try:
                    prediction = call_lambda_prediction()
                    st.session_state.prediction = prediction
                    #st.success(f"Prediction: {prediction['prediction']}")
                    #st.json(prediction['probabilities'])
                except Exception as e:
                    st.error(f"Error: {e}")

        # Show prediction if exist
        if "prediction" in st.session_state:
            result = st.session_state.prediction

            # Main prediction with color coding
            prediction_class = result['prediction']
            confidence = result['confidence']

            if prediction_class == 'bullish':
                st.success(f"🟢 BULLISH ({confidence:.1%})")
            elif prediction_class == 'bearish':
                st.error(f"🔴 BEARISH ({confidence:.1%})")
            else:
                st.warning(f"🟡 NEUTRAL ({confidence:.1%})")

            # Show all probabilities
            st.subheader("Detailed Probabilities")

            col1a, col2a, col3a = st.columns(3)

            with col1a:
                st.metric(
                    label="Bearish",
                    value=f"{result['probabilities']['bearish']:.1%}",
                    delta=None
                )

            with col2a:
                st.metric(
                    label="Neutral",
                    value=f"{result['probabilities']['neutral']:.1%}",
                    delta=None
                )

            with col3a:
                st.metric(
                    label="Bullish",
                    value=f"{result['probabilities']['bullish']:.1%}",
                    delta=None
                )

            # Show prediction details
            st.caption(f"Made at: {result['timestamp']}")
            #st.caption(f"Last data update: {result['date']}")

    with col3:
        st.markdown("""
        **What it does:**
        This system predicts significant EUR/USD moves for the next day. Instead of chasing noise, it focuses only on substantial price action worth trading.
        
        **How it works:**
        Analyzes 13 financial markets (stocks, bonds, commodities, currencies) with 90+ technical indicators.  Ensemble of three complementary models: linear (Logistic Regression) and two tree-based algorithms (CatBoost, LightGBM) for different pattern recognition approaches.
        
    
        
        **Key Performance Insights:**
        - **System has bullish bias** - excels at predicting upward moves (93.5% AUC) vs downward (88.9% AUC)
        - **Outstanding neutral detection** - 87.9% precision for sideways markets, crucial for avoiding bad trades
        - **Bearish signals need caution** - lower precision (62.3%) means more false sell signals
        - **EUR/USD uptrends are more predictable** than downtrends, making this ideal for long-biased strategies
        
        **Trading Implications:**
        - Highest confidence in bullish and neutral signals
        - Bearish alerts require additional confirmation
        - System works best during trending and consolidation periods
        - Designed for moves large enough to beat spreads and generate real profits
        
        **Data Foundation:**
        20 years of training data + daily fresh market data from Yahoo Finance, filtered to 75th percentile moves only.
        """)

with tab2:

    st.subheader("Model Performance Comparison")
    with open('models/performance_plots.json', 'r') as f:
        plots = json.load(f)

    # Plots
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("ROC Curves")
        fig_roc = pio.from_json(plots['roc_fig'])
        st.plotly_chart(fig_roc, use_container_width=True)

    with col2:
        st.subheader("Precision-Recall Curves")
        fig_pr = pio.from_json(plots['pr_fig'])
        st.plotly_chart(fig_pr, use_container_width=True)

    st.markdown("---")  

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
        ### ROC Results:
        - **Bearish**: 0.889 AUC - strong downside detection
        - **Neutral**: 0.787 AUC - good sideways identification  
        - **Bullish**: 0.935 AUC - excellent upside prediction

        ### Key Trading Insight:
        **System has bullish bias** - EUR/USD uptrends are significantly more predictable than downtrends. This makes the system ideal for long-biased trading strategies and trend-following approaches.
        """)

    with col2:
        st.markdown("""
        ### Precision-Recall Results:
        - **Bearish**: 0.623 AP - requires additional confirmation
        - **Neutral**: 0.879 AP - outstanding neutral market detection
        - **Bullish**: 0.757 AP - solid upside precision

        ### Practical Application:
        **Highest confidence** in bullish and neutral signals. Bearish alerts generate more false signals (62.3% precision) - use with caution or combine with other indicators for confirmation.
        """)

with tab3:
    st.header("About the project")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.info("""
        **🔧 Technical Architecture**

        Complete serverless ML pipeline for forex prediction:

        • **Frontend:** Streamlit + Plotly charts
        
        • **Backend:** Python 3.11 + pandas + numpy
        
        • **Cloud:** AWS Lambda + ECR + EC2
        
        • **Infrastructure:** Ubuntu 24.04 + systemd monitoring
        
        • **ML Ensemble:** CatBoost + LightGBM + Logistic Regression
        
        • **Data Sources:** yfinance API (13 assets)
        
        • **Features:** 90+ technical indicators
        
        • **Data Pipeline:** Time-aware splits, 20 years EUR/USD data
        """)

    with col2:
        st.info("""
        **⚠️ For Traders & Investors**

        **The Challenge:**
        
        • EUR/USD movements are hard to predict
        
        • Most systems chase noise anc can give meaningless signals
        
        • Single-asset analysis misses market context
        
        • Transaction costs eat small profits

        **The Solution:**
        
        • Multi-asset analysis across 13 markets
        
        • Focus on 75th percentile moves only
        
        • Ensemble approach for balanced predictions
        
        • Educational system - not financial advice
        

        **Perfect for:**
        
        • Learning ML in finance
        
        • Understanding forex dynamics
        
        • Portfolio demonstration
        
        • Educational purposes
        """)

    with col3:
        st.info("""
        **🚀 Future Enhancements**
        

        • **Additional currency pairs** for broader coverage
        
        • **Alternative data** (news sentiment, economics)
        
        • **Deep learning models** (LSTM, Transformers)
    
        """)

