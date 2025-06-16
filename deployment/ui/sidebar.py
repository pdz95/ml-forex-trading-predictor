import streamlit as st


def create_sidebar():
    """Side bar with additional information"""
    with st.sidebar:
        st.markdown("#### 📋 About the project")
        st.markdown("""
        **Version:** 1.1

        **Author:** Paweł

        **Year-month:** 2025/06
        """)

        st.markdown("---")

        # Linki
        st.markdown("#### 🔗 Links")

        # GitHub
        st.markdown("""
        [![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/twoj-username/projekt)
        """)

        # Email
        st.markdown("""
        [![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:pdzialak55@gmail.com)
        """)

        st.markdown("---")

        # Informacje techniczne
        st.markdown("#### ⚙️ Tech stack")
        st.markdown("""
        **Frontend:** Streamlit + Plotly

        **Backend:** Python 3.11 + pandas + numpy

        **Cloud:** AWS services (Lambda, ECR, EC2)
        
        **Infrastructure:** Ubuntu 24.04 + systemd

        **Ensemble ML model composed of:**: 
        

        - CatBoost
        - LightGBM
        - scikit-learn (Logistic Regression)

        **Data Sources:** yfinance API (13 assets)

        **Features:** 90+ technical indicators

        **Data Pipeline:** Time-aware splits, 20 years EUR/USD data
        """)

        st.markdown("---")

        # Footer
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; color: #666; font-size: 12px;'>
                Made with ❤️ using Streamlit

                Kraken2: Wood et al. 2019, Improved metagenomic analysis with Kraken 2. Genome Biol 20, 257 (2019).

                NCBI samples: SRX28808346, SRX28808342<br>
            </div>
            """,
            unsafe_allow_html=True
        )