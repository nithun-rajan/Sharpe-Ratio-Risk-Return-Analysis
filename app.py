import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from modules.portfolio_optimizer import PortfolioOptimizer
from modules.risk_analyzer import RiskAnalyzer
from modules.stock_predictor import StockPredictor
from modules.rl_agent import RLAgent
from modules.backtest_report import BacktestReport
from modules.utils import load_custom_css, get_stock_data

# Page configuration
st.set_page_config(
    page_title="FinOptima | AI-Powered Financial Suite",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS for fintech styling
load_custom_css()

def main():
    # Sidebar Navigation
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
            <h1>📈 FinOptima</h1>
            <p>AI-Powered Financial Optimization Suite</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Date Range Selector
        st.markdown("### 📅 Date Range")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=365*2),
                help="Select the start date for analysis"
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now(),
                help="Select the end date for analysis"
            )
        
        # Stock Ticker Selector
        st.markdown("### 🎯 Stock Selection")
        default_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'SPY']
        selected_tickers = st.multiselect(
            "Select Stocks",
            options=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'SPY', 'QQQ', 'JPM', 'BAC', 'JNJ', 'PG', 'KO'],
            default=default_tickers[:4],
            help="Choose stocks for analysis (max 10 recommended)"
        )
        
        # Navigation Menu
        st.markdown("### 🧭 Navigation")
        selected_section = st.radio(
            "Dashboard Sections",
            [
                "📊 Portfolio Optimizer",
                "⚠️ Risk Analyzer", 
                "📈 Stock Predictor",
                "🤖 RL Trading Agent",
                "📉 Backtest Report",
                "📘 About Project"
            ]
        )
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div class="sidebar-footer">
            <p>Built with ❤️ using Streamlit</p>
            <p>© 2024 FinOptima Suite</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main Content Area
    if not selected_tickers:
        st.error("⚠️ Please select at least one stock ticker from the sidebar to continue.")
        return
    
    # Load data for selected tickers
    with st.spinner(f"Loading data for {len(selected_tickers)} stocks..."):
        stock_data = get_stock_data(selected_tickers, start_date, end_date)
    
    # Route to appropriate section
    if selected_section == "📊 Portfolio Optimizer":
        portfolio_optimizer = PortfolioOptimizer()
        portfolio_optimizer.render(stock_data, selected_tickers, start_date, end_date)
    
    elif selected_section == "⚠️ Risk Analyzer":
        risk_analyzer = RiskAnalyzer()
        risk_analyzer.render(stock_data, selected_tickers)
    
    elif selected_section == "📈 Stock Predictor":
        stock_predictor = StockPredictor()
        stock_predictor.render(selected_tickers, start_date, end_date)
    
    elif selected_section == "🤖 RL Trading Agent":
        rl_agent = RLAgent()
        rl_agent.render(selected_tickers, start_date, end_date)
    
    elif selected_section == "📉 Backtest Report":
        backtest_report = BacktestReport()
        backtest_report.render(stock_data, selected_tickers, start_date, end_date)
    
    elif selected_section == "📘 About Project":
        render_about_section()

def render_about_section():
    st.markdown("""
    <div class="main-header">
        <h1>📘 About FinOptima Project</h1>
        <p>Comprehensive AI-Powered Financial Analysis Suite</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Project Architecture
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🏗️ Project Architecture")
        st.markdown("""
        **FinOptima** is a comprehensive financial analysis suite that combines traditional quantitative finance
        with cutting-edge machine learning and reinforcement learning techniques.
        
        #### 🧩 Core Components:
        
        1. **Portfolio Optimizer** - Markowitz Mean-Variance Optimization with constraints
        2. **Risk Analyzer** - Multi-method Value-at-Risk (VaR) calculations
        3. **Stock Predictor** - LSTM & XGBoost time series forecasting
        4. **RL Trading Agent** - Deep Q-Network (DQN) trading strategies
        5. **Backtest Engine** - Comprehensive strategy performance analysis
        
        #### 🛠️ Technology Stack:
        - **Frontend**: Streamlit with custom CSS styling
        - **Data Processing**: Pandas, NumPy, Yahoo Finance API
        - **Visualization**: Plotly, Seaborn, Matplotlib
        - **ML/DL**: TensorFlow, XGBoost, Scikit-learn
        - **RL Framework**: Stable-Baselines3, OpenAI Gym
        - **Optimization**: CVXPY, SciPy
        - **Reporting**: FPDF2 for PDF generation
        """)
    
    with col2:
        st.markdown("### 📊 Key Features")
        st.info("""
        ✅ **Modern Portfolio Theory**
        - Efficient Frontier Calculation
        - Sharpe Ratio Optimization
        - Risk-Return Analysis
        
        ✅ **Advanced Risk Management**
        - Historical VaR
        - Monte Carlo VaR
        - Parametric VaR
        
        ✅ **AI-Powered Predictions**
        - LSTM Neural Networks
        - XGBoost Ensemble
        - Performance Metrics
        
        ✅ **Reinforcement Learning**
        - DQN Trading Agent
        - Environment Simulation
        - Strategy Backtesting
        """)
    
    # Technical Implementation
    st.markdown("### 💻 Technical Implementation")
    
    tab1, tab2, tab3 = st.tabs(["🧠 Models", "📈 Data Flow", "🎨 Design"])
    
    with tab1:
        st.markdown("""
        #### Machine Learning Models
        
        **LSTM Networks**
        ```python
        # Sequential LSTM architecture for time series prediction
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(lookback, features)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        ```
        
        **XGBoost Ensemble**
        ```python
        # Gradient boosting with technical indicators
        xgb_model = XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8
        )
        ```
        
        **Reinforcement Learning**
        ```python
        # DQN Agent for trading decisions
        from stable_baselines3 import DQN
        
        model = DQN(
            'MlpPolicy',
            env,
            learning_rate=1e-3,
            buffer_size=1000000,
            target_update_interval=1000
        )
        ```
        """)
    
    with tab2:
        st.markdown("""
        #### Data Processing Pipeline
        
        1. **Data Acquisition** 📡
           - Yahoo Finance API integration
           - Real-time price feeds
           - Historical data retrieval
        
        2. **Feature Engineering** ⚙️
           - Technical indicators (RSI, MACD, Bollinger Bands)
           - Price transformations (returns, log returns)
           - Rolling statistics (volatility, moving averages)
        
        3. **Model Training** 🎯
           - Train/validation/test splits
           - Cross-validation for robustness
           - Hyperparameter optimization
        
        4. **Backtesting** 📊
           - Out-of-sample testing
           - Performance metrics calculation
           - Risk-adjusted returns analysis
        """)
    
    with tab3:
        st.markdown("""
        #### Design Language & UX
        
        **Color Palette**
        - Primary: `#00C49F` (Teal Green) - Success, Growth
        - Secondary: `#1F2633` (Dark Slate) - Professional, Modern
        - Accent: `#FF8042` (Orange) - Attention, Warnings
        
        **Typography**
        - Headers: Montserrat (Clean, Modern)
        - Body: Roboto (Readable, Professional)
        
        **UX Principles**
        - Progressive disclosure with collapsible sections
        - Contextual tooltips for financial metrics
        - Responsive layout for different screen sizes
        - Interactive visualizations with Plotly
        """)
    
    # Resume Bullet Points
    st.markdown("### 🎯 Key Achievements & Skills Demonstrated")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 📈 Quantitative Finance
        - Implemented Modern Portfolio Theory with efficient frontier optimization
        - Developed multi-method Value-at-Risk (VaR) calculation system
        - Created comprehensive backtesting framework with risk-adjusted metrics
        - Built Monte Carlo simulation engine for risk assessment
        """)
    
    with col2:
        st.markdown("""
        #### 🤖 Machine Learning & AI
        - Designed LSTM neural networks for time series forecasting
        - Implemented XGBoost ensemble models with feature engineering
        - Developed reinforcement learning trading agent using DQN
        - Created automated model comparison and selection pipeline
        """)
    
    # GitHub and Contact
    st.markdown("---")
    st.markdown("### 🔗 Links & Resources")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🔗 Repository**
        - [GitHub Repository](https://github.com/yourusername/finoptima)
        - [Documentation](https://finoptima-docs.com)
        """)
    
    with col2:
        st.markdown("""
        **📊 Live Demo**
        - [Streamlit Cloud](https://finoptima.streamlit.app)
        - [Video Demo](https://youtube.com/watch?v=demo)
        """)
    
    with col3:
        st.markdown("""
        **💼 Professional**
        - [LinkedIn Profile](https://linkedin.com/in/yourprofile)
        - [Portfolio Website](https://yourportfolio.com)
        """)

if __name__ == "__main__":
    main()