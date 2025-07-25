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
    """Main application function"""
    # Sidebar configuration
    with st.sidebar:
        # Logo and title
        st.markdown("""
        <div class="sidebar-header">
            <h1>📈 FinOptima</h1>
            <p>AI-Powered Financial Optimization Suite</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation
        st.markdown("### 🧭 Navigation")
        section = st.radio(
            "Dashboard Sections",
            [
                "🏠 Overview",
                "📊 Portfolio Optimizer", 
                "⚠️ Risk Analyzer",
                "📈 AI Stock Predictor",
                "🤖 RL Trading Agent", 
                "📉 Backtest Report",
                "📘 About Project"
            ],
            key="navigation"
        )
        
        st.markdown("---")
        
        # Global settings
        st.markdown("### ⚙️ Global Settings")
        
        # Date range selector
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=365*2),
                key="start_date"
            )
        with col2:
            end_date = st.date_input(
                "End Date", 
                value=datetime.now(),
                key="end_date"
            )
        
        # Stock ticker selector
        default_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX"]
        
        tickers = st.multiselect(
            "Select Stock Tickers",
            options=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", 
                    "BRK-B", "V", "JNJ", "WMT", "JPM", "PG", "UNH", "DIS", "HD", "MA",
                    "CRM", "BAC", "ADBE", "CMCSA", "XOM", "VZ", "KO", "PEP", "TMO", "ABT"],
            default=default_tickers[:4],
            help="Select stocks for analysis (3-8 recommended)"
        )
        
        # Risk-free rate
        risk_free_rate = st.slider(
            "Risk-Free Rate (%)",
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            step=0.1,
            help="Current risk-free rate for calculations"
        ) / 100
        
        st.markdown("---")
        
        # Quick stats
        if tickers:
            with st.spinner("Loading market data..."):
                try:
                    stock_data = get_stock_data(tickers, start_date, end_date)
                    if stock_data:
                        st.markdown("### 📊 Quick Stats")
                        
                        # Market summary
                        total_return = 0
                        for ticker in tickers:
                            if ticker in stock_data:
                                data = stock_data[ticker]
                                if len(data) > 1:
                                    ret = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100
                                    total_return += ret / len(tickers)
                        
                        st.metric("Avg Portfolio Return", f"{total_return:.2f}%")
                        st.metric("Analysis Period", f"{len(stock_data[tickers[0]]) if tickers[0] in stock_data else 0} days")
                        st.metric("Active Tickers", len([t for t in tickers if t in stock_data]))
                        
                except Exception as e:
                    st.error("⚠️ Error loading market data")
        
        st.markdown("---")
        st.markdown("""
        <div class="sidebar-footer">
            <p>💡 <strong>Pro Tip:</strong> Use multiple modules together for comprehensive analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content area
    if not tickers:
        # Landing page
        st.markdown("""
        <div class="main-header">
            <h1>📈 FinOptima</h1>
            <h2>AI-Powered Financial Optimization Suite</h2>
            <p>Welcome to the most comprehensive financial analysis platform built with cutting-edge AI and modern portfolio theory.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.warning("🔍 Please select stock tickers in the sidebar to begin analysis.")
        
        # Feature overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 📊 Portfolio Optimization
            - Markowitz Mean-Variance Optimization
            - Efficient Frontier Analysis
            - Risk-Return Optimization
            - Constraint-Based Allocation
            """)
            
        with col2:
            st.markdown("""
            ### 🤖 AI Prediction Models
            - LSTM Neural Networks
            - XGBoost Ensemble
            - CNN-LSTM Hybrid
            - Feature Engineering
            """)
            
        with col3:
            st.markdown("""
            ### ⚠️ Risk Management
            - Value-at-Risk (VaR)
            - Stress Testing
            - Monte Carlo Simulation
            - Correlation Analysis
            """)
        
        # Technology showcase
        st.markdown("---")
        st.markdown("### 🚀 Technology Stack")
        
        tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)
        
        with tech_col1:
            st.info("**Frontend**\n\nStreamlit\nPlotly\nCustom CSS")
        with tech_col2:
            st.info("**Data Science**\n\nPandas\nNumPy\nSciPy")
        with tech_col3:
            st.info("**Machine Learning**\n\nTensorFlow\nXGBoost\nStable-Baselines3")
        with tech_col4:
            st.info("**Finance**\n\nYahoo Finance\nCVXPY\nTA-Lib")
        
        return
    
    # Load stock data
    with st.spinner("🔄 Loading financial data..."):
        try:
            stock_data = get_stock_data(tickers, start_date, end_date)
            if not stock_data:
                st.error("❌ Failed to load stock data. Please check your tickers and date range.")
                return
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            return
    
    # Route to appropriate section
    if section == "🏠 Overview":
        render_overview(stock_data, tickers, start_date, end_date)
    
    elif section == "📊 Portfolio Optimizer":
        optimizer = PortfolioOptimizer()
        optimizer.render(stock_data, tickers, start_date, end_date, risk_free_rate)
    
    elif section == "⚠️ Risk Analyzer":
        analyzer = RiskAnalyzer()
        analyzer.render(stock_data, tickers)
    
    elif section == "📈 AI Stock Predictor":
        predictor = StockPredictor()
        predictor.render(tickers, start_date, end_date)
    
    elif section == "🤖 RL Trading Agent":
        rl_agent = RLAgent()
        rl_agent.render(tickers, start_date, end_date)
    
    elif section == "📉 Backtest Report":
        backtest = BacktestReport()
        backtest.render(stock_data, tickers, start_date, end_date)
    
    elif section == "📘 About Project":
        render_about_project()

def render_overview(stock_data, tickers, start_date, end_date):
    """Render the overview dashboard"""
    st.markdown("""
    <div class="main-header">
        <h1>🏠 Portfolio Overview</h1>
        <p>Comprehensive market analysis and portfolio insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate portfolio metrics
    portfolio_return = 0
    portfolio_volatility = 0
    total_value = 0
    
    for ticker in tickers:
        if ticker in stock_data:
            data = stock_data[ticker]
            if len(data) > 1:
                returns = data['Close'].pct_change().dropna()
                ret = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100
                vol = returns.std() * np.sqrt(252) * 100
                portfolio_return += ret / len(tickers)
                portfolio_volatility += vol / len(tickers)
                total_value += data['Close'].iloc[-1]
    
    with col1:
        st.metric(
            "Portfolio Return",
            f"{portfolio_return:.2f}%",
            delta=f"{portfolio_return - 5:.2f}% vs Market" if portfolio_return > 5 else None
        )
    
    with col2:
        st.metric(
            "Portfolio Volatility", 
            f"{portfolio_volatility:.2f}%",
            help="Annualized volatility"
        )
    
    with col3:
        sharpe = (portfolio_return - 2) / portfolio_volatility if portfolio_volatility > 0 else 0
        st.metric(
            "Sharpe Ratio",
            f"{sharpe:.3f}",
            help="Risk-adjusted return metric"
        )
    
    with col4:
        st.metric(
            "Total Portfolio Value",
            f"${total_value:,.2f}",
            help="Sum of current stock prices"
        )
    
    # Charts section
    col1, col2 = st.columns(2)
    
    with col1:
        # Price performance chart
        st.markdown("### 📈 Price Performance")
        
        fig = go.Figure()
        
        for ticker in tickers:
            if ticker in stock_data:
                data = stock_data[ticker]
                normalized_prices = (data['Close'] / data['Close'].iloc[0]) * 100
                
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=normalized_prices,
                    mode='lines',
                    name=ticker,
                    line=dict(width=2)
                ))
        
        fig.update_layout(
            title="Normalized Price Performance (Base = 100)",
            xaxis_title="Date",
            yaxis_title="Normalized Price",
            template="plotly_dark",
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Volume analysis
        st.markdown("### 📊 Volume Analysis")
        
        fig = go.Figure()
        
        for ticker in tickers:
            if ticker in stock_data:
                data = stock_data[ticker]
                avg_volume = data['Volume'].rolling(20).mean()
                
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=avg_volume,
                    mode='lines',
                    name=f"{ticker} Volume",
                    line=dict(width=2)
                ))
        
        fig.update_layout(
            title="20-Day Average Volume",
            xaxis_title="Date", 
            yaxis_title="Volume",
            template="plotly_dark",
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Returns distribution
    st.markdown("### 📊 Returns Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Daily returns histogram
        fig = go.Figure()
        
        for ticker in tickers:
            if ticker in stock_data:
                data = stock_data[ticker]
                returns = data['Close'].pct_change().dropna() * 100
                
                fig.add_trace(go.Histogram(
                    x=returns,
                    name=ticker,
                    opacity=0.7,
                    nbinsx=50
                ))
        
        fig.update_layout(
            title="Daily Returns Distribution",
            xaxis_title="Daily Return (%)",
            yaxis_title="Frequency",
            template="plotly_dark",
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            barmode='overlay'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Correlation heatmap
        if len(tickers) > 1:
            returns_data = {}
            for ticker in tickers:
                if ticker in stock_data:
                    returns_data[ticker] = stock_data[ticker]['Close'].pct_change().dropna()
            
            if len(returns_data) > 1:
                returns_df = pd.DataFrame(returns_data)
                correlation_matrix = returns_df.corr()
                
                fig = go.Figure(data=go.Heatmap(
                    z=correlation_matrix.values,
                    x=correlation_matrix.columns,
                    y=correlation_matrix.index,
                    colorscale='RdBu',
                    zmid=0,
                    text=correlation_matrix.round(3).values,
                    texttemplate="%{text}",
                    textfont={"size": 12},
                    hoverangles="closest"
                ))
                
                fig.update_layout(
                    title="Returns Correlation Matrix",
                    template="plotly_dark",
                    height=400,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Market data table
    st.markdown("### 📋 Market Data Summary")
    
    summary_data = []
    for ticker in tickers:
        if ticker in stock_data:
            data = stock_data[ticker]
            if len(data) > 1:
                current_price = data['Close'].iloc[-1]
                returns = data['Close'].pct_change().dropna()
                total_return = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100
                volatility = returns.std() * np.sqrt(252) * 100
                avg_volume = data['Volume'].mean()
                
                summary_data.append({
                    'Ticker': ticker,
                    'Current Price': f"${current_price:.2f}",
                    'Total Return': f"{total_return:.2f}%",
                    'Volatility': f"{volatility:.2f}%",
                    'Avg Volume': f"{avg_volume:,.0f}",
                    'Data Points': len(data)
                })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)

def render_about_project():
    """Render the About Project section"""
    st.markdown("""
    <div class="main-header">
        <h1>📘 About FinOptima</h1>
        <p>AI-Powered Financial Optimization Suite</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Project overview
    st.markdown("""
    ## 🚀 Project Overview
    
    FinOptima is a comprehensive financial analysis platform that combines cutting-edge artificial intelligence 
    with modern portfolio theory to provide institutional-grade financial analytics and optimization tools.
    
    ### 🎯 Key Features
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 📊 Portfolio Optimization
        - **Markowitz Mean-Variance Optimization**
        - **Efficient Frontier Analysis**
        - **Risk-Return Optimization**
        - **Constraint-Based Asset Allocation**
        
        #### 🤖 AI & Machine Learning
        - **LSTM Neural Networks** for price prediction
        - **XGBoost Ensemble** models
        - **CNN-LSTM Hybrid** architectures
        - **Reinforcement Learning** trading agents
        """)
    
    with col2:
        st.markdown("""
        #### ⚠️ Risk Management
        - **Value-at-Risk (VaR)** calculation
        - **Monte Carlo Simulation**
        - **Stress Testing**
        - **Correlation Analysis**
        
        #### 📈 Advanced Analytics
        - **Comprehensive Backtesting**
        - **Performance Attribution**
        - **Drawdown Analysis**
        - **Professional Reporting**
        """)
    
    # Technology architecture
    st.markdown("---")
    st.markdown("## 🏗️ Technology Architecture")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### Frontend
        - **Streamlit**: Interactive web application
        - **Plotly**: Professional financial charts
        - **Custom CSS**: Modern fintech styling
        """)
    
    with col2:
        st.markdown("""
        ### Data Science
        - **Pandas**: Financial data manipulation
        - **NumPy**: Numerical computations
        - **SciPy**: Scientific computing
        """)
    
    with col3:
        st.markdown("""
        ### Machine Learning
        - **TensorFlow**: Deep learning models
        - **XGBoost**: Gradient boosting
        - **Stable-Baselines3**: Reinforcement learning
        """)
    
    # Model descriptions
    st.markdown("---")
    st.markdown("## 🧠 AI Models & Algorithms")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Portfolio Optimization", "Stock Prediction", "RL Trading", "Risk Analysis"])
    
    with tab1:
        st.markdown("""
        ### 📊 Modern Portfolio Theory
        
        **Markowitz Mean-Variance Optimization:**
        - Maximizes expected return for a given level of risk
        - Minimizes risk for a given level of expected return
        - Calculates efficient frontier of optimal portfolios
        
        **Implementation:**
        ```python
        # Objective function: minimize portfolio variance
        w^T * Σ * w
        
        # Subject to constraints:
        # - Sum of weights = 1
        # - Optional: No short selling (w_i >= 0)
        # - Optional: Maximum position size limits
        ```
        
        **Key Metrics:**
        - Expected Return
        - Portfolio Variance/Volatility
        - Sharpe Ratio
        - Maximum Drawdown
        """)
    
    with tab2:
        st.markdown("""
        ### 📈 AI Stock Prediction Models
        
        **LSTM Neural Networks:**
        - Long Short-Term Memory networks for time series
        - Captures long-term dependencies in price data
        - Advanced feature engineering with technical indicators
        
        **XGBoost Ensemble:**
        - Gradient boosting for structured data
        - Feature importance analysis
        - Hyperparameter optimization
        
        **CNN-LSTM Hybrid:**
        - Convolutional layers for pattern extraction
        - LSTM layers for sequence modeling
        - State-of-the-art architecture for financial time series
        
        **Performance Metrics:**
        - R² Score (Coefficient of Determination)
        - MAPE (Mean Absolute Percentage Error)
        - Directional Accuracy
        - Sharpe Ratio of predictions
        """)
    
    with tab3:
        st.markdown("""
        ### 🤖 Reinforcement Learning Trading
        
        **Supported Algorithms:**
        - **DQN**: Deep Q-Network for discrete actions
        - **PPO**: Proximal Policy Optimization
        - **A2C**: Advantage Actor-Critic
        - **SAC**: Soft Actor-Critic
        
        **Custom Trading Environment:**
        - Realistic transaction costs
        - Market impact modeling
        - Risk management constraints
        - Performance-based reward function
        
        **Action Space:**
        - Buy: Increase position
        - Hold: Maintain current position
        - Sell: Decrease position
        
        **State Space:**
        - Price history
        - Technical indicators
        - Market context features
        - Portfolio status
        """)
    
    with tab4:
        st.markdown("""
        ### ⚠️ Risk Management Systems
        
        **Value-at-Risk (VaR) Methods:**
        - **Historical VaR**: Based on historical returns
        - **Monte Carlo VaR**: Simulation-based approach
        - **Parametric VaR**: Assumes normal distribution
        
        **Advanced Risk Metrics:**
        - Conditional VaR (Expected Shortfall)
        - Maximum Drawdown
        - Volatility clustering analysis
        - Correlation breakdown
        
        **Stress Testing:**
        - Scenario analysis
        - Historical stress events
        - Custom shock scenarios
        - Portfolio resilience testing
        """)
    
    # Contact and links
    st.markdown("---")
    st.markdown("## 🔗 Links & Contact")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📚 Documentation
        - [GitHub Repository](https://github.com/yourusername/finoptima)
        - [Technical Documentation](https://finoptima.readthedocs.io)
        - [API Reference](https://api.finoptima.com)
        """)
    
    with col2:
        st.markdown("""
        ### 🤝 Connect
        - [LinkedIn Profile](https://linkedin.com/in/yourprofile)
        - [Personal Website](https://yourwebsite.com)
        - [Email Contact](mailto:your.email@domain.com)
        """)
    
    with col3:
        st.markdown("""
        ### 🏆 Achievements
        - Production-ready financial platform
        - 6,700+ lines of code
        - Enterprise-grade architecture
        - Industry-standard algorithms
        """)

# Footer
def render_footer():
    """Render application footer"""
    st.markdown("---")
    st.markdown("""
    <div class="app-footer">
        <p>📈 FinOptima | AI-Powered Financial Optimization Suite</p>
        <p>Built with Streamlit, TensorFlow, and Modern Portfolio Theory</p>
        <p>© 2024 FinOptima. Designed for financial professionals and quantitative analysts.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    render_footer()