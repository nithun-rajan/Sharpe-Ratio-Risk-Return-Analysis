import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

def load_custom_css():
    """Load custom CSS for FinOptima fintech styling"""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;600;700&family=Roboto:wght@300;400;500&display=swap');
    
    /* Root variables for theme colors */
    :root {
        --primary-color: #00C49F;
        --secondary-color: #1F2633;
        --accent-color: #FF8042;
        --background-dark: #0E1117;
        --text-light: #FAFAFA;
        --text-secondary: #A0A0A0;
        --card-background: #1E1E1E;
        --border-color: #333333;
    }
    
    /* Main app styling */
    .stApp {
        background: linear-gradient(135deg, var(--background-dark) 0%, #1a1a1a 100%);
        font-family: 'Roboto', sans-serif;
    }
    
    /* Sidebar styling */
    .sidebar-header {
        text-align: center;
        padding: 1rem 0;
        border-bottom: 2px solid var(--primary-color);
        margin-bottom: 1rem;
    }
    
    .sidebar-header h1 {
        color: var(--primary-color);
        font-family: 'Montserrat', sans-serif;
        font-weight: 700;
        font-size: 2rem;
        margin: 0;
    }
    
    .sidebar-header p {
        color: var(--text-secondary);
        font-size: 0.9rem;
        margin: 0.5rem 0 0 0;
    }
    
    .sidebar-footer {
        text-align: center;
        padding: 1rem 0;
        border-top: 1px solid var(--border-color);
        margin-top: 1rem;
    }
    
    .sidebar-footer p {
        color: var(--text-secondary);
        font-size: 0.8rem;
        margin: 0.2rem 0;
    }
    
    /* Main content headers */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 2rem;
        background: linear-gradient(45deg, var(--primary-color), var(--accent-color));
        border-radius: 15px;
        color: white;
    }
    
    .main-header h1 {
        font-family: 'Montserrat', sans-serif;
        font-weight: 700;
        font-size: 2.5rem;
        margin: 0;
    }
    
    .main-header p {
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Metric cards */
    .metric-card {
        background: var(--card-background);
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: var(--primary-color);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary-color);
        font-family: 'Montserrat', sans-serif;
    }
    
    .metric-label {
        color: var(--text-secondary);
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    
    /* Success/Error states */
    .success-text {
        color: var(--primary-color);
        font-weight: 600;
    }
    
    .error-text {
        color: var(--accent-color);
        font-weight: 600;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, var(--primary-color), #00A085);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(45deg, #00A085, var(--primary-color));
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 196, 159, 0.3);
    }
    
    /* Plotly chart container */
    .plotly-chart {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Custom tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: var(--card-background);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        color: var(--text-light);
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--primary-color);
        border-color: var(--primary-color);
    }
    
    /* Info boxes */
    .stInfo {
        background: rgba(0, 196, 159, 0.1);
        border-left: 4px solid var(--primary-color);
    }
    
    .stWarning {
        background: rgba(255, 128, 66, 0.1);
        border-left: 4px solid var(--accent-color);
    }
    
    .stError {
        background: rgba(255, 128, 66, 0.1);
        border-left: 4px solid var(--accent-color);
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    </style>
    """, unsafe_allow_html=True)

def get_stock_data(tickers, start_date, end_date):
    """
    Fetch stock data for given tickers and date range
    
    Args:
        tickers (list): List of stock ticker symbols
        start_date (datetime): Start date for data
        end_date (datetime): End date for data
    
    Returns:
        dict: Dictionary with ticker as key and DataFrame as value
    """
    stock_data = {}
    
    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if not data.empty:
                stock_data[ticker] = data
            else:
                st.warning(f"No data found for {ticker}")
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {str(e)}")
    
    return stock_data

def calculate_returns(prices):
    """Calculate daily returns from price series"""
    return prices.pct_change().dropna()

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """
    Calculate Sharpe ratio
    
    Args:
        returns (pd.Series): Daily returns
        risk_free_rate (float): Annual risk-free rate
    
    Returns:
        float: Sharpe ratio
    """
    excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
    return np.sqrt(252) * excess_returns.mean() / returns.std()

def calculate_sortino_ratio(returns, risk_free_rate=0.02):
    """
    Calculate Sortino ratio (downside deviation)
    
    Args:
        returns (pd.Series): Daily returns
        risk_free_rate (float): Annual risk-free rate
    
    Returns:
        float: Sortino ratio
    """
    excess_returns = returns - (risk_free_rate / 252)
    downside_returns = excess_returns[excess_returns < 0]
    downside_deviation = np.sqrt(252) * downside_returns.std()
    
    if downside_deviation == 0:
        return np.inf
    
    return np.sqrt(252) * excess_returns.mean() / downside_deviation

def calculate_max_drawdown(cumulative_returns):
    """
    Calculate maximum drawdown
    
    Args:
        cumulative_returns (pd.Series): Cumulative returns series
    
    Returns:
        float: Maximum drawdown percentage
    """
    rolling_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    return drawdown.min()

def format_percentage(value, decimals=2):
    """Format value as percentage"""
    return f"{value * 100:.{decimals}f}%"

def format_currency(value, decimals=2):
    """Format value as currency"""
    return f"${value:,.{decimals}f}"

def create_metric_card(title, value, delta=None, help_text=None):
    """
    Create a styled metric card
    
    Args:
        title (str): Metric title
        value (str): Metric value
        delta (str, optional): Change indicator
        help_text (str, optional): Help tooltip text
    """
    delta_html = ""
    if delta:
        delta_color = "var(--primary-color)" if "+" in str(delta) else "var(--accent-color)"
        delta_html = f'<div style="color: {delta_color}; font-size: 0.9rem; margin-top: 0.5rem;">{delta}</div>'
    
    help_html = ""
    if help_text:
        help_html = f'<div style="color: var(--text-secondary); font-size: 0.8rem; margin-top: 0.5rem;">ℹ️ {help_text}</div>'
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{title}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
        {help_html}
    </div>
    """, unsafe_allow_html=True)

def plot_candlestick(data, title, ticker):
    """
    Create candlestick chart
    
    Args:
        data (pd.DataFrame): OHLCV data
        title (str): Chart title
        ticker (str): Stock ticker
    
    Returns:
        plotly.graph_objects.Figure: Candlestick chart
    """
    fig = go.Figure(data=go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name=ticker
    ))
    
    fig.update_layout(
        title=title,
        yaxis_title="Price ($)",
        xaxis_title="Date",
        template="plotly_dark",
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig

def plot_correlation_heatmap(returns_data):
    """
    Create correlation heatmap
    
    Args:
        returns_data (pd.DataFrame): Returns data for multiple assets
    
    Returns:
        plotly.graph_objects.Figure: Correlation heatmap
    """
    correlation_matrix = returns_data.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        colorscale='RdYlGn',
        zmid=0,
        text=correlation_matrix.round(2).values,
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverangles="closest"
    ))
    
    fig.update_layout(
        title="Asset Correlation Matrix",
        template="plotly_dark",
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig

def add_technical_indicators(data):
    """
    Add technical indicators to price data
    
    Args:
        data (pd.DataFrame): OHLCV data
    
    Returns:
        pd.DataFrame: Data with technical indicators
    """
    # Moving averages
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['EMA_12'] = data['Close'].ewm(span=12).mean()
    data['EMA_26'] = data['Close'].ewm(span=26).mean()
    
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
    data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
    
    # Bollinger Bands
    data['BB_Middle'] = data['Close'].rolling(window=20).mean()
    bb_std = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
    data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
    
    # Volatility
    data['Volatility'] = data['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
    
    return data

def validate_data(data, min_periods=252):
    """
    Validate if data has sufficient periods for analysis
    
    Args:
        data (pd.DataFrame): Stock data
        min_periods (int): Minimum required periods
    
    Returns:
        bool: True if data is valid
    """
    return len(data) >= min_periods

def get_risk_free_rate():
    """
    Get current risk-free rate (10-year Treasury)
    
    Returns:
        float: Risk-free rate
    """
    try:
        treasury = yf.download('^TNX', period='1d', progress=False)
        if not treasury.empty:
            return treasury['Close'].iloc[-1] / 100
    except:
        pass
    
    # Default risk-free rate if fetch fails
    return 0.02