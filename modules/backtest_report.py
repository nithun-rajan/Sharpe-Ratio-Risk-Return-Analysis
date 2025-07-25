import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from fpdf import FPDF
import io
from datetime import datetime, timedelta
from .utils import (calculate_returns, calculate_sharpe_ratio, calculate_sortino_ratio, 
                   calculate_max_drawdown, format_percentage, format_currency, create_metric_card)

class BacktestReport:
    def __init__(self):
        self.performance_data = {}
        
    def render(self, stock_data, tickers, start_date, end_date):
        """Render the comprehensive backtest report section"""
        st.markdown("""
        <div class="main-header">
            <h1>📉 Backtest Report</h1>
            <p>Comprehensive Strategy Performance Analytics & Risk Assessment</p>
        </div>
        """, unsafe_allow_html=True)
        
        if not stock_data:
            st.error("⚠️ No stock data available. Please check your ticker selection.")
            return
        
        # Strategy configuration
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### ⚙️ Backtest Configuration")
            
            # Strategy selection
            strategy_type = st.selectbox(
                "Strategy Type",
                [
                    "Buy & Hold",
                    "Moving Average Crossover",
                    "RSI Mean Reversion",
                    "Bollinger Bands",
                    "MACD Strategy",
                    "Momentum Strategy",
                    "Custom Portfolio"
                ],
                help="Select trading strategy for backtesting"
            )
            
            # Portfolio allocation
            if strategy_type == "Custom Portfolio":
                st.markdown("#### 💼 Portfolio Allocation")
                portfolio_weights = {}
                remaining_weight = 100.0
                
                for i, ticker in enumerate(tickers):
                    if i == len(tickers) - 1:
                        # Last ticker gets remaining weight
                        weight = remaining_weight
                        st.write(f"{ticker}: {weight:.1f}% (auto-assigned)")
                    else:
                        weight = st.slider(
                            f"{ticker} Weight (%)",
                            min_value=0.0,
                            max_value=remaining_weight,
                            value=min(100.0 / len(tickers), remaining_weight),
                            step=1.0
                        )
                        remaining_weight -= weight
                    
                    portfolio_weights[ticker] = weight / 100.0
            
            # Backtest parameters
            st.markdown("#### 📊 Parameters")
            
            initial_capital = st.number_input(
                "Initial Capital ($)",
                min_value=10000,
                max_value=10000000,
                value=100000,
                step=10000
            )
            
            transaction_cost = st.slider(
                "Transaction Cost (%)",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.05
            ) / 100
            
            # Strategy-specific parameters
            if strategy_type == "Moving Average Crossover":
                short_window = st.slider("Short MA Period", 5, 50, 20)
                long_window = st.slider("Long MA Period", 20, 200, 50)
            
            elif strategy_type == "RSI Mean Reversion":
                rsi_period = st.slider("RSI Period", 5, 30, 14)
                oversold = st.slider("Oversold Threshold", 10, 40, 30)
                overbought = st.slider("Overbought Threshold", 60, 90, 70)
            
            elif strategy_type == "Bollinger Bands":
                bb_period = st.slider("BB Period", 10, 50, 20)
                bb_std = st.slider("Standard Deviations", 1.0, 3.0, 2.0, 0.1)
            
            elif strategy_type == "MACD Strategy":
                fast_ema = st.slider("Fast EMA", 5, 20, 12)
                slow_ema = st.slider("Slow EMA", 20, 50, 26)
                signal_ema = st.slider("Signal EMA", 5, 15, 9)
        
        with col2:
            # Run backtest
            if st.button("🚀 Run Comprehensive Backtest", type="primary"):
                with st.spinner("Running detailed backtest analysis..."):
                    # Prepare strategy parameters
                    strategy_params = {
                        'initial_capital': initial_capital,
                        'transaction_cost': transaction_cost
                    }
                    
                    if strategy_type == "Moving Average Crossover":
                        strategy_params.update({
                            'short_window': short_window,
                            'long_window': long_window
                        })
                    elif strategy_type == "RSI Mean Reversion":
                        strategy_params.update({
                            'rsi_period': rsi_period,
                            'oversold': oversold,
                            'overbought': overbought
                        })
                    elif strategy_type == "Custom Portfolio":
                        strategy_params['weights'] = portfolio_weights
                    
                    # Run backtest
                    backtest_results = self._run_backtest(
                        stock_data, strategy_type, strategy_params, tickers
                    )
                    
                    if backtest_results:
                        st.session_state.backtest_results = backtest_results
                        st.session_state.strategy_type = strategy_type
                        st.success("✅ Backtest completed successfully!")
            
            # Display summary if available
            if hasattr(st.session_state, 'backtest_results'):
                self._display_backtest_summary()
        
        # Backtest Analytics Tabs
        st.markdown("---")
        st.markdown("### 📊 Backtest Analytics")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Performance Overview",
            "📊 Risk Analysis", 
            "🎯 Strategy Comparison",
            "📉 Drawdown Analysis",
            "📁 Reports & Export"
        ])
        
        with tab1:
            self._render_performance_tab()
        
        with tab2:
            self._render_risk_analysis_tab()
        
        with tab3:
            self._render_strategy_comparison_tab(stock_data, tickers)
        
        with tab4:
            self._render_drawdown_tab()
        
        with tab5:
            self._render_export_tab()
    
    def _run_backtest(self, stock_data, strategy_type, params, tickers):
        """Run comprehensive backtest analysis"""
        try:
            if strategy_type == "Buy & Hold":
                return self._backtest_buy_hold(stock_data, params, tickers)
            elif strategy_type == "Moving Average Crossover":
                return self._backtest_ma_crossover(stock_data, params, tickers[0])
            elif strategy_type == "RSI Mean Reversion":
                return self._backtest_rsi_strategy(stock_data, params, tickers[0])
            elif strategy_type == "Bollinger Bands":
                return self._backtest_bollinger_bands(stock_data, params, tickers[0])
            elif strategy_type == "MACD Strategy":
                return self._backtest_macd_strategy(stock_data, params, tickers[0])
            elif strategy_type == "Momentum Strategy":
                return self._backtest_momentum_strategy(stock_data, params, tickers[0])
            elif strategy_type == "Custom Portfolio":
                return self._backtest_custom_portfolio(stock_data, params, tickers)
            
        except Exception as e:
            st.error(f"❌ Backtest failed: {str(e)}")
            return None
    
    def _backtest_buy_hold(self, stock_data, params, tickers):
        """Backtest buy and hold strategy"""
        initial_capital = params['initial_capital']
        transaction_cost = params['transaction_cost']
        
        # Equal weight allocation
        weight_per_stock = 1.0 / len(tickers)
        portfolio_values = []
        trades = []
        
        # Get aligned data
        aligned_data = {}
        min_date = None
        max_date = None
        
        for ticker in tickers:
            data = stock_data[ticker]
            aligned_data[ticker] = data
            if min_date is None or data.index[0] > min_date:
                min_date = data.index[0]
            if max_date is None or data.index[-1] < max_date:
                max_date = data.index[-1]
        
        # Create date range
        dates = pd.date_range(start=min_date, end=max_date, freq='D')
        dates = [d for d in dates if any(d in aligned_data[t].index for t in tickers)]
        
        # Calculate portfolio value over time
        for date in dates:
            total_value = 0
            for ticker in tickers:
                if date in aligned_data[ticker].index:
                    price = aligned_data[ticker].loc[date, 'Close']
                    shares = (initial_capital * weight_per_stock) / aligned_data[ticker].iloc[0]['Close']
                    total_value += shares * price
            
            portfolio_values.append(total_value)
        
        # Calculate performance metrics
        returns = pd.Series(portfolio_values).pct_change().dropna()
        
        return {
            'strategy': 'Buy & Hold',
            'portfolio_values': portfolio_values,
            'dates': dates,
            'returns': returns,
            'trades': trades,
            'initial_capital': initial_capital,
            'final_value': portfolio_values[-1] if portfolio_values else initial_capital,
            'total_return': (portfolio_values[-1] - initial_capital) / initial_capital if portfolio_values else 0,
            'sharpe_ratio': calculate_sharpe_ratio(returns),
            'max_drawdown': calculate_max_drawdown(pd.Series(portfolio_values) / initial_capital),
            'volatility': returns.std() * np.sqrt(252)
        }
    
    def _backtest_ma_crossover(self, stock_data, params, ticker):
        """Backtest moving average crossover strategy"""
        data = stock_data[ticker].copy()
        initial_capital = params['initial_capital']
        short_window = params['short_window']
        long_window = params['long_window']
        transaction_cost = params['transaction_cost']
        
        # Calculate moving averages
        data['MA_Short'] = data['Close'].rolling(window=short_window).mean()
        data['MA_Long'] = data['Close'].rolling(window=long_window).mean()
        
        # Generate signals
        data['Signal'] = 0
        data.loc[data['MA_Short'] > data['MA_Long'], 'Signal'] = 1
        data['Position'] = data['Signal'].diff()
        
        # Backtest
        portfolio_value = initial_capital
        cash = initial_capital
        shares = 0
        portfolio_values = []
        trades = []
        
        for i, (date, row) in enumerate(data.iterrows()):
            if i < long_window:  # Skip initial period
                portfolio_values.append(portfolio_value)
                continue
            
            # Check for buy signal
            if row['Position'] == 1 and cash > 0:
                shares_to_buy = int(cash / (row['Close'] * (1 + transaction_cost)))
                if shares_to_buy > 0:
                    cost = shares_to_buy * row['Close'] * (1 + transaction_cost)
                    cash -= cost
                    shares += shares_to_buy
                    trades.append({
                        'date': date,
                        'action': 'BUY',
                        'price': row['Close'],
                        'shares': shares_to_buy,
                        'value': cost
                    })
            
            # Check for sell signal
            elif row['Position'] == -1 and shares > 0:
                revenue = shares * row['Close'] * (1 - transaction_cost)
                cash += revenue
                trades.append({
                    'date': date,
                    'action': 'SELL',
                    'price': row['Close'],
                    'shares': shares,
                    'value': revenue
                })
                shares = 0
            
            # Calculate portfolio value
            portfolio_value = cash + shares * row['Close']
            portfolio_values.append(portfolio_value)
        
        # Calculate metrics
        returns = pd.Series(portfolio_values).pct_change().dropna()
        
        return {
            'strategy': f'MA Crossover ({short_window}/{long_window})',
            'portfolio_values': portfolio_values,
            'dates': data.index.tolist(),
            'returns': returns,
            'trades': trades,
            'signals': data[['MA_Short', 'MA_Long', 'Signal']],
            'initial_capital': initial_capital,
            'final_value': portfolio_values[-1],
            'total_return': (portfolio_values[-1] - initial_capital) / initial_capital,
            'sharpe_ratio': calculate_sharpe_ratio(returns),
            'max_drawdown': calculate_max_drawdown(pd.Series(portfolio_values) / initial_capital),
            'volatility': returns.std() * np.sqrt(252),
            'num_trades': len(trades)
        }
    
    def _backtest_rsi_strategy(self, stock_data, params, ticker):
        """Backtest RSI mean reversion strategy"""
        data = stock_data[ticker].copy()
        initial_capital = params['initial_capital']
        rsi_period = params['rsi_period']
        oversold = params['oversold']
        overbought = params['overbought']
        transaction_cost = params['transaction_cost']
        
        # Calculate RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Generate signals
        data['Signal'] = 0
        data.loc[data['RSI'] < oversold, 'Signal'] = 1  # Buy signal
        data.loc[data['RSI'] > overbought, 'Signal'] = -1  # Sell signal
        
        # Backtest logic similar to MA crossover
        portfolio_value = initial_capital
        cash = initial_capital
        shares = 0
        portfolio_values = []
        trades = []
        position = 0  # 0: no position, 1: long
        
        for i, (date, row) in enumerate(data.iterrows()):
            if i < rsi_period:
                portfolio_values.append(portfolio_value)
                continue
            
            # Buy signal (oversold)
            if row['Signal'] == 1 and position == 0:
                shares_to_buy = int(cash / (row['Close'] * (1 + transaction_cost)))
                if shares_to_buy > 0:
                    cost = shares_to_buy * row['Close'] * (1 + transaction_cost)
                    cash -= cost
                    shares += shares_to_buy
                    position = 1
                    trades.append({
                        'date': date,
                        'action': 'BUY',
                        'price': row['Close'],
                        'shares': shares_to_buy,
                        'rsi': row['RSI']
                    })
            
            # Sell signal (overbought)
            elif row['Signal'] == -1 and position == 1:
                revenue = shares * row['Close'] * (1 - transaction_cost)
                cash += revenue
                trades.append({
                    'date': date,
                    'action': 'SELL',
                    'price': row['Close'],
                    'shares': shares,
                    'rsi': row['RSI']
                })
                shares = 0
                position = 0
            
            portfolio_value = cash + shares * row['Close']
            portfolio_values.append(portfolio_value)
        
        returns = pd.Series(portfolio_values).pct_change().dropna()
        
        return {
            'strategy': f'RSI Mean Reversion ({oversold}/{overbought})',
            'portfolio_values': portfolio_values,
            'dates': data.index.tolist(),
            'returns': returns,
            'trades': trades,
            'indicators': data[['RSI', 'Signal']],
            'initial_capital': initial_capital,
            'final_value': portfolio_values[-1],
            'total_return': (portfolio_values[-1] - initial_capital) / initial_capital,
            'sharpe_ratio': calculate_sharpe_ratio(returns),
            'max_drawdown': calculate_max_drawdown(pd.Series(portfolio_values) / initial_capital),
            'volatility': returns.std() * np.sqrt(252),
            'num_trades': len(trades)
        }
    
    def _backtest_custom_portfolio(self, stock_data, params, tickers):
        """Backtest custom weighted portfolio"""
        weights = params['weights']
        initial_capital = params['initial_capital']
        
        # Calculate portfolio returns
        returns_data = {}
        for ticker in tickers:
            if ticker in stock_data and ticker in weights:
                returns_data[ticker] = calculate_returns(stock_data[ticker]['Close'])
        
        # Align dates
        returns_df = pd.DataFrame(returns_data).dropna()
        
        # Calculate portfolio returns
        portfolio_returns = sum(returns_df[ticker] * weights[ticker] 
                              for ticker in weights.keys() if ticker in returns_df.columns)
        
        # Calculate portfolio values
        portfolio_values = [initial_capital]
        for ret in portfolio_returns:
            portfolio_values.append(portfolio_values[-1] * (1 + ret))
        
        return {
            'strategy': 'Custom Weighted Portfolio',
            'portfolio_values': portfolio_values[1:],  # Remove initial value
            'dates': portfolio_returns.index.tolist(),
            'returns': portfolio_returns,
            'trades': [],  # No individual trades for portfolio
            'weights': weights,
            'initial_capital': initial_capital,
            'final_value': portfolio_values[-1],
            'total_return': (portfolio_values[-1] - initial_capital) / initial_capital,
            'sharpe_ratio': calculate_sharpe_ratio(portfolio_returns),
            'max_drawdown': calculate_max_drawdown(pd.Series(portfolio_values[1:]) / initial_capital),
            'volatility': portfolio_returns.std() * np.sqrt(252)
        }
    
    def _display_backtest_summary(self):
        """Display backtest summary metrics"""
        if not hasattr(st.session_state, 'backtest_results'):
            return
        
        results = st.session_state.backtest_results
        
        st.markdown("### 🎯 Backtest Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            create_metric_card(
                "Total Return",
                format_percentage(results['total_return']),
                help_text="Total portfolio return"
            )
        
        with col2:
            create_metric_card(
                "Sharpe Ratio",
                f"{results['sharpe_ratio']:.3f}",
                help_text="Risk-adjusted return"
            )
        
        with col3:
            create_metric_card(
                "Max Drawdown",
                format_percentage(results['max_drawdown']),
                help_text="Maximum portfolio decline"
            )
        
        with col4:
            create_metric_card(
                "Volatility",
                format_percentage(results['volatility']),
                help_text="Annualized volatility"
            )
    
    def _render_performance_tab(self):
        """Render performance overview tab"""
        if not hasattr(st.session_state, 'backtest_results'):
            st.info("🔄 Run a backtest first to see performance analysis.")
            return
        
        results = st.session_state.backtest_results
        
        # Portfolio value chart
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Portfolio Value Over Time',
                'Rolling Returns (30-day)',
                'Monthly Returns Heatmap',
                'Return Distribution'
            ),
            specs=[[{'colspan': 2}, None],
                   [{'type': 'xy'}, {'type': 'xy'}]]
        )
        
        # Portfolio value
        fig.add_trace(
            go.Scatter(
                x=results['dates'],
                y=results['portfolio_values'],
                name='Portfolio Value',
                line=dict(color='#00C49F', width=2)
            ),
            row=1, col=1
        )
        
        # Benchmark (buy and hold equal weight)
        benchmark_value = results['initial_capital']
        benchmark_values = [benchmark_value * (1 + results['returns'].iloc[:i+1].sum()) 
                          for i in range(len(results['returns']))]
        
        fig.add_trace(
            go.Scatter(
                x=results['dates'][-len(benchmark_values):],
                y=benchmark_values,
                name='Benchmark',
                line=dict(color='#FF8042', width=2, dash='dash')
            ),
            row=1, col=1
        )
        
        # Rolling returns
        rolling_returns = results['returns'].rolling(30).sum()
        fig.add_trace(
            go.Scatter(
                x=results['dates'][-len(rolling_returns):],
                y=rolling_returns * 100,
                name='30-day Rolling Return (%)',
                line=dict(color='#00C49F', width=1)
            ),
            row=2, col=1
        )
        
        # Return distribution
        fig.add_trace(
            go.Histogram(
                x=results['returns'] * 100,
                nbinsx=30,
                name='Daily Returns (%)',
                marker_color='#00C49F',
                opacity=0.7
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=700,
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance metrics table
        st.markdown("#### 📊 Detailed Performance Metrics")
        
        # Calculate additional metrics
        win_rate = (results['returns'] > 0).mean() * 100 if len(results['returns']) > 0 else 0
        avg_win = results['returns'][results['returns'] > 0].mean() * 100 if len(results['returns'][results['returns'] > 0]) > 0 else 0
        avg_loss = results['returns'][results['returns'] < 0].mean() * 100 if len(results['returns'][results['returns'] < 0]) > 0 else 0
        
        metrics_data = {
            'Metric': [
                'Total Return', 'Annualized Return', 'Sharpe Ratio', 'Sortino Ratio',
                'Max Drawdown', 'Volatility', 'Win Rate', 'Average Win', 'Average Loss',
                'Final Value', 'Number of Trades'
            ],
            'Value': [
                format_percentage(results['total_return']),
                format_percentage(results['total_return'] * 252 / len(results['returns']) if len(results['returns']) > 0 else 0),
                f"{results['sharpe_ratio']:.3f}",
                f"{calculate_sortino_ratio(results['returns']):.3f}",
                format_percentage(results['max_drawdown']),
                format_percentage(results['volatility']),
                f"{win_rate:.1f}%",
                f"{avg_win:.2f}%" if avg_win > 0 else "0.00%",
                f"{avg_loss:.2f}%" if avg_loss < 0 else "0.00%",
                format_currency(results['final_value']),
                str(results.get('num_trades', 'N/A'))
            ]
        }
        
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True)
    
    def _render_risk_analysis_tab(self):
        """Render risk analysis tab"""
        if not hasattr(st.session_state, 'backtest_results'):
            st.info("🔄 Run a backtest first to see risk analysis.")
            return
        
        results = st.session_state.backtest_results
        returns = results['returns']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # VaR calculation
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)
            cvar_95 = returns[returns <= var_95].mean()
            
            st.markdown("#### ⚠️ Value at Risk (VaR)")
            
            var_metrics = {
                'VaR (95%)': format_percentage(var_95),
                'VaR (99%)': format_percentage(var_99),
                'CVaR (95%)': format_percentage(cvar_95),
                'Worst Day': format_percentage(returns.min()),
                'Best Day': format_percentage(returns.max())
            }
            
            for metric, value in var_metrics.items():
                create_metric_card(metric, value)
        
        with col2:
            # Risk-return scatter
            fig = go.Figure()
            
            # Calculate rolling risk-return metrics
            window = 30
            rolling_returns = returns.rolling(window).mean() * 252
            rolling_vol = returns.rolling(window).std() * np.sqrt(252)
            
            fig.add_trace(go.Scatter(
                x=rolling_vol,
                y=rolling_returns,
                mode='markers',
                name='Risk-Return Profile',
                marker=dict(
                    size=8,
                    color=range(len(rolling_vol)),
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Time Period")
                )
            ))
            
            fig.update_layout(
                title="Rolling Risk-Return Profile (30-day windows)",
                xaxis_title="Volatility (Annualized)",
                yaxis_title="Return (Annualized)",
                template="plotly_dark",
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Drawdown analysis
        portfolio_values = pd.Series(results['portfolio_values'])
        rolling_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - rolling_max) / rolling_max
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=results['dates'],
            y=drawdown * 100,
            fill='tonexty',
            mode='lines',
            name='Drawdown (%)',
            line=dict(color='#FF8042'),
            fillcolor='rgba(255, 128, 66, 0.3)'
        ))
        
        fig.update_layout(
            title="Portfolio Drawdown Over Time",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            template="plotly_dark",
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_strategy_comparison_tab(self, stock_data, tickers):
        """Render strategy comparison tab"""
        st.markdown("### 🎯 Strategy Comparison")
        
        if st.button("🔄 Run Multi-Strategy Comparison"):
            with st.spinner("Running comparison across multiple strategies..."):
                comparison_results = self._run_strategy_comparison(stock_data, tickers)
                
                if comparison_results:
                    st.session_state.comparison_results = comparison_results
        
        if hasattr(st.session_state, 'comparison_results'):
            results = st.session_state.comparison_results
            
            # Comparison table
            comparison_df = pd.DataFrame(results).T
            comparison_df = comparison_df.round(4)
            
            st.markdown("#### 📊 Strategy Performance Comparison")
            st.dataframe(comparison_df, use_container_width=True)
            
            # Performance chart
            fig = go.Figure()
            
            metrics = ['Total Return', 'Sharpe Ratio', 'Max Drawdown']
            strategies = list(results.keys())
            
            for metric in metrics:
                values = [results[strategy][metric] for strategy in strategies]
                fig.add_trace(go.Bar(
                    name=metric,
                    x=strategies,
                    y=values,
                    text=[f"{v:.3f}" for v in values],
                    textposition='auto'
                ))
            
            fig.update_layout(
                title="Strategy Performance Comparison",
                template="plotly_dark",
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                barmode='group'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _run_strategy_comparison(self, stock_data, tickers):
        """Run comparison across multiple strategies"""
        strategies = {
            'Buy & Hold': {'initial_capital': 100000, 'transaction_cost': 0.001},
            'MA Crossover (20/50)': {'initial_capital': 100000, 'transaction_cost': 0.001, 'short_window': 20, 'long_window': 50},
            'RSI (30/70)': {'initial_capital': 100000, 'transaction_cost': 0.001, 'rsi_period': 14, 'oversold': 30, 'overbought': 70}
        }
        
        results = {}
        
        for strategy_name, params in strategies.items():
            try:
                if 'MA Crossover' in strategy_name:
                    result = self._backtest_ma_crossover(stock_data, params, tickers[0])
                elif 'RSI' in strategy_name:
                    result = self._backtest_rsi_strategy(stock_data, params, tickers[0])
                else:
                    result = self._backtest_buy_hold(stock_data, params, tickers)
                
                results[strategy_name] = {
                    'Total Return': result['total_return'],
                    'Sharpe Ratio': result['sharpe_ratio'],
                    'Max Drawdown': result['max_drawdown'],
                    'Volatility': result['volatility'],
                    'Final Value': result['final_value']
                }
            except:
                continue
        
        return results
    
    def _render_drawdown_tab(self):
        """Render detailed drawdown analysis"""
        if not hasattr(st.session_state, 'backtest_results'):
            st.info("🔄 Run a backtest first to see drawdown analysis.")
            return
        
        results = st.session_state.backtest_results
        portfolio_values = pd.Series(results['portfolio_values'])
        
        # Calculate drawdown metrics
        rolling_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - rolling_max) / rolling_max
        
        # Drawdown periods
        drawdown_periods = []
        in_drawdown = False
        start_date = None
        
        for i, (date, dd) in enumerate(zip(results['dates'], drawdown)):
            if dd < 0 and not in_drawdown:
                in_drawdown = True
                start_date = date
                peak_value = rolling_max.iloc[i]
            elif dd == 0 and in_drawdown:
                in_drawdown = False
                recovery_date = date
                trough_value = portfolio_values.iloc[i-1]
                duration = i - results['dates'].index(start_date)
                
                drawdown_periods.append({
                    'Start': start_date,
                    'End': recovery_date,
                    'Duration (days)': duration,
                    'Peak Value': peak_value,
                    'Trough Value': trough_value,
                    'Drawdown %': ((trough_value - peak_value) / peak_value) * 100
                })
        
        # Display drawdown summary
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📉 Drawdown Statistics")
            
            dd_stats = {
                'Max Drawdown': format_percentage(drawdown.min()),
                'Average Drawdown': format_percentage(drawdown[drawdown < 0].mean()) if len(drawdown[drawdown < 0]) > 0 else "0.00%",
                'Drawdown Periods': str(len(drawdown_periods)),
                'Days in Drawdown': str(sum(1 for dd in drawdown if dd < 0)),
                'Recovery Factor': f"{abs(results['total_return'] / drawdown.min()):.2f}" if drawdown.min() != 0 else "N/A"
            }
            
            for stat, value in dd_stats.items():
                create_metric_card(stat, value)
        
        with col2:
            # Underwater curve
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=results['dates'],
                y=drawdown * 100,
                fill='tonexty',
                mode='lines',
                name='Drawdown %',
                line=dict(color='#FF8042', width=1),
                fillcolor='rgba(255, 128, 66, 0.3)'
            ))
            
            fig.update_layout(
                title="Underwater Curve",
                xaxis_title="Date", 
                yaxis_title="Drawdown (%)",
                template="plotly_dark",
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Drawdown periods table
        if drawdown_periods:
            st.markdown("#### 📋 Major Drawdown Periods")
            dd_df = pd.DataFrame(drawdown_periods)
            dd_df = dd_df.sort_values('Drawdown %').head(10)  # Top 10 worst drawdowns
            st.dataframe(dd_df, use_container_width=True)
    
    def _render_export_tab(self):
        """Render export and reporting tab"""
        if not hasattr(st.session_state, 'backtest_results'):
            st.info("🔄 Run a backtest first to access export options.")
            return
        
        st.markdown("### 📁 Reports & Export")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Data Export")
            
            # Export backtest results
            if st.button("📈 Export Backtest Data"):
                results = st.session_state.backtest_results
                
                export_data = pd.DataFrame({
                    'Date': results['dates'],
                    'Portfolio_Value': results['portfolio_values'],
                    'Daily_Return': results['returns'].values if len(results['returns']) == len(results['dates']) else [0] + list(results['returns'].values),
                    'Cumulative_Return': [(val / results['initial_capital'] - 1) * 100 for val in results['portfolio_values']]
                })
                
                csv = export_data.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"backtest_results_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            
            # Export trades
            if 'trades' in st.session_state.backtest_results and st.session_state.backtest_results['trades']:
                if st.button("💼 Export Trade Log"):
                    trades_df = pd.DataFrame(st.session_state.backtest_results['trades'])
                    csv = trades_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Trades CSV",
                        data=csv,
                        file_name=f"trade_log_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
        
        with col2:
            st.markdown("#### 📋 Professional Reports")
            
            # Generate PDF report
            if st.button("📄 Generate PDF Report"):
                pdf_buffer = self._generate_pdf_report()
                
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_buffer,
                    file_name=f"backtest_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf"
                )
            
            # Generate executive summary
            if st.button("📝 Generate Executive Summary"):
                summary = self._generate_executive_summary()
                
                st.download_button(
                    label="📥 Download Summary",
                    data=summary,
                    file_name=f"executive_summary_{datetime.now().strftime('%Y%m%d')}.md",
                    mime="text/markdown"
                )
        
        # Report preview
        st.markdown("---")
        st.markdown("#### 👀 Report Preview")
        
        if st.button("🔍 Preview Executive Summary"):
            summary = self._generate_executive_summary()
            st.markdown(summary)
    
    def _generate_pdf_report(self):
        """Generate professional PDF report"""
        # Note: This would require additional PDF generation libraries
        # For now, return a placeholder
        results = st.session_state.backtest_results
        
        report_text = f"""
BACKTEST PERFORMANCE REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Strategy: {results['strategy']}
Period: {results['dates'][0]} to {results['dates'][-1]}

PERFORMANCE SUMMARY:
- Total Return: {format_percentage(results['total_return'])}
- Sharpe Ratio: {results['sharpe_ratio']:.3f}
- Maximum Drawdown: {format_percentage(results['max_drawdown'])}
- Volatility: {format_percentage(results['volatility'])}
- Final Portfolio Value: {format_currency(results['final_value'])}

This report was generated by FinOptima Backtest Engine.
        """
        
        return report_text.encode('utf-8')
    
    def _generate_executive_summary(self):
        """Generate executive summary"""
        results = st.session_state.backtest_results
        strategy = st.session_state.strategy_type
        
        # Performance grade
        if results['sharpe_ratio'] > 1.5:
            grade = "A+"
        elif results['sharpe_ratio'] > 1.0:
            grade = "A"
        elif results['sharpe_ratio'] > 0.5:
            grade = "B"
        else:
            grade = "C"
        
        summary = f"""
# Executive Summary: {strategy} Strategy

## 📊 Performance Overview
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**Strategy Grade**: {grade}

## 🎯 Key Performance Indicators

| Metric | Value | Benchmark |
|--------|-------|-----------|
| Total Return | {format_percentage(results['total_return'])} | Market dependent |
| Sharpe Ratio | {results['sharpe_ratio']:.3f} | >1.0 excellent |
| Max Drawdown | {format_percentage(results['max_drawdown'])} | <-20% acceptable |
| Volatility | {format_percentage(results['volatility'])} | Market dependent |

## 💼 Investment Thesis

The **{strategy}** strategy demonstrated {'strong' if results['sharpe_ratio'] > 1.0 else 'moderate' if results['sharpe_ratio'] > 0.5 else 'weak'} risk-adjusted performance over the testing period.

### ✅ Strengths:
- {'Excellent' if results['total_return'] > 0.2 else 'Good' if results['total_return'] > 0.1 else 'Modest'} absolute returns
- {'Low' if results['max_drawdown'] > -0.1 else 'Moderate' if results['max_drawdown'] > -0.2 else 'High'} drawdown risk
- {'Consistent' if results['sharpe_ratio'] > 1.0 else 'Variable'} performance profile

### ⚠️ Risk Considerations:
- Market volatility: {format_percentage(results['volatility'])}
- Maximum loss period: {format_percentage(results['max_drawdown'])}
- Strategy concentration risk

## 🚀 Deployment Readiness

**Recommendation**: {'Deploy with confidence' if grade in ['A+', 'A'] else 'Consider with caution' if grade == 'B' else 'Requires optimization'}

**Next Steps**:
1. {'✅' if results['sharpe_ratio'] > 1.0 else '⚠️'} Risk management review
2. {'✅' if results['max_drawdown'] > -0.2 else '⚠️'} Capital allocation planning  
3. {'✅' if len(results.get('trades', [])) > 10 else '⚠️'} Implementation strategy
4. 🔄 Ongoing monitoring framework

---
*This analysis was generated by FinOptima Professional Analytics Suite*
        """
        
        return summary