import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize
import cvxpy as cp
from .utils import calculate_returns, calculate_sharpe_ratio, format_percentage, format_currency, create_metric_card

class PortfolioOptimizer:
    def __init__(self):
        self.returns_data = None
        self.expected_returns = None
        self.cov_matrix = None
        
    def render(self, stock_data, tickers, start_date, end_date):
        """Render the Portfolio Optimizer section"""
        st.markdown("""
        <div class="main-header">
            <h1>📊 Portfolio Optimizer</h1>
            <p>Modern Portfolio Theory & Efficient Frontier Analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        if not stock_data:
            st.error("⚠️ No stock data available. Please check your ticker selection.")
            return
        
        # Prepare data
        self._prepare_data(stock_data)
        
        # Input controls
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### ⚙️ Optimization Settings")
            
            # Risk-free rate input
            risk_free_rate = st.slider(
                "Risk-Free Rate (%)",
                min_value=0.0,
                max_value=10.0,
                value=2.0,
                step=0.1,
                help="Annual risk-free rate (e.g., 10-year Treasury rate)"
            ) / 100
            
            # Optimization objective
            objective = st.selectbox(
                "Optimization Objective",
                ["Max Sharpe Ratio", "Min Volatility", "Max Return"],
                help="Choose the optimization criteria"
            )
            
            # Constraints
            st.markdown("#### 🔒 Constraints")
            
            no_short_selling = st.checkbox(
                "No Short Selling",
                value=True,
                help="Prevent negative weights (no short positions)"
            )
            
            equal_weights = st.checkbox(
                "Equal Weight Constraint",
                value=False,
                help="Force equal allocation across all assets"
            )
            
            max_weight = st.slider(
                "Max Weight per Asset (%)",
                min_value=10,
                max_value=100,
                value=50,
                step=5,
                disabled=equal_weights,
                help="Maximum allocation per individual asset"
            ) / 100
            
            # Target return for min volatility
            target_return = None
            if objective == "Min Volatility":
                target_return = st.slider(
                    "Target Return (%)",
                    min_value=float(self.expected_returns.min() * 100),
                    max_value=float(self.expected_returns.max() * 100),
                    value=float(self.expected_returns.mean() * 100),
                    step=0.1
                ) / 100
        
        with col2:
            # Run optimization
            if st.button("🚀 Optimize Portfolio", type="primary"):
                with st.spinner("Running optimization..."):
                    optimal_weights, performance = self._optimize_portfolio(
                        objective, risk_free_rate, no_short_selling, 
                        equal_weights, max_weight, target_return
                    )
                    
                    if optimal_weights is not None:
                        st.session_state.optimal_weights = optimal_weights
                        st.session_state.performance = performance
                        st.success("✅ Optimization completed successfully!")
            
            # Display results if available
            if hasattr(st.session_state, 'optimal_weights'):
                self._display_results(
                    st.session_state.optimal_weights,
                    st.session_state.performance,
                    risk_free_rate
                )
        
        # Efficient Frontier Analysis
        st.markdown("---")
        st.markdown("### 📈 Efficient Frontier Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Generate efficient frontier
            if st.button("📊 Generate Efficient Frontier"):
                with st.spinner("Calculating efficient frontier..."):
                    frontier_data = self._calculate_efficient_frontier(risk_free_rate)
                    fig = self._plot_efficient_frontier(frontier_data, risk_free_rate)
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Asset correlation heatmap
            if st.button("🔗 Show Correlation Matrix"):
                correlation_matrix = self.returns_data.corr()
                fig = self._plot_correlation_heatmap(correlation_matrix)
                st.plotly_chart(fig, use_container_width=True)
        
        # Additional Analysis
        st.markdown("---")
        self._render_additional_analysis()
    
    def _prepare_data(self, stock_data):
        """Prepare returns data and statistics"""
        # Calculate returns for all stocks
        returns_dict = {}
        for ticker, data in stock_data.items():
            returns = calculate_returns(data['Close'])
            returns_dict[ticker] = returns
        
        # Create returns DataFrame
        self.returns_data = pd.DataFrame(returns_dict).dropna()
        
        # Calculate expected returns (annualized)
        self.expected_returns = self.returns_data.mean() * 252
        
        # Calculate covariance matrix (annualized)
        self.cov_matrix = self.returns_data.cov() * 252
    
    def _optimize_portfolio(self, objective, risk_free_rate, no_short_selling, 
                          equal_weights, max_weight, target_return=None):
        """Optimize portfolio based on specified criteria"""
        n_assets = len(self.expected_returns)
        
        try:
            if equal_weights:
                # Equal weight portfolio
                weights = np.array([1/n_assets] * n_assets)
            else:
                # Define optimization variables
                w = cp.Variable(n_assets)
                
                # Objective function
                if objective == "Max Sharpe Ratio":
                    # Maximize Sharpe ratio (minimize negative Sharpe)
                    portfolio_return = self.expected_returns.values @ w
                    portfolio_variance = cp.quad_form(w, self.cov_matrix.values)
                    
                    # Use auxiliary variable for Sharpe ratio maximization
                    kappa = cp.Variable()
                    objective_func = cp.Minimize(-kappa)
                    
                    constraints = [
                        cp.sum(w) == kappa,
                        portfolio_return - risk_free_rate * kappa >= 1,
                        portfolio_variance <= kappa,
                        kappa >= 0
                    ]
                    
                elif objective == "Min Volatility":
                    portfolio_variance = cp.quad_form(w, self.cov_matrix.values)
                    objective_func = cp.Minimize(portfolio_variance)
                    
                    constraints = [cp.sum(w) == 1]
                    
                    if target_return is not None:
                        portfolio_return = self.expected_returns.values @ w
                        constraints.append(portfolio_return >= target_return)
                
                elif objective == "Max Return":
                    portfolio_return = self.expected_returns.values @ w
                    objective_func = cp.Maximize(portfolio_return)
                    
                    constraints = [cp.sum(w) == 1]
                
                # Add constraints
                if no_short_selling:
                    constraints.append(w >= 0)
                
                if not equal_weights:
                    constraints.append(w <= max_weight)
                
                # Solve optimization problem
                if objective == "Max Sharpe Ratio":
                    prob = cp.Problem(objective_func, constraints)
                    prob.solve(solver=cp.ECOS)
                    
                    if prob.status == cp.OPTIMAL:
                        weights = w.value / np.sum(w.value)  # Normalize weights
                    else:
                        return None, None
                else:
                    prob = cp.Problem(objective_func, constraints)
                    prob.solve()
                    
                    if prob.status == cp.OPTIMAL:
                        weights = w.value
                    else:
                        return None, None
            
            # Calculate portfolio performance
            portfolio_return = np.sum(weights * self.expected_returns)
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std
            
            performance = {
                'return': portfolio_return,
                'volatility': portfolio_std,
                'sharpe_ratio': sharpe_ratio
            }
            
            return weights, performance
            
        except Exception as e:
            st.error(f"Optimization failed: {str(e)}")
            return None, None
    
    def _display_results(self, weights, performance, risk_free_rate):
        """Display optimization results"""
        st.markdown("### 🎯 Optimal Portfolio Results")
        
        # Performance metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            create_metric_card(
                "Expected Return",
                format_percentage(performance['return']),
                help_text="Annualized expected return"
            )
        
        with col2:
            create_metric_card(
                "Volatility (Risk)",
                format_percentage(performance['volatility']),
                help_text="Annualized standard deviation"
            )
        
        with col3:
            create_metric_card(
                "Sharpe Ratio",
                f"{performance['sharpe_ratio']:.3f}",
                help_text="Risk-adjusted return metric"
            )
        
        # Asset allocation
        st.markdown("#### 🥧 Asset Allocation")
        
        # Create allocation DataFrame
        allocation_df = pd.DataFrame({
            'Asset': self.expected_returns.index,
            'Weight': weights,
            'Weight (%)': weights * 100
        }).sort_values('Weight', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart
            fig = go.Figure(data=[go.Pie(
                labels=allocation_df['Asset'],
                values=allocation_df['Weight'],
                hole=0.3,
                textinfo='label+percent',
                textposition='outside'
            )])
            
            fig.update_traces(
                hovertemplate='<b>%{label}</b><br>Weight: %{percent}<extra></extra>',
                marker=dict(colors=px.colors.qualitative.Set3)
            )
            
            fig.update_layout(
                title="Portfolio Allocation",
                template="plotly_dark",
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Allocation table
            st.dataframe(
                allocation_df.style.format({
                    'Weight': '{:.3f}',
                    'Weight (%)': '{:.1f}%'
                }),
                use_container_width=True
            )
    
    def _calculate_efficient_frontier(self, risk_free_rate, num_portfolios=100):
        """Calculate efficient frontier portfolios"""
        n_assets = len(self.expected_returns)
        
        # Define return range
        min_ret = self.expected_returns.min()
        max_ret = self.expected_returns.max()
        target_returns = np.linspace(min_ret, max_ret, num_portfolios)
        
        efficient_portfolios = []
        
        for target_ret in target_returns:
            try:
                # Minimize volatility for target return
                w = cp.Variable(n_assets)
                portfolio_variance = cp.quad_form(w, self.cov_matrix.values)
                portfolio_return = self.expected_returns.values @ w
                
                objective = cp.Minimize(portfolio_variance)
                constraints = [
                    cp.sum(w) == 1,
                    portfolio_return >= target_ret,
                    w >= 0  # No short selling
                ]
                
                prob = cp.Problem(objective, constraints)
                prob.solve()
                
                if prob.status == cp.OPTIMAL:
                    weights = w.value
                    ret = np.sum(weights * self.expected_returns)
                    vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
                    sharpe = (ret - risk_free_rate) / vol
                    
                    efficient_portfolios.append({
                        'return': ret,
                        'volatility': vol,
                        'sharpe_ratio': sharpe
                    })
            except:
                continue
        
        return pd.DataFrame(efficient_portfolios)
    
    def _plot_efficient_frontier(self, frontier_data, risk_free_rate):
        """Plot efficient frontier with optimal portfolio"""
        fig = go.Figure()
        
        # Efficient frontier
        fig.add_trace(go.Scatter(
            x=frontier_data['volatility'],
            y=frontier_data['return'],
            mode='lines',
            name='Efficient Frontier',
            line=dict(color='#00C49F', width=3)
        ))
        
        # Individual assets
        fig.add_trace(go.Scatter(
            x=np.sqrt(np.diag(self.cov_matrix)),
            y=self.expected_returns,
            mode='markers+text',
            name='Individual Assets',
            text=self.expected_returns.index,
            textposition='top center',
            marker=dict(size=10, color='#FF8042')
        ))
        
        # Optimal portfolio (max Sharpe ratio)
        if hasattr(st.session_state, 'performance'):
            perf = st.session_state.performance
            fig.add_trace(go.Scatter(
                x=[perf['volatility']],
                y=[perf['return']],
                mode='markers',
                name='Optimal Portfolio',
                marker=dict(size=15, color='#FFD700', symbol='star')
            ))
        
        # Capital Allocation Line (CAL)
        if not frontier_data.empty:
            max_sharpe_idx = frontier_data['sharpe_ratio'].idxmax()
            optimal_vol = frontier_data.loc[max_sharpe_idx, 'volatility']
            optimal_ret = frontier_data.loc[max_sharpe_idx, 'return']
            
            # Extend CAL
            cal_vol = np.linspace(0, frontier_data['volatility'].max() * 1.2, 100)
            cal_ret = risk_free_rate + (optimal_ret - risk_free_rate) / optimal_vol * cal_vol
            
            fig.add_trace(go.Scatter(
                x=cal_vol,
                y=cal_ret,
                mode='lines',
                name='Capital Allocation Line',
                line=dict(color='#FF8042', dash='dash', width=2)
            ))
        
        fig.update_layout(
            title="Efficient Frontier & Portfolio Optimization",
            xaxis_title="Volatility (Risk)",
            yaxis_title="Expected Return",
            template="plotly_dark",
            height=500,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return fig
    
    def _plot_correlation_heatmap(self, correlation_matrix):
        """Plot asset correlation heatmap"""
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
    
    def _render_additional_analysis(self):
        """Render additional portfolio analysis"""
        st.markdown("### 📊 Additional Analysis")
        
        tab1, tab2, tab3 = st.tabs(["📈 Risk-Return", "📉 Drawdown", "💼 Portfolio Stats"])
        
        with tab1:
            self._render_risk_return_analysis()
        
        with tab2:
            self._render_drawdown_analysis()
        
        with tab3:
            self._render_portfolio_statistics()
    
    def _render_risk_return_analysis(self):
        """Risk-return scatter plot"""
        fig = go.Figure()
        
        returns = self.expected_returns
        risks = np.sqrt(np.diag(self.cov_matrix))
        
        fig.add_trace(go.Scatter(
            x=risks,
            y=returns,
            mode='markers+text',
            text=returns.index,
            textposition='top center',
            marker=dict(
                size=12,
                color=returns,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Expected Return")
            ),
            hovertemplate='<b>%{text}</b><br>Risk: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Risk vs Return Analysis",
            xaxis_title="Risk (Volatility)",
            yaxis_title="Expected Return",
            template="plotly_dark",
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_drawdown_analysis(self):
        """Portfolio drawdown analysis"""
        if not hasattr(st.session_state, 'optimal_weights'):
            st.info("Please optimize a portfolio first to see drawdown analysis.")
            return
        
        # Calculate portfolio returns
        weights = st.session_state.optimal_weights
        portfolio_returns = (self.returns_data * weights).sum(axis=1)
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        # Calculate drawdowns
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown * 100,
            fill='tonexty',
            mode='lines',
            name='Drawdown',
            line=dict(color='#FF8042'),
            fillcolor='rgba(255, 128, 66, 0.3)'
        ))
        
        fig.update_layout(
            title="Portfolio Drawdown Analysis",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            template="plotly_dark",
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Drawdown statistics
        max_dd = drawdown.min()
        avg_dd = drawdown[drawdown < 0].mean()
        
        col1, col2 = st.columns(2)
        with col1:
            create_metric_card("Max Drawdown", format_percentage(max_dd))
        with col2:
            create_metric_card("Average Drawdown", format_percentage(avg_dd))
    
    def _render_portfolio_statistics(self):
        """Comprehensive portfolio statistics"""
        if not hasattr(st.session_state, 'optimal_weights'):
            st.info("Please optimize a portfolio first to see detailed statistics.")
            return
        
        weights = st.session_state.optimal_weights
        returns = self.returns_data
        
        # Portfolio statistics
        portfolio_returns = (returns * weights).sum(axis=1)
        
        stats = {
            'Annualized Return': portfolio_returns.mean() * 252,
            'Annualized Volatility': portfolio_returns.std() * np.sqrt(252),
            'Skewness': portfolio_returns.skew(),
            'Kurtosis': portfolio_returns.kurtosis(),
            'Max Daily Return': portfolio_returns.max(),
            'Min Daily Return': portfolio_returns.min(),
            'Positive Days (%)': (portfolio_returns > 0).mean() * 100,
            'VaR (95%)': portfolio_returns.quantile(0.05),
            'CVaR (95%)': portfolio_returns[portfolio_returns <= portfolio_returns.quantile(0.05)].mean()
        }
        
        # Display statistics
        col1, col2 = st.columns(2)
        
        for i, (metric, value) in enumerate(stats.items()):
            col = col1 if i % 2 == 0 else col2
            
            if 'Return' in metric or 'Volatility' in metric or 'VaR' in metric:
                formatted_value = format_percentage(value)
            elif metric == 'Positive Days (%)':
                formatted_value = f"{value:.1f}%"
            else:
                formatted_value = f"{value:.3f}"
            
            with col:
                create_metric_card(metric, formatted_value)