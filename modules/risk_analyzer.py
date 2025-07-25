import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from scipy.stats import norm, t
from .utils import calculate_returns, format_percentage, create_metric_card

class RiskAnalyzer:
    def __init__(self):
        self.returns_data = None
        
    def render(self, stock_data, tickers):
        """Render the Risk Analyzer section"""
        st.markdown("""
        <div class="main-header">
            <h1>⚠️ Risk Analyzer</h1>
            <p>Value-at-Risk (VaR) & Advanced Risk Metrics</p>
        </div>
        """, unsafe_allow_html=True)
        
        if not stock_data:
            st.error("⚠️ No stock data available. Please check your ticker selection.")
            return
        
        # Prepare returns data
        self._prepare_data(stock_data)
        
        # Input controls
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### ⚙️ Risk Analysis Settings")
            
            # Asset selection
            selected_asset = st.selectbox(
                "Select Asset",
                options=list(self.returns_data.columns),
                help="Choose asset for detailed risk analysis"
            )
            
            # VaR method
            var_method = st.selectbox(
                "VaR Method",
                ["Historical", "Monte Carlo", "Parametric"],
                help="Choose Value-at-Risk calculation method"
            )
            
            # Confidence level
            confidence_level = st.slider(
                "Confidence Level (%)",
                min_value=90,
                max_value=99,
                value=95,
                step=1,
                help="Statistical confidence level for VaR calculation"
            )
            
            # Time horizon
            time_horizon = st.selectbox(
                "Time Horizon",
                [1, 5, 10, 22],
                index=0,
                format_func=lambda x: f"{x} day{'s' if x > 1 else ''}",
                help="Investment time horizon for risk calculation"
            )
            
            # Portfolio value (for VaR calculation)
            portfolio_value = st.number_input(
                "Portfolio Value ($)",
                min_value=1000,
                max_value=10000000,
                value=100000,
                step=10000,
                help="Total portfolio value for risk calculation"
            )
            
            # Monte Carlo parameters (if selected)
            if var_method == "Monte Carlo":
                st.markdown("#### 🎲 Monte Carlo Settings")
                num_simulations = st.number_input(
                    "Number of Simulations",
                    min_value=1000,
                    max_value=100000,
                    value=10000,
                    step=1000
                )
        
        with col2:
            # Calculate and display VaR
            if st.button("📊 Calculate Risk Metrics", type="primary"):
                with st.spinner("Calculating risk metrics..."):
                    var_results = self._calculate_var(
                        selected_asset, var_method, confidence_level,
                        time_horizon, portfolio_value,
                        num_simulations if var_method == "Monte Carlo" else 10000
                    )
                    
                    if var_results:
                        st.session_state.var_results = var_results
                        st.session_state.selected_asset = selected_asset
                        st.success("✅ Risk analysis completed!")
            
            # Display results if available
            if hasattr(st.session_state, 'var_results'):
                self._display_var_results(
                    st.session_state.var_results,
                    st.session_state.selected_asset,
                    confidence_level,
                    time_horizon,
                    portfolio_value
                )
        
        # Risk visualization
        st.markdown("---")
        st.markdown("### 📈 Risk Visualization")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Returns Distribution",
            "📉 Rolling VaR",
            "🔄 Volatility Analysis",
            "🌡️ Risk Heatmap"
        ])
        
        with tab1:
            self._render_returns_distribution(selected_asset, confidence_level)
        
        with tab2:
            self._render_rolling_var(selected_asset, confidence_level)
        
        with tab3:
            self._render_volatility_analysis()
        
        with tab4:
            self._render_risk_heatmap()
    
    def _prepare_data(self, stock_data):
        """Prepare returns data"""
        returns_dict = {}
        for ticker, data in stock_data.items():
            returns = calculate_returns(data['Close'])
            returns_dict[ticker] = returns
        
        self.returns_data = pd.DataFrame(returns_dict).dropna()
    
    def _calculate_var(self, asset, method, confidence_level, time_horizon, 
                      portfolio_value, num_simulations=10000):
        """Calculate Value-at-Risk using different methods"""
        returns = self.returns_data[asset].dropna()
        alpha = (100 - confidence_level) / 100
        
        try:
            if method == "Historical":
                var_result = self._historical_var(returns, alpha, time_horizon)
            elif method == "Monte Carlo":
                var_result = self._monte_carlo_var(returns, alpha, time_horizon, num_simulations)
            elif method == "Parametric":
                var_result = self._parametric_var(returns, alpha, time_horizon)
            
            # Convert to dollar amounts
            var_dollar = var_result['var'] * portfolio_value
            cvar_dollar = var_result['cvar'] * portfolio_value
            
            var_result.update({
                'var_dollar': var_dollar,
                'cvar_dollar': cvar_dollar,
                'portfolio_value': portfolio_value,
                'method': method,
                'confidence_level': confidence_level,
                'time_horizon': time_horizon
            })
            
            return var_result
            
        except Exception as e:
            st.error(f"VaR calculation failed: {str(e)}")
            return None
    
    def _historical_var(self, returns, alpha, time_horizon):
        """Calculate Historical VaR"""
        # Scale returns for time horizon
        scaled_returns = returns * np.sqrt(time_horizon)
        
        # Calculate VaR and CVaR
        var = np.percentile(scaled_returns, alpha * 100)
        cvar = scaled_returns[scaled_returns <= var].mean()
        
        return {
            'var': var,
            'cvar': cvar,
            'method_details': f"Based on {len(returns)} historical observations"
        }
    
    def _monte_carlo_var(self, returns, alpha, time_horizon, num_simulations):
        """Calculate Monte Carlo VaR"""
        # Fit distribution parameters
        mu = returns.mean()
        sigma = returns.std()
        
        # Generate random scenarios
        np.random.seed(42)  # For reproducibility
        simulated_returns = np.random.normal(
            mu * time_horizon, 
            sigma * np.sqrt(time_horizon), 
            num_simulations
        )
        
        # Calculate VaR and CVaR
        var = np.percentile(simulated_returns, alpha * 100)
        cvar = simulated_returns[simulated_returns <= var].mean()
        
        return {
            'var': var,
            'cvar': cvar,
            'simulated_returns': simulated_returns,
            'method_details': f"Based on {num_simulations} Monte Carlo simulations"
        }
    
    def _parametric_var(self, returns, alpha, time_horizon):
        """Calculate Parametric VaR"""
        # Assume normal distribution
        mu = returns.mean()
        sigma = returns.std()
        
        # Scale for time horizon
        scaled_mu = mu * time_horizon
        scaled_sigma = sigma * np.sqrt(time_horizon)
        
        # Calculate VaR using normal distribution
        var = norm.ppf(alpha, scaled_mu, scaled_sigma)
        
        # Calculate CVaR (Expected Shortfall)
        cvar = scaled_mu - scaled_sigma * norm.pdf(norm.ppf(alpha)) / alpha
        
        # Alternative using t-distribution (more robust)
        df = len(returns) - 1
        t_var = scaled_mu + scaled_sigma * t.ppf(alpha, df)
        t_cvar = scaled_mu - scaled_sigma * (df + t.ppf(alpha, df)**2) / (df - 1) * t.pdf(t.ppf(alpha, df), df) / alpha
        
        return {
            'var': var,
            'cvar': cvar,
            't_var': t_var,
            't_cvar': t_cvar,
            'mu': scaled_mu,
            'sigma': scaled_sigma,
            'method_details': f"Normal distribution (μ={scaled_mu:.4f}, σ={scaled_sigma:.4f})"
        }
    
    def _display_var_results(self, var_results, asset, confidence_level, 
                           time_horizon, portfolio_value):
        """Display VaR calculation results"""
        st.markdown(f"### 🎯 Risk Results for {asset}")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            create_metric_card(
                "Value-at-Risk",
                format_percentage(var_results['var']),
                help_text=f"{confidence_level}% VaR over {time_horizon} day(s)"
            )
        
        with col2:
            create_metric_card(
                "VaR (Dollar)",
                f"${abs(var_results['var_dollar']):,.0f}",
                help_text="Potential loss in dollar terms"
            )
        
        with col3:
            create_metric_card(
                "Conditional VaR",
                format_percentage(var_results['cvar']),
                help_text="Expected loss beyond VaR threshold"
            )
        
        with col4:
            create_metric_card(
                "CVaR (Dollar)",
                f"${abs(var_results['cvar_dollar']):,.0f}",
                help_text="Expected shortfall in dollar terms"
            )
        
        # Method details
        st.info(f"**Method**: {var_results['method']} | {var_results['method_details']}")
        
        # Risk interpretation
        self._display_risk_interpretation(var_results, confidence_level)
    
    def _display_risk_interpretation(self, var_results, confidence_level):
        """Display risk interpretation and recommendations"""
        var_pct = abs(var_results['var']) * 100
        
        if var_pct < 1:
            risk_level = "🟢 Low"
            interpretation = "The asset shows relatively low risk with minimal potential losses."
        elif var_pct < 3:
            risk_level = "🟡 Moderate"
            interpretation = "The asset has moderate risk levels typical for equity investments."
        elif var_pct < 5:
            risk_level = "🟠 High"
            interpretation = "The asset exhibits high volatility and significant downside risk."
        else:
            risk_level = "🔴 Very High"
            interpretation = "The asset is extremely volatile with substantial loss potential."
        
        st.markdown(f"""
        #### 🔍 Risk Assessment
        
        **Risk Level**: {risk_level}
        
        **Interpretation**: {interpretation}
        
        **What this means**: There is a {confidence_level}% probability that losses will not exceed 
        {format_percentage(abs(var_results['var']))} over the next {var_results['time_horizon']} day(s).
        
        **Risk Management Recommendations**:
        - Consider position sizing based on VaR limits
        - Implement stop-loss orders around the VaR threshold
        - Monitor portfolio concentration in high-risk assets
        - Regular stress testing and scenario analysis
        """)
    
    def _render_returns_distribution(self, asset, confidence_level):
        """Render returns distribution with VaR visualization"""
        returns = self.returns_data[asset].dropna()
        alpha = (100 - confidence_level) / 100
        
        # Create histogram
        fig = go.Figure()
        
        # Returns histogram
        fig.add_trace(go.Histogram(
            x=returns,
            nbinsx=50,
            name='Returns Distribution',
            opacity=0.7,
            marker_color='#00C49F'
        ))
        
        # VaR threshold line
        var_threshold = np.percentile(returns, alpha * 100)
        fig.add_vline(
            x=var_threshold,
            line_dash="dash",
            line_color="#FF8042",
            annotation_text=f"VaR ({confidence_level}%): {var_threshold:.3f}",
            annotation_position="top left"
        )
        
        # Normal distribution overlay
        x_range = np.linspace(returns.min(), returns.max(), 100)
        normal_dist = norm.pdf(x_range, returns.mean(), returns.std())
        
        # Scale normal distribution to match histogram
        hist_counts, _ = np.histogram(returns, bins=50)
        scaling_factor = hist_counts.max() / normal_dist.max()
        
        fig.add_trace(go.Scatter(
            x=x_range,
            y=normal_dist * scaling_factor,
            mode='lines',
            name='Normal Distribution',
            line=dict(color='#FF8042', dash='dot')
        ))
        
        fig.update_layout(
            title=f"Returns Distribution - {asset}",
            xaxis_title="Daily Returns",
            yaxis_title="Frequency",
            template="plotly_dark",
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Distribution statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            create_metric_card("Mean", f"{returns.mean():.4f}")
        with col2:
            create_metric_card("Std Dev", f"{returns.std():.4f}")
        with col3:
            create_metric_card("Skewness", f"{returns.skew():.3f}")
        with col4:
            create_metric_card("Kurtosis", f"{returns.kurtosis():.3f}")
    
    def _render_rolling_var(self, asset, confidence_level):
        """Render rolling VaR analysis"""
        returns = self.returns_data[asset].dropna()
        alpha = (100 - confidence_level) / 100
        
        # Calculate rolling VaR
        window = 252  # 1 year rolling window
        rolling_var = returns.rolling(window=window).quantile(alpha)
        
        fig = go.Figure()
        
        # Rolling VaR
        fig.add_trace(go.Scatter(
            x=rolling_var.index,
            y=rolling_var * 100,
            mode='lines',
            name=f'Rolling VaR ({confidence_level}%)',
            line=dict(color='#FF8042', width=2)
        ))
        
        # Add shaded regions for crisis periods (example)
        fig.add_vrect(
            x0="2020-02-01", x1="2020-05-01",
            fillcolor="red", opacity=0.1,
            annotation_text="COVID-19 Crisis",
            annotation_position="top left"
        )
        
        fig.update_layout(
            title=f"Rolling {confidence_level}% VaR - {asset}",
            xaxis_title="Date",
            yaxis_title="VaR (%)",
            template="plotly_dark",
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Rolling VaR statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            create_metric_card("Current VaR", format_percentage(rolling_var.iloc[-1]))
        with col2:
            create_metric_card("Average VaR", format_percentage(rolling_var.mean()))
        with col3:
            create_metric_card("Max VaR", format_percentage(rolling_var.min()))
    
    def _render_volatility_analysis(self):
        """Render volatility comparison across assets"""
        # Calculate rolling volatility
        volatilities = {}
        for asset in self.returns_data.columns:
            vol = self.returns_data[asset].rolling(window=30).std() * np.sqrt(252)
            volatilities[asset] = vol
        
        vol_df = pd.DataFrame(volatilities)
        
        # Volatility time series
        fig = go.Figure()
        
        for asset in vol_df.columns:
            fig.add_trace(go.Scatter(
                x=vol_df.index,
                y=vol_df[asset] * 100,
                mode='lines',
                name=asset,
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title="Rolling 30-Day Volatility (Annualized)",
            xaxis_title="Date",
            yaxis_title="Volatility (%)",
            template="plotly_dark",
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Current volatility comparison
        current_vol = vol_df.iloc[-1].sort_values(ascending=False)
        
        fig2 = go.Figure(data=[go.Bar(
            x=current_vol.index,
            y=current_vol * 100,
            marker_color='#00C49F'
        )])
        
        fig2.update_layout(
            title="Current Volatility Comparison",
            xaxis_title="Asset",
            yaxis_title="Volatility (%)",
            template="plotly_dark",
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    def _render_risk_heatmap(self):
        """Render risk metrics heatmap"""
        # Calculate various risk metrics
        risk_metrics = {}
        
        for asset in self.returns_data.columns:
            returns = self.returns_data[asset].dropna()
            
            risk_metrics[asset] = {
                'Volatility': returns.std() * np.sqrt(252),
                'VaR_95': np.percentile(returns, 5),
                'VaR_99': np.percentile(returns, 1),
                'Max_Drawdown': self._calculate_max_drawdown(returns),
                'Downside_Dev': self._calculate_downside_deviation(returns),
                'Skewness': returns.skew(),
                'Kurtosis': returns.kurtosis()
            }
        
        risk_df = pd.DataFrame(risk_metrics).T
        
        # Normalize data for heatmap (0-1 scale)
        risk_df_norm = (risk_df - risk_df.min()) / (risk_df.max() - risk_df.min())
        
        fig = go.Figure(data=go.Heatmap(
            z=risk_df_norm.values,
            x=risk_df_norm.columns,
            y=risk_df_norm.index,
            colorscale='Reds',
            text=risk_df.round(3).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverangles="closest"
        ))
        
        fig.update_layout(
            title="Risk Metrics Heatmap (Normalized)",
            template="plotly_dark",
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk ranking table
        st.markdown("#### 🏆 Risk Ranking")
        
        # Calculate composite risk score
        risk_scores = risk_df_norm.mean(axis=1).sort_values(ascending=False)
        
        ranking_df = pd.DataFrame({
            'Asset': risk_scores.index,
            'Risk Score': risk_scores.values,
            'Risk Level': ['High' if x > 0.7 else 'Medium' if x > 0.4 else 'Low' for x in risk_scores.values]
        })
        
        st.dataframe(
            ranking_df.style.format({'Risk Score': '{:.3f}'}),
            use_container_width=True
        )
    
    def _calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        return drawdown.min()
    
    def _calculate_downside_deviation(self, returns, target_return=0):
        """Calculate downside deviation"""
        downside_returns = returns[returns < target_return]
        return downside_returns.std() * np.sqrt(252)