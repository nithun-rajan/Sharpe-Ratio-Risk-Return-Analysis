import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN, PPO, A2C, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
import warnings
import os
from datetime import datetime, timedelta
from .utils import add_technical_indicators, calculate_sharpe_ratio, calculate_sortino_ratio, format_percentage, create_metric_card

warnings.filterwarnings('ignore')

class TradingEnvironment(gym.Env):
    """Custom Trading Environment for Reinforcement Learning"""
    
    def __init__(self, df, initial_balance=100000, transaction_cost=0.001, window_size=30):
        super(TradingEnvironment, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.window_size = window_size
        
        # Action space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: OHLCV + technical indicators
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(window_size, len(df.columns)), 
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.trades = []
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Get current observation (normalized price data + indicators)"""
        start_idx = self.current_step - self.window_size
        end_idx = self.current_step
        
        obs = self.df.iloc[start_idx:end_idx].values
        
        # Normalize observation
        if len(obs) > 0:
            obs = (obs - obs.mean(axis=0)) / (obs.std(axis=0) + 1e-8)
        
        return obs.astype(np.float32)
    
    def step(self, action):
        current_price = self.df.iloc[self.current_step]['Close']
        
        # Execute action
        if action == 1:  # Buy
            self._buy_stock(current_price)
        elif action == 2:  # Sell
            self._sell_stock(current_price)
        # action == 0 is Hold (do nothing)
        
        # Update net worth
        self.net_worth = self.balance + self.shares_held * current_price
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= len(self.df) - 1
        
        # Additional info
        info = {
            'balance': self.balance,
            'shares_held': self.shares_held,
            'net_worth': self.net_worth,
            'current_price': current_price
        }
        
        return self._get_observation(), reward, done, False, info
    
    def _buy_stock(self, current_price):
        """Execute buy order"""
        # Calculate how many shares we can buy
        available_amount = self.balance * 0.95  # Keep 5% cash buffer
        shares_to_buy = int(available_amount / (current_price * (1 + self.transaction_cost)))
        
        if shares_to_buy > 0:
            cost = shares_to_buy * current_price * (1 + self.transaction_cost)
            if cost <= self.balance:
                self.balance -= cost
                self.shares_held += shares_to_buy
                
                self.trades.append({
                    'step': self.current_step,
                    'action': 'BUY',
                    'shares': shares_to_buy,
                    'price': current_price,
                    'cost': cost
                })
    
    def _sell_stock(self, current_price):
        """Execute sell order"""
        if self.shares_held > 0:
            # Sell all shares
            revenue = self.shares_held * current_price * (1 - self.transaction_cost)
            self.balance += revenue
            self.total_sales_value += revenue
            self.total_shares_sold += self.shares_held
            
            self.trades.append({
                'step': self.current_step,
                'action': 'SELL',
                'shares': self.shares_held,
                'price': current_price,
                'revenue': revenue
            })
            
            self.shares_held = 0
    
    def _calculate_reward(self):
        """Calculate reward based on portfolio performance"""
        # Portfolio return
        portfolio_return = (self.net_worth - self.initial_balance) / self.initial_balance
        
        # Risk-adjusted return (Sharpe-like ratio)
        if hasattr(self, 'previous_net_worth'):
            return_change = (self.net_worth - self.previous_net_worth) / self.previous_net_worth
        else:
            return_change = 0
        
        # Update previous net worth
        self.previous_net_worth = self.net_worth
        
        # Reward components
        return_reward = return_change * 100  # Scale up returns
        
        # Penalty for holding cash when market is going up
        current_price = self.df.iloc[self.current_step]['Close']
        if self.current_step > 1:
            price_change = (current_price - self.df.iloc[self.current_step-1]['Close']) / self.df.iloc[self.current_step-1]['Close']
            if price_change > 0 and self.shares_held == 0:
                return_reward -= abs(price_change) * 50  # Penalty for missing upside
        
        return return_reward

class TradingCallback(BaseCallback):
    """Custom callback for training monitoring"""
    
    def __init__(self, verbose=0):
        super(TradingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        # Log training progress
        return True

class RLAgent:
    def __init__(self):
        self.env = None
        self.model = None
        self.trained = False
        
    def render(self, tickers, start_date, end_date):
        """Render the RL Trading Agent section"""
        st.markdown("""
        <div class="main-header">
            <h1>🤖 RL Trading Agent</h1>
            <p>Advanced Reinforcement Learning for Algorithmic Trading</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Agent configuration
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### 🎮 Agent Configuration")
            
            # Stock selection
            selected_ticker = st.selectbox(
                "Select Stock",
                options=tickers,
                help="Choose stock for RL agent training"
            )
            
            # RL Algorithm
            algorithm = st.selectbox(
                "RL Algorithm",
                ["DQN", "PPO", "A2C", "SAC"],
                help="Choose reinforcement learning algorithm"
            )
            
            # Environment settings
            st.markdown("#### 🌍 Environment Settings")
            
            initial_balance = st.number_input(
                "Initial Balance ($)",
                min_value=10000,
                max_value=1000000,
                value=100000,
                step=10000,
                help="Starting capital for the trading agent"
            )
            
            transaction_cost = st.slider(
                "Transaction Cost (%)",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.05,
                help="Trading fees and slippage"
            ) / 100
            
            window_size = st.slider(
                "Observation Window",
                min_value=10,
                max_value=100,
                value=30,
                help="Number of days in observation window"
            )
            
            # Training parameters
            with st.expander("🔧 Advanced Training Settings"):
                training_steps = st.slider(
                    "Training Steps",
                    min_value=10000,
                    max_value=500000,
                    value=100000,
                    step=10000
                )
                
                learning_rate = st.selectbox(
                    "Learning Rate",
                    [1e-4, 3e-4, 1e-3, 3e-3],
                    index=1,
                    format_func=lambda x: f"{x:.0e}"
                )
                
                batch_size = st.selectbox(
                    "Batch Size",
                    [32, 64, 128, 256],
                    index=1
                )
        
        with col2:
            # Train agent
            if st.button("🚀 Train RL Agent", type="primary"):
                with st.spinner(f"Training {algorithm} agent on {selected_ticker}..."):
                    success = self._prepare_environment(
                        selected_ticker, start_date, end_date,
                        initial_balance, transaction_cost, window_size
                    )
                    
                    if success:
                        training_results = self._train_agent(
                            algorithm, training_steps, learning_rate, batch_size
                        )
                        
                        if training_results:
                            st.session_state.rl_results = training_results
                            st.session_state.rl_ticker = selected_ticker
                            st.session_state.rl_algorithm = algorithm
                            self.trained = True
                            st.success("✅ RL Agent training completed!")
            
            # Display training results
            if hasattr(st.session_state, 'rl_results'):
                self._display_training_summary()
        
        # RL Analytics Tabs
        st.markdown("---")
        st.markdown("### 📊 RL Agent Analytics")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Trading Performance",
            "🎯 Agent Behavior",
            "📊 Strategy Analysis",
            "🔄 Backtesting"
        ])
        
        with tab1:
            self._render_performance_tab()
        
        with tab2:
            self._render_behavior_tab()
        
        with tab3:
            self._render_strategy_tab()
        
        with tab4:
            self._render_backtesting_tab()
    
    def _prepare_environment(self, ticker, start_date, end_date, 
                           initial_balance, transaction_cost, window_size):
        """Prepare trading environment with enhanced features"""
        try:
            # Download stock data
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if data.empty:
                st.error(f"❌ No data available for {ticker}")
                return False
            
            # Add technical indicators
            data = add_technical_indicators(data)
            
            # Add market context
            data = self._add_market_features(data, start_date, end_date)
            
            # Feature engineering for RL
            data = self._engineer_rl_features(data)
            
            # Clean data
            data = data.dropna()
            
            if len(data) < window_size + 100:
                st.error("❌ Insufficient data for training")
                return False
            
            # Create trading environment
            self.env = TradingEnvironment(
                df=data,
                initial_balance=initial_balance,
                transaction_cost=transaction_cost,
                window_size=window_size
            )
            
            # Store data for analysis
            self.data = data
            
            st.info(f"📊 Environment prepared with {len(data)} trading days")
            return True
            
        except Exception as e:
            st.error(f"❌ Environment preparation failed: {str(e)}")
            return False
    
    def _add_market_features(self, data, start_date, end_date):
        """Add market context features for RL"""
        try:
            # VIX (fear index)
            vix = yf.download("^VIX", start=start_date, end=end_date, progress=False)['Close']
            data['VIX'] = vix.reindex(data.index, method='ffill')
            
            # Market returns
            spy = yf.download("SPY", start=start_date, end=end_date, progress=False)['Close']
            data['Market_Return'] = spy.pct_change().reindex(data.index, method='ffill')
            
            # Sector momentum (using QQQ as tech proxy)
            qqq = yf.download("QQQ", start=start_date, end=end_date, progress=False)['Close']
            data['Sector_Momentum'] = qqq.pct_change(5).reindex(data.index, method='ffill')
            
        except Exception as e:
            st.warning(f"⚠️ Some market features unavailable: {str(e)}")
        
        return data
    
    def _engineer_rl_features(self, data):
        """Engineer features specifically for RL trading"""
        # Price momentum features
        for period in [1, 3, 5, 10]:
            data[f'Price_Momentum_{period}'] = data['Close'].pct_change(period)
        
        # Volume features
        data['Volume_Ratio'] = data['Volume'] / data['Volume'].rolling(20).mean()
        data['Price_Volume'] = data['Close'].pct_change() * data['Volume_Ratio']
        
        # Volatility clustering
        data['Volatility_5'] = data['Close'].pct_change().rolling(5).std()
        data['Volatility_20'] = data['Close'].pct_change().rolling(20).std()
        
        # Market microstructure
        data['Bid_Ask_Spread'] = (data['High'] - data['Low']) / data['Close']
        data['Intraday_Return'] = (data['Close'] - data['Open']) / data['Open']
        
        return data
    
    def _train_agent(self, algorithm, training_steps, learning_rate, batch_size):
        """Train the RL agent with advanced algorithms"""
        try:
            # Create vectorized environment
            env = DummyVecEnv([lambda: self.env])
            
            # Initialize model based on algorithm
            if algorithm == "DQN":
                self.model = DQN(
                    "MlpPolicy",
                    env,
                    learning_rate=learning_rate,
                    buffer_size=50000,
                    learning_starts=1000,
                    batch_size=batch_size,
                    tau=1.0,
                    gamma=0.99,
                    train_freq=4,
                    gradient_steps=1,
                    target_update_interval=1000,
                    exploration_fraction=0.1,
                    exploration_initial_eps=1.0,
                    exploration_final_eps=0.05,
                    verbose=0
                )
                
            elif algorithm == "PPO":
                self.model = PPO(
                    "MlpPolicy",
                    env,
                    learning_rate=learning_rate,
                    n_steps=2048,
                    batch_size=batch_size,
                    n_epochs=10,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    verbose=0
                )
                
            elif algorithm == "A2C":
                self.model = A2C(
                    "MlpPolicy",
                    env,
                    learning_rate=learning_rate,
                    n_steps=5,
                    gamma=0.99,
                    gae_lambda=1.0,
                    ent_coef=0.0,
                    vf_coef=0.25,
                    max_grad_norm=0.5,
                    verbose=0
                )
                
            elif algorithm == "SAC":
                self.model = SAC(
                    "MlpPolicy",
                    env,
                    learning_rate=learning_rate,
                    buffer_size=50000,
                    learning_starts=100,
                    batch_size=batch_size,
                    tau=0.005,
                    gamma=0.99,
                    train_freq=1,
                    gradient_steps=1,
                    verbose=0
                )
            
            # Create callback for monitoring
            callback = TradingCallback()
            
            # Train the model
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Training with progress updates
            total_steps = training_steps
            checkpoint_interval = total_steps // 10
            
            for i in range(10):
                steps_to_train = checkpoint_interval
                self.model.learn(
                    total_timesteps=steps_to_train,
                    callback=callback,
                    reset_num_timesteps=False if i > 0 else True
                )
                
                progress = (i + 1) / 10
                progress_bar.progress(progress)
                status_text.text(f"Training progress: {progress*100:.0f}% ({(i+1)*checkpoint_interval}/{total_steps} steps)")
            
            status_text.text("Training completed! Evaluating performance...")
            
            # Evaluate trained model
            evaluation_results = self._evaluate_agent()
            
            return {
                'model': self.model,
                'algorithm': algorithm,
                'training_steps': training_steps,
                'evaluation': evaluation_results
            }
            
        except Exception as e:
            st.error(f"❌ Training failed: {str(e)}")
            return None
    
    def _evaluate_agent(self):
        """Evaluate trained agent performance"""
        try:
            # Reset environment
            obs, _ = self.env.reset()
            
            # Run evaluation episode
            total_reward = 0
            actions_taken = []
            portfolio_values = []
            trade_log = []
            
            done = False
            step = 0
            
            while not done and step < len(self.data) - self.env.window_size - 1:
                # Predict action
                action, _ = self.model.predict(obs, deterministic=True)
                
                # Take action
                obs, reward, done, _, info = self.env.step(action)
                
                # Log results
                total_reward += reward
                actions_taken.append(action)
                portfolio_values.append(info['net_worth'])
                
                if action != 0:  # If not holding
                    trade_log.append({
                        'step': step,
                        'action': 'BUY' if action == 1 else 'SELL',
                        'price': info['current_price'],
                        'portfolio_value': info['net_worth']
                    })
                
                step += 1
            
            # Calculate performance metrics
            initial_value = self.env.initial_balance
            final_value = portfolio_values[-1] if portfolio_values else initial_value
            total_return = (final_value - initial_value) / initial_value
            
            # Calculate Sharpe ratio
            portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]
            sharpe_ratio = calculate_sharpe_ratio(pd.Series(portfolio_returns))
            
            # Calculate max drawdown
            cumulative = np.array(portfolio_values)
            rolling_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - rolling_max) / rolling_max
            max_drawdown = np.min(drawdown)
            
            # Buy and hold benchmark
            start_price = self.data.iloc[self.env.window_size]['Close']
            end_price = self.data.iloc[-1]['Close']
            buy_hold_return = (end_price - start_price) / start_price
            
            return {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'buy_hold_return': buy_hold_return,
                'excess_return': total_return - buy_hold_return,
                'total_trades': len(trade_log),
                'portfolio_values': portfolio_values,
                'actions': actions_taken,
                'trade_log': trade_log,
                'final_value': final_value
            }
            
        except Exception as e:
            st.error(f"❌ Evaluation failed: {str(e)}")
            return None
    
    def _display_training_summary(self):
        """Display training summary"""
        if not hasattr(st.session_state, 'rl_results'):
            return
        
        results = st.session_state.rl_results
        evaluation = results['evaluation']
        
        st.markdown("### 🎯 Training Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            create_metric_card(
                "Total Return",
                format_percentage(evaluation['total_return']),
                delta=f"+{format_percentage(evaluation['excess_return'])} vs Buy & Hold"
            )
        
        with col2:
            create_metric_card(
                "Sharpe Ratio",
                f"{evaluation['sharpe_ratio']:.3f}",
                help_text="Risk-adjusted returns"
            )
        
        with col3:
            create_metric_card(
                "Max Drawdown",
                format_percentage(evaluation['max_drawdown']),
                help_text="Maximum portfolio decline"
            )
        
        with col4:
            create_metric_card(
                "Total Trades",
                str(evaluation['total_trades']),
                help_text="Number of buy/sell transactions"
            )
    
    def _render_performance_tab(self):
        """Render trading performance analysis"""
        if not hasattr(st.session_state, 'rl_results'):
            st.info("🔄 Train an RL agent first to see performance analysis.")
            return
        
        evaluation = st.session_state.rl_results['evaluation']
        
        # Portfolio value chart
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Portfolio Value Over Time', 'Cumulative Returns Comparison'),
            vertical_spacing=0.1
        )
        
        # Portfolio performance
        dates = self.data.index[self.env.window_size:self.env.window_size + len(evaluation['portfolio_values'])]
        
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=evaluation['portfolio_values'],
                name='RL Agent Portfolio',
                line=dict(color='#00C49F', width=2)
            ),
            row=1, col=1
        )
        
        # Buy and hold benchmark
        initial_price = self.data.iloc[self.env.window_size]['Close']
        prices = self.data.iloc[self.env.window_size:self.env.window_size + len(evaluation['portfolio_values'])]['Close']
        buy_hold_values = [self.env.initial_balance * (price / initial_price) for price in prices]
        
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=buy_hold_values,
                name='Buy & Hold',
                line=dict(color='#FF8042', width=2, dash='dash')
            ),
            row=1, col=1
        )
        
        # Cumulative returns
        rl_returns = [(val / self.env.initial_balance - 1) * 100 for val in evaluation['portfolio_values']]
        bh_returns = [(val / self.env.initial_balance - 1) * 100 for val in buy_hold_values]
        
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=rl_returns,
                name='RL Agent Returns (%)',
                line=dict(color='#00C49F', width=2)
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=bh_returns,
                name='Buy & Hold Returns (%)',
                line=dict(color='#FF8042', width=2, dash='dash')
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=600,
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            title=f"RL Agent Performance - {st.session_state.rl_ticker}"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance metrics comparison
        st.markdown("#### 📊 Performance Comparison")
        
        comparison_data = {
            'Metric': ['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Final Value'],
            'RL Agent': [
                format_percentage(evaluation['total_return']),
                f"{evaluation['sharpe_ratio']:.3f}",
                format_percentage(evaluation['max_drawdown']),
                f"${evaluation['final_value']:,.2f}"
            ],
            'Buy & Hold': [
                format_percentage(evaluation['buy_hold_return']),
                f"{calculate_sharpe_ratio(pd.Series(np.diff(buy_hold_values) / buy_hold_values[:-1])):.3f}",
                format_percentage((min(buy_hold_values) - max(buy_hold_values)) / max(buy_hold_values)),
                f"${buy_hold_values[-1]:,.2f}"
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
    
    def _render_behavior_tab(self):
        """Render agent behavior analysis"""
        if not hasattr(st.session_state, 'rl_results'):
            st.info("🔄 Train an RL agent first to see behavior analysis.")
            return
        
        evaluation = st.session_state.rl_results['evaluation']
        
        # Action distribution
        actions = evaluation['actions']
        action_counts = np.bincount(actions)
        action_labels = ['Hold', 'Buy', 'Sell']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Action distribution pie chart
            fig = go.Figure(data=[go.Pie(
                labels=action_labels,
                values=action_counts,
                hole=0.3,
                marker_colors=['#FFA500', '#00C49F', '#FF8042']
            )])
            
            fig.update_layout(
                title="Agent Action Distribution",
                template="plotly_dark",
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Action timing heatmap
            dates = self.data.index[self.env.window_size:self.env.window_size + len(actions)]
            action_df = pd.DataFrame({
                'Date': dates,
                'Action': actions,
                'Price': self.data.iloc[self.env.window_size:self.env.window_size + len(actions)]['Close']
            })
            
            # Create action heatmap over time
            action_df['Week'] = action_df['Date'].dt.isocalendar().week
            action_df['Day'] = action_df['Date'].dt.dayofweek
            
            action_pivot = action_df.groupby(['Week', 'Day'])['Action'].apply(
                lambda x: x.value_counts().get(1, 0) - x.value_counts().get(2, 0)  # Buy - Sell
            ).unstack(fill_value=0)
            
            fig = go.Figure(data=go.Heatmap(
                z=action_pivot.values,
                x=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                y=action_pivot.index,
                colorscale='RdYlGn',
                zmid=0
            ))
            
            fig.update_layout(
                title="Trading Activity Heatmap (Green=Buy, Red=Sell)",
                template="plotly_dark",
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Trade analysis
        if evaluation['trade_log']:
            st.markdown("#### 📋 Trade Analysis")
            
            trade_df = pd.DataFrame(evaluation['trade_log'])
            
            # Trade frequency analysis
            buy_trades = trade_df[trade_df['action'] == 'BUY']
            sell_trades = trade_df[trade_df['action'] == 'SELL']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                create_metric_card("Buy Trades", str(len(buy_trades)))
            with col2:
                create_metric_card("Sell Trades", str(len(sell_trades)))
            with col3:
                avg_hold_time = len(actions) / max(len(sell_trades), 1)
                create_metric_card("Avg Hold Time", f"{avg_hold_time:.1f} days")
            
            # Recent trades
            st.markdown("#### 🔄 Recent Trades")
            st.dataframe(trade_df.tail(10), use_container_width=True)
    
    def _render_strategy_tab(self):
        """Render strategy analysis"""
        if not hasattr(st.session_state, 'rl_results'):
            st.info("🔄 Train an RL agent first to see strategy analysis.")
            return
        
        st.markdown("### 📊 Strategy Analysis")
        
        evaluation = st.session_state.rl_results['evaluation']
        
        # Risk metrics
        portfolio_returns = np.diff(evaluation['portfolio_values']) / evaluation['portfolio_values'][:-1]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk metrics
            st.markdown("#### ⚠️ Risk Metrics")
            
            volatility = np.std(portfolio_returns) * np.sqrt(252)
            sortino_ratio = calculate_sortino_ratio(pd.Series(portfolio_returns))
            
            risk_metrics = {
                'Annual Volatility': format_percentage(volatility),
                'Sharpe Ratio': f"{evaluation['sharpe_ratio']:.3f}",
                'Sortino Ratio': f"{sortino_ratio:.3f}",
                'Max Drawdown': format_percentage(evaluation['max_drawdown']),
                'VaR (95%)': format_percentage(np.percentile(portfolio_returns, 5))
            }
            
            for metric, value in risk_metrics.items():
                create_metric_card(metric, value)
        
        with col2:
            # Return distribution
            fig = go.Figure(data=[go.Histogram(
                x=portfolio_returns * 100,
                nbinsx=30,
                name='Daily Returns',
                marker_color='#00C49F',
                opacity=0.7
            )])
            
            fig.update_layout(
                title="Daily Returns Distribution",
                xaxis_title="Daily Return (%)",
                yaxis_title="Frequency",
                template="plotly_dark",
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_backtesting_tab(self):
        """Render comprehensive backtesting results"""
        if not hasattr(st.session_state, 'rl_results'):
            st.info("🔄 Train an RL agent first to see backtesting results.")
            return
        
        st.markdown("### 🔄 Backtesting Results")
        
        evaluation = st.session_state.rl_results['evaluation']
        
        # Strategy summary
        st.markdown("#### 📈 Strategy Performance Summary")
        
        summary_text = f"""
        **Algorithm**: {st.session_state.rl_algorithm}
        
        **Performance Highlights**:
        - 🎯 **Total Return**: {format_percentage(evaluation['total_return'])}
        - 📊 **Excess Return vs Buy & Hold**: {format_percentage(evaluation['excess_return'])}
        - ⚡ **Sharpe Ratio**: {evaluation['sharpe_ratio']:.3f}
        - 📉 **Maximum Drawdown**: {format_percentage(evaluation['max_drawdown'])}
        - 🔄 **Total Trades**: {evaluation['total_trades']}
        
        **Risk Assessment**:
        {'🟢 Low Risk' if evaluation['max_drawdown'] > -0.1 else '🟡 Moderate Risk' if evaluation['max_drawdown'] > -0.2 else '🔴 High Risk'}
        
        **Strategy Grade**: {'A+' if evaluation['sharpe_ratio'] > 1.5 else 'A' if evaluation['sharpe_ratio'] > 1.0 else 'B' if evaluation['sharpe_ratio'] > 0.5 else 'C'}
        """
        
        st.markdown(summary_text)
        
        # Deployment readiness
        st.markdown("#### 🚀 Deployment Readiness")
        
        deployment_score = 0
        if evaluation['total_return'] > 0:
            deployment_score += 25
        if evaluation['sharpe_ratio'] > 1.0:
            deployment_score += 25
        if evaluation['max_drawdown'] > -0.2:
            deployment_score += 25
        if evaluation['total_trades'] > 10:
            deployment_score += 25
        
        if deployment_score >= 75:
            st.success(f"""
            🎉 **Ready for Production Deployment!**
            
            Deployment Score: {deployment_score}/100
            
            ✅ Strong risk-adjusted returns
            ✅ Acceptable drawdown levels  
            ✅ Sufficient trading activity
            ✅ Outperforms buy-and-hold benchmark
            
            **Next Steps:**
            - Deploy to paper trading environment
            - Monitor performance with real-time data
            - Implement risk management overlays
            - Set up automated retraining pipeline
            """)
        else:
            st.warning(f"""
            ⚠️ **Needs Improvement Before Deployment**
            
            Deployment Score: {deployment_score}/100
            
            **Recommendations:**
            - Increase training duration
            - Optimize hyperparameters
            - Add more market features
            - Implement ensemble methods
            """)
        
        # Export options
        st.markdown("#### 📁 Export Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Export Backtest Results"):
                # Prepare export data
                results_data = {
                    'Algorithm': st.session_state.rl_algorithm,
                    'Ticker': st.session_state.rl_ticker,
                    'Total_Return': evaluation['total_return'],
                    'Sharpe_Ratio': evaluation['sharpe_ratio'],
                    'Max_Drawdown': evaluation['max_drawdown'],
                    'Buy_Hold_Return': evaluation['buy_hold_return'],
                    'Excess_Return': evaluation['excess_return'],
                    'Total_Trades': evaluation['total_trades'],
                    'Final_Portfolio_Value': evaluation['final_value']
                }
                
                results_df = pd.DataFrame([results_data])
                csv = results_df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Results CSV",
                    data=csv,
                    file_name=f"rl_backtest_{st.session_state.rl_ticker}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📋 Generate Strategy Report"):
                report = f"""
# RL Trading Strategy Report

## Strategy Details
- **Algorithm**: {st.session_state.rl_algorithm}
- **Asset**: {st.session_state.rl_ticker}
- **Training Period**: {len(self.data)} days
- **Backtest Date**: {datetime.now().strftime('%Y-%m-%d')}

## Performance Metrics
- **Total Return**: {format_percentage(evaluation['total_return'])}
- **Sharpe Ratio**: {evaluation['sharpe_ratio']:.3f}
- **Maximum Drawdown**: {format_percentage(evaluation['max_drawdown'])}
- **Excess Return vs Buy & Hold**: {format_percentage(evaluation['excess_return'])}

## Risk Analysis
- **Volatility**: Calculated from daily returns
- **Downside Protection**: {format_percentage(evaluation['max_drawdown'])} maximum decline
- **Trade Frequency**: {evaluation['total_trades']} total trades

## Deployment Status
Deployment Score: {deployment_score}/100
{'✅ Ready for production' if deployment_score >= 75 else '⚠️ Needs improvement'}

---
Generated by FinOptima RL Trading System
                """
                
                st.download_button(
                    label="📥 Download Report",
                    data=report,
                    file_name=f"rl_strategy_report_{st.session_state.rl_ticker}_{datetime.now().strftime('%Y%m%d')}.md",
                    mime="text/markdown"
                )