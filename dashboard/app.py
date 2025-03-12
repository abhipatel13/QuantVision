"""Main dashboard application."""
import streamlit as st
import sys
from pathlib import Path

# Add the parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from dashboard.config.settings import PAGE_CONFIG
from dashboard.styles.theme import CUSTOM_CSS
from dashboard.components.sidebar import render_sidebar
from dashboard.components.charts import (
    create_price_volume_chart,
    create_signal_gauge,
    create_rsi_chart,
    create_macd_chart,
    create_equity_curve_chart
)
from dashboard.utils.market_data import (
    calculate_market_metrics,
    calculate_technical_indicators,
    get_strategy_metrics,
    calculate_equity_curve
)

class Dashboard:
    def __init__(self):
        self.initialize_session_state()
        self.setup_page()
        
    @staticmethod
    def initialize_session_state():
        """Initialize session state variables"""
        if 'market_data' not in st.session_state:
            st.session_state.market_data = None
        if 'model' not in st.session_state:
            st.session_state.model = None
            
    @staticmethod
    def setup_page():
        """Setup page configuration and styling"""
        st.set_page_config(**PAGE_CONFIG)
        st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
            
    def main(self):
        """Main dashboard layout"""
        st.title("📈 Advanced Stock Market Analytics")
        
        # Render sidebar and get user inputs
        inputs = render_sidebar()
        
        if st.session_state.market_data is not None:
            self.render_main_content(inputs)
        else:
            self.render_welcome_screen()
            
    @staticmethod
    def render_welcome_screen():
        """Display welcome screen when no data is loaded"""
        st.markdown("""
        ### 👋 Welcome to Stock Market Analytics
        
        Get started by following these steps:
        1. Enter a stock symbol in the sidebar (e.g., AAPL, GOOGL)
        2. Select your desired date range
        3. Click 'Fetch Data' to begin analysis
        
        #### Features:
        - Real-time market data analysis
        - Technical indicators
        - Trading signals
        - Performance metrics
        - Advanced charting
        """)
            
    def render_main_content(self, inputs: dict):
        """Render main dashboard content"""
        # Calculate metrics and indicators
        metrics = calculate_market_metrics(st.session_state.market_data)
        indicators = calculate_technical_indicators(
            st.session_state.market_data,
            inputs['selected_indicators']
        )
        
        # Render metrics
        self.render_key_metrics(metrics)
        
        # Charts section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Price Action & Volume")
            fig = create_price_volume_chart(st.session_state.market_data, indicators)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Trading Signals")
            fig = create_signal_gauge(metrics['signal_strength'])
            st.plotly_chart(fig, use_container_width=True)
        
        # Technical Analysis Section
        st.header("📊 Technical Analysis")
        self.render_technical_analysis(indicators)
        
        # Performance Metrics
        st.header("🎯 Strategy Performance")
        self.render_strategy_performance()
        
    @staticmethod
    def render_key_metrics(metrics: dict):
        """Render key metrics at the top of the dashboard"""
        cols = st.columns(4)
        
        with cols[0]:
            st.metric(
                "Current Price",
                f"${st.session_state.market_data['Close'].iloc[-1]:.2f}",
                f"{metrics['price_change']:.2f}%"
            )
        
        with cols[1]:
            st.metric(
                "Trading Volume",
                f"{st.session_state.market_data['Volume'].iloc[-1]:,.0f}",
                f"{metrics['volume_change']:.2f}%"
            )
            
        with cols[2]:
            st.metric(
                "Signal Strength",
                f"{metrics['signal_strength']:.1f}%",
                f"{metrics['signal_delta']:.1f}%"
            )
            
        with cols[3]:
            st.metric(
                "Volatility",
                f"{metrics['volatility']:.2f}%",
                f"{metrics['volatility_change']:.2f}%"
            )
            
    def render_technical_analysis(self, indicators: dict):
        """Render technical analysis section"""
        col1, col2 = st.columns(2)
        
        with col1:
            if 'RSI' in indicators:
                fig = create_rsi_chart(st.session_state.market_data, indicators['RSI'])
                st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            if all(k in indicators for k in ['MACD', 'MACD_signal']):
                fig = create_macd_chart(
                    st.session_state.market_data,
                    indicators['MACD'],
                    indicators['MACD_signal']
                )
                st.plotly_chart(fig, use_container_width=True)
                
    def render_strategy_performance(self):
        """Render strategy performance metrics"""
        metrics = get_strategy_metrics()
        
        cols = st.columns(3)
        
        with cols[0]:
            st.metric(
                "Win Rate",
                f"{metrics['win_rate']:.1f}%",
                f"{metrics['win_rate_delta']:.1f}%"
            )
            
        with cols[1]:
            st.metric(
                "Profit Factor",
                f"{metrics['profit_factor']:.2f}",
                f"{metrics['profit_factor_delta']:.2f}"
            )
            
        with cols[2]:
            st.metric(
                "Sharpe Ratio",
                f"{metrics['sharpe_ratio']:.2f}",
                f"{metrics['sharpe_ratio_delta']:.2f}"
            )
            
        # Equity curve
        equity_curve = calculate_equity_curve(st.session_state.market_data)
        fig = create_equity_curve_chart(st.session_state.market_data, equity_curve)
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    dashboard = Dashboard()
    dashboard.main()