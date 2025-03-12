"""Sidebar component for the dashboard."""
import streamlit as st
from datetime import datetime, timedelta
from dashboard.config.settings import DEFAULTS, TECHNICAL_INDICATORS, TIMEFRAMES
from dashboard.utils.market_data import fetch_market_data

def render_sidebar() -> dict:
    """Render sidebar and return user inputs."""
    st.sidebar.title("Navigation")
    
    # Stock symbol input with validation
    symbol = st.sidebar.text_input(
        "Enter Stock Symbol",
        value=DEFAULTS["symbol"],
        help="Enter a valid stock symbol (e.g., AAPL, GOOGL)"
    ).upper()
    
    # Date range selection
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            datetime.now() - timedelta(days=DEFAULTS["lookback_days"]),
            max_value=datetime.now()
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            datetime.now(),
            max_value=datetime.now()
        )
    
    # Analysis settings
    st.sidebar.header("Analysis Settings")
    
    # Timeframe selection
    timeframe = st.sidebar.selectbox(
        "Select Timeframe",
        TIMEFRAMES,
        help="Select the timeframe for analysis"
    )
    
    # Technical indicators
    selected_indicators = st.sidebar.multiselect(
        "Select Technical Indicators",
        TECHNICAL_INDICATORS,
        default=DEFAULTS["indicators"],
        help="Choose technical indicators to display"
    )
    
    # Advanced settings
    with st.sidebar.expander("Advanced Settings"):
        sma_period = st.slider(
            "SMA Period",
            5, 50,
            DEFAULTS["sma_period"]
        )
        rsi_period = st.slider(
            "RSI Period",
            5, 30,
            DEFAULTS["rsi_period"]
        )
        signal_threshold = st.slider(
            "Signal Threshold",
            0.0, 1.0,
            DEFAULTS["signal_threshold"]
        )
    
    # Fetch data button with loading state
    if st.sidebar.button("Fetch Data", type="primary"):
        with st.sidebar.status("Fetching data...", expanded=True) as status:
            try:
                data = fetch_market_data(symbol, start_date, end_date)
                st.session_state.market_data = data
                status.update(label="Data fetched successfully!", state="complete")
            except Exception as e:
                status.update(label=str(e), state="error")
                st.session_state.market_data = None
    
    return {
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
        "timeframe": timeframe,
        "selected_indicators": selected_indicators,
        "settings": {
            "sma_period": sma_period,
            "rsi_period": rsi_period,
            "signal_threshold": signal_threshold
        }
    } 