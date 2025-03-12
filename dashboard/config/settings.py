"""Dashboard configuration settings."""

# Page configuration
PAGE_CONFIG = {
    "page_title": "Stock Market Analytics Dashboard",
    "page_icon": "📈",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Default values
DEFAULTS = {
    "symbol": "AAPL",
    "lookback_days": 365,
    "indicators": ["SMA", "RSI"],
    "sma_period": 20,
    "rsi_period": 14,
    "signal_threshold": 0.5
}

# Technical indicators options
TECHNICAL_INDICATORS = [
    "SMA",
    "EMA",
    "MACD",
    "RSI",
    "Bollinger Bands"
]

# Timeframe options
TIMEFRAMES = [
    "1D",
    "5D",
    "1M",
    "3M",
    "6M",
    "1Y",
    "YTD"
]

# Chart settings
CHART_SETTINGS = {
    "price_chart_height": 600,
    "indicator_chart_height": 300,
    "chart_margins": dict(l=0, r=0, t=30, b=0)
} 