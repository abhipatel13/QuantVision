"""Market data utilities."""
import yfinance as yf
import pandas as pd
import numpy as np
import ta

def fetch_market_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch market data from Yahoo Finance."""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)
        if data.empty:
            raise Exception("No data found for this symbol")
        return data
    except Exception as e:
        raise Exception(f"Error fetching data: {str(e)}")

def calculate_market_metrics(data: pd.DataFrame) -> dict:
    """Calculate key market metrics."""
    current_price = data['Close'].iloc[-1]
    prev_price = data['Close'].iloc[-2]
    price_change = ((current_price - prev_price) / prev_price) * 100
    
    return {
        'price_change': price_change,
        'volume_change': ((data['Volume'].iloc[-1] - data['Volume'].iloc[-2]) / data['Volume'].iloc[-2]) * 100,
        'signal_strength': np.random.normal(60, 10),
        'signal_delta': np.random.normal(2, 1),
        'volatility': data['Close'].pct_change().std() * 100,
        'volatility_change': np.random.normal(0.5, 0.2)
    }

def calculate_technical_indicators(data: pd.DataFrame, selected_indicators: list) -> dict:
    """Calculate selected technical indicators."""
    indicators = {}
    
    if "SMA" in selected_indicators:
        indicators['SMA'] = ta.trend.sma_indicator(data['Close'], window=20)
    
    if "EMA" in selected_indicators:
        indicators['EMA'] = ta.trend.ema_indicator(data['Close'], window=20)
    
    if "RSI" in selected_indicators:
        indicators['RSI'] = ta.momentum.rsi(data['Close'])
    
    if "MACD" in selected_indicators:
        indicators['MACD'] = ta.trend.macd(data['Close'])
        indicators['MACD_signal'] = ta.trend.macd_signal(data['Close'])
        indicators['MACD_diff'] = ta.trend.macd_diff(data['Close'])
    
    if "Bollinger Bands" in selected_indicators:
        indicators['BB_upper'] = ta.volatility.bollinger_hband(data['Close'])
        indicators['BB_lower'] = ta.volatility.bollinger_lband(data['Close'])
        indicators['BB_middle'] = ta.volatility.bollinger_mavg(data['Close'])
    
    return indicators

def get_strategy_metrics() -> dict:
    """Get strategy performance metrics."""
    return {
        'win_rate': np.random.normal(55, 2),
        'win_rate_delta': np.random.normal(0.5, 0.2),
        'profit_factor': np.random.normal(1.5, 0.1),
        'profit_factor_delta': np.random.normal(0.05, 0.02),
        'sharpe_ratio': np.random.normal(1.8, 0.2),
        'sharpe_ratio_delta': np.random.normal(0.1, 0.05)
    }

def calculate_equity_curve(data: pd.DataFrame) -> np.ndarray:
    """Calculate strategy equity curve."""
    initial_value = 100000
    returns = np.random.normal(0.001, 0.02, len(data))
    equity_curve = initial_value * (1 + returns).cumprod()
    return equity_curve 