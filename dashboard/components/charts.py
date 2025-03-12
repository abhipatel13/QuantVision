"""Chart components for the dashboard."""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from dashboard.styles.theme import PLOTLY_TEMPLATE, CHART_COLORS
from dashboard.config.settings import CHART_SETTINGS

def create_price_volume_chart(data: pd.DataFrame, indicators: dict) -> go.Figure:
    """Create price and volume chart."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3]
    )
    
    # Add candlestick
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="Price"
        ),
        row=1, col=1
    )
    
    # Add indicators
    for name, indicator_data in indicators.items():
        if name in ['SMA', 'EMA', 'BB_upper', 'BB_lower', 'BB_middle']:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=indicator_data,
                    name=name,
                    line=dict(color=CHART_COLORS['secondary'])
                ),
                row=1, col=1
            )
    
    # Add volume bars
    colors = ['red' if row['Open'] - row['Close'] >= 0 
             else 'green' for index, row in data.iterrows()]
    
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['Volume'],
            name="Volume",
            marker_color=colors,
            opacity=0.7
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=CHART_SETTINGS['price_chart_height'],
        template=PLOTLY_TEMPLATE,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        margin=CHART_SETTINGS['chart_margins']
    )
    
    return fig

def create_signal_gauge(signal_strength: float) -> go.Figure:
    """Create signal strength gauge."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=signal_strength,
        delta={'reference': 50},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': CHART_COLORS['primary']},
            'steps': [
                {'range': [0, 30], 'color': CHART_COLORS['danger']},
                {'range': [30, 70], 'color': CHART_COLORS['warning']},
                {'range': [70, 100], 'color': CHART_COLORS['success']}
            ]
        },
        title={'text': "Signal Strength"}
    ))
    
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=CHART_SETTINGS['indicator_chart_height'],
        margin=CHART_SETTINGS['chart_margins']
    )
    
    return fig

def create_rsi_chart(data: pd.DataFrame, rsi_data: pd.Series) -> go.Figure:
    """Create RSI chart."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=rsi_data,
        name="RSI",
        line=dict(color=CHART_COLORS['primary'])
    ))
    
    # Add RSI levels
    fig.add_hline(y=70, line_dash="dash", line_color=CHART_COLORS['danger'], opacity=0.5)
    fig.add_hline(y=30, line_dash="dash", line_color=CHART_COLORS['success'], opacity=0.5)
    
    fig.update_layout(
        title="RSI (14)",
        template=PLOTLY_TEMPLATE,
        height=CHART_SETTINGS['indicator_chart_height'],
        margin=CHART_SETTINGS['chart_margins'],
        yaxis=dict(range=[0, 100])
    )
    
    return fig

def create_macd_chart(data: pd.DataFrame, macd: pd.Series, signal: pd.Series) -> go.Figure:
    """Create MACD chart."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=macd,
        name="MACD",
        line=dict(color=CHART_COLORS['primary'])
    ))
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=signal,
        name="Signal",
        line=dict(color=CHART_COLORS['danger'])
    ))
    
    fig.update_layout(
        title="MACD",
        template=PLOTLY_TEMPLATE,
        height=CHART_SETTINGS['indicator_chart_height'],
        margin=CHART_SETTINGS['chart_margins']
    )
    
    return fig

def create_equity_curve_chart(data: pd.DataFrame, equity_curve: pd.Series) -> go.Figure:
    """Create equity curve chart."""
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=equity_curve,
            name="Strategy Performance",
            fill='tozeroy',
            line=dict(color=CHART_COLORS['primary'])
        )
    )
    
    fig.update_layout(
        title="Strategy Equity Curve",
        template=PLOTLY_TEMPLATE,
        height=CHART_SETTINGS['indicator_chart_height'],
        margin=CHART_SETTINGS['chart_margins']
    )
    
    return fig 