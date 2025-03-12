"""Theme configuration for the dashboard."""

CUSTOM_CSS = """
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stButton>button {
        background-color: #1f6feb;
        color: white;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        border: none;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #388bfd;
        border-color: #388bfd;
    }
    .sidebar .sidebar-content {
        background-color: #161b22;
    }
    .st-emotion-cache-1y4p8pa {
        padding: 2rem 1rem;
    }
    .st-emotion-cache-1wmy9hl {
        max-width: 100%;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
    }
    div[data-testid="stMetricDelta"] {
        font-size: 16px;
    }
    .plot-container {
        background-color: #161b22;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    h1, h2, h3 {
        color: #ffffff;
    }
    .stSelectbox label {
        color: #ffffff;
    }
    </style>
"""

PLOTLY_TEMPLATE = 'plotly_dark'

CHART_COLORS = {
    'primary': '#1f6feb',
    'secondary': '#388bfd',
    'success': '#4caf50',
    'warning': '#ffb74d',
    'danger': '#ff4b4b',
} 