from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, precision_score
import pandas as pd
import numpy as np
import ta

class MarketDataPreprocessor(BaseEstimator, TransformerMixin):
    """Custom transformer for market data preprocessing"""
    
    def __init__(self, feature_engineering=True):
        self.feature_engineering = feature_engineering
        self.technical_features = {}
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        
        if self.feature_engineering:
            # Add technical indicators
            # Trend indicators
            df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
            df['ema_20'] = ta.trend.ema_indicator(df['close'], window=20)
            df['macd'] = ta.trend.macd_diff(df['close'])
            
            # Momentum indicators
            df['rsi'] = ta.momentum.rsi(df['close'])
            df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
            
            # Volatility indicators
            df['bb_high'] = ta.volatility.bollinger_hband(df['close'])
            df['bb_low'] = ta.volatility.bollinger_lband(df['close'])
            df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
            
            # Volume indicators
            df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
            df['adi'] = ta.volume.acc_dist_index(df['high'], df['low'], df['close'], df['volume'])
            
            # Custom features
            df['price_range'] = df['high'] - df['low']
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(window=20).std()
            
        # Handle missing values
        df = df.fillna(method='ffill')
        df = df.fillna(0)  # For any remaining NaN values
        
        return df

class TechnicalFeatureSelector(BaseEstimator, TransformerMixin):
    """Advanced feature selection for technical indicators"""
    
    def __init__(self, n_features_to_select='auto', selection_method='importance'):
        self.n_features_to_select = n_features_to_select
        self.selection_method = selection_method
        self.selected_features_ = None
        
    def fit(self, X, y):
        if self.selection_method == 'importance':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            selector = SelectFromModel(
                model,
                max_features=self.n_features_to_select if self.n_features_to_select != 'auto' else None,
                threshold='median'
            )
            selector.fit(X, y)
            self.selected_features_ = selector.get_support()
        
        return self
    
    def transform(self, X):
        return X[:, self.selected_features_]

def create_market_pipeline(price_features, technical_features):
    """Creates an advanced market data processing pipeline"""
    
    # Price data preprocessing
    price_transformer = Pipeline(steps=[
        ('scaler', RobustScaler())  # More robust to outliers
    ])

    # Technical indicator preprocessing
    technical_transformer = Pipeline(steps=[
        ('preprocessor', MarketDataPreprocessor(feature_engineering=True)),
        ('scaler', StandardScaler())
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('price', price_transformer, price_features),
            ('technical', technical_transformer, technical_features)
        ])

    # Create full pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('feature_selector', TechnicalFeatureSelector()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    return pipeline

def train_market_model(X, y, pipeline, prediction_horizon=5):
    """Trains the market model with time series cross-validation"""
    
    # Define parameter grid for hyperparameter optimization
    param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5, 10],
        'feature_selector__n_features_to_select': ['auto', 0.5, 0.7]
    }
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Custom scoring for financial metrics
    def financial_score(y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        return 0.7 * accuracy + 0.3 * precision  # Custom weight for financial importance
    
    # Perform grid search with time series cross-validation
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=tscv,
        scoring=make_scorer(financial_score),
        n_jobs=-1,
        verbose=1
    )
    
    # Fit the model
    grid_search.fit(X, y)
    
    return grid_search.best_estimator_

class MarketSignalGenerator:
    """Generates trading signals based on model predictions and market conditions"""
    
    def __init__(self, model, risk_threshold=0.5):
        self.model = model
        self.risk_threshold = risk_threshold
        
    def generate_signals(self, market_data):
        # Get model predictions
        predictions = self.model.predict_proba(market_data)
        
        # Calculate signal strength
        signal_strength = np.zeros(len(predictions))
        for i, pred in enumerate(predictions):
            if pred[1] > self.risk_threshold:  # Buy signal
                signal_strength[i] = pred[1]
            elif pred[0] > self.risk_threshold:  # Sell signal
                signal_strength[i] = -pred[0]
                
        return signal_strength

def create_signal_generator(trained_model, risk_threshold=0.5):
    """Creates a signal generator based on the trained model"""
    return MarketSignalGenerator(trained_model, risk_threshold) 