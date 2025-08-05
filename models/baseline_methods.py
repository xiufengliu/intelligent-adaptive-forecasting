"""
Baseline forecasting methods for comparison with I-ASNH
Includes statistical, neural, and selection baseline methods
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

class BaselineMethod:
    """Base class for all forecasting methods"""
    
    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
    
    def fit(self, train_data: np.ndarray):
        """Fit the model to training data"""
        raise NotImplementedError
    
    def predict(self, horizon: int) -> np.ndarray:
        """Generate forecasts for specified horizon"""
        raise NotImplementedError
    
    def fit_predict(self, train_data: np.ndarray, horizon: int) -> np.ndarray:
        """Fit model and generate predictions"""
        self.fit(train_data)
        return self.predict(horizon)

class NaiveMethod(BaselineMethod):
    """Naive forecasting method - repeats last value"""
    
    def __init__(self):
        super().__init__("Naive")
        self.last_value = None
    
    def fit(self, train_data: np.ndarray):
        self.last_value = train_data[-1]
        self.is_fitted = True
    
    def predict(self, horizon: int) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return np.full(horizon, self.last_value)

class SeasonalNaiveMethod(BaselineMethod):
    """Seasonal naive method - repeats seasonal pattern"""
    
    def __init__(self, season_length: int = 24):
        super().__init__("Seasonal_Naive")
        self.season_length = season_length
        self.seasonal_pattern = None
    
    def fit(self, train_data: np.ndarray):
        if len(train_data) >= self.season_length:
            self.seasonal_pattern = train_data[-self.season_length:]
        else:
            self.seasonal_pattern = train_data
        self.is_fitted = True
    
    def predict(self, horizon: int) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = []
        pattern_len = len(self.seasonal_pattern)
        
        for i in range(horizon):
            predictions.append(self.seasonal_pattern[i % pattern_len])
        
        return np.array(predictions)

class LinearMethod(BaselineMethod):
    """Linear regression forecasting method"""
    
    def __init__(self, lookback: int = 24):
        super().__init__("Linear")
        self.lookback = lookback
        self.model = LinearRegression()
        self.train_data = None
    
    def fit(self, train_data: np.ndarray):
        self.train_data = train_data
        
        # Create features and targets for supervised learning
        X, y = [], []
        for i in range(self.lookback, len(train_data)):
            X.append(train_data[i-self.lookback:i])
            y.append(train_data[i])
        
        if len(X) > 0:
            X = np.array(X)
            y = np.array(y)
            self.model.fit(X, y)
            self.is_fitted = True
        else:
            # Fallback to naive if insufficient data
            self.last_value = train_data[-1]
            self.is_fitted = True
    
    def predict(self, horizon: int) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if hasattr(self, 'last_value'):  # Fallback case
            return np.full(horizon, self.last_value)
        
        predictions = []
        current_window = self.train_data[-self.lookback:].copy()
        
        for _ in range(horizon):
            pred = self.model.predict(current_window.reshape(1, -1))[0]
            predictions.append(pred)
            # Update window for next prediction
            current_window = np.roll(current_window, -1)
            current_window[-1] = pred
        
        return np.array(predictions)

class DLinearMethod(BaselineMethod):
    """DLinear method with decomposition"""
    
    def __init__(self, seq_len: int = 96, pred_len: int = 24):
        super().__init__("DLinear")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.trend_model = LinearRegression()
        self.seasonal_model = LinearRegression()
        self.train_data = None
    
    def decompose(self, data: np.ndarray) -> tuple:
        """Simple moving average decomposition"""
        # Trend component using moving average
        window = min(25, len(data) // 4)
        if window < 3:
            trend = np.full_like(data, np.mean(data))
        else:
            trend = pd.Series(data).rolling(window=window, center=True).mean().fillna(method='bfill').fillna(method='ffill').values
        
        # Seasonal component (residual)
        seasonal = data - trend
        
        return trend, seasonal
    
    def fit(self, train_data: np.ndarray):
        self.train_data = train_data
        
        # Decompose training data
        trend, seasonal = self.decompose(train_data)
        
        # Prepare features for both components
        X_trend, y_trend = [], []
        X_seasonal, y_seasonal = [], []
        
        lookback = min(self.seq_len, len(train_data) - 1)
        
        for i in range(lookback, len(train_data)):
            X_trend.append(trend[i-lookback:i])
            y_trend.append(trend[i])
            X_seasonal.append(seasonal[i-lookback:i])
            y_seasonal.append(seasonal[i])
        
        if len(X_trend) > 0:
            X_trend = np.array(X_trend)
            y_trend = np.array(y_trend)
            X_seasonal = np.array(X_seasonal)
            y_seasonal = np.array(y_seasonal)
            
            self.trend_model.fit(X_trend, y_trend)
            self.seasonal_model.fit(X_seasonal, y_seasonal)
            self.is_fitted = True
        else:
            self.last_value = train_data[-1]
            self.is_fitted = True
    
    def predict(self, horizon: int) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if hasattr(self, 'last_value'):  # Fallback case
            return np.full(horizon, self.last_value)
        
        # Decompose recent data for prediction
        recent_data = self.train_data[-self.seq_len:]
        trend, seasonal = self.decompose(recent_data)
        
        predictions = []
        current_trend = trend.copy()
        current_seasonal = seasonal.copy()
        
        for _ in range(horizon):
            # Predict trend and seasonal components
            trend_pred = self.trend_model.predict(current_trend[-self.seq_len:].reshape(1, -1))[0]
            seasonal_pred = self.seasonal_model.predict(current_seasonal[-self.seq_len:].reshape(1, -1))[0]
            
            # Combine components
            pred = trend_pred + seasonal_pred
            predictions.append(pred)
            
            # Update windows
            current_trend = np.append(current_trend, trend_pred)[-self.seq_len:]
            current_seasonal = np.append(current_seasonal, seasonal_pred)[-self.seq_len:]
        
        return np.array(predictions)

class ARIMAMethod(BaselineMethod):
    """ARIMA forecasting method with automatic parameter selection"""
    
    def __init__(self, max_p: int = 3, max_d: int = 2, max_q: int = 3):
        super().__init__("ARIMA")
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.model = None
        self.fitted_model = None
    
    def fit(self, train_data: np.ndarray):
        try:
            # Simple auto ARIMA - try common configurations
            best_aic = float('inf')
            best_params = (1, 1, 1)
            
            for p in range(self.max_p + 1):
                for d in range(self.max_d + 1):
                    for q in range(self.max_q + 1):
                        try:
                            model = ARIMA(train_data, order=(p, d, q))
                            fitted = model.fit()
                            if fitted.aic < best_aic:
                                best_aic = fitted.aic
                                best_params = (p, d, q)
                        except:
                            continue
            
            # Fit best model
            self.model = ARIMA(train_data, order=best_params)
            self.fitted_model = self.model.fit()
            self.is_fitted = True
            
        except Exception as e:
            # Fallback to naive method
            self.last_value = train_data[-1]
            self.is_fitted = True
    
    def predict(self, horizon: int) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if hasattr(self, 'last_value'):  # Fallback case
            return np.full(horizon, self.last_value)
        
        try:
            forecast = self.fitted_model.forecast(steps=horizon)
            return forecast.values if hasattr(forecast, 'values') else forecast
        except:
            # Fallback to last value
            return np.full(horizon, self.fitted_model.fittedvalues[-1])

class ETSMethod(BaselineMethod):
    """Exponential Smoothing (ETS) method"""
    
    def __init__(self):
        super().__init__("ETS")
        self.fitted_model = None
    
    def fit(self, train_data: np.ndarray):
        try:
            model = ETSModel(train_data, error='add', trend='add', seasonal='add', seasonal_periods=24)
            self.fitted_model = model.fit()
            self.is_fitted = True
        except:
            try:
                # Try simpler model
                model = ETSModel(train_data, error='add', trend='add', seasonal=None)
                self.fitted_model = model.fit()
                self.is_fitted = True
            except:
                # Fallback to naive
                self.last_value = train_data[-1]
                self.is_fitted = True
    
    def predict(self, horizon: int) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if hasattr(self, 'last_value'):  # Fallback case
            return np.full(horizon, self.last_value)
        
        try:
            forecast = self.fitted_model.forecast(steps=horizon)
            return forecast.values if hasattr(forecast, 'values') else forecast
        except:
            return np.full(horizon, self.fitted_model.fittedvalues[-1])

class ProphetMethod(BaselineMethod):
    """Facebook Prophet forecasting method"""
    
    def __init__(self):
        super().__init__("Prophet")
        self.model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=False,
            yearly_seasonality=False,
            interval_width=0.8
        )
        self.train_data = None
    
    def fit(self, train_data: np.ndarray):
        try:
            # Prepare data for Prophet
            dates = pd.date_range(start='2020-01-01', periods=len(train_data), freq='H')
            df = pd.DataFrame({
                'ds': dates,
                'y': train_data
            })
            
            self.model.fit(df)
            self.train_data = train_data
            self.is_fitted = True
            
        except Exception as e:
            # Fallback to naive method
            self.last_value = train_data[-1]
            self.is_fitted = True
    
    def predict(self, horizon: int) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if hasattr(self, 'last_value'):  # Fallback case
            return np.full(horizon, self.last_value)
        
        try:
            # Create future dataframe
            last_date = pd.date_range(start='2020-01-01', periods=len(self.train_data), freq='H')[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=horizon, freq='H')
            future_df = pd.DataFrame({'ds': future_dates})
            
            forecast = self.model.predict(future_df)
            return forecast['yhat'].values
            
        except Exception as e:
            return np.full(horizon, self.train_data[-1])

class LSTMMethod(BaselineMethod):
    """LSTM neural network forecasting method"""

    def __init__(self, seq_len: int = 96, hidden_size: int = 64, num_layers: int = 2):
        super().__init__("LSTM")
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler_mean = 0
        self.scaler_std = 1

    def _create_model(self):
        """Create LSTM model"""
        class LSTMNet(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, output_size):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.1)
                self.fc = nn.Linear(hidden_size, output_size)

            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                output = self.fc(lstm_out[:, -1, :])  # Use last time step
                return output

        return LSTMNet(1, self.hidden_size, self.num_layers, 1)

    def fit(self, train_data: np.ndarray):
        try:
            # Normalize data
            self.scaler_mean = np.mean(train_data)
            self.scaler_std = np.std(train_data) + 1e-8
            normalized_data = (train_data - self.scaler_mean) / self.scaler_std

            # Create sequences
            X, y = [], []
            for i in range(self.seq_len, len(normalized_data)):
                X.append(normalized_data[i-self.seq_len:i])
                y.append(normalized_data[i])

            if len(X) < 10:  # Insufficient data
                self.last_value = train_data[-1]
                self.is_fitted = True
                return

            X = torch.FloatTensor(X).unsqueeze(-1).to(self.device)  # Add feature dimension
            y = torch.FloatTensor(y).to(self.device)

            # Create and train model
            self.model = self._create_model().to(self.device)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            criterion = nn.MSELoss()

            # Training loop
            self.model.train()
            for epoch in range(50):  # Quick training
                optimizer.zero_grad()
                outputs = self.model(X).squeeze()
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

            self.train_data = normalized_data
            self.is_fitted = True

        except Exception as e:
            self.last_value = train_data[-1]
            self.is_fitted = True

    def predict(self, horizon: int) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if hasattr(self, 'last_value'):  # Fallback case
            return np.full(horizon, self.last_value)

        try:
            self.model.eval()
            predictions = []
            current_seq = torch.FloatTensor(self.train_data[-self.seq_len:]).unsqueeze(0).unsqueeze(-1).to(self.device)

            with torch.no_grad():
                for _ in range(horizon):
                    pred = self.model(current_seq).item()
                    predictions.append(pred)

                    # Update sequence
                    new_seq = torch.cat([current_seq[:, 1:, :], torch.FloatTensor([[[pred]]]).to(self.device)], dim=1)
                    current_seq = new_seq

            # Denormalize predictions
            predictions = np.array(predictions) * self.scaler_std + self.scaler_mean
            return predictions

        except Exception as e:
            return np.full(horizon, self.last_value)

class TransformerMethod(BaselineMethod):
    """Transformer-based forecasting method"""

    def __init__(self, seq_len: int = 96, d_model: int = 64, nhead: int = 4, num_layers: int = 2):
        super().__init__("Transformer")
        self.seq_len = seq_len
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler_mean = 0
        self.scaler_std = 1

    def _create_model(self):
        """Create Transformer model"""
        class TransformerNet(nn.Module):
            def __init__(self, d_model, nhead, num_layers, seq_len):
                super().__init__()
                self.d_model = d_model
                self.input_projection = nn.Linear(1, d_model)
                self.pos_encoding = nn.Parameter(torch.randn(seq_len, d_model))

                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=d_model*4,
                    dropout=0.1,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
                self.output_projection = nn.Linear(d_model, 1)

            def forward(self, x):
                # x shape: [batch_size, seq_len, 1]
                x = self.input_projection(x)  # [batch_size, seq_len, d_model]
                x = x + self.pos_encoding.unsqueeze(0)  # Add positional encoding

                transformer_out = self.transformer(x)  # [batch_size, seq_len, d_model]
                output = self.output_projection(transformer_out[:, -1, :])  # Use last time step
                return output

        return TransformerNet(self.d_model, self.nhead, self.num_layers, self.seq_len)

    def fit(self, train_data: np.ndarray):
        try:
            # Normalize data
            self.scaler_mean = np.mean(train_data)
            self.scaler_std = np.std(train_data) + 1e-8
            normalized_data = (train_data - self.scaler_mean) / self.scaler_std

            # Create sequences
            X, y = [], []
            for i in range(self.seq_len, len(normalized_data)):
                X.append(normalized_data[i-self.seq_len:i])
                y.append(normalized_data[i])

            if len(X) < 10:  # Insufficient data
                self.last_value = train_data[-1]
                self.is_fitted = True
                return

            X = torch.FloatTensor(X).unsqueeze(-1).to(self.device)  # Add feature dimension
            y = torch.FloatTensor(y).to(self.device)

            # Create and train model
            self.model = self._create_model().to(self.device)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            criterion = nn.MSELoss()

            # Training loop
            self.model.train()
            for epoch in range(30):  # Quick training
                optimizer.zero_grad()
                outputs = self.model(X).squeeze()
                loss = criterion(outputs, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

            self.train_data = normalized_data
            self.is_fitted = True

        except Exception as e:
            self.last_value = train_data[-1]
            self.is_fitted = True

    def predict(self, horizon: int) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if hasattr(self, 'last_value'):  # Fallback case
            return np.full(horizon, self.last_value)

        try:
            self.model.eval()
            predictions = []
            current_seq = torch.FloatTensor(self.train_data[-self.seq_len:]).unsqueeze(0).unsqueeze(-1).to(self.device)

            with torch.no_grad():
                for _ in range(horizon):
                    pred = self.model(current_seq).item()
                    predictions.append(pred)

                    # Update sequence
                    new_seq = torch.cat([current_seq[:, 1:, :], torch.FloatTensor([[[pred]]]).to(self.device)], dim=1)
                    current_seq = new_seq

            # Denormalize predictions
            predictions = np.array(predictions) * self.scaler_std + self.scaler_mean
            return predictions

        except Exception as e:
            return np.full(horizon, self.last_value)

def get_all_baseline_methods():
    """Get all baseline forecasting methods"""
    return {
        'Naive': NaiveMethod(),
        'Seasonal_Naive': SeasonalNaiveMethod(),
        'Linear': LinearMethod(),
        'DLinear': DLinearMethod(),
        'ARIMA': ARIMAMethod(),
        'ETS': ETSMethod(),
        'Prophet': ProphetMethod(),
        'LSTM': LSTMMethod(),
        'Transformer': TransformerMethod()
    }
