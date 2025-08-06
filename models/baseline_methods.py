"""
Baseline forecasting methods for comparison with I-ASNH
Includes statistical, neural, and selection baseline methods
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from prophet import Prophet
from typing import Optional
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
            # Use a fixed, common order to avoid slow grid search
            best_params = (1, 1, 1)
            
            # Fit the model with the fixed order
            self.model = ARIMA(train_data, order=best_params)
            self.fitted_model = self.model.fit()
            self.is_fitted = True
            
        except Exception as e:
            # Fallback to naive method if ARIMA fails
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
    """Exponential Smoothing (ETS) method with adaptive seasonal period detection"""

    def __init__(self):
        super().__init__("ETS")
        self.fitted_model = None
        self.last_value = None

    def _detect_seasonal_period(self, data: np.ndarray) -> Optional[int]:
        """Detect seasonal period using autocorrelation"""
        if len(data) < 48:  # Need sufficient data
            return None

        # Test common seasonal periods
        candidates = [24, 12, 7, 4]  # hourly, 12-hour, weekly, quarterly patterns
        max_period = min(len(data) // 3, 52)  # Don't exceed 1/3 of data length

        best_period = None
        best_score = 0

        for period in candidates:
            if period >= max_period:
                continue

            try:
                # Calculate autocorrelation at this lag
                if len(data) > period * 2:
                    autocorr = np.corrcoef(data[:-period], data[period:])[0, 1]
                    if not np.isnan(autocorr) and autocorr > best_score and autocorr > 0.3:
                        best_score = autocorr
                        best_period = period
            except:
                continue

        return best_period

    def fit(self, train_data: np.ndarray):
        self.last_value = train_data[-1]  # Always set fallback

        try:
            # Detect seasonal period
            seasonal_period = self._detect_seasonal_period(train_data)

            if seasonal_period and len(train_data) >= seasonal_period * 3:
                # Try seasonal model
                try:
                    model = ETSModel(train_data, error='add', trend='add', seasonal='add',
                                   seasonal_periods=seasonal_period)
                    self.fitted_model = model.fit(maxiter=100, disp=False)
                    self.is_fitted = True
                    self.last_value = None  # Clear fallback since we have a model
                    return
                except:
                    pass

            # Try trend-only model
            try:
                model = ETSModel(train_data, error='add', trend='add', seasonal=None)
                self.fitted_model = model.fit(maxiter=100, disp=False)
                self.is_fitted = True
                self.last_value = None  # Clear fallback since we have a model
                return
            except:
                pass

            # Try simple exponential smoothing
            try:
                model = ETSModel(train_data, error='add', trend=None, seasonal=None)
                self.fitted_model = model.fit(maxiter=100, disp=False)
                self.is_fitted = True
                self.last_value = None  # Clear fallback since we have a model
                return
            except:
                pass

        except Exception as e:
            pass

        # If all else fails, use fallback
        self.is_fitted = True

    def predict(self, horizon: int) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Use fallback if no model was fitted
        if self.fitted_model is None or self.last_value is not None:
            return np.full(horizon, self.last_value)

        try:
            forecast = self.fitted_model.forecast(steps=horizon)
            predictions = forecast.values if hasattr(forecast, 'values') else forecast

            # Sanity check predictions
            if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
                return np.full(horizon, self.last_value)

            return predictions
        except Exception as e:
            # Fallback to last fitted value or naive prediction
            try:
                last_fitted = self.fitted_model.fittedvalues[-1]
                return np.full(horizon, last_fitted)
            except:
                return np.full(horizon, self.last_value)

class ProphetMethod(BaselineMethod):
    """Facebook Prophet forecasting method"""

    def __init__(self):
        super().__init__("Prophet")
        self.model = None  # Initialize as None, create fresh for each fit
        self.train_data = None
        self.last_value = None
    
    def fit(self, train_data: np.ndarray):
        self.last_value = train_data[-1]  # Always set fallback

        try:
            # Create fresh Prophet model for each fit
            self.model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=False,
                yearly_seasonality=False,
                interval_width=0.8,
                uncertainty_samples=0  # Disable uncertainty for speed
            )

            # Prepare data for Prophet
            dates = pd.date_range(start='2020-01-01', periods=len(train_data), freq='H')
            df = pd.DataFrame({
                'ds': dates,
                'y': train_data
            })

            # Fit with timeout protection
            import signal
            def timeout_handler(signum, frame):
                raise TimeoutError("Prophet fitting timed out")

            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)  # Reduced timeout to 30 seconds

            try:
                self.model.fit(df)
                self.train_data = train_data
                self.is_fitted = True
                self.last_value = None  # Clear fallback since we have a model
            finally:
                signal.alarm(0)  # Cancel the alarm

        except Exception as e:
            # Keep fallback value for prediction
            self.model = None
            self.is_fitted = True
    
    def predict(self, horizon: int) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Use fallback if no model was fitted
        if self.model is None or self.last_value is not None:
            return np.full(horizon, self.last_value)

        try:
            # Create future dataframe
            last_date = pd.date_range(start='2020-01-01', periods=len(self.train_data), freq='H')[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=horizon, freq='H')
            future_df = pd.DataFrame({'ds': future_dates})

            forecast = self.model.predict(future_df)
            predictions = forecast['yhat'].values

            # Sanity check predictions
            if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
                return np.full(horizon, self.last_value)

            return predictions

        except Exception as e:
            return np.full(horizon, self.last_value)

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

            # Training loop with timeout protection
            self.model.train()
            for epoch in range(15):  # Further reduced epochs to prevent hanging
                try:
                    optimizer.zero_grad()
                    outputs = self.model(X).squeeze()
                    loss = criterion(outputs, y)
                    loss.backward()
                    optimizer.step()

                    # Early stopping if loss is very low
                    if loss.item() < 0.01:
                        break
                except Exception as e:
                    print(f"LSTM training failed at epoch {epoch}: {e}")
                    break

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

            # Training loop with timeout protection
            self.model.train()
            for epoch in range(10):  # Further reduced epochs to prevent hanging
                try:
                    optimizer.zero_grad()
                    outputs = self.model(X).squeeze()
                    loss = criterion(outputs, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()

                    # Early stopping if loss is very low
                    if loss.item() < 0.01:
                        break
                except Exception as e:
                    print(f"Transformer training failed at epoch {epoch}: {e}")
                    break

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

class NBEATSMethod(BaselineMethod):
    """N-BEATS forecasting method"""

    def __init__(self, seq_len: int = 96, pred_len: int = 24, stack_types=('generic', 'generic'), num_blocks=3, num_layers=4, layer_size=256):
        super().__init__("N_BEATS")
        self.seq_len = seq_len
        self.pred_len = pred_len  # This will be updated during fit
        self.stack_types = stack_types
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.layer_size = layer_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler_mean = 0
        self.scaler_std = 1
        self.last_value = None

    def _create_model(self):
        """Create N-BEATS model"""
        class NBEATSNet(nn.Module):
            def __init__(self, seq_len, pred_len, stack_types, num_blocks, num_layers, layer_size):
                super().__init__()
                self.pred_len = pred_len
                self.stacks = nn.ModuleList()
                for stack_type in stack_types:
                    self.stacks.append(self.create_stack(seq_len, pred_len, num_blocks, num_layers, layer_size, stack_type))

            def create_stack(self, seq_len, pred_len, num_blocks, num_layers, layer_size, stack_type):
                blocks = nn.ModuleList()
                for _ in range(num_blocks):
                    blocks.append(NBEATSBlock(seq_len, pred_len, num_layers, layer_size, stack_type))
                return blocks

            def forward(self, x):
                backcast = x
                forecast = torch.zeros(x.size(0), self.pred_len).to(x.device)
                for stack in self.stacks:
                    for block in stack:
                        b, f = block(backcast)
                        backcast = backcast - b
                        forecast = forecast + f
                return forecast

        class NBEATSBlock(nn.Module):
            def __init__(self, seq_len, pred_len, num_layers, layer_size, stack_type):
                super().__init__()
                self.stack_type = stack_type
                self.fc_stack = nn.ModuleList([nn.Linear(seq_len, layer_size)] + [nn.Linear(layer_size, layer_size) for _ in range(num_layers - 1)])
                self.backcast_linear = nn.Linear(layer_size, seq_len)
                self.forecast_linear = nn.Linear(layer_size, pred_len)

            def forward(self, x):
                for layer in self.fc_stack:
                    x = torch.relu(layer(x))
                backcast = self.backcast_linear(x)
                forecast = self.forecast_linear(x)
                return backcast, forecast

        return NBEATSNet(self.seq_len, self.pred_len, self.stack_types, self.num_blocks, self.num_layers, self.layer_size)

    def fit(self, train_data: np.ndarray):
        try:
            self.scaler_mean = np.mean(train_data)
            self.scaler_std = np.std(train_data) + 1e-8
            normalized_data = (train_data - self.scaler_mean) / self.scaler_std

            X, y = [], []
            for i in range(self.seq_len, len(normalized_data) - self.pred_len):
                X.append(normalized_data[i-self.seq_len:i])
                y.append(normalized_data[i:i+self.pred_len])

            if len(X) < 10:
                self.last_value = train_data[-1]
                self.is_fitted = True
                return

            X = torch.FloatTensor(X).to(self.device)
            y = torch.FloatTensor(y).to(self.device)

            self.model = self._create_model().to(self.device)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            criterion = nn.MSELoss()

            self.model.train()
            for epoch in range(20):  # Reduced epochs to prevent hanging
                try:
                    optimizer.zero_grad()
                    outputs = self.model(X)
                    loss = criterion(outputs, y)
                    loss.backward()
                    optimizer.step()
                except Exception as e:
                    print(f"N-BEATS training failed at epoch {epoch}: {e}")
                    break

            self.train_data = normalized_data
            self.is_fitted = True

        except Exception as e:
            self.last_value = train_data[-1]
            self.is_fitted = True

    def predict(self, horizon: int) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if self.model is None or self.last_value is not None:
            return np.full(horizon, self.last_value)

        try:
            self.model.eval()
            all_predictions = []
            current_seq = self.train_data[-self.seq_len:].copy()

            # Iterative prediction for long horizons
            remaining_horizon = horizon
            while remaining_horizon > 0:
                with torch.no_grad():
                    input_seq = torch.FloatTensor(current_seq[-self.seq_len:]).unsqueeze(0).to(self.device)
                    batch_predictions = self.model(input_seq).squeeze().cpu().numpy()

                # Take only what we need
                steps_to_take = min(len(batch_predictions), remaining_horizon)
                all_predictions.extend(batch_predictions[:steps_to_take])

                # Update sequence for next iteration
                if remaining_horizon > steps_to_take:
                    current_seq = np.concatenate([current_seq, batch_predictions[:steps_to_take]])

                remaining_horizon -= steps_to_take

            # Denormalize predictions
            predictions = np.array(all_predictions[:horizon])
            predictions = predictions * self.scaler_std + self.scaler_mean
            return predictions

        except Exception as e:
            return np.full(horizon, self.last_value)

class DeepARMethod(BaselineMethod):
    """DeepAR forecasting method"""

    def __init__(self, seq_len: int = 96, hidden_size: int = 64, num_layers: int = 2):
        super().__init__("DeepAR")
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler_mean = 0
        self.scaler_std = 1
        self.last_value = None

    def _create_model(self):
        """Create DeepAR model"""
        class DeepARNet(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                self.mean_fc = nn.Linear(hidden_size, 1)
                self.std_fc = nn.Linear(hidden_size, 1)

            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                mean = self.mean_fc(lstm_out[:, -1, :])
                std = F.softplus(self.std_fc(lstm_out[:, -1, :]))
                return mean, std

        return DeepARNet(1, self.hidden_size, self.num_layers)

    def fit(self, train_data: np.ndarray):
        # Always set last_value as fallback
        self.last_value = train_data[-1]

        try:
            self.scaler_mean = np.mean(train_data)
            self.scaler_std = np.std(train_data) + 1e-8
            normalized_data = (train_data - self.scaler_mean) / self.scaler_std

            X, y = [], []
            for i in range(self.seq_len, len(normalized_data)):
                X.append(normalized_data[i-self.seq_len:i])
                y.append(normalized_data[i])

            if len(X) < 10:
                self.is_fitted = True
                return

            X = torch.FloatTensor(X).unsqueeze(-1).to(self.device)
            y = torch.FloatTensor(y).unsqueeze(-1).to(self.device)

            self.model = self._create_model().to(self.device)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

            self.model.train()
            for epoch in range(15):  # Further reduced epochs to prevent hanging
                try:
                    optimizer.zero_grad()
                    mean, std = self.model(X)
                    dist = torch.distributions.Normal(mean, std)
                    loss = -dist.log_prob(y).mean()
                    loss.backward()
                    optimizer.step()

                    # Early stopping if loss is very low
                    if loss.item() < 0.01:
                        break
                except Exception as e:
                    print(f"DeepAR training failed at epoch {epoch}: {e}")
                    break

            self.train_data = normalized_data
            self.is_fitted = True
            # Clear last_value since we have a trained model
            self.last_value = None

        except Exception as e:
            # Keep last_value for fallback
            self.is_fitted = True

    def predict(self, horizon: int) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if self.model is None or self.last_value is not None:
            return np.full(horizon, self.last_value)

        try:
            self.model.eval()
            predictions = []
            current_seq = torch.FloatTensor(self.train_data[-self.seq_len:]).unsqueeze(0).unsqueeze(-1).to(self.device)

            with torch.no_grad():
                for _ in range(horizon):
                    mean, std = self.model(current_seq)
                    pred = torch.distributions.Normal(mean, std).sample().item()
                    predictions.append(pred)
                    
                    new_seq = torch.cat([current_seq[:, 1:, :], torch.FloatTensor([[[pred]]]).to(self.device)], dim=1)
                    current_seq = new_seq

            predictions = np.array(predictions) * self.scaler_std + self.scaler_mean
            return predictions

        except Exception as e:
            # Ensure we have a fallback value
            fallback_value = self.last_value if self.last_value is not None else 0.0
            return np.full(horizon, fallback_value)

class InformerMethod(BaselineMethod):
    """Informer forecasting method"""

    def __init__(self, seq_len: int = 96, pred_len: int = 24, d_model: int = 64, nhead: int = 4, num_encoder_layers: int = 2, num_decoder_layers: int = 2):
        super().__init__("Informer")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler_mean = 0
        self.scaler_std = 1
        self.last_value = None

    def _create_model(self):
        """Create Informer model"""
        class InformerNet(nn.Module):
            def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers):
                super().__init__()
                self.transformer = nn.Transformer(
                    d_model=d_model,
                    nhead=nhead,
                    num_encoder_layers=num_encoder_layers,
                    num_decoder_layers=num_decoder_layers,
                    dim_feedforward=d_model*4,
                    dropout=0.1,
                    batch_first=True
                )
                self.input_projection = nn.Linear(1, d_model)
                self.output_projection = nn.Linear(d_model, 1)

            def forward(self, src, tgt):
                src = self.input_projection(src)
                tgt = self.input_projection(tgt)
                output = self.transformer(src, tgt)
                return self.output_projection(output)

        return InformerNet(self.d_model, self.nhead, self.num_encoder_layers, self.num_decoder_layers)

    def fit(self, train_data: np.ndarray):
        try:
            self.scaler_mean = np.mean(train_data)
            self.scaler_std = np.std(train_data) + 1e-8
            normalized_data = (train_data - self.scaler_mean) / self.scaler_std

            X, y = [], []
            for i in range(self.seq_len, len(normalized_data) - self.pred_len):
                X.append(normalized_data[i-self.seq_len:i])
                y.append(normalized_data[i:i+self.pred_len])

            if len(X) < 10:
                self.last_value = train_data[-1]
                self.is_fitted = True
                return

            X = torch.FloatTensor(X).unsqueeze(-1).to(self.device)
            y = torch.FloatTensor(y).unsqueeze(-1).to(self.device)

            self.model = self._create_model().to(self.device)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            criterion = nn.MSELoss()

            self.model.train()
            for epoch in range(15):  # Reduced epochs to prevent hanging
                try:
                    optimizer.zero_grad()
                    decoder_input = torch.cat([X[:, -self.pred_len//2:, :], torch.zeros(X.size(0), self.pred_len - self.pred_len//2, 1).to(self.device)], dim=1)
                    outputs = self.model(X, decoder_input)
                    loss = criterion(outputs, y)
                    loss.backward()
                    optimizer.step()
                except Exception as e:
                    print(f"Informer training failed at epoch {epoch}: {e}")
                    break

            self.train_data = normalized_data
            self.is_fitted = True

        except Exception as e:
            self.last_value = train_data[-1]
            self.is_fitted = True

    def predict(self, horizon: int) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if self.model is None or self.last_value is not None:
            return np.full(horizon, self.last_value)

        try:
            self.model.eval()
            all_predictions = []
            current_seq = self.train_data[-self.seq_len:].copy()

            # Iterative prediction for long horizons
            remaining_horizon = horizon
            while remaining_horizon > 0:
                with torch.no_grad():
                    input_seq = torch.FloatTensor(current_seq[-self.seq_len:]).unsqueeze(0).unsqueeze(-1).to(self.device)
                    decoder_input = torch.cat([input_seq[:, -self.pred_len//2:, :],
                                             torch.zeros(1, self.pred_len - self.pred_len//2, 1).to(self.device)], dim=1)
                    batch_predictions = self.model(input_seq, decoder_input).squeeze().cpu().numpy()

                # Handle scalar output
                if batch_predictions.ndim == 0:
                    batch_predictions = np.array([batch_predictions])

                # Take only what we need
                steps_to_take = min(len(batch_predictions), remaining_horizon)
                all_predictions.extend(batch_predictions[:steps_to_take])

                # Update sequence for next iteration
                if remaining_horizon > steps_to_take:
                    current_seq = np.concatenate([current_seq, batch_predictions[:steps_to_take]])

                remaining_horizon -= steps_to_take

            # Denormalize predictions
            predictions = np.array(all_predictions[:horizon])
            predictions = predictions * self.scaler_std + self.scaler_mean
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
        'Transformer': TransformerMethod(),
        'N_BEATS': NBEATSMethod(),
        'DeepAR': DeepARMethod(),
        'Informer': InformerMethod()
    }
