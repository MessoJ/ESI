# ml-pipeline/src/models/lstm_predictor.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)

class AttentionLSTM(nn.Module):
    """LSTM with multi-head attention for stress prediction"""
    
    def __init__(self, 
                 input_size: int = 8, 
                 hidden_size: int = 128, 
                 num_layers: int = 3, 
                 dropout: float = 0.2,
                 num_heads: int = 8):
        super(AttentionLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Input projection
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # Bidirectional
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        
        # Output layers with residual connections
        self.output_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.GELU(),
                nn.Dropout(dropout),
            ),
            nn.Linear(hidden_size // 2, 1)
        ])
        
        # Confidence estimation head
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Project input features
        x = self.input_projection(x)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention mechanism
        attn_out, attention_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Residual connection and layer norm
        lstm_out = self.layer_norm(lstm_out + attn_out)
        
        # Use last timestep for prediction
        final_hidden = lstm_out[:, -1, :]
        
        # Output prediction
        x = self.output_layers[0](final_hidden)
        stress_prediction = torch.sigmoid(self.output_layers[1](x)) * 100
        
        # Confidence estimation
        confidence = self.confidence_head(final_hidden)
        
        return stress_prediction, confidence, attention_weights

class StressPredictor:
    """Main stress prediction class with ensemble methods"""
    
    def __init__(self, model_dir: str = "models/", sequence_length: int = 50):
        self.model_dir = Path(model_dir)
        self.sequence_length = sequence_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Models
        self.lstm_model = None
        self.transformer_model = None
        self.ensemble_weights = [0.6, 0.4]  # LSTM, Transformer
        
        # Preprocessing
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
        # Feature engineering
        self.feature_history = []
        self.max_history = 200
        
    async def load_models(self):
        """Load all trained models"""
        try:
            # Load LSTM model
            lstm_path = self.model_dir / "lstm_attention_model.pth"
            if lstm_path.exists():
                self.lstm_model = AttentionLSTM()
                checkpoint = torch.load(lstm_path, map_location=self.device)
                self.lstm_model.load_state_dict(checkpoint['model_state_dict'])
                self.lstm_model.to(self.device)
                self.lstm_model.eval()
                logger.info("LSTM model loaded successfully")
            
            # Load scalers
            scaler_path = self.model_dir / "feature_scaler.joblib"
            if scaler_path.exists():
                self.feature_scaler = joblib.load(scaler_path)
                
            target_scaler_path = self.model_dir / "target_scaler.joblib"
            if target_scaler_path.exists():
                self.target_scaler = joblib.load(target_scaler_path)
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    async def predict_stress(self, indicators: Dict) -> float:
        """Predict stress index using ensemble of ML models"""
        if not self.lstm_model:
            await self.load_models()
        
        try:
            # Feature engineering
            features = self._engineer_features(indicators)
            
            # Update feature history
            self._update_feature_history(features)
            
            # Create sequence for LSTM
            if len(self.feature_history) >= self.sequence_length:
                sequence = np.array(self.feature_history[-self.sequence_length:])
                sequence = sequence.reshape(1, self.sequence_length, -1)
                
                # Convert to tensor
                input_tensor = torch.FloatTensor(sequence).to(self.device)
                
                # LSTM prediction
                lstm_pred = 50.0  # Default
                if self.lstm_model:
                    with torch.no_grad():
                        lstm_pred, confidence, attention = self.lstm_model(input_tensor)
                        lstm_pred = lstm_pred.item()
                
                # Ensemble prediction (would include transformer here)
                final_prediction = lstm_pred  # Simplified for now
                
                return max(0, min(100, final_prediction))
            else:
                # Not enough history, use simple calculation
                return self._simple_stress_calculation(indicators)
                
        except Exception as e:
            logger.error(f"Error in ML prediction: {e}")
            return 50.0
    
    def _engineer_features(self, indicators: Dict) -> np.ndarray:
        """Engineer features from raw indicators"""
        base_features = []
        feature_names = ['vix', 'credit_spreads', 'currency_volatility', 'unemployment', 
                        'inflation', 'interest_rates', 'oil_volatility', 'consumer_confidence']
        
        # Base indicators
        for name in feature_names:
            base_features.append(indicators.get(name, 0))
        
        # Technical features
        if len(self.feature_history) > 20:
            recent_history = np.array(self.feature_history[-20:])
            
            # Moving averages
            ma_5 = np.mean(recent_history[-5:], axis=0)
            ma_20 = np.mean(recent_history, axis=0)
            
            # Momentum features
            momentum = recent_history[-1] - recent_history[-5] if len(recent_history) >= 5 else np.zeros_like(recent_history[-1])
            
            # Volatility features
            volatility = np.std(recent_history, axis=0)
            
            # Combine all features
            engineered_features = np.concatenate([
                base_features,
                ma_5.flatten() if ma_5.ndim > 1 else ma_5,
                ma_20.flatten() if ma_20.ndim > 1 else ma_20,