# backend/app/main.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from contextlib import asynccontextmanager
import asyncio
import json
from datetime import datetime
import logging

from app.core.config import settings
from app.core.database import init_db
from app.api.v1.api import api_router
from app.services.websocket_manager import WebSocketManager
from app.services.stress_calculator import StressCalculator
from app.services.data_collector import DataCollector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
websocket_manager = WebSocketManager()
stress_calculator = StressCalculator()
data_collector = DataCollector()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Economic Stress Index API...")
    await init_db()
    
    # Start background tasks
    asyncio.create_task(real_time_stress_calculation())
    asyncio.create_task(data_collection_task())
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")

app = FastAPI(
    title="Economic Stress Index API",
    description="Real-time macroeconomic volatility tracking system",
    version="1.0.0",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Include API routes
app.include_router(api_router, prefix="/api/v1")

@app.websocket("/ws/stress-index")
async def websocket_endpoint(websocket: WebSocket):
    await websocket_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)

async def real_time_stress_calculation():
    """Background task for real-time stress calculation"""
    while True:
        try:
            # Calculate current stress index
            stress_data = await stress_calculator.calculate_current_stress()
            
            # Broadcast to all connected clients
            await websocket_manager.broadcast(json.dumps({
                "type": "stress_update",
                "data": stress_data,
                "timestamp": datetime.utcnow().isoformat()
            }))
            
            await asyncio.sleep(5)  # Update every 5 seconds
        except Exception as e:
            logger.error(f"Error in stress calculation: {e}")
            await asyncio.sleep(10)

async def data_collection_task():
    """Background task for data collection"""
    while True:
        try:
            await data_collector.collect_all_indicators()
            await asyncio.sleep(60)  # Collect every minute
        except Exception as e:
            logger.error(f"Error in data collection: {e}")
            await asyncio.sleep(60)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        workers=4 if not settings.DEBUG else 1
    )

# backend/app/core/config.py
from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "postgresql://user:password@localhost/esi_db"
    REDIS_URL: str = "redis://localhost:6379"
    
    # External APIs
    ALPHA_VANTAGE_API_KEY: str = ""
    FRED_API_KEY: str = ""
    POLYGON_API_KEY: str = ""
    YAHOO_FINANCE_ENABLED: bool = True
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    ALLOWED_HOSTS: List[str] = ["http://localhost:3000", "https://yourdomain.com"]
    
    # Application
    DEBUG: bool = False
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Economic Stress Index"
    
    # ML Models
    MODEL_UPDATE_INTERVAL: int = 3600  # 1 hour
    BATCH_SIZE: int = 1000
    
    # Kafka
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_STRESS_TOPIC: str = "stress-index-updates"
    KAFKA_INDICATORS_TOPIC: str = "economic-indicators"
    
    # Monitoring
    PROMETHEUS_ENABLED: bool = True
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"

settings = Settings()

# backend/app/services/stress_calculator.py
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.models.indicators import EconomicIndicator
from app.ml.inference.predictor import StressPredictor

class StressCalculator:
    def __init__(self):
        self.predictor = StressPredictor()
        self.weights = {
            'vix': 0.25,
            'credit_spreads': 0.20,
            'currency_volatility': 0.15,
            'unemployment': 0.15,
            'inflation': 0.10,
            'interest_rates': 0.08,
            'oil_volatility': 0.04,
            'consumer_confidence': 0.03
        }
        
    async def calculate_current_stress(self) -> Dict:
        """Calculate current stress index with ML enhancement"""
        try:
            # Get latest indicator values
            indicators = await self._get_latest_indicators()
            
            # Calculate base stress index
            base_stress = await self._calculate_weighted_stress(indicators)
            
            # ML-enhanced prediction
            ml_stress = await self.predictor.predict_stress(indicators)
            
            # Combine base and ML predictions (80% ML, 20% traditional)
            final_stress = 0.8 * ml_stress + 0.2 * base_stress
            
            # Calculate components
            components = await self._calculate_components(indicators)
            
            # Detect anomalies
            anomaly_score = await self.predictor.detect_anomaly(indicators)
            
            return {
                "stress_index": round(final_stress, 2),
                "base_stress": round(base_stress, 2),
                "ml_stress": round(ml_stress, 2),
                "components": components,
                "anomaly_score": round(anomaly_score, 3),
                "indicators": indicators,
                "level": self._get_stress_level(final_stress),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating stress: {e}")
            return {"error": str(e), "stress_index": 0}
    
    async def _get_latest_indicators(self) -> Dict:
        """Fetch latest economic indicators from database"""
        # This would connect to your database
        # Simulated data for now
        return {
            'vix': np.random.normal(20, 5),
            'credit_spreads': np.random.normal(2.5, 0.8),
            'currency_volatility': np.random.normal(12, 3),
            'unemployment': np.random.normal(4.5, 1.2),
            'inflation': np.random.normal(2.8, 1.5),
            'interest_rates': np.random.normal(1.2, 0.5),
            'oil_volatility': np.random.normal(25, 8),
            'consumer_confidence': np.random.normal(95, 15)
        }
    
    async def _calculate_weighted_stress(self, indicators: Dict) -> float:
        """Traditional weighted stress calculation"""
        normalized_indicators = self._normalize_indicators(indicators)
        
        stress = 0
        for key, value in normalized_indicators.items():
            if key in self.weights:
                stress += value * self.weights[key]
        
        return min(100, max(0, stress * 100))
    
    def _normalize_indicators(self, indicators: Dict) -> Dict:
        """Normalize indicators to 0-1 scale based on historical ranges"""
        normalized = {}
        
        # Define normalization ranges (historical min/max values)
        ranges = {
            'vix': (10, 80),
            'credit_spreads': (0.5, 10),
            'currency_volatility': (5, 30),
            'unemployment': (2, 15),
            'inflation': (-2, 10),
            'interest_rates': (0, 5),
            'oil_volatility': (10, 60),
            'consumer_confidence': (50, 130)
        }
        
        for key, value in indicators.items():
            if key in ranges:
                min_val, max_val = ranges[key]
                # For consumer confidence, reverse the scale (lower is worse)
                if key == 'consumer_confidence':
                    normalized[key] = 1 - (value - min_val) / (max_val - min_val)
                else:
                    normalized[key] = (value - min_val) / (max_val - min_val)
                normalized[key] = max(0, min(1, normalized[key]))
        
        return normalized
    
    async def _calculate_components(self, indicators: Dict) -> Dict:
        """Calculate individual component contributions"""
        normalized = self._normalize_indicators(indicators)
        components = {}
        
        for key, norm_value in normalized.items():
            if key in self.weights:
                components[key] = {
                    "value": indicators[key],
                    "normalized": round(norm_value, 3),
                    "contribution": round(norm_value * self.weights[key] * 100, 2),
                    "weight": self.weights[key]
                }
        
        return components
    
    def _get_stress_level(self, stress_index: float) -> str:
        """Determine stress level category"""
        if stress_index >= 80:
            return "CRITICAL"
        elif stress_index >= 60:
            return "HIGH"
        elif stress_index >= 30:
            return "MODERATE"
        else:
            return "LOW"

# backend/app/ml/models/lstm_predictor.py
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
import joblib
from pathlib import Path

class LSTMStressPredictor(nn.Module):
    def __init__(self, input_size: int = 8, hidden_size: int = 128, num_layers: int = 3, dropout: float = 0.2):
        super(LSTMStressPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers with dropout
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layers
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention
        attn_out, attention_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use last timestep output
        final_output = attn_out[:, -1, :]
        
        # Pass through fully connected layers
        output = self.fc_layers(final_output)
        
        return output * 100  # Scale to 0-100

class EnsembleStressModel:
    def __init__(self, model_dir: str = "models/"):
        self.model_dir = Path(model_dir)
        self.lstm_model = None
        self.scaler = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    async def load_models(self):
        """Load trained models"""
        try:
            # Load LSTM model
            self.lstm_model = LSTMStressPredictor()
            model_path = self.model_dir / "lstm_stress_model.pth"
            if model_path.exists():
                self.lstm_model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.lstm_model.to(self.device)
                self.lstm_model.eval()
            
            # Load scaler
            scaler_path = self.model_dir / "feature_scaler.joblib"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    async def predict(self, indicators: Dict) -> float:
        """Predict stress index using ensemble of models"""
        if not self.lstm_model or not self.scaler:
            await self.load_models()
        
        try:
            # Prepare input data
            features = self._prepare_features(indicators)
            
            # Scale features
            if self.scaler:
                features_scaled = self.scaler.transform(features.reshape(1, -1))
                features = features_scaled.reshape(-1)
            
            # Create sequence (assume we have 50 timesteps)
            sequence = np.tile(features, (50, 1)).reshape(1, 50, len(features))
            
            # Convert to tensor
            input_tensor = torch.FloatTensor(sequence).to(self.device)
            
            # Predict with LSTM
            with torch.no_grad():
                lstm_prediction = self.lstm_model(input_tensor).item()
            
            return max(0, min(100, lstm_prediction))
            
        except Exception as e:
            logger.error(f"Error in ML prediction: {e}")
            return 50.0  # Default moderate stress
    
    def _prepare_features(self, indicators: Dict) -> np.ndarray:
        """Convert indicators dict to feature array"""
        feature_order = ['vix', 'credit_spreads', 'currency_volatility', 'unemployment', 
                        'inflation', 'interest_rates', 'oil_volatility', 'consumer_confidence']
        
        features = []
        for key in feature_order:
            features.append(indicators.get(key, 0))
        
        return np.array(features, dtype=np.float32)

# backend/app/services/data_collector.py
import aiohttp
import asyncio
from typing import Dict, List
import logging
from datetime import datetime
import json

from app.core.config import settings
from app.data.sources.alpha_vantage import AlphaVantageClient
from app.data.sources.fred_api import FREDClient
from app.data.sources.yahoo_finance import YahooFinanceClient

logger = logging.getLogger(__name__)

class DataCollector:
    def __init__(self):
        self.alpha_vantage = AlphaVantageClient(settings.ALPHA_VANTAGE_API_KEY)
        self.fred = FREDClient(settings.FRED_API_KEY)
        self.yahoo = YahooFinanceClient()
        self.last_collection = {}
        
    async def collect_all_indicators(self) -> Dict:
        """Collect all economic indicators from various sources"""
        try:
            # Collect data concurrently
            tasks = [
                self._collect_market_data(),
                self._collect_economic_data(),
                self._collect_volatility_data()
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine results
            combined_data = {}
            for result in results:
                if isinstance(result, dict):
                    combined_data.update(result)
                else:
                    logger.error(f"Data collection error: {result}")
            
            # Store in cache/database
            await self._store_indicators(combined_data)
            
            return combined_data
            
        except Exception as e:
            logger.error(f"Error in data collection: {e}")
            return {}
    
    async def _collect_market_data(self) -> Dict:
        """Collect market-related indicators"""
        data = {}
        
        try:
            # VIX data
            vix_data = await self.yahoo.get_vix()
            if vix_data:
                data['vix'] = vix_data['close']
                data['vix_change'] = vix_data['change_percent']
            
            # Credit spreads (using corporate bond ETF vs Treasury)
            spread_data = await self._calculate_credit_spreads()
            if spread_data:
                data.update(spread_data)
                
        except Exception as e:
            logger.error(f"Error collecting market data: {e}")
        
        return data
    
    async def _collect_economic_data(self) -> Dict:
        """Collect economic indicators from FRED"""
        data = {}
        
        try:
            # Unemployment rate
            unemployment = await self.fred.get_series('UNRATE')
            if unemployment:
                data['unemployment'] = unemployment['value']
                data['unemployment_change'] = unemployment['change']
            
            # Inflation (CPI)
            cpi = await self.fred.get_series('CPIAUCSL')
            if cpi:
                data['inflation'] = cpi['yoy_change']
                data['inflation_change'] = cpi['mom_change']
            
            # Federal funds rate
            fed_rate = await self.fred.get_series('FEDFUNDS')
            if fed_rate:
                data['interest_rates'] = fed_rate['value']
                data['interest_rate_change'] = fed_rate['change']
            
            # Consumer confidence
            confidence = await self.fred.get_series('UMCSENT')
            if confidence:
                data['consumer_confidence'] = confidence['value']
                data['confidence_change'] = confidence['change']
                
        except Exception as e:
            logger.error(f"Error collecting economic data: {e}")
        
        return data
    
    async def _collect_volatility_data(self) -> Dict:
        """Collect volatility-related indicators"""
        data = {}
        
        try:
            # Currency volatility (DXY - Dollar Index)
            dxy_data = await self.yahoo.get_symbol('DX-Y.NYB')
            if dxy_data:
                volatility = await self._calculate_volatility(dxy_data['prices'])
                data['currency_volatility'] = volatility
            
            # Oil price volatility
            oil_data = await self.yahoo.get_symbol('CL=F')
            if oil_data:
                oil_volatility = await self._calculate_volatility(oil_data['prices'])
                data['oil_volatility'] = oil_volatility
                
        except Exception as e:
            logger.error(f"Error collecting volatility data: {e}")
        
        return data
    
    async def _calculate_credit_spreads(self) -> Dict:
        """Calculate credit spreads using bond ETFs"""
        try:
            # Corporate bond ETF (LQD) vs Treasury ETF (IEF)
            lqd_data = await self.yahoo.get_symbol('LQD')
            ief_data = await self.yahoo.get_symbol('IEF')
            
            if lqd_data and ief_data:
                # Simplified spread calculation using yield approximation
                spread = abs(lqd_data.get('yield', 0) - ief_data.get('yield', 0))
                return {
                    'credit_spreads': spread,
                    'credit_spread_change': 0  # Would calculate from historical data
                }
        except Exception as e:
            logger.error(f"Error calculating credit spreads: {e}")
        
        return {}
    
    async def _calculate_volatility(self, prices: List[float], window: int = 20) -> float:
        """Calculate rolling volatility"""
        if len(prices) < window:
            return 0
        
        returns = np.diff(np.log(prices[-window:]))
        volatility = np.std(returns) * np.sqrt(252) * 100  # Annualized volatility %
        return volatility
    
    async def _store_indicators(self, data: Dict):
        """Store indicators in database and cache"""
        try:
            # Store in Redis for real-time access
            import redis
            r = redis.from_url(settings.REDIS_URL)
            
            timestamp = datetime.utcnow().isoformat()
            cache_data = {
                **data,
                "timestamp": timestamp,
                "collection_time": datetime.utcnow().timestamp()
            }
            
            await r.set("latest_indicators", json.dumps(cache_data), ex=300)  # 5-min expiry
            
            # Store in database for historical analysis
            # Database storage implementation would go here
            
        except Exception as e:
            logger.error(f"Error storing indicators: {e}")

# backend/app/data/sources/alpha_vantage.py
import aiohttp
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class AlphaVantageClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        
    async def get_vix_data(self) -> Optional[Dict]:
        """Get VIX volatility index data"""
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': 'VIX',
            'apikey': self.api_key
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_time_series(data)
        except Exception as e:
            logger.error(f"Error fetching VIX data: {e}")
        
        return None
    
    async def get_forex_data(self, from_currency: str = "USD", to_currency: str = "EUR") -> Optional[Dict]:
        """Get forex exchange rate data"""
        params = {
            'function': 'FX_DAILY',
            'from_symbol': from_currency,
            'to_symbol': to_currency,
            'apikey': self.api_key
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_fx_data(data)
        except Exception as e:
            logger.error(f"Error fetching forex data: {e}")
        
        return None
    
    def _parse_time_series(self, data: Dict) -> Dict:
        """Parse Alpha Vantage time series response"""
        try:
            time_series = data.get('Time Series (Daily)', {})
            if not time_series:
                return {}
            
            latest_date = max(time_series.keys())
            latest_data = time_series[latest_date]
            
            return {
                'date': latest_date,
                'open': float(latest_data['1. open']),
                'high': float(latest_data['2. high']),
                'low': float(latest_data['3. low']),
                'close': float(latest_data['4. close']),
                'volume': int(latest_data['5. volume'])
            }
        except Exception as e:
            logger.error(f"Error parsing time series data: {e}")
            return {}

# backend/app/api/v1/endpoints/stress_index.py
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import List, Dict
from datetime import datetime, timedelta

from app.services.stress_calculator import StressCalculator
from app.services.alert_service import AlertService
from app.schemas.stress_index import StressIndexResponse, HistoricalStressResponse

router = APIRouter()

stress_calculator = StressCalculator()
alert_service = AlertService()

@router.get("/current", response_model=StressIndexResponse)
async def get_current_stress():
    """Get current stress index with all components"""
    try:
        stress_data = await stress_calculator.calculate_current_stress()
        return StressIndexResponse(**stress_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/historical", response_model=List[HistoricalStressResponse])
async def get_historical_stress(
    days: int = 30,
    resolution: str = "1h"  # 1h, 1d, 1w
):
    """Get historical stress index data"""
    try:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # Implementation would query database
        # For now, return simulated data
        historical_data = []
        current = start_date
        
        while current <= end_date:
            stress_value = 30 + 20 * np.sin(current.timestamp() / 86400) + np.random.normal(0, 5)
            historical_data.append({
                "timestamp": current.isoformat(),
                "stress_index": max(0, min(100, stress_value)),
                "level": _get_stress_level(stress_value)
            })
            
            if resolution == "1h":
                current += timedelta(hours=1)
            elif resolution == "1d":
                current += timedelta(days=1)
            else:
                current += timedelta(weeks=1)
        
        return historical_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/alerts/subscribe")
async def subscribe_to_alerts(
    background_tasks: BackgroundTasks,
    email: str,
    threshold: float = 60.0
):
    """Subscribe to stress index alerts"""
    try:
        await alert_service.subscribe_user(email, threshold)
        return {"message": "Successfully subscribed to alerts"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _get_stress_level(stress_index: float) -> str:
    """Helper function to determine stress level"""
    if stress_index >= 80:
        return "CRITICAL"
    elif stress_index >= 60:
        return "HIGH"
    elif stress_index >= 30:
        return "MODERATE"
    else:
        return "LOW"

# backend/requirements.txt
# Core Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0

# Database
asyncpg==0.29.0
sqlalchemy==2.0.23
alembic==1.13.1
timescaledb==2.13.0

# Cache & Messaging
redis==5.0.1
aioredis==2.0.1
kafka-python==2.0.2
celery==5.3.4

# ML/AI Libraries
torch==2.1.1
torchvision==0.16.1
scikit-learn==1.3.2
pandas==2.1.4
numpy==1.24.4
scipy==1.11.4
mlflow==2.8.1
onnx==1.15.0
onnxruntime==1.16.3

# Data Processing
aiohttp==3.9.1
httpx==0.25.2
beautifulsoup4==4.12.2
lxml==4.9.3

# Financial Data
yfinance==0.2.28
alpha-vantage==2.3.1
fredapi==0.5.1
polygon-api-client==1.12.0

# Monitoring & Logging
prometheus-client==0.19.0
structlog==23.2.0
sentry-sdk==1.38.0

# Security
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-multipart==0.0.6

# Development
pytest==7.4.3
pytest-asyncio==0.21.1
black==23.11.0
isort==5.12.0
mypy==1.7.1
pre-commit==3.6.0

# backend/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]