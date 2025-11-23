# Hidden Regime Production Deployment Guide

**Complete guide for deploying Hidden Regime in production environments**

This guide covers everything needed to successfully deploy and maintain Hidden Regime models in production, from initial setup to monitoring and optimization.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Installation & Configuration](#installation--configuration)
4. [Production Architecture](#production-architecture)
5. [Deployment Strategies](#deployment-strategies)
6. [Monitoring & Alerting](#monitoring--alerting)
7. [Performance Optimization](#performance-optimization)
8. [Security Considerations](#security-considerations)
9. [Maintenance & Updates](#maintenance--updates)
10. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

**Minimum Requirements:**
- **CPU**: 4 cores, 2.5GHz+
- **RAM**: 8GB (16GB recommended)
- **Storage**: 50GB available space
- **OS**: Linux (Ubuntu 20.04+), macOS 10.15+, Windows 10+

**Recommended Production Requirements:**
- **CPU**: 8+ cores, 3.0GHz+
- **RAM**: 32GB+ 
- **Storage**: 200GB+ SSD
- **OS**: Linux (Ubuntu 22.04 LTS)

### Software Dependencies

**Core Requirements:**
```bash
Python >= 3.8, <= 3.11
pip >= 21.0
virtualenv or conda
```

**System Packages (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install -y \
    python3-dev \
    python3-pip \
    python3-venv \
    build-essential \
    libatlas-base-dev \
    gfortran \
    pkg-config \
    libhdf5-dev
```

**System Packages (RHEL/CentOS):**
```bash
sudo yum install -y \
    python3-devel \
    python3-pip \
    gcc \
    gcc-c++ \
    atlas-devel \
    gcc-gfortran \
    pkgconfig \
    hdf5-devel
```

## Environment Setup

### 1. Virtual Environment Creation

**Using venv (Recommended):**
```bash
# Create project directory
mkdir /opt/hidden-regime-prod
cd /opt/hidden-regime-prod

# Create virtual environment
python3 -m venv venv

# Activate environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

**Using conda:**
```bash
# Create conda environment
conda create -n hidden-regime python=3.10
conda activate hidden-regime
```

### 2. Environment Variables

Create a `.env` file for production configuration:

```bash
# Production .env file
cat << EOF > /opt/hidden-regime-prod/.env
# Environment
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Data Sources
YFINANCE_RATE_LIMIT=60
ALPHA_VANTAGE_API_KEY=your_api_key_here
CACHE_TTL_SECONDS=3600

# Model Configuration
DEFAULT_REGIME_TYPE=3_state
MODEL_RETRAIN_INTERVAL_HOURS=24
MAX_MODEL_AGE_DAYS=30

# Database (if using)
DATABASE_URL=postgresql://user:pass@localhost:5432/hidden_regime
REDIS_URL=redis://localhost:6379/0

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
ALERT_WEBHOOK_URL=https://your-alerts.com/webhook

# Security
SECRET_KEY=your-secret-key-here
API_KEY_REQUIRED=true
ALLOWED_HOSTS=localhost,127.0.0.1,your-domain.com

EOF
```

### 3. Directory Structure

```bash
/opt/hidden-regime-prod/
├── .env                    # Environment configuration
├── venv/                   # Virtual environment
├── app/                    # Application code
│   ├── models/            # Trained model storage
│   ├── data/              # Data cache
│   ├── logs/              # Application logs
│   └── config/            # Configuration files
├── scripts/               # Deployment scripts
├── monitoring/            # Monitoring configuration
├── backups/               # Model and data backups
└── requirements.txt       # Python dependencies
```

## Installation & Configuration

### 1. Install Hidden Regime

```bash
# Activate virtual environment
source venv/bin/activate

# Install from PyPI
pip install hidden-regime

# Or install development version
pip install git+https://github.com/your-org/hidden-regime.git

# Install optional production dependencies
pip install \
    redis \
    psycopg2-binary \
    prometheus-client \
    gunicorn \
    uvicorn \
    fastapi
```

### 2. Production Requirements File

```bash
# Create requirements.txt for production
cat << EOF > requirements.txt
hidden-regime>=0.1.0
fastapi>=0.68.0
uvicorn>=0.15.0
gunicorn>=20.1.0
redis>=4.0.0
psycopg2-binary>=2.9.0
prometheus-client>=0.12.0
pydantic>=1.8.0
python-multipart>=0.0.5
python-jose[cryptography]>=3.3.0
passlib[bcrypt]>=1.7.4
aiofiles>=0.7.0
schedule>=1.1.0
EOF

pip install -r requirements.txt
```

### 3. Configuration Files

**Production Configuration (`app/config/production.py`):**
```python
import os
from typing import List, Optional
from pydantic import BaseSettings

class Settings(BaseSettings):
    # Environment
    environment: str = "production"
    debug: bool = False
    log_level: str = "INFO"
    
    # API Configuration
    api_title: str = "Hidden Regime API"
    api_version: str = "1.0.0"
    api_prefix: str = "/api/v1"
    
    # Data Sources
    yfinance_rate_limit: int = 60
    alpha_vantage_api_key: Optional[str] = None
    cache_ttl_seconds: int = 3600
    
    # Model Configuration
    default_regime_type: str = "3_state"
    model_retrain_interval_hours: int = 24
    max_model_age_days: int = 30
    model_storage_path: str = "/opt/hidden-regime-prod/app/models"
    
    # Database
    database_url: Optional[str] = None
    redis_url: str = "redis://localhost:6379/0"
    
    # Security
    secret_key: str
    api_key_required: bool = True
    allowed_hosts: List[str] = ["localhost", "127.0.0.1"]
    cors_origins: List[str] = ["http://localhost", "http://localhost:3000"]
    
    # Performance
    max_workers: int = 4
    worker_timeout: int = 300
    max_requests_per_worker: int = 1000
    
    # Monitoring
    prometheus_port: int = 9090
    metrics_enabled: bool = True
    health_check_timeout: int = 30
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
```

## Production Architecture

### 1. Application Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Load Balancer │────│   Web Server     │────│   Application   │
│   (nginx)       │    │   (gunicorn)     │    │   (FastAPI)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                ┌───────────────┼───────────────┐
                │               │               │
        ┌───────▼────┐  ┌──────▼──────┐  ┌────▼─────┐
        │   Cache    │  │  Database   │  │   Queue  │
        │  (Redis)   │  │(PostgreSQL) │  │  (Redis) │
        └────────────┘  └─────────────┘  └──────────┘
                                │
                        ┌───────▼────────┐
                        │   Monitoring   │
                        │ (Prometheus +  │
                        │   Grafana)     │
                        └────────────────┘
```

### 2. FastAPI Application Structure

**Main Application (`app/main.py`):**
```python
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import uvicorn
import logging
from contextlib import asynccontextmanager

from .config.production import settings
from .routers import regime_detection, health, models
from .services import ModelService, DataService, CacheService
from .monitoring import setup_metrics, metrics_middleware

# Setup logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Services
model_service = ModelService()
data_service = DataService()
cache_service = CacheService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup/shutdown lifecycle"""
    # Startup
    logger.info("Starting Hidden Regime API...")
    await model_service.initialize()
    await cache_service.initialize()
    setup_metrics()
    logger.info("Application started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Hidden Regime API...")
    await model_service.cleanup()
    await cache_service.cleanup()
    logger.info("Application shutdown complete")

# Create FastAPI app
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(metrics_middleware)

# Routes
app.include_router(health.router, prefix=settings.api_prefix, tags=["health"])
app.include_router(regime_detection.router, prefix=settings.api_prefix, tags=["regime"])
app.include_router(models.router, prefix=settings.api_prefix, tags=["models"])

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=settings.max_workers,
        log_level=settings.log_level.lower()
    )
```

### 3. Service Layer Implementation

**Model Service (`app/services/model_service.py`):**
```python
import asyncio
import pickle
import json
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime, timedelta
import logging

from hidden_regime import HiddenMarkovModel, HMMConfig, StateStandardizer
from ..config.production import settings

logger = logging.getLogger(__name__)

class ModelService:
    """Production model management service"""
    
    def __init__(self):
        self.models: Dict[str, HiddenMarkovModel] = {}
        self.model_metadata: Dict[str, dict] = {}
        self.storage_path = Path(settings.model_storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
    async def initialize(self):
        """Initialize model service"""
        logger.info("Initializing model service...")
        await self.load_existing_models()
        
    async def load_existing_models(self):
        """Load existing models from disk"""
        for model_file in self.storage_path.glob("*.pkl"):
            try:
                ticker = model_file.stem.split("_")[0]
                await self.load_model(ticker)
                logger.info(f"Loaded existing model for {ticker}")
            except Exception as e:
                logger.error(f"Failed to load model {model_file}: {e}")
    
    async def get_or_create_model(self, ticker: str, regime_type: str = None) -> HiddenMarkovModel:
        """Get existing model or create new one"""
        if ticker in self.models:
            # Check if model needs retraining
            metadata = self.model_metadata.get(ticker, {})
            last_update = metadata.get('last_update')
            
            if last_update:
                last_update_dt = datetime.fromisoformat(last_update)
                if datetime.now() - last_update_dt > timedelta(hours=settings.model_retrain_interval_hours):
                    logger.info(f"Model for {ticker} needs retraining")
                    return await self.retrain_model(ticker, regime_type)
            
            return self.models[ticker]
        
        return await self.create_model(ticker, regime_type)
    
    async def create_model(self, ticker: str, regime_type: str = None) -> HiddenMarkovModel:
        """Create new model for ticker"""
        regime_type = regime_type or settings.default_regime_type
        
        config = HMMConfig.for_standardized_regimes(regime_type=regime_type)
        model = HiddenMarkovModel(config)
        
        self.models[ticker] = model
        self.model_metadata[ticker] = {
            'created': datetime.now().isoformat(),
            'regime_type': regime_type,
            'status': 'created'
        }
        
        logger.info(f"Created new model for {ticker} with regime type {regime_type}")
        return model
    
    async def save_model(self, ticker: str):
        """Save model to disk"""
        if ticker not in self.models:
            raise ValueError(f"Model for {ticker} not found")
        
        model = self.models[ticker]
        metadata = self.model_metadata[ticker]
        
        # Save model
        model_path = self.storage_path / f"{ticker}_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save metadata
        metadata_path = self.storage_path / f"{ticker}_metadata.json"
        metadata['last_saved'] = datetime.now().isoformat()
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        logger.info(f"Saved model for {ticker}")
    
    async def load_model(self, ticker: str) -> HiddenMarkovModel:
        """Load model from disk"""
        model_path = self.storage_path / f"{ticker}_model.pkl"
        metadata_path = self.storage_path / f"{ticker}_metadata.json"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load metadata
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        self.models[ticker] = model
        self.model_metadata[ticker] = metadata
        
        return model
    
    async def retrain_model(self, ticker: str, regime_type: str = None) -> HiddenMarkovModel:
        """Retrain existing model with new data"""
        # This would integrate with DataService to get fresh data
        # For now, just update metadata
        if ticker in self.model_metadata:
            self.model_metadata[ticker]['last_retrain'] = datetime.now().isoformat()
            self.model_metadata[ticker]['status'] = 'retrained'
        
        logger.info(f"Retrained model for {ticker}")
        return self.models[ticker]
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up model service...")
        # Save all models before shutdown
        for ticker in self.models.keys():
            try:
                await self.save_model(ticker)
            except Exception as e:
                logger.error(f"Failed to save model {ticker}: {e}")
```

## Deployment Strategies

### 1. Docker Deployment

**Dockerfile:**
```dockerfile
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev \
    gfortran \
    pkg-config \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -r appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Start application
CMD ["gunicorn", "app.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000"]
```

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - REDIS_URL=redis://redis:6379/0
      - DATABASE_URL=postgresql://postgres:password@db:5432/hidden_regime
    volumes:
      - ./app/models:/app/models
      - ./app/data:/app/data
      - ./logs:/app/logs
    depends_on:
      - redis
      - db
    restart: unless-stopped
    
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    
  db:
    image: postgres:15-alpine
    ports:
      - "5432:5432"
    environment:
      POSTGRES_DB: hidden_regime
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - app
    restart: unless-stopped

volumes:
  redis_data:
  postgres_data:
```

### 2. Kubernetes Deployment

**deployment.yaml:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hidden-regime-api
  labels:
    app: hidden-regime-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: hidden-regime-api
  template:
    metadata:
      labels:
        app: hidden-regime-api
    spec:
      containers:
      - name: hidden-regime-api
        image: your-registry/hidden-regime:latest
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: hidden-regime-secrets
              key: redis-url
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: hidden-regime-secrets
              key: database-url
        - name: SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: hidden-regime-secrets
              key: secret-key
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /api/v1/health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /api/v1/health/ready
            port: 8000
          initialDelaySeconds: 15
          periodSeconds: 5
        volumeMounts:
        - name: model-storage
          mountPath: /app/models
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: hidden-regime-models

---
apiVersion: v1
kind: Service
metadata:
  name: hidden-regime-api-service
spec:
  selector:
    app: hidden-regime-api
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
```

### 3. Systemd Service Deployment

**systemd service file (`/etc/systemd/system/hidden-regime.service`):**
```ini
[Unit]
Description=Hidden Regime API
After=network.target
Wants=network-online.target

[Service]
Type=exec
User=hidden-regime
Group=hidden-regime
WorkingDirectory=/opt/hidden-regime-prod
Environment=PATH=/opt/hidden-regime-prod/venv/bin
EnvironmentFile=/opt/hidden-regime-prod/.env
ExecStart=/opt/hidden-regime-prod/venv/bin/gunicorn \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 300 \
    --max-requests 1000 \
    --max-requests-jitter 100 \
    --access-logfile /opt/hidden-regime-prod/logs/access.log \
    --error-logfile /opt/hidden-regime-prod/logs/error.log \
    --log-level info \
    app.main:app

ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Enable and start service:**
```bash
sudo systemctl daemon-reload
sudo systemctl enable hidden-regime
sudo systemctl start hidden-regime
sudo systemctl status hidden-regime
```

## Monitoring & Alerting

### 1. Health Check Endpoints

**Health Check Router (`app/routers/health.py`):**
```python
from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime
import psutil
import redis
from ..services import ModelService, CacheService
from ..config.production import settings

router = APIRouter()

@router.get("/health")
async def health_check():
    """Basic health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": settings.api_version
    }

@router.get("/health/detailed")
async def detailed_health_check(
    model_service: ModelService = Depends(),
    cache_service: CacheService = Depends()
):
    """Detailed health check with service status"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {}
    }
    
    # Check system resources
    try:
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        disk_percent = psutil.disk_usage('/').percent
        
        health_status["system"] = {
            "cpu_usage": f"{cpu_percent}%",
            "memory_usage": f"{memory_percent}%",
            "disk_usage": f"{disk_percent}%"
        }
        
        # Alert if resources are high
        if cpu_percent > 90 or memory_percent > 90 or disk_percent > 90:
            health_status["status"] = "warning"
            
    except Exception as e:
        health_status["system"] = {"error": str(e)}
        health_status["status"] = "unhealthy"
    
    # Check Redis connection
    try:
        await cache_service.ping()
        health_status["services"]["redis"] = "healthy"
    except Exception as e:
        health_status["services"]["redis"] = f"unhealthy: {e}"
        health_status["status"] = "unhealthy"
    
    # Check model service
    try:
        model_count = len(model_service.models)
        health_status["services"]["models"] = {
            "status": "healthy",
            "loaded_models": model_count
        }
    except Exception as e:
        health_status["services"]["models"] = f"unhealthy: {e}"
        health_status["status"] = "unhealthy"
    
    if health_status["status"] == "unhealthy":
        raise HTTPException(status_code=503, detail=health_status)
    
    return health_status

@router.get("/health/ready")
async def readiness_check():
    """Readiness check for load balancer"""
    # Add specific readiness checks here
    return {"status": "ready", "timestamp": datetime.now().isoformat()}
```

### 2. Prometheus Metrics

**Metrics Setup (`app/monitoring/metrics.py`):**
```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
from functools import wraps

# Define metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_MODELS = Gauge('active_models_count', 'Number of active models')
MODEL_PREDICTIONS = Counter('model_predictions_total', 'Total model predictions', ['ticker', 'regime_type'])
CACHE_HITS = Counter('cache_hits_total', 'Cache hits')
CACHE_MISSES = Counter('cache_misses_total', 'Cache misses')
ERROR_COUNT = Counter('errors_total', 'Total errors', ['error_type'])

def setup_metrics():
    """Start Prometheus metrics server"""
    start_http_server(9090)

def track_requests(func):
    """Decorator to track request metrics"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            REQUEST_COUNT.labels(method='GET', endpoint=func.__name__).inc()
            return result
        except Exception as e:
            ERROR_COUNT.labels(error_type=type(e).__name__).inc()
            raise
        finally:
            REQUEST_DURATION.observe(time.time() - start_time)
    return wrapper
```

### 3. Grafana Dashboard Configuration

**grafana-dashboard.json:**
```json
{
  "dashboard": {
    "title": "Hidden Regime API Monitoring",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph", 
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.50, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "50th percentile"
          }
        ]
      },
      {
        "title": "Active Models",
        "type": "stat",
        "targets": [
          {
            "expr": "active_models_count",
            "legendFormat": "Models"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(errors_total[5m])",
            "legendFormat": "{{error_type}}"
          }
        ]
      }
    ]
  }
}
```

## Performance Optimization

### 1. Database Optimization

**PostgreSQL Configuration:**
```sql
-- Create indexes for performance
CREATE INDEX CONCURRENTLY idx_market_data_ticker_date 
ON market_data (ticker, date);

CREATE INDEX CONCURRENTLY idx_model_predictions_ticker_timestamp 
ON model_predictions (ticker, timestamp);

-- Partitioning for large datasets
CREATE TABLE market_data_partitioned (
    LIKE market_data INCLUDING ALL
) PARTITION BY RANGE (date);

-- Create monthly partitions
CREATE TABLE market_data_2024_01 PARTITION OF market_data_partitioned
FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
```

### 2. Redis Caching Strategy

**Cache Configuration:**
```python
import redis.asyncio as redis
import json
import pickle
from typing import Any, Optional

class CacheService:
    def __init__(self):
        self.redis_client = None
        
    async def initialize(self):
        """Initialize Redis connection"""
        self.redis_client = redis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=False,  # Handle binary data
            max_connections=20
        )
        
    async def get_json(self, key: str) -> Optional[dict]:
        """Get JSON data from cache"""
        try:
            data = await self.redis_client.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
            
    async def set_json(self, key: str, data: dict, expire: int = None):
        """Set JSON data in cache"""
        try:
            await self.redis_client.set(
                key, 
                json.dumps(data), 
                ex=expire or settings.cache_ttl_seconds
            )
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            
    async def get_model(self, key: str):
        """Get pickled model from cache"""
        try:
            data = await self.redis_client.get(key)
            return pickle.loads(data) if data else None
        except Exception as e:
            logger.error(f"Model cache get error: {e}")
            return None
            
    async def set_model(self, key: str, model: Any, expire: int = None):
        """Set pickled model in cache"""
        try:
            await self.redis_client.set(
                key,
                pickle.dumps(model),
                ex=expire or (24 * 60 * 60)  # 24 hours for models
            )
        except Exception as e:
            logger.error(f"Model cache set error: {e}")
```

### 3. Connection Pooling

**Database Connection Pool:**
```python
import asyncpg
from contextlib import asynccontextmanager

class DatabaseService:
    def __init__(self):
        self.pool = None
        
    async def initialize(self):
        """Initialize connection pool"""
        self.pool = await asyncpg.create_pool(
            settings.database_url,
            min_size=5,
            max_size=20,
            max_queries=50000,
            max_inactive_connection_lifetime=300.0,
            command_timeout=60.0
        )
        
    @asynccontextmanager
    async def get_connection(self):
        """Get database connection from pool"""
        async with self.pool.acquire() as connection:
            yield connection
            
    async def cleanup(self):
        """Close connection pool"""
        if self.pool:
            await self.pool.close()
```

## Security Considerations

### 1. API Security

**Authentication Middleware:**
```python
from fastapi import HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from ..config.production import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    """Verify API key"""
    if not settings.api_key_required:
        return True
        
    if not api_key:
        raise HTTPException(status_code=401, detail="API Key required")
        
    # In production, check against database or secure store
    valid_api_keys = os.getenv("VALID_API_KEYS", "").split(",")
    if api_key not in valid_api_keys:
        raise HTTPException(status_code=401, detail="Invalid API Key")
        
    return True
```

### 2. Input Validation

**Data Validation:**
```python
from pydantic import BaseModel, validator
from typing import List, Optional
from datetime import datetime, date

class RegimeDetectionRequest(BaseModel):
    ticker: str
    start_date: date
    end_date: date
    regime_type: Optional[str] = "3_state"
    
    @validator('ticker')
    def validate_ticker(cls, v):
        if not v or len(v) > 10:
            raise ValueError('Invalid ticker symbol')
        return v.upper()
    
    @validator('end_date')
    def validate_date_range(cls, v, values):
        if 'start_date' in values and v <= values['start_date']:
            raise ValueError('End date must be after start date')
        if v > date.today():
            raise ValueError('End date cannot be in the future')
        return v
    
    @validator('regime_type')
    def validate_regime_type(cls, v):
        if v not in ['3_state', '4_state', '5_state', 'auto']:
            raise ValueError('Invalid regime type')
        return v
```

### 3. Rate Limiting

**Rate Limiting Middleware:**
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import redis

# Initialize limiter
limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=settings.redis_url
)

# Add to FastAPI app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Apply to routes
@router.post("/detect-regimes")
@limiter.limit("10/minute")  # 10 requests per minute
async def detect_regimes(request: Request, ...):
    pass
```

## Maintenance & Updates

### 1. Model Retraining Schedule

**Scheduled Tasks (`app/tasks/scheduler.py`):**
```python
import schedule
import asyncio
from datetime import datetime
import logging
from ..services import ModelService, DataService

logger = logging.getLogger(__name__)

class ModelRetrainingScheduler:
    def __init__(self, model_service: ModelService, data_service: DataService):
        self.model_service = model_service
        self.data_service = data_service
        
    async def retrain_all_models(self):
        """Retrain all models with fresh data"""
        logger.info("Starting scheduled model retraining...")
        
        for ticker in self.model_service.models.keys():
            try:
                # Get fresh data
                end_date = datetime.now().date()
                start_date = end_date.replace(year=end_date.year - 2)  # 2 years of data
                
                data = await self.data_service.load_data(ticker, start_date, end_date)
                
                # Retrain model
                model = await self.model_service.get_or_create_model(ticker)
                model.fit(data['log_return'].values)
                
                # Save updated model
                await self.model_service.save_model(ticker)
                
                logger.info(f"Successfully retrained model for {ticker}")
                
            except Exception as e:
                logger.error(f"Failed to retrain model for {ticker}: {e}")
        
        logger.info("Completed scheduled model retraining")
    
    def start_scheduler(self):
        """Start background scheduler"""
        # Schedule daily at 2 AM
        schedule.every().day.at("02:00").do(
            lambda: asyncio.create_task(self.retrain_all_models())
        )
        
        # Run scheduler in background
        def run_scheduler():
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        
        import threading
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
```

### 2. Backup Strategy

**Backup Script (`scripts/backup.sh`):**
```bash
#!/bin/bash

# Configuration
BACKUP_DIR="/opt/hidden-regime-prod/backups"
DATE=$(date +%Y%m%d_%H%M%S)
MODEL_DIR="/opt/hidden-regime-prod/app/models"
DB_NAME="hidden_regime"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Backup models
echo "Backing up models..."
tar -czf "$BACKUP_DIR/models_$DATE.tar.gz" -C "$MODEL_DIR" .

# Backup database
echo "Backing up database..."
pg_dump "$DB_NAME" > "$BACKUP_DIR/database_$DATE.sql"

# Cleanup old backups (keep 30 days)
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +30 -delete
find "$BACKUP_DIR" -name "*.sql" -mtime +30 -delete

echo "Backup completed: $DATE"
```

**Cron job for automated backups:**
```bash
# Run daily at 1 AM
0 1 * * * /opt/hidden-regime-prod/scripts/backup.sh >> /var/log/hidden-regime-backup.log 2>&1
```

### 3. Rolling Updates

**Zero-downtime deployment script:**
```bash
#!/bin/bash

REPO_URL="https://github.com/your-org/hidden-regime.git"
DEPLOY_DIR="/opt/hidden-regime-prod"
VENV_DIR="$DEPLOY_DIR/venv"

echo "Starting deployment..."

# Create backup of current version
cp -r "$DEPLOY_DIR" "$DEPLOY_DIR.backup.$(date +%s)"

# Pull latest code
cd "$DEPLOY_DIR"
git pull origin main

# Update dependencies
source "$VENV_DIR/bin/activate"
pip install -r requirements.txt

# Run database migrations (if any)
# python manage.py migrate

# Test the application
python -m pytest tests/ -x

# Reload application (graceful restart)
sudo systemctl reload hidden-regime

# Health check
sleep 10
curl -f http://localhost:8000/api/v1/health || {
    echo "Health check failed, rolling back..."
    sudo systemctl stop hidden-regime
    rm -rf "$DEPLOY_DIR"
    mv "$DEPLOY_DIR.backup."* "$DEPLOY_DIR"
    sudo systemctl start hidden-regime
    exit 1
}

echo "Deployment successful"

# Cleanup backup after successful deployment
rm -rf "$DEPLOY_DIR.backup."*
```

## Performance Benchmarks

### Expected Performance Metrics

**API Response Times:**
- Health check: < 10ms
- Regime detection (cached): < 50ms
- Regime detection (fresh): < 2s
- Model training: 10-60s (depending on data size)

**Throughput:**
- Concurrent requests: 100+ req/s
- Model predictions: 1000+ predictions/s (cached models)
- Data loading: 10-50 assets/minute (depending on data source)

**Resource Usage:**
- Memory: 1-4GB per worker process
- CPU: 20-80% during model training
- Storage: 1-10MB per trained model

## Conclusion

This deployment guide provides a comprehensive foundation for running Hidden Regime in production. Key considerations:

1. **Scalability**: Use horizontal scaling with load balancers and multiple workers
2. **Reliability**: Implement proper monitoring, health checks, and backup strategies  
3. **Security**: Use API keys, input validation, and rate limiting
4. **Maintenance**: Schedule regular model retraining and system updates
5. **Performance**: Optimize with caching, connection pooling, and efficient data structures

Adapt these configurations to your specific infrastructure and requirements.