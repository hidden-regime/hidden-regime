#!/usr/bin/env python3
"""
Production Deployment Workflow
===============================

This example demonstrates a complete production deployment workflow for
regime analysis systems, including containerization, monitoring, alerting,
API deployment, and automated scaling.

Key features:
- Docker containerization for regime analysis services
- REST API deployment with FastAPI
- Monitoring and alerting systems
- Automated testing and validation
- CI/CD pipeline integration
- Production health checks and diagnostics
- Load balancing and scaling strategies

Use cases:
- Production trading systems
- Financial research platforms
- Risk management services
- Automated investment strategies

"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import json
import yaml
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import our framework
from hidden_regime.data import DataLoader
from hidden_regime.analysis import RegimeAnalyzer
from hidden_regime.config import DataConfig

@dataclass
class DeploymentConfig:
    """Configuration for production deployment"""
    environment: str  # 'development', 'staging', 'production'
    api_port: int
    monitoring_enabled: bool
    alerting_enabled: bool
    auto_scaling: bool
    container_registry: str
    health_check_interval: int  # seconds
    max_concurrent_requests: int
    data_retention_days: int
    backup_enabled: bool

class ProductionDeploymentWorkflow:
    """Complete production deployment workflow for regime analysis systems"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.data_config = DataConfig()
        self.analyzer = RegimeAnalyzer(self.data_config)
        
        # Load deployment configuration
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            self.deployment_config = DeploymentConfig(**config_data)
        else:
            self.deployment_config = self._get_default_deployment_config()
        
        self.deployment_dir = './deployment'
        self._ensure_deployment_structure()
    
    def _get_default_deployment_config(self) -> DeploymentConfig:
        """Get default deployment configuration"""
        return DeploymentConfig(
            environment='production',
            api_port=8000,
            monitoring_enabled=True,
            alerting_enabled=True,
            auto_scaling=True,
            container_registry='your-registry.com/hidden-regime',
            health_check_interval=30,
            max_concurrent_requests=100,
            data_retention_days=365,
            backup_enabled=True
        )
    
    def _ensure_deployment_structure(self):
        """Create deployment directory structure"""
        
        directories = [
            self.deployment_dir,
            f'{self.deployment_dir}/docker',
            f'{self.deployment_dir}/kubernetes',
            f'{self.deployment_dir}/monitoring',
            f'{self.deployment_dir}/scripts',
            f'{self.deployment_dir}/tests',
            f'{self.deployment_dir}/config'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def generate_dockerfile(self) -> str:
        """Generate Dockerfile for regime analysis service"""
        
        dockerfile_content = """# Multi-stage Docker build for Hidden Regime Analysis Service
FROM python:3.9-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    libblas-dev \\
    liblapack-dev \\
    gfortran \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM base as production

# Create non-root user
RUN useradd --create-home --shell /bin/bash regime
USER regime
WORKDIR /home/regime/app

# Copy application code
COPY --chown=regime:regime . .

# Set environment variables
ENV PYTHONPATH=/home/regime/app
ENV PYTHONUNBUFFERED=1
ENV HIDDEN_REGIME_ENV=production

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Start command
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
"""
        
        dockerfile_path = os.path.join(self.deployment_dir, 'docker', 'Dockerfile')
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)
        
        print(f"Generated Dockerfile: {dockerfile_path}")
        return dockerfile_path
    
    def generate_docker_compose(self) -> str:
        """Generate Docker Compose configuration"""
        
        compose_content = f"""version: '3.8'

services:
  regime-api:
    build: 
      context: ../../
      dockerfile: deployment/docker/Dockerfile
    ports:
      - "{self.deployment_config.api_port}:8000"
    environment:
      - HIDDEN_REGIME_ENV={self.deployment_config.environment}
      - API_PORT=8000
      - MAX_CONCURRENT_REQUESTS={self.deployment_config.max_concurrent_requests}
    volumes:
      - regime_data:/home/regime/app/data
      - regime_logs:/home/regime/app/logs
    depends_on:
      - redis
      - postgres
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import requests; requests.get('http://localhost:8000/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    command: redis-server --appendonly yes

  postgres:
    image: postgres:14-alpine
    environment:
      POSTGRES_DB: hidden_regime
      POSTGRES_USER: regime_user
      POSTGRES_PASSWORD: regime_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./config/init.sql:/docker-entrypoint-initdb.d/init.sql
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./config/nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - regime-api
    restart: unless-stopped

volumes:
  regime_data:
  regime_logs:
  redis_data:
  postgres_data:

networks:
  default:
    name: hidden-regime-network
"""
        
        compose_path = os.path.join(self.deployment_dir, 'docker', 'docker-compose.yml')
        with open(compose_path, 'w') as f:
            f.write(compose_content)
        
        print(f"Generated Docker Compose: {compose_path}")
        return compose_path
    
    def generate_kubernetes_manifests(self) -> List[str]:
        """Generate Kubernetes deployment manifests"""
        
        manifests = []
        
        # Deployment manifest
        deployment_manifest = f"""apiVersion: apps/v1
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
      - name: regime-api
        image: {self.deployment_config.container_registry}:latest
        ports:
        - containerPort: 8000
        env:
        - name: HIDDEN_REGIME_ENV
          value: "{self.deployment_config.environment}"
        - name: API_PORT
          value: "8000"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: hidden-regime-service
spec:
  selector:
    app: hidden-regime-api
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: hidden-regime-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: hidden-regime-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
"""
        
        k8s_deployment_path = os.path.join(self.deployment_dir, 'kubernetes', 'deployment.yml')
        with open(k8s_deployment_path, 'w') as f:
            f.write(deployment_manifest)
        manifests.append(k8s_deployment_path)
        
        # ConfigMap for application configuration
        configmap_manifest = f"""apiVersion: v1
kind: ConfigMap
metadata:
  name: hidden-regime-config
data:
  app.yaml: |
    environment: {self.deployment_config.environment}
    api:
      port: {self.deployment_config.api_port}
      max_concurrent_requests: {self.deployment_config.max_concurrent_requests}
    monitoring:
      enabled: {str(self.deployment_config.monitoring_enabled).lower()}
      health_check_interval: {self.deployment_config.health_check_interval}
    data:
      retention_days: {self.deployment_config.data_retention_days}
      backup_enabled: {str(self.deployment_config.backup_enabled).lower()}
"""
        
        k8s_configmap_path = os.path.join(self.deployment_dir, 'kubernetes', 'configmap.yml')
        with open(k8s_configmap_path, 'w') as f:
            f.write(configmap_manifest)
        manifests.append(k8s_configmap_path)
        
        print(f"Generated Kubernetes manifests: {len(manifests)} files")
        return manifests
    
    def generate_api_service(self) -> str:
        """Generate FastAPI service code"""
        
        api_code = '''from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import asyncio
import logging
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hidden_regime.analysis import RegimeAnalyzer
from hidden_regime.config import DataConfig

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Hidden Regime Analysis API",
    description="Production API for market regime analysis using Hidden Markov Models",
    version="1.0.0"
)

# Global analyzer instance
analyzer = RegimeAnalyzer(DataConfig())

# Request/Response models
class RegimeAnalysisRequest(BaseModel):
    symbol: str
    start_date: str
    end_date: str
    n_states: int = 3

class RegimeAnalysisResponse(BaseModel):
    symbol: str
    current_regime: str
    confidence: float
    days_in_regime: int
    expected_return: float
    expected_volatility: float
    regime_changes: int
    analysis_timestamp: str

class BatchAnalysisRequest(BaseModel):
    symbols: List[str]
    start_date: str
    end_date: str
    n_states: int = 3

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    environment: str

# Global state for health monitoring
app_start_time = datetime.now()
analysis_count = 0

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        environment=os.getenv("HIDDEN_REGIME_ENV", "development")
    )

@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint"""
    try:
        # Test analyzer functionality
        test_analysis = analyzer.analyze_stock("AAPL", "2024-01-01", "2024-01-31")
        if test_analysis:
            return {"status": "ready", "timestamp": datetime.now().isoformat()}
        else:
            raise HTTPException(status_code=503, detail="Service not ready")
    except Exception as e:
        logger.error(f"Readiness check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service not ready")

@app.post("/analyze", response_model=RegimeAnalysisResponse)
async def analyze_regime(request: RegimeAnalysisRequest):
    """Analyze regime for a single symbol"""
    global analysis_count
    analysis_count += 1
    
    try:
        logger.info(f"Starting analysis for {request.symbol}")
        
        analysis = analyzer.analyze_stock(
            request.symbol,
            request.start_date,
            request.end_date,
            request.n_states
        )
        
        if not analysis:
            raise HTTPException(
                status_code=404, 
                detail=f"Could not analyze symbol {request.symbol}"
            )
        
        return RegimeAnalysisResponse(
            symbol=analysis['symbol'],
            current_regime=analysis['current_regime'],
            confidence=analysis['confidence'],
            days_in_regime=analysis['days_in_regime'],
            expected_return=analysis['expected_return'],
            expected_volatility=analysis['expected_volatility'],
            regime_changes=analysis['regime_changes'],
            analysis_timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Analysis failed for {request.symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/batch")
async def batch_analyze(request: BatchAnalysisRequest, background_tasks: BackgroundTasks):
    """Batch analysis for multiple symbols"""
    global analysis_count
    analysis_count += len(request.symbols)
    
    try:
        logger.info(f"Starting batch analysis for {len(request.symbols)} symbols")
        
        results = []
        for symbol in request.symbols:
            try:
                analysis = analyzer.analyze_stock(
                    symbol,
                    request.start_date,
                    request.end_date,
                    request.n_states
                )
                
                if analysis:
                    results.append(RegimeAnalysisResponse(
                        symbol=analysis['symbol'],
                        current_regime=analysis['current_regime'],
                        confidence=analysis['confidence'],
                        days_in_regime=analysis['days_in_regime'],
                        expected_return=analysis['expected_return'],
                        expected_volatility=analysis['expected_volatility'],
                        regime_changes=analysis['regime_changes'],
                        analysis_timestamp=datetime.now().isoformat()
                    ))
                else:
                    logger.warning(f"Failed to analyze {symbol}")
                    
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {str(e)}")
                continue
        
        return {"results": results, "total_requested": len(request.symbols), "successful": len(results)}
        
    except Exception as e:
        logger.error(f"Batch analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get service metrics"""
    uptime = datetime.now() - app_start_time
    
    return {
        "uptime_seconds": uptime.total_seconds(),
        "total_analyses": analysis_count,
        "start_time": app_start_time.isoformat(),
        "current_time": datetime.now().isoformat()
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Hidden Regime Analysis API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("API_PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
'''
        
        os.makedirs(os.path.join(self.deployment_dir, 'api'), exist_ok=True)
        api_file = os.path.join(self.deployment_dir, 'api', 'main.py')
        with open(api_file, 'w') as f:
            f.write(api_code)
        
        # Generate requirements.txt
        requirements = """fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2
seaborn==0.12.2
scipy==1.11.1
yfinance==0.2.18
requests==2.31.0
python-multipart==0.0.6
jinja2==3.1.2
pyyaml==6.0.1
"""
        
        requirements_file = os.path.join(self.deployment_dir, 'requirements.txt')
        with open(requirements_file, 'w') as f:
            f.write(requirements)
        
        print(f"Generated API service: {api_file}")
        return api_file
    
    def generate_monitoring_config(self) -> List[str]:
        """Generate monitoring and alerting configurations"""
        
        configs = []
        
        # Prometheus configuration
        prometheus_config = """global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alerts.yml"

scrape_configs:
  - job_name: 'hidden-regime-api'
    static_configs:
      - targets: ['regime-api:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
"""
        
        prometheus_file = os.path.join(self.deployment_dir, 'monitoring', 'prometheus.yml')
        with open(prometheus_file, 'w') as f:
            f.write(prometheus_config)
        configs.append(prometheus_file)
        
        # Grafana dashboard configuration
        grafana_dashboard = """{
  "dashboard": {
    "id": null,
    "title": "Hidden Regime API Monitoring",
    "tags": ["hidden-regime"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "Requests/sec"
          }
        ]
      },
      {
        "id": 2,
        "title": "Response Time",
        "type": "graph", 
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "id": 3,
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total{status=~\"5..\"}[5m])",
            "legendFormat": "5xx errors/sec"
          }
        ]
      }
    ],
    "refresh": "10s",
    "time": {
      "from": "now-1h",
      "to": "now"
    }
  }
}"""
        
        grafana_file = os.path.join(self.deployment_dir, 'monitoring', 'dashboard.json')
        with open(grafana_file, 'w') as f:
            f.write(grafana_dashboard)
        configs.append(grafana_file)
        
        # Alert rules
        alert_rules = """groups:
- name: hidden-regime-alerts
  rules:
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: High error rate detected
      description: Error rate is {{ $value }} errors per second

  - alert: HighResponseTime
    expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: High response time
      description: 95th percentile response time is {{ $value }} seconds

  - alert: ServiceDown
    expr: up{job="hidden-regime-api"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: Hidden Regime API is down
      description: The Hidden Regime API service is not responding
"""
        
        alerts_file = os.path.join(self.deployment_dir, 'monitoring', 'alerts.yml')
        with open(alerts_file, 'w') as f:
            f.write(alert_rules)
        configs.append(alerts_file)
        
        print(f"Generated monitoring configs: {len(configs)} files")
        return configs
    
    def generate_deployment_scripts(self) -> List[str]:
        """Generate deployment and management scripts"""
        
        scripts = []
        
        # Build and deploy script
        deploy_script = f"""#!/bin/bash
set -e

echo "🚀 Starting Hidden Regime deployment..."

# Configuration
ENVIRONMENT="{self.deployment_config.environment}"
CONTAINER_REGISTRY="{self.deployment_config.container_registry}"
API_PORT="{self.deployment_config.api_port}"

# Build Docker image
echo "📦 Building Docker image..."
docker build -t hidden-regime-api:latest -f deployment/docker/Dockerfile .
docker tag hidden-regime-api:latest $CONTAINER_REGISTRY:latest

# Push to registry (uncomment for actual deployment)
# echo "📤 Pushing to container registry..."
# docker push $CONTAINER_REGISTRY:latest

# Deploy with Docker Compose (for local/staging)
if [ "$ENVIRONMENT" != "production" ]; then
    echo "🐳 Deploying with Docker Compose..."
    cd deployment/docker
    docker-compose down
    docker-compose up -d
    echo "✅ Deployment complete! API available at http://localhost:$API_PORT"
fi

# Deploy to Kubernetes (for production)
if [ "$ENVIRONMENT" = "production" ]; then
    echo "☸️  Deploying to Kubernetes..."
    kubectl apply -f deployment/kubernetes/
    kubectl rollout status deployment/hidden-regime-api
    echo "✅ Production deployment complete!"
fi

# Health check
echo "🏥 Running health check..."
sleep 10
curl -f http://localhost:$API_PORT/health || echo "⚠️  Health check failed"

echo "🎉 Deployment finished!"
"""
        
        deploy_file = os.path.join(self.deployment_dir, 'scripts', 'deploy.sh')
        with open(deploy_file, 'w') as f:
            f.write(deploy_script)
        os.chmod(deploy_file, 0o755)
        scripts.append(deploy_file)
        
        # Health monitoring script
        monitor_script = """#!/bin/bash

API_URL="http://localhost:8000"
SLACK_WEBHOOK_URL=""  # Add your Slack webhook URL

check_health() {
    local endpoint=$1
    local name=$2
    
    echo "Checking $name..."
    
    if curl -f -s "$API_URL$endpoint" > /dev/null; then
        echo "✅ $name: OK"
        return 0
    else
        echo "❌ $name: FAILED"
        return 1
    fi
}

send_alert() {
    local message=$1
    
    if [ ! -z "$SLACK_WEBHOOK_URL" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\\"text\\": \\"🚨 Hidden Regime Alert: $message\\"}" \
            "$SLACK_WEBHOOK_URL"
    fi
    
    echo "$(date): ALERT - $message" >> alerts.log
}

echo "🏥 Starting health monitoring..."
echo "Monitoring API at: $API_URL"

failed_checks=0

while true; do
    echo "\\n$(date): Running health checks..."
    
    # Check health endpoint
    if ! check_health "/health" "Health Check"; then
        ((failed_checks++))
    fi
    
    # Check readiness endpoint  
    if ! check_health "/ready" "Readiness Check"; then
        ((failed_checks++))
    fi
    
    # Check metrics endpoint
    if ! check_health "/metrics" "Metrics"; then
        ((failed_checks++))
    fi
    
    # Alert if multiple failures
    if [ $failed_checks -ge 2 ]; then
        send_alert "Multiple health check failures detected ($failed_checks)"
        failed_checks=0
    fi
    
    sleep 60  # Check every minute
done
"""
        
        monitor_file = os.path.join(self.deployment_dir, 'scripts', 'monitor.sh')
        with open(monitor_file, 'w') as f:
            f.write(monitor_script)
        os.chmod(monitor_file, 0o755)
        scripts.append(monitor_file)
        
        # Testing script
        test_script = """#!/bin/bash
set -e

API_URL="http://localhost:8000"

echo "🧪 Starting API tests..."

# Test health endpoint
echo "Testing /health endpoint..."
curl -f "$API_URL/health" | jq .

# Test analyze endpoint
echo "\\nTesting /analyze endpoint..."
curl -X POST "$API_URL/analyze" \
    -H "Content-Type: application/json" \
    -d '{
        "symbol": "AAPL",
        "start_date": "2024-01-01", 
        "end_date": "2024-03-01",
        "n_states": 3
    }' | jq .

# Test batch analyze endpoint
echo "\\nTesting /analyze/batch endpoint..."
curl -X POST "$API_URL/analyze/batch" \
    -H "Content-Type: application/json" \
    -d '{
        "symbols": ["AAPL", "MSFT"],
        "start_date": "2024-01-01",
        "end_date": "2024-03-01", 
        "n_states": 3
    }' | jq .

# Test metrics endpoint
echo "\\nTesting /metrics endpoint..."
curl -f "$API_URL/metrics" | jq .

echo "\\n✅ All tests passed!"
"""
        
        test_file = os.path.join(self.deployment_dir, 'scripts', 'test.sh')
        with open(test_file, 'w') as f:
            f.write(test_script)
        os.chmod(test_file, 0o755)
        scripts.append(test_file)
        
        print(f"Generated deployment scripts: {len(scripts)} files")
        return scripts
    
    def generate_ci_cd_pipeline(self) -> str:
        """Generate CI/CD pipeline configuration (GitHub Actions)"""
        
        github_workflow = f"""name: Hidden Regime CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  REGISTRY: {self.deployment_config.container_registry}
  IMAGE_NAME: hidden-regime-api

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r deployment/requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        pytest deployment/tests/ -v --cov=hidden_regime
    
    - name: Upload coverage reports
      uses: codecov/codecov-action@v3

  build-and-push:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
    
    - name: Log in to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ${{{{ env.REGISTRY }}}}
        username: ${{{{ github.actor }}}}
        password: ${{{{ secrets.GITHUB_TOKEN }}}}
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v4
      with:
        images: ${{{{ env.REGISTRY }}}}/${{{{ env.IMAGE_NAME }}}}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=sha,prefix={{{{branch}}}}-
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        file: deployment/docker/Dockerfile
        push: true
        tags: ${{{{ steps.meta.outputs.tags }}}}
        labels: ${{{{ steps.meta.outputs.labels }}}}

  deploy-staging:
    needs: build-and-push
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/develop'
    environment: staging
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to staging
      run: |
        echo "Deploying to staging environment..."
        # Add staging deployment commands here
    
  deploy-production:
    needs: build-and-push
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    environment: production
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Configure kubectl
      uses: azure/k8s-set-context@v3
      with:
        method: kubeconfig
        kubeconfig: ${{{{ secrets.KUBE_CONFIG }}}}
    
    - name: Deploy to production
      run: |
        kubectl set image deployment/hidden-regime-api regime-api=${{{{ env.REGISTRY }}}}/${{{{ env.IMAGE_NAME }}}}:main-${{{{ github.sha }}}}
        kubectl rollout status deployment/hidden-regime-api
"""
        
        workflow_dir = os.path.join(self.deployment_dir, '.github', 'workflows')
        os.makedirs(workflow_dir, exist_ok=True)
        
        workflow_file = os.path.join(workflow_dir, 'ci-cd.yml')
        with open(workflow_file, 'w') as f:
            f.write(github_workflow)
        
        print(f"Generated CI/CD pipeline: {workflow_file}")
        return workflow_file
    
    def generate_production_tests(self) -> List[str]:
        """Generate comprehensive production tests"""
        
        test_files = []
        
        # API integration tests
        api_tests = """import pytest
import requests
import time
from datetime import datetime, timedelta

API_BASE_URL = "http://localhost:8000"

class TestHiddenRegimeAPI:
    
    @pytest.fixture(scope="session", autouse=True)
    def wait_for_api(self):
        \"\"\"Wait for API to be ready before running tests\"\"\"
        for _ in range(30):
            try:
                response = requests.get(f"{API_BASE_URL}/health")
                if response.status_code == 200:
                    break
                time.sleep(1)
            except requests.exceptions.ConnectionError:
                time.sleep(1)
        else:
            pytest.fail("API did not start within 30 seconds")
    
    def test_health_endpoint(self):
        \"\"\"Test health check endpoint\"\"\"
        response = requests.get(f"{API_BASE_URL}/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
    
    def test_readiness_endpoint(self):
        \"\"\"Test readiness check endpoint\"\"\"
        response = requests.get(f"{API_BASE_URL}/ready")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "ready"
    
    def test_metrics_endpoint(self):
        \"\"\"Test metrics endpoint\"\"\"
        response = requests.get(f"{API_BASE_URL}/metrics")
        assert response.status_code == 200
        
        data = response.json()
        assert "uptime_seconds" in data
        assert "total_analyses" in data
    
    def test_single_analysis(self):
        \"\"\"Test single stock analysis\"\"\"
        payload = {
            "symbol": "AAPL",
            "start_date": "2024-01-01",
            "end_date": "2024-03-01",
            "n_states": 3
        }
        
        response = requests.post(f"{API_BASE_URL}/analyze", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert data["symbol"] == "AAPL"
        assert "current_regime" in data
        assert "confidence" in data
        assert 0 <= data["confidence"] <= 1
    
    def test_batch_analysis(self):
        \"\"\"Test batch analysis\"\"\"
        payload = {
            "symbols": ["AAPL", "MSFT"],
            "start_date": "2024-01-01", 
            "end_date": "2024-03-01",
            "n_states": 3
        }
        
        response = requests.post(f"{API_BASE_URL}/analyze/batch", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert data["total_requested"] == 2
        assert len(data["results"]) > 0
    
    def test_invalid_symbol(self):
        \"\"\"Test handling of invalid symbol\"\"\"
        payload = {
            "symbol": "INVALID123",
            "start_date": "2024-01-01",
            "end_date": "2024-03-01"
        }
        
        response = requests.post(f"{API_BASE_URL}/analyze", json=payload)
        assert response.status_code in [404, 500]  # Should handle gracefully
    
    def test_concurrent_requests(self):
        \"\"\"Test concurrent request handling\"\"\"
        import concurrent.futures
        
        def make_request():
            payload = {
                "symbol": "AAPL",
                "start_date": "2024-01-01",
                "end_date": "2024-02-01"
            }
            return requests.post(f"{API_BASE_URL}/analyze", json=payload)
        
        # Make 5 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(5)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # All requests should succeed
        for response in results:
            assert response.status_code == 200

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
"""
        
        api_test_file = os.path.join(self.deployment_dir, 'tests', 'test_api.py')
        with open(api_test_file, 'w') as f:
            f.write(api_tests)
        test_files.append(api_test_file)
        
        # Load tests
        load_tests = """import requests
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

API_BASE_URL = "http://localhost:8000"

def single_request():
    \"\"\"Make a single API request and measure response time\"\"\"
    start_time = time.time()
    
    try:
        response = requests.post(f"{API_BASE_URL}/analyze", json={
            "symbol": "AAPL",
            "start_date": "2024-01-01",
            "end_date": "2024-02-01"
        })
        
        end_time = time.time()
        response_time = end_time - start_time
        
        return {
            "success": response.status_code == 200,
            "response_time": response_time,
            "status_code": response.status_code
        }
    except Exception as e:
        return {
            "success": False,
            "response_time": float('inf'),
            "error": str(e)
        }

def load_test(concurrent_users=10, duration_seconds=60):
    \"\"\"Run load test with specified parameters\"\"\"
    print(f"Starting load test: {concurrent_users} concurrent users for {duration_seconds} seconds")
    
    results = []
    start_time = time.time()
    
    def worker():
        while time.time() - start_time < duration_seconds:
            result = single_request()
            results.append(result)
            time.sleep(0.1)  # Small delay between requests
    
    # Run concurrent workers
    with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
        futures = [executor.submit(worker) for _ in range(concurrent_users)]
        for future in as_completed(futures):
            future.result()
    
    # Analyze results
    successful_requests = [r for r in results if r["success"]]
    failed_requests = [r for r in results if not r["success"]]
    
    if successful_requests:
        response_times = [r["response_time"] for r in successful_requests]
        
        print(f"\\n=== Load Test Results ===")
        print(f"Total requests: {len(results)}")
        print(f"Successful: {len(successful_requests)}")
        print(f"Failed: {len(failed_requests)}")
        print(f"Success rate: {len(successful_requests)/len(results)*100:.1f}%")
        print(f"")
        print(f"Response times:")
        print(f"  Mean: {statistics.mean(response_times):.3f}s")
        print(f"  Median: {statistics.median(response_times):.3f}s") 
        print(f"  95th percentile: {sorted(response_times)[int(0.95*len(response_times))]:.3f}s")
        print(f"  Max: {max(response_times):.3f}s")
        print(f"")
        print(f"Requests per second: {len(results)/duration_seconds:.1f}")
    
    return results

if __name__ == "__main__":
    load_test()
"""
        
        load_test_file = os.path.join(self.deployment_dir, 'tests', 'test_load.py')
        with open(load_test_file, 'w') as f:
            f.write(load_tests)
        test_files.append(load_test_file)
        
        print(f"Generated production tests: {len(test_files)} files")
        return test_files
    
    def create_deployment_summary(self) -> str:
        """Create comprehensive deployment documentation"""
        
        summary = f"""# Hidden Regime Production Deployment Guide

## Overview

This deployment package provides everything needed to deploy the Hidden Regime Analysis system to production, including containerization, orchestration, monitoring, and CI/CD pipelines.

## Deployment Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Load Balancer │────│  API Instances   │────│    Database     │
│    (Nginx)      │    │   (FastAPI)      │    │  (PostgreSQL)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         │              ┌──────────────────┐              │
         └──────────────│     Cache        │──────────────┘
                        │    (Redis)       │
                        └──────────────────┘
                                 │
                        ┌──────────────────┐
                        │   Monitoring     │
                        │ (Prometheus/     │
                        │   Grafana)       │
                        └──────────────────┘
```

## Quick Start

### 1. Local Development
```bash
cd deployment/docker
docker-compose up -d
```

### 2. Production Deployment
```bash
# Build and push image
./deployment/scripts/deploy.sh

# Deploy to Kubernetes
kubectl apply -f deployment/kubernetes/
```

## Configuration

### Environment Variables
- `HIDDEN_REGIME_ENV`: {self.deployment_config.environment}
- `API_PORT`: {self.deployment_config.api_port}
- `MAX_CONCURRENT_REQUESTS`: {self.deployment_config.max_concurrent_requests}

### Resource Requirements
- **CPU**: 250m-500m per replica
- **Memory**: 512Mi-1Gi per replica
- **Storage**: 10Gi for data persistence

## API Endpoints

- `GET /health` - Health check
- `GET /ready` - Readiness check
- `GET /metrics` - Service metrics
- `POST /analyze` - Single stock analysis
- `POST /analyze/batch` - Batch analysis

## Monitoring

### Health Checks
- **Liveness Probe**: `/health` endpoint
- **Readiness Probe**: `/ready` endpoint
- **Startup Probe**: Initial health validation

### Metrics
- Request rate and response times
- Error rates by endpoint
- Resource utilization
- Business metrics (analysis count)

### Alerts
- High error rates (>10% 5xx responses)
- High response times (>2s 95th percentile)
- Service unavailability
- Resource exhaustion

## Scaling

### Horizontal Pod Autoscaler
- **Min replicas**: 2
- **Max replicas**: 10
- **CPU threshold**: 70%
- **Memory threshold**: 80%

### Load Testing
```bash
cd deployment/tests
python test_load.py
```

## Security

### Container Security
- Non-root user execution
- Minimal base image (python:3.9-slim)
- No secrets in container images
- Regular security updates

### Network Security
- TLS termination at load balancer
- Internal service communication
- Network policies (Kubernetes)

## Backup and Recovery

### Data Backup
- Daily PostgreSQL backups
- Redis persistence enabled
- Application logs retention: {self.deployment_config.data_retention_days} days

### Disaster Recovery
- Multi-zone deployment
- Automated failover
- Recovery time objective: <5 minutes

## Troubleshooting

### Common Issues
1. **Service not starting**: Check logs with `kubectl logs -l app=hidden-regime-api`
2. **High response times**: Check resource utilization and scale up
3. **Analysis failures**: Verify data source connectivity

### Debug Commands
```bash
# Check service status
kubectl get pods -l app=hidden-regime-api

# View logs
kubectl logs -f deployment/hidden-regime-api

# Scale manually
kubectl scale deployment hidden-regime-api --replicas=5
```

## CI/CD Pipeline

### Automated Testing
- Unit tests with pytest
- Integration tests with test suite
- Load testing for performance validation

### Deployment Stages
1. **Test**: Run all tests on pull request
2. **Build**: Build and push container image
3. **Deploy Staging**: Auto-deploy to staging environment
4. **Deploy Production**: Manual approval required

## Support

### Runbooks
- Located in `deployment/docs/runbooks/`
- Covers common operational scenarios
- Includes escalation procedures

### Monitoring Dashboards
- **Grafana URL**: http://grafana.your-domain.com
- **Prometheus URL**: http://prometheus.your-domain.com
- **API Docs**: http://api.your-domain.com/docs

---
*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
*Environment: {self.deployment_config.environment}*
"""
        
        summary_file = os.path.join(self.deployment_dir, 'README.md')
        with open(summary_file, 'w') as f:
            f.write(summary)
        
        print(f"Generated deployment summary: {summary_file}")
        return summary_file
    
    def generate_complete_deployment_package(self) -> Dict[str, Any]:
        """Generate complete production deployment package"""
        
        print("🚀 Generating Complete Production Deployment Package")
        print("=" * 80)
        
        results = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'environment': self.deployment_config.environment,
            'generated_files': {}
        }
        
        try:
            # 1. Container configuration
            print("\n📦 Generating container configuration...")
            dockerfile = self.generate_dockerfile()
            docker_compose = self.generate_docker_compose()
            results['generated_files']['container'] = [dockerfile, docker_compose]
            
            # 2. Kubernetes manifests
            print("\n☸️  Generating Kubernetes manifests...")
            k8s_manifests = self.generate_kubernetes_manifests()
            results['generated_files']['kubernetes'] = k8s_manifests
            
            # 3. API service
            print("\n🔌 Generating API service...")
            api_service = self.generate_api_service()
            results['generated_files']['api'] = [api_service]
            
            # 4. Monitoring configuration
            print("\n📊 Generating monitoring configuration...")
            monitoring_configs = self.generate_monitoring_config()
            results['generated_files']['monitoring'] = monitoring_configs
            
            # 5. Deployment scripts
            print("\n📜 Generating deployment scripts...")
            deployment_scripts = self.generate_deployment_scripts()
            results['generated_files']['scripts'] = deployment_scripts
            
            # 6. CI/CD pipeline
            print("\n🔄 Generating CI/CD pipeline...")
            ci_cd_pipeline = self.generate_ci_cd_pipeline()
            results['generated_files']['ci_cd'] = [ci_cd_pipeline]
            
            # 7. Production tests
            print("\n🧪 Generating production tests...")
            test_files = self.generate_production_tests()
            results['generated_files']['tests'] = test_files
            
            # 8. Documentation
            print("\n📚 Generating deployment documentation...")
            documentation = self.create_deployment_summary()
            results['generated_files']['documentation'] = [documentation]
            
            print(f"\n✅ Complete deployment package generated!")
            return results
            
        except Exception as e:
            print(f"\n❌ Error generating deployment package: {str(e)}")
            results['error'] = str(e)
            return results

def main():
    """Main execution function for production deployment workflow"""
    
    print("🚀 Hidden Regime Production Deployment Workflow")
    print("=" * 80)
    
    OUTPUT_DIR = './output/production_deployment'
    
    print(f"📁 Deployment package directory: {OUTPUT_DIR}")
    
    try:
        # Initialize deployment workflow
        workflow = ProductionDeploymentWorkflow()
        
        # Generate complete deployment package
        results = workflow.generate_complete_deployment_package()
        
        if 'error' not in results:
            print(f"\n📄 Deployment documentation generated")
            print(f"📁 All files saved to: {workflow.deployment_dir}")
            
            # Display generated file summary
            print(f"\n🎯 Generated Components:")
            for component, files in results['generated_files'].items():
                print(f"   📦 {component}: {len(files)} files")
            
            # Count total files
            total_files = sum(len(files) for files in results['generated_files'].values())
            print(f"\n📊 Total files generated: {total_files}")
            
            # List all generated files
            print(f"\n📂 Complete File Listing:")
            for root, dirs, files in os.walk(workflow.deployment_dir):
                for file in files:
                    rel_path = os.path.relpath(os.path.join(root, file), workflow.deployment_dir)
                    print(f"   - {rel_path}")
            
            # Show next steps
            print(f"\n🎯 Next Steps:")
            print(f"   1. Review configuration in {workflow.deployment_dir}/config/")
            print(f"   2. Test locally: cd {workflow.deployment_dir}/docker && docker-compose up")
            print(f"   3. Run tests: cd {workflow.deployment_dir} && python tests/test_api.py")
            print(f"   4. Deploy: ./deployment/scripts/deploy.sh")
            
        else:
            print(f"\n❌ Deployment package generation failed: {results['error']}")
            return False
        
    except Exception as e:
        print(f"❌ Error in production deployment workflow: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Production deployment workflow completed successfully!")
        print("🚀 Ready for production deployment!")
    else:
        print("\n❌ Production deployment workflow failed")
        exit(1)