# QuantConnect LEAN Installation Guide

Complete guide for setting up Hidden-Regime with QuantConnect LEAN in under 5 minutes.

---

## Table of Contents

1. [Quick Start (Automated)](#quick-start-automated)
2. [Manual Installation](#manual-installation)
3. [Verification](#verification)
4. [Running Your First Backtest](#running-your-first-backtest)
5. [Troubleshooting](#troubleshooting)

---

## Quick Start (Automated)

### One-Command Setup ⚡

```bash
./scripts/setup_quantconnect.sh
```

This script will:
1. ✅ Check prerequisites (Docker, .NET)
2. ✅ Install LEAN CLI (if needed)
3. ✅ Build custom Docker image
4. ✅ Configure LEAN to use hidden-regime
5. ✅ Display next steps

**Time:** ~5 minutes (depending on download speed)

---

## Manual Installation

### Prerequisites

Before installing, ensure you have:

- **Docker Desktop** (20.10+)
  - Download: https://docs.docker.com/get-docker/
  - Verify: `docker --version`

- **.NET SDK** (6.0+) *(optional, for LEAN CLI)*
  - Download: https://dotnet.microsoft.com/download
  - Verify: `dotnet --version`

- **Git** *(for cloning repository)*
  - Verify: `git --version`

---

### Installation Methods

Choose one of three installation methods based on your needs:

#### Method 1: LEAN CLI (Recommended) 🌟

**Best for:** Regular backtesting, local development

**Steps:**

1. **Install LEAN CLI**
   ```bash
   dotnet tool install -g QuantConnect.Lean.CLI
   ```

2. **Clone hidden-regime**
   ```bash
   git clone https://github.com/hidden-regime/hidden-regime.git
   cd hidden-regime
   ```

3. **Build Docker image**
   ```bash
   ./scripts/build_docker.sh
   ```

4. **Configure LEAN**
   ```bash
   lean config set engine-image quantconnect/lean:hidden-regime-latest
   ```

5. **Verify installation**
   ```bash
   lean --version
   docker images | grep hidden-regime
   ```

---

#### Method 2: Docker Compose 🐳

**Best for:** Running multiple services, Jupyter notebooks

**Steps:**

1. **Clone repository**
   ```bash
   git clone https://github.com/hidden-regime/hidden-regime.git
   cd hidden-regime
   ```

2. **Build and start services**
   ```bash
   cd docker
   docker-compose up -d
   ```

3. **Verify services**
   ```bash
   docker-compose ps
   docker-compose logs lean-hidden-regime
   ```

**Services included:**
- `lean-hidden-regime`: LEAN engine with hidden-regime
- `lean-research`: Jupyter research environment (port 8888)

---

#### Method 3: Direct Docker 🔧

**Best for:** Quick testing, CI/CD pipelines

**Steps:**

1. **Clone repository**
   ```bash
   git clone https://github.com/hidden-regime/hidden-regime.git
   cd hidden-regime
   ```

2. **Build image**
   ```bash
   docker build -f docker/Dockerfile -t quantconnect/lean:hidden-regime .
   ```

3. **Run backtest**
   ```bash
   docker run --rm \
     -v $(pwd)/quantconnect_templates:/Lean/Algorithm.Python \
     quantconnect/lean:hidden-regime
   ```

---

## Verification

### 1. Check Docker Image

```bash
docker images quantconnect/lean:hidden-regime-latest
```

**Expected output:**
```
REPOSITORY           TAG                   IMAGE ID       SIZE
quantconnect/lean    hidden-regime-latest  abc123def456   2.5GB
```

### 2. Verify Hidden-Regime Installation

```bash
docker run --rm quantconnect/lean:hidden-regime-latest \
  python -c "import hidden_regime; print(hidden_regime.__version__)"
```

**Expected output:**
```
<version number>
```

### 3. Verify QC Integration

```bash
docker run --rm quantconnect/lean:hidden-regime-latest \
  python -c "from hidden_regime.quantconnect import HiddenRegimeAlgorithm; print('OK')"
```

**Expected output:**
```
OK
```

### 4. Check LEAN CLI (if installed)

```bash
lean --version
lean config list
```

Should show `engine-image` configured to `quantconnect/lean:hidden-regime-latest`

---

## Running Your First Backtest

### Option 1: Quick Backtest Script ⚡

**Fastest way** to run a backtest:

```bash
./scripts/quick_backtest.sh basic_regime_switching MyFirstStrategy
```

**What this does:**
1. Creates a new LEAN project
2. Copies the basic regime switching template
3. Runs backtest
4. Shows results

**Time:** < 2 minutes

---

### Option 2: LEAN CLI (Step by Step)

1. **Create project**
   ```bash
   lean project-create MyRegimeStrategy
   cd MyRegimeStrategy
   ```

2. **Copy template**
   ```bash
   cp ../quantconnect_templates/basic_regime_switching.py main.py
   ```

3. **Run backtest**
   ```bash
   lean backtest MyRegimeStrategy
   ```

4. **View results**
   ```bash
   open backtests/*/index.html  # macOS
   xdg-open backtests/*/index.html  # Linux
   ```

---

### Option 3: Docker Compose

1. **Copy algorithm**
   ```bash
   cp quantconnect_templates/basic_regime_switching.py \
      docker/algorithms/main.py
   ```

2. **Start backtest**
   ```bash
   docker-compose up lean-hidden-regime
   ```

3. **View logs**
   ```bash
   docker-compose logs -f lean-hidden-regime
   ```

---

## Configuration

### Customizing Docker Build

#### Build with PyPI Version

Use when hidden-regime is published:

```bash
./scripts/build_docker.sh --pypi
```

#### Custom Tag

```bash
./scripts/build_docker.sh --tag v1.0.0
```

#### No Cache Build

Force fresh build:

```bash
./scripts/build_docker.sh --no-cache
```

---

### Environment Variables

Create `.env` file in `docker/` directory:

```env
# LEAN Configuration
LEAN_MODE=backtest
LEAN_ENVIRONMENT=docker

# Hidden-Regime Settings
HR_DEFAULT_STATES=3
HR_LOOKBACK_DAYS=252
HR_RETRAIN_FREQUENCY=weekly

# Docker Build
BUILD_DATE=2025-01-17
VCS_REF=main
```

---

## Advanced Setup

### Custom LEAN Configuration

1. **Create config directory**
   ```bash
   mkdir -p docker/config
   ```

2. **Add custom config.json**
   ```json
   {
     "algorithm-type-name": "BasicRegimeSwitching",
     "algorithm-language": "Python",
     "algorithm-location": "/Lean/Algorithm.Python/main.py",
     "data-folder": "/Lean/Data"
   }
   ```

3. **Mount in docker-compose.yml**
   ```yaml
   volumes:
     - ./config:/Lean/config:ro
   ```

---

### Jupyter Research Environment

Access Jupyter for research:

1. **Start research container**
   ```bash
   docker-compose up -d lean-research
   ```

2. **Open browser**
   ```
   http://localhost:8888
   ```

3. **Test hidden-regime**
   ```python
   import hidden_regime as hr
   pipeline = hr.create_financial_pipeline('SPY', n_states=3)
   result = pipeline.update()
   print(result.tail())
   ```

---

## Troubleshooting

### Docker Build Fails

**Problem:** Build fails with "no space left on device"

**Solution:**
```bash
docker system prune -a
./scripts/build_docker.sh
```

---

### Permission Denied

**Problem:** Cannot execute scripts

**Solution:**
```bash
chmod +x scripts/*.sh
```

---

### LEAN CLI Not Found

**Problem:** `lean: command not found`

**Solution:**
```bash
# Add to PATH (bash)
echo 'export PATH="$PATH:$HOME/.dotnet/tools"' >> ~/.bashrc
source ~/.bashrc

# Add to PATH (zsh)
echo 'export PATH="$PATH:$HOME/.dotnet/tools"' >> ~/.zshrc
source ~/.zshrc
```

---

### Image Not Found

**Problem:** `Error: Unable to find image 'quantconnect/lean:hidden-regime-latest'`

**Solution:**
```bash
# Rebuild image
./scripts/build_docker.sh

# Or pull base image first
docker pull quantconnect/lean:latest
./scripts/build_docker.sh
```

---

### Import Errors in Algorithm

**Problem:** `ModuleNotFoundError: No module named 'hidden_regime'`

**Solution:**

1. **Verify installation in container**
   ```bash
   docker run --rm quantconnect/lean:hidden-regime-latest \
     pip list | grep hidden-regime
   ```

2. **Rebuild if needed**
   ```bash
   ./scripts/build_docker.sh --no-cache
   ```

---

### Backtest Runs But No Trades

**Problem:** Algorithm runs but doesn't place any trades

**Common causes:**
1. Insufficient warm-up period
2. Low confidence threshold
3. No regime changes in backtest period

**Solution:**
```python
# In Initialize():
self.SetWarmUp(timedelta(days=252))  # Ensure warm-up

# Lower confidence threshold
self.initialize_regime_detection(
    ticker="SPY",
    min_confidence=0.5  # Lower from default 0.6
)
```

---

### Slow Backtest Performance

**Problem:** Backtest takes too long

**Solutions:**

1. **Reduce retraining frequency**
   ```python
   retrain_frequency="monthly"  # Instead of "weekly"
   ```

2. **Smaller lookback window**
   ```python
   lookback_days=90  # Instead of 252
   ```

3. **Disable retraining after initial**
   ```python
   retrain_frequency="never"
   ```

---

## Performance Benchmarks

Typical performance on standard hardware:

| Operation | Time | Notes |
|-----------|------|-------|
| Docker build | 3-5 min | First time (with downloads) |
| Docker build | 30-60s | Cached build |
| Create project | 5-10s | LEAN CLI |
| Backtest (1 year daily) | 10-30s | SPY, 3 states |
| Backtest (5 years daily) | 30-90s | SPY, 3 states |
| Multi-asset (4 assets, 1 year) | 30-60s | 4 states each |

---

## Next Steps

### Learn More

- 📖 **Templates Guide**: `quantconnect_templates/README.md`
- 🗺️ **Roadmap**: `QC_ROADMAP.md`
- ✅ **Phase 1 Summary**: `QUANTCONNECT_PHASE1_COMPLETE.md`

### Try Advanced Features

1. **Multi-asset rotation**
   ```bash
   ./scripts/quick_backtest.sh multi_asset_rotation
   ```

2. **Custom configurations**
   - Edit regime allocations
   - Adjust confidence thresholds
   - Modify retraining frequency

3. **QC Framework integration**
   - Use `HiddenRegimeAlphaModel`
   - Implement custom universe selection

---

## Support

### Getting Help

- **Documentation**: https://hiddenregime.com/docs/quantconnect
- **Issues**: https://github.com/hidden-regime/hidden-regime/issues
- **QuantConnect Forums**: https://www.quantconnect.com/forum
- **Discord**: [Join our community](#)

### Common Resources

- **QuantConnect Docs**: https://www.quantconnect.com/docs
- **LEAN GitHub**: https://github.com/QuantConnect/Lean
- **LEAN CLI Docs**: https://www.quantconnect.com/docs/v2/lean-cli

---

**Happy Trading! 🚀**

*Installation guide last updated: 2025-01-17*
