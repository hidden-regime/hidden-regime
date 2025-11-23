# Docker Infrastructure

Docker setup for running Hidden-Regime with QuantConnect LEAN.

## Quick Start

### Build Image

```bash
cd ..
./scripts/build_docker.sh
```

### Run with Docker Compose

```bash
docker-compose up -d
```

## Files

### Dockerfiles

- **`Dockerfile`** - Development build (from local source)
- **`Dockerfile.pypi`** - Production build (from PyPI)

### Configuration

- **`docker-compose.yml`** - Orchestration configuration
- **`config/`** - LEAN configuration files (create as needed)

## Services

### lean-hidden-regime

Main LEAN engine with hidden-regime installed.

**Ports:**
- 5678: Debugger port

**Volumes:**
- `../quantconnect_templates` → `/Lean/Algorithm.Python`
- `lean-data` → `/Lean/Data`
- `lean-results` → `/Results`

### lean-research

Jupyter notebook environment for research.

**Ports:**
- 8888: Jupyter web interface

**Access:**
```
http://localhost:8888
```

## Usage

### Start Services

```bash
docker-compose up -d
```

### View Logs

```bash
docker-compose logs -f lean-hidden-regime
```

### Stop Services

```bash
docker-compose down
```

### Rebuild

```bash
docker-compose build --no-cache
docker-compose up -d
```

## Customization

### Environment Variables

Create `.env` file:

```env
BUILD_DATE=2025-01-17
VCS_REF=main
LEAN_MODE=backtest
```

### Custom Configuration

1. Create `config/` directory
2. Add custom LEAN config files
3. Mount in `docker-compose.yml`

## Troubleshooting

### Build Fails

```bash
docker system prune -a
docker-compose build --no-cache
```

### Container Won't Start

```bash
docker-compose logs lean-hidden-regime
```

### Reset Everything

```bash
docker-compose down -v
docker rmi quantconnect/lean:hidden-regime
docker-compose up -d
```

## Image Details

### Base Image

`quantconnect/lean:latest`

### Added Packages

- hidden-regime (with dependencies)
- numpy, pandas, scipy
- scikit-learn
- matplotlib, yfinance, ta

### Size

~2.5 GB (includes LEAN + .NET + Python + packages)

## Development

### Local Development Build

Uses `Dockerfile` which copies local source:

```bash
./scripts/build_docker.sh
```

### Production Build

Uses `Dockerfile.pypi` which installs from PyPI:

```bash
./scripts/build_docker.sh --pypi
```

### Build Arguments

```bash
docker build \
  --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
  --build-arg VCS_REF=$(git rev-parse --short HEAD) \
  -f Dockerfile \
  -t quantconnect/lean:hidden-regime \
  ..
```

## Advanced

### Multi-Stage Builds

For smaller images, consider multi-stage builds:

```dockerfile
FROM quantconnect/lean:latest AS builder
# Build steps...

FROM quantconnect/lean:latest
COPY --from=builder /packages /packages
```

### Caching

Optimize build times with BuildKit:

```bash
DOCKER_BUILDKIT=1 docker build ...
```

### Registry

Push to registry:

```bash
docker tag quantconnect/lean:hidden-regime \
  your-registry/lean:hidden-regime

docker push your-registry/lean:hidden-regime
```
