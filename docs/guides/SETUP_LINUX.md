# Hidden Regime MCP Setup for Linux

This guide provides step-by-step instructions for setting up Hidden Regime with Claude Desktop on Linux (Ubuntu, Debian, Fedora, Arch, etc.).

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Distribution-Specific Setup](#distribution-specific-setup)
3. [Common Installation](#common-installation)
4. [Claude Desktop Configuration](#claude-desktop-configuration)
5. [Verification and Testing](#verification-and-testing)
6. [Troubleshooting](#troubleshooting)
7. [Docker Setup](#docker-setup-optional)

---

## Prerequisites

### System Requirements
- **Linux Kernel**: 5.4+ (for Claude Desktop compatibility)
- **CPU**: x86-64, ARM64 (Raspberry Pi 3+)
- **RAM**: Minimum 4GB, recommended 8GB+
- **Disk Space**: ~500MB for Python environment and dependencies
- **Internet**: Required for downloading packages and market data

### Supported Distributions
- **Debian-based**: Ubuntu (22.04 LTS+), Debian (12+)
- **RHEL-based**: Fedora (38+), CentOS (8+), RHEL (8+)
- **Arch-based**: Arch Linux, Manjaro
- **Alpine**: Alpine Linux (3.18+)
- **Other**: Any distribution with Python 3.10+ and pip

---

## Distribution-Specific Setup

### Ubuntu / Debian

#### Step 1: Install Python and Build Tools

```bash
# Update package manager
sudo apt update
sudo apt upgrade -y

# Install Python and dependencies
sudo apt install -y python3 python3-pip python3-venv build-essential

# Verify
python3 --version
# Output: Python 3.10 or higher
```

#### Step 2: Create Project Directory

```bash
# Create directory
mkdir -p ~/hidden-regime
cd ~/hidden-regime

# Create virtual environment
python3 -m venv venv

# Activate
source venv/bin/activate
```

#### Step 3: Install Hidden Regime

```bash
# Upgrade pip
pip install --upgrade pip

# Install package
pip install hidden-regime

# Verify
python -c "import hidden_regime; print(hidden_regime.__version__)"
```

### Fedora / RHEL / CentOS

#### Step 1: Install Python and Build Tools

```bash
# Install Python and dependencies
sudo dnf install -y python3 python3-pip python3-devel gcc

# Verify
python3 --version
```

#### Step 2: Create Project Directory

```bash
mkdir -p ~/hidden-regime
cd ~/hidden-regime

# Create virtual environment
python3 -m venv venv

# Activate
source venv/bin/activate
```

#### Step 3: Install Hidden Regime

```bash
pip install --upgrade pip
pip install hidden-regime

# Verify
python -c "import hidden_regime; print(hidden_regime.__version__)"
```

### Arch Linux / Manjaro

#### Step 1: Install Python and Build Tools

```bash
# Install Python and dependencies
sudo pacman -S python python-pip base-devel

# Verify
python --version
```

#### Step 2: Create Project Directory

```bash
mkdir -p ~/hidden-regime
cd ~/hidden-regime

# Create virtual environment
python -m venv venv

# Activate
source venv/bin/activate
```

#### Step 3: Install Hidden Regime

```bash
pip install --upgrade pip
pip install hidden-regime

# Verify
python -c "import hidden_regime; print(hidden_regime.__version__)"
```

### Alpine Linux

#### Step 1: Install Python

```bash
# Alpine uses a different package manager
sudo apk add --no-cache python3 py3-pip python3-dev gcc musl-dev

# Verify
python3 --version
```

#### Step 2-3: Follow Ubuntu steps above

### Raspberry Pi / ARM64

#### Step 1: Update System

```bash
sudo apt update
sudo apt upgrade -y

# Install dependencies
sudo apt install -y python3 python3-pip python3-venv build-essential libopenblas-dev
```

#### Step 2-3: Follow Ubuntu steps above

---

## Common Installation

### Step 1: Get Your Python Path

After activating your virtual environment:

```bash
# Get absolute path to Python
which python
# Output: /home/username/hidden-regime/venv/bin/python

# Or verify with:
python -c "import sys; print(sys.executable)"
```

### Step 2: Create Installation Verification Script

Create a script to verify everything works:

```bash
# Create verification script
cat > ~/hidden-regime/verify.sh << 'EOF'
#!/bin/bash

echo "=== Hidden Regime Installation Verification ==="
echo ""

# Check Python
echo "1. Python Installation:"
source venv/bin/activate
python --version
python -c "import sys; print(f'  Location: {sys.executable}')"
echo ""

# Check Hidden Regime
echo "2. Hidden Regime Package:"
python -c "import hidden_regime; print(f'  Version: {hidden_regime.__version__}')"
python -c "import hidden_regime_mcp; print('  MCP Server: OK')"
echo ""

# Check Dependencies
echo "3. Core Dependencies:"
python -c "import pandas; print(f'  pandas: OK')"
python -c "import yfinance; print(f'  yfinance: OK')"
python -c "import sklearn; print(f'  scikit-learn: OK')"
echo ""

# Test MCP startup
echo "4. MCP Server Startup Test (5 seconds):"
timeout 5 python -m hidden_regime_mcp || echo "  Server timeout (expected - server is running)"
echo ""

echo "=== Verification Complete ==="
EOF

# Make executable
chmod +x ~/hidden-regime/verify.sh

# Run it
~/hidden-regime/verify.sh
```

---

## Claude Desktop Configuration

### Step 1: Locate Configuration File

Claude Desktop config on Linux:

```bash
# The config file location (using XDG standard)
~/.config/Claude/claude_desktop_config.json

# Or the alternate location
~/.claude/claude_desktop_config.json

# Create directory if it doesn't exist
mkdir -p ~/.config/Claude
```

### Step 2: Edit Configuration File

```bash
# Using nano
nano ~/.config/Claude/claude_desktop_config.json

# Using vim
vim ~/.config/Claude/claude_desktop_config.json

# Using your preferred editor
gedit ~/.config/Claude/claude_desktop_config.json

# Or VS Code
code ~/.config/Claude/claude_desktop_config.json
```

### Step 3: Add Hidden Regime MCP

```json
{
  "mcpServers": {
    "hidden-regime": {
      "command": "/home/username/hidden-regime/venv/bin/python",
      "args": ["-m", "hidden_regime_mcp"]
    }
  }
}
```

**Replace `/home/username` with your actual username.**

### Step 4: Save and Restart Claude Desktop

```bash
# Close Claude Desktop completely
pkill -f "Claude"

# Wait a moment
sleep 2

# Reopen Claude Desktop
# (from Applications menu or terminal, depending on your setup)
```

---

## Verification and Testing

### Command Line Test

```bash
# Navigate to project
cd ~/hidden-regime

# Activate venv
source venv/bin/activate

# Test MCP server
python -m hidden_regime_mcp

# You should see output about the MCP server starting
# Press Ctrl+C to stop
```

### Claude Desktop Test

1. Open Claude Desktop
2. Start a new conversation
3. Use this prompt:
   ```
   Detect the current market regime for SPY (S&P 500 ETF).
   What's the current regime and how confident is the model?
   ```

4. Expected: Detailed analysis with regime name, confidence, duration, etc.

### Full Diagnostic

```bash
cd ~/hidden-regime
source venv/bin/activate

cat > test_full.py << 'EOF'
import sys
print("=== Full Diagnostic ===")
print(f"Python: {sys.version}")
print(f"Executable: {sys.executable}")

# Test imports
try:
    import hidden_regime
    print(f"hidden-regime: {hidden_regime.__version__}")
except Exception as e:
    print(f"hidden-regime: ERROR - {e}")

try:
    import hidden_regime_mcp
    print("hidden-regime-mcp: OK")
except Exception as e:
    print(f"hidden-regime-mcp: ERROR - {e}")

try:
    from hidden_regime_mcp.cache import get_cache
    cache = get_cache()
    print(f"Cache system: OK")
except Exception as e:
    print(f"Cache system: ERROR - {e}")

try:
    import yfinance
    print("yfinance: OK")
except Exception as e:
    print(f"yfinance: ERROR - {e}")

print("=== End Diagnostic ===")
EOF

python test_full.py
```

---

## Troubleshooting

### Issue: "Python: command not found"

**Solution 1: Use python3**
```bash
# On some systems, python3 is available but not python
python3 --version

# Use in venv if needed
python3 -m venv venv
```

**Solution 2: Check PATH**
```bash
# Check if Python is in PATH
which python3

# If not found, reinstall:
sudo apt install python3  # Ubuntu/Debian
sudo dnf install python3  # Fedora/RHEL
```

### Issue: "ModuleNotFoundError: No module named 'hidden_regime_mcp'"

**Solution:**
```bash
cd ~/hidden-regime
source venv/bin/activate

# Verify the module
python -c "import hidden_regime_mcp; print(hidden_regime_mcp.__file__)"

# If not found, reinstall
pip uninstall hidden-regime
pip install --upgrade hidden-regime

# Try again
python -c "import hidden_regime_mcp"
```

### Issue: "Permission denied" or "No such file or directory"

**Solution 1: Fix file permissions**
```bash
# Make Python executable
chmod +x ~/hidden-regime/venv/bin/python

# Make scripts executable
chmod +x ~/hidden-regime/venv/bin/*
```

**Solution 2: Verify path in config**
```bash
# Check if path exists
ls -la ~/hidden-regime/venv/bin/python

# If it doesn't exist, you may need to recreate venv
cd ~/hidden-regime
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install hidden-regime
```

### Issue: "ImportError: No module named 'pandas'"

**Solution:**
```bash
cd ~/hidden-regime
source venv/bin/activate

# Reinstall with all dependencies
pip install --upgrade --force-reinstall hidden-regime

# Verify dependencies
pip list | grep -E "pandas|numpy|scikit-learn|yfinance"
```

### Issue: Claude Desktop won't connect to MCP

**Solution 1: Check if Claude Desktop supports your environment**
- Claude Desktop is primarily for macOS and Windows
- For Linux, ensure you have the AppImage version

**Solution 2: Verify config syntax**
```bash
# Check JSON is valid
python3 -m json.tool ~/.config/Claude/claude_desktop_config.json

# Should output: OK if valid, error if invalid
```

**Solution 3: Check logs**
```bash
# Look for Claude Desktop logs
find ~/.cache -name "*claude*" -type f 2>/dev/null

# Or system logs
journalctl -xe | grep -i claude

# Or dmesg
dmesg | tail -20
```

### Issue: "Slow response" or "request timeout"

**Solution 1: Check internet**
```bash
# Test if yfinance can download data
python -c "import yfinance as yf; print(yf.download('SPY', period='1d'))"

# If this fails, you have network issues
```

**Solution 2: Increase timeout in config**
```json
{
  "mcpServers": {
    "hidden-regime": {
      "command": "/home/username/hidden-regime/venv/bin/python",
      "args": ["-m", "hidden_regime_mcp"],
      "timeout": 60000
    }
  }
}
```

### Issue: Out of memory when loading large datasets

**Solution:**
```bash
# Increase system swap
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Check available memory
free -h

# Or limit HMM to fewer states
# Set n_states=2 instead of 3 in your queries
```

---

## Docker Setup (Optional)

For containerized deployment:

### Step 1: Create Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /app/venv

# Activate venv in PATH
ENV PATH="/app/venv/bin:$PATH"

# Install hidden-regime
RUN pip install --upgrade pip && \
    pip install hidden-regime

# Expose MCP on port 3000
EXPOSE 3000

# Run MCP server
CMD ["python", "-m", "hidden_regime_mcp"]
```

### Step 2: Build and Run

```bash
# Build image
docker build -t hidden-regime:latest .

# Run container
docker run -d \
  --name hidden-regime \
  -p 3000:3000 \
  hidden-regime:latest

# Check logs
docker logs hidden-regime

# Stop container
docker stop hidden-regime
```

---

## Performance Tuning

### For Raspberry Pi / Low-RAM Systems

```bash
# Limit Python memory usage
export PYTHONUNBUFFERED=1

# Use minimal dependencies
pip install --no-cache-dir hidden-regime

# Monitor resources
watch free -h
top -p $(pgrep -f "python -m hidden_regime_mcp")
```

### For High-Performance Servers

```bash
# Use production-grade server
pip install gunicorn

# Run with multiple workers
gunicorn --workers 4 --bind 0.0.0.0:3000 hidden_regime_mcp.asgi:app
```

---

## Next Steps

After successful setup:

1. **Test the MCP**: Ask Claude about market regimes
2. **Read API docs**: [README_MCP.md](../README_MCP.md)
3. **Explore examples**: `examples/` directory in the package
4. **Join community**: [github.com/hidden-regime](https://github.com/hidden-regime)

---

## Getting Help

If you encounter issues:

1. **Check this guide**: Review [Troubleshooting](#troubleshooting)
2. **Run diagnostics**:
   ```bash
   python test_full.py  # From section above
   ```
3. **GitHub Issues**: [github.com/hidden-regime/issues](https://github.com/hidden-regime/issues)
4. **Distribution forums**: Ask in your distro's community

---

**Last Updated**: November 2025
**Version**: 1.1.0+
**Status**: Production Ready
**Tested On**:
- Ubuntu 22.04 LTS, 24.04 LTS
- Debian 12
- Fedora 39, 40
- Arch Linux (latest)
- Raspberry Pi OS (Bullseye, Bookworm)
