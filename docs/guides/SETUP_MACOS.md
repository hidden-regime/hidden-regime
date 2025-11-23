# Hidden Regime MCP Setup for macOS

This guide provides step-by-step instructions for setting up Hidden Regime with Claude Desktop on macOS (Intel and Apple Silicon).

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation Steps](#installation-steps)
3. [Claude Desktop Configuration](#claude-desktop-configuration)
4. [Verification and Testing](#verification-and-testing)
5. [Troubleshooting](#troubleshooting)
6. [macOS-Specific Notes](#macos-specific-notes)

---

## Prerequisites

### System Requirements
- **macOS**: 12 (Monterey) or later
- **Processor**: Intel or Apple Silicon (M1/M2/M3)
- **RAM**: Minimum 4GB, recommended 8GB+
- **Disk Space**: ~500MB for Python environment and dependencies
- **Internet**: Required for downloading packages and market data

### Required Software
- **Xcode Command Line Tools**: For building Python packages
- **Python 3.10+**: From [python.org](https://www.python.org/downloads/) or Homebrew
- **Claude Desktop**: Download from [claude.ai](https://claude.ai/download)
- **Text Editor**: Any code editor (VS Code, Sublime, Xcode recommended)

---

## Installation Steps

### Step 1: Install Xcode Command Line Tools

These are required for building Python packages from source.

```bash
# In Terminal, run:
xcode-select --install

# You'll see a popup asking to install. Click "Install"
# This takes 5-10 minutes

# Verify installation
xcode-select -p
# Output should be: /Applications/Xcode.app/Contents/Developer
```

### Step 2: Install Python

Choose one method:

#### Option A: Using Homebrew (Recommended)

```bash
# If you don't have Homebrew, install it first:
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.12

# Verify
python3 --version
# Output: Python 3.12.x
```

#### Option B: Using Official Python Installer

1. Download Python 3.12 from [python.org/downloads/macos/](https://www.python.org/downloads/macos/)
2. Choose the appropriate version:
   - **ARM64** if you have M1/M2/M3 Mac
   - **Intel** if you have an Intel Mac
3. Run the installer
4. Verify:
   ```bash
   python3 --version
   ```

### Step 3: Create Project Directory and Virtual Environment

```bash
# Create a dedicated directory for hidden-regime
mkdir -p ~/hidden-regime
cd ~/hidden-regime

# Create a virtual environment (recommended)
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# You should see (venv) at the beginning of your terminal prompt
```

### Step 4: Install Hidden Regime Package

```bash
# Make sure your venv is activated
source venv/bin/activate

# Upgrade pip first
pip install --upgrade pip

# Install hidden-regime
pip install hidden-regime

# Verify installation
python -c "import hidden_regime; print(hidden_regime.__version__)"
# Output: 1.1.0 (or higher)
```

### Step 5: Get Your Python Path

You'll need this for Claude Desktop configuration:

```bash
# Get the full path to your Python executable
which python
# Example output: /Users/yourname/hidden-regime/venv/bin/python

# Or use:
python -c "import sys; print(sys.executable)"
# Example output: /Users/yourname/hidden-regime/venv/bin/python

# Copy this path - you'll need it next
```

---

## Claude Desktop Configuration

### Step 1: Locate Configuration File

macOS stores Claude Desktop config here:
```bash
# The config file location
~/Library/Application\ Support/Claude/claude_desktop_config.json

# Or use this shortcut to open in Finder:
open ~/Library/Application\ Support/Claude/
```

If the file doesn't exist, create it with the content below.

### Step 2: Edit Configuration File

Open the config file with your text editor:

```bash
# Using nano (built-in editor)
nano ~/Library/Application\ Support/Claude/claude_desktop_config.json

# Or using VS Code
code ~/Library/Application\ Support/Claude/claude_desktop_config.json

# Or using default editor
open ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

### Step 3: Add Hidden Regime MCP

Add the following to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "hidden-regime": {
      "command": "/Users/yourname/hidden-regime/venv/bin/python",
      "args": ["-m", "hidden_regime_mcp"]
    }
  }
}
```

**Replace `/Users/yourname/hidden-regime/venv/bin/python` with your actual path from Step 5 above.**

**Complete example:**
```json
{
  "mcpServers": {
    "hidden-regime": {
      "command": "/Users/alice/hidden-regime/venv/bin/python",
      "args": ["-m", "hidden_regime_mcp"]
    }
  }
}
```

### Step 4: Save and Restart Claude Desktop

1. Save the file (Cmd+S if using a text editor)
2. Fully close Claude Desktop (Cmd+Q)
3. Wait 2-3 seconds
4. Reopen Claude Desktop
5. You should see a small indicator showing the MCP is connected

---

## Verification and Testing

### Command Line Test

Before testing in Claude Desktop, verify the MCP works from the command line:

```bash
# Navigate to your project
cd ~/hidden-regime

# Activate the virtual environment
source venv/bin/activate

# Test the MCP server
python -m hidden_regime_mcp

# You should see output indicating the MCP server is running
# Press Ctrl+C to stop
```

### Claude Desktop Test

1. Open Claude Desktop
2. In a new conversation, try this prompt:
   ```
   Detect the current market regime for SPY (S&P 500 ETF).
   What's the current regime and how confident is the model?
   ```

3. Expected response: A detailed analysis with:
   - Current regime name (bull, bear, sideways)
   - Confidence level
   - Expected duration
   - Price performance metrics
   - Interpretation

---

## Troubleshooting

### Issue: "Python: command not found"

**Solution 1: Check installation**
```bash
# Verify Python is installed
which python3

# If it's not found, reinstall:
brew install python@3.12

# Add to PATH (if needed):
export PATH="/usr/local/bin:$PATH"
```

**Solution 2: Use full path in config**
```bash
# Instead of "python", use the full path
/usr/local/bin/python3 -m hidden_regime_mcp
```

### Issue: "ModuleNotFoundError: No module named 'hidden_regime_mcp'"

**Solution:**
```bash
# Make sure you're in the venv
cd ~/hidden-regime
source venv/bin/activate

# Reinstall the package
pip uninstall hidden-regime
pip install --upgrade hidden-regime

# Verify
python -c "import hidden_regime_mcp"
```

### Issue: MCP server won't start in Claude Desktop

**Solution 1: Check file permissions**
```bash
# The Python executable should be executable
chmod +x ~/hidden-regime/venv/bin/python
```

**Solution 2: Verify the path in config**
```bash
# Check if the path exists and is correct
ls -la ~/hidden-regime/venv/bin/python

# If it doesn't exist, you may have the wrong path
```

**Solution 3: Clear Claude Desktop cache**
```bash
# Close Claude Desktop
# Then run:
rm -rf ~/Library/Caches/Claude

# Reopen Claude Desktop
```

### Issue: "Slow response" or "request timeout"

**Solution 1: Check internet connection**
```bash
# Test yfinance connectivity
python -c "import yfinance as yf; print(yf.download('SPY', period='1d'))"
```

**Solution 2: Restart the MCP**
- Fully close Claude Desktop (Cmd+Q)
- Wait 5 seconds
- Reopen Claude Desktop

**Solution 3: Increase timeout**
Edit your config to increase timeout:
```json
{
  "mcpServers": {
    "hidden-regime": {
      "command": "/Users/yourname/hidden-regime/venv/bin/python",
      "args": ["-m", "hidden_regime_mcp"],
      "timeout": 60000
    }
  }
}
```

### Issue: "Insufficient data" error

**Solution:**
Some stocks may have limited historical data. Try:
- Large cap stocks: SPY, QQQ, IWM, AAPL
- ETFs: BND, GLD, USO
- Avoid: Very new stocks, penny stocks, delisted stocks

### Issue: "yfinance not available"

**Solution:**
```bash
cd ~/hidden-regime
source venv/bin/activate

# Install yfinance
pip install yfinance

# Reinstall hidden-regime
pip install --upgrade hidden-regime
```

---

## macOS-Specific Notes

### Intel vs Apple Silicon

**Apple Silicon (M1/M2/M3) Macs:**
- Download ARM64 version of Python from python.org
- Or use: `brew install python@3.12` (automatically correct version)
- Performance is excellent

**Intel Macs:**
- Download the Intel version from python.org
- Or use: `brew install python@3.12` (automatically correct version)

### Gatekeeper Warning

If you see "Cannot open because developer cannot be verified":

1. Go to **System Settings** → **Privacy & Security**
2. Scroll to "Security"
3. Click "Open Anyway" next to the blocked app
4. Or allow Python:
   ```bash
   xattr -d com.apple.quarantine ~/hidden-regime/venv/bin/python
   ```

### Homebrew Path Issues

If Homebrew is installed but commands not found:

```bash
# Check Homebrew installation
brew --version

# Add Homebrew to PATH (for Intel Macs)
export PATH="/usr/local/bin:$PATH"

# Add Homebrew to PATH (for Apple Silicon)
export PATH="/opt/homebrew/bin:$PATH"

# Make it permanent by adding to ~/.zprofile or ~/.bash_profile
```

### Virtual Environment Best Practices

Keep your venv organized:

```bash
# Never commit venv to git
echo "venv/" >> ~/.gitignore

# Make it easy to activate
# Add this to ~/.zprofile:
alias activate_regime="source ~/hidden-regime/venv/bin/activate"

# Then you can just type:
activate_regime
```

---

## Advanced Configuration

### Using Python from Conda

If you prefer conda environments:

```bash
# Create conda environment
conda create -n hidden-regime python=3.12

# Activate
conda activate hidden-regime

# Install
pip install hidden-regime

# Get Python path
which python
# /Users/yourname/opt/anaconda3/envs/hidden-regime/bin/python

# Use that path in Claude Desktop config
```

### Multiple Python Versions

To use a specific Python version:

```bash
# Install multiple versions with Homebrew
brew install python@3.11 python@3.12

# Use specific version in venv
/usr/local/bin/python3.12 -m venv venv

# Or update your config to use full path:
"/usr/local/bin/python3.12"
```

---

## Next Steps

After successful setup:

1. **Test the MCP**: Ask Claude about market regimes
2. **Read the API docs**: [README_MCP.md](../README_MCP.md)
3. **Explore examples**: Check `examples/` in the package
4. **Join community**: [github.com/hidden-regime](https://github.com/hidden-regime)

---

## Getting Help

If you encounter issues:

1. **Check this guide**: Review the [Troubleshooting](#troubleshooting) section
2. **Run diagnostic**:
   ```bash
   python -c "import sys; print(sys.version); import hidden_regime; print(hidden_regime.__version__)"
   ```
3. **GitHub Issues**: [github.com/hidden-regime/issues](https://github.com/hidden-regime/issues)
4. **Documentation**: [README_MCP.md](../README_MCP.md)

---

**Last Updated**: November 2025
**Version**: 1.1.0+
**Status**: Production Ready
**Tested On**:
- macOS 14 (Sonoma) - Apple Silicon
- macOS 13 (Ventura) - Intel
- macOS 12 (Monterey) - Apple Silicon
