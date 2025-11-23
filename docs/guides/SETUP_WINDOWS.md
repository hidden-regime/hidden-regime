# Hidden Regime MCP Setup for Windows

This guide provides step-by-step instructions for setting up Hidden Regime with Claude Desktop on Windows (both WSL2 and native Python).

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation Options](#installation-options)
3. [Claude Desktop Configuration](#claude-desktop-configuration)
4. [Verification and Testing](#verification-and-testing)
5. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements
- **OS**: Windows 10/11 (21H2 or later recommended)
- **RAM**: Minimum 4GB, recommended 8GB+
- **Disk Space**: ~500MB for Python environment and dependencies
- **Internet**: Required for downloading packages and market data

### Required Software
- **Python 3.10+**: Download from [python.org](https://www.python.org/downloads/)
- **Claude Desktop**: Download from [claude.ai](https://claude.ai/download)
- **Text Editor**: Any code editor (VS Code recommended, but Notepad works)

---

## Installation Options

### Option 1: Native Windows Python (Recommended for Simplicity)

This approach installs Python directly on Windows without WSL2.

#### Step 1: Install Python

1. Download Python 3.11 or 3.12 from [python.org](https://www.python.org/downloads/windows/)
2. Run the installer:
   - **Check "Add python.exe to PATH"** ✓ (This is important!)
   - Click "Install Now"
3. Verify installation:
   ```bash
   python --version
   # Output should be: Python 3.11.x or 3.12.x
   ```

#### Step 2: Install Hidden Regime Package

1. Open Command Prompt (Win+R, type `cmd`, press Enter)
2. Run the following commands:
   ```bash
   # Create a dedicated project directory
   mkdir C:\Users\%USERNAME%\hidden-regime
   cd C:\Users\%USERNAME%\hidden-regime

   # Create a virtual environment
   python -m venv venv

   # Activate the virtual environment
   venv\Scripts\activate

   # Install Hidden Regime
   pip install hidden-regime
   ```

3. Verify installation:
   ```bash
   python -c "import hidden_regime; print(hidden_regime.__version__)"
   # Output: 1.1.0 (or higher)
   ```

#### Step 3: Find Python and venv Paths

You'll need these paths for Claude Desktop configuration:

```bash
# While still in the activated venv, run:
python -c "import sys; print(sys.executable)"
# Output example: C:\Users\YourUsername\hidden-regime\venv\Scripts\python.exe

# Get the hidden-regime package location:
python -c "import hidden_regime; print(hidden_regime.__file__)"
```

**Note the paths** - you'll need them in Step 4.

---

### Option 2: WSL2 (Windows Subsystem for Linux)

This approach uses Windows Subsystem for Linux 2 for a Unix-like environment.

#### Step 1: Install WSL2

1. Open PowerShell as Administrator (Right-click → Run as Administrator)
2. Run:
   ```powershell
   wsl --install
   ```
3. Restart your computer
4. After restart, Ubuntu will launch - create a username and password

#### Step 2: Install Python on Ubuntu

In the Ubuntu terminal:
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv

# Verify
python3 --version
```

#### Step 3: Install Hidden Regime

```bash
# Create project directory
mkdir -p ~/hidden-regime
cd ~/hidden-regime

# Create virtual environment
python3 -m venv venv

# Activate
source venv/bin/activate

# Install Hidden Regime
pip install hidden-regime

# Verify
python -c "import hidden_regime; print(hidden_regime.__version__)"
```

#### Step 4: Find WSL Paths

```bash
# While in activated venv:
which python
# Output example: /home/username/hidden-regime/venv/bin/python

# Verify the package
python -c "import hidden_regime; print(hidden_regime.__file__)"
```

---

## Claude Desktop Configuration

### Configuration File Location

The MCP configuration file differs by installation method:

**Native Windows:**
- Path: `%APPDATA%\Claude\claude_desktop_config.json`
- Expanded example: `C:\Users\YourUsername\AppData\Roaming\Claude\claude_desktop_config.json`

**WSL2:**
- Path: Same as above (`%APPDATA%\Claude\claude_desktop_config.json`)
- But the command references WSL paths

### Step 1: Locate Configuration File

1. Press `Win+R`
2. Type: `%APPDATA%\Claude`
3. Press Enter
4. You should see `claude_desktop_config.json`
5. Right-click → Open with → Notepad

### Step 2: Add Hidden Regime MCP

#### For Native Windows Python:

Find this line in your config:
```json
"mcpServers": {
```

Add the hidden-regime entry:
```json
"mcpServers": {
  "hidden-regime": {
    "command": "C:\\Users\\YourUsername\\hidden-regime\\venv\\Scripts\\python.exe",
    "args": ["-m", "hidden_regime_mcp"]
  }
}
```

**Replace `C:\Users\YourUsername\hidden-regime\venv\Scripts\python.exe` with your actual path from Step 3 above.**

#### For WSL2:

```json
"mcpServers": {
  "hidden-regime": {
    "command": "wsl",
    "args": [
      "bash",
      "-c",
      "source /home/username/hidden-regime/venv/bin/activate && python -m hidden_regime_mcp"
    ]
  }
}
```

**Replace `/home/username/hidden-regime` with your actual WSL path.**

### Step 3: Save Configuration

1. Save the file (Ctrl+S)
2. Close Notepad
3. Restart Claude Desktop completely (close and reopen)

### Step 4: Verify Connection

1. Open Claude Desktop
2. Start a new conversation
3. In the prompt field, click the gear icon (⚙️) at the bottom
4. You should see "hidden-regime" in the list of available MCPs
5. If it shows an error icon, check the [Troubleshooting](#troubleshooting) section

---

## Verification and Testing

### Manual Testing (Command Line)

```bash
# Activate your virtual environment first
cd C:\Users\%USERNAME%\hidden-regime
venv\Scripts\activate

# Test the MCP server starts
python -m hidden_regime_mcp

# If successful, you should see output about the MCP server starting
# Press Ctrl+C to stop
```

### Claude Desktop Testing

1. Open Claude Desktop
2. Ask: "What is the current regime for SPY?"
3. Expected response: A detailed analysis of the S&P 500's current market regime

Example prompt to test:
```
Using the hidden-regime tool, detect the current market regime for SPY (S&P 500).
Tell me:
1. What's the current regime?
2. How confident is the model?
3. What's the expected duration?
```

---

## Troubleshooting

### Issue: "MCP server failed to load" or error icon in Claude Desktop

**Solution 1: Check Python path**
- Verify the path in your config is correct
- Use forward slashes: `C:/Users/YourUsername/...` OR escaped backslashes: `C:\\Users\\YourUsername\\...`
- Never use raw backslashes: `C:\Users\...` (this breaks the JSON)

**Solution 2: Reinstall hidden-regime package**
```bash
venv\Scripts\activate
pip uninstall hidden-regime
pip install --upgrade hidden-regime
```

**Solution 3: Check virtual environment**
```bash
# Verify venv is properly created
venv\Scripts\python.exe --version

# Should output: Python 3.x.x
```

### Issue: "ModuleNotFoundError: No module named 'hidden_regime_mcp'"

**Solution:**
```bash
venv\Scripts\activate
pip install --upgrade hidden-regime

# Verify the module exists
python -c "import hidden_regime_mcp; print(hidden_regime_mcp.__file__)"
```

### Issue: "yfinance not available"

**Solution:**
```bash
venv\Scripts\activate
pip install yfinance
pip install --upgrade hidden-regime
```

### Issue: "No data available for ticker"

**Common causes:**
- Internet connection problem - check your connection
- Invalid ticker symbol - verify the stock symbol is correct
- Market data not available - some tickers may have limited data

**Test your connection:**
```bash
python -c "import yfinance as yf; data = yf.download('SPY', period='1mo'); print(f'Downloaded {len(data)} rows')"
```

### Issue: "Connection timeout" or slow responses

**Solutions:**
1. Check your internet connection speed
2. Try again after a few seconds (rate limiting)
3. Restart Claude Desktop
4. Restart the MCP server:
   ```bash
   # Close Claude Desktop completely
   # Wait 10 seconds
   # Reopen Claude Desktop
   ```

### Issue: Claude Desktop won't find the MCP after config changes

**Solution:**
1. Save the config file
2. **Close Claude Desktop completely** (not just minimize)
3. Wait 5 seconds
4. Open Claude Desktop again

---

## System-Specific Notes

### Windows 11 with Defender

If Windows Defender blocks the Python process:
1. Go to Settings → Privacy & Security → Virus & threat protection
2. Click "Manage settings"
3. Allow `python.exe` through Defender

### Antivirus Software

Some antivirus software may interfere with yfinance data downloads:
- Whitelist `python.exe` in your antivirus
- Or add exception for `%APPDATA%\.venv` directory

### Network/Proxy Issues

If you're behind a corporate proxy:
1. Configure pip to use your proxy:
   ```bash
   pip install --proxy [user:passwd@]proxy.server:port hidden-regime
   ```
2. Or create/edit `%APPDATA%\pip\pip.ini`:
   ```ini
   [global]
   proxy = [user:passwd@]proxy.server:port
   ```

---

## Uninstallation

To remove Hidden Regime and its virtual environment:

```bash
# Delete the entire project directory
rmdir /s C:\Users\%USERNAME%\hidden-regime

# Remove from Claude Desktop config (follow step "Claude Desktop Configuration" above)
```

---

## Next Steps

After successful setup:

1. **Test the MCP**: Ask Claude about market regimes for your favorite stocks
2. **Read the docs**: Check out [README_MCP.md](../README_MCP.md) for detailed tool documentation
3. **Try examples**: Browse the `examples/` directory in the package
4. **Join the community**: Visit [github.com/hidden-regime](https://github.com/hidden-regime)

---

## Getting Help

If you encounter issues:

1. **Check this guide**: Review the [Troubleshooting](#troubleshooting) section
2. **GitHub Issues**: [github.com/hidden-regime/issues](https://github.com/hidden-regime/issues)
3. **Documentation**: [README_MCP.md](../README_MCP.md) for API details
4. **Community**: [Discussions](https://github.com/hidden-regime/discussions)

---

**Last Updated**: November 2025
**Version**: 1.1.0+
**Status**: Production Ready
