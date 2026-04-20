#!/bin/zsh

# --- AngioPy Launcher for macOS ---
# This script will set up a virtual environment, install requirements, and run the app.

# Navigate to the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "--------------------------------------------------------"
echo "🚀 Starting AngioPy Segmentation..."
echo "--------------------------------------------------------"
echo "📂 Working directory: $SCRIPT_DIR"

# Check if virtual environment exists and is valid for this machine
NEEDS_RECREATE=false

if [ ! -d ".venv" ]; then
    NEEDS_RECREATE=true
elif [ ! -f ".venv/bin/python3" ]; then
    NEEDS_RECREATE=true
elif [ ! -f ".venv/pyvenv.cfg" ]; then
    NEEDS_RECREATE=true
else
    # Check if the venv was created with a Python that still exists
    VENV_HOME=$(grep "^home" .venv/pyvenv.cfg | cut -d'=' -f2 | tr -d ' ')
    if [ ! -x "$VENV_HOME/python3" ]; then
        echo "⚠️  Virtual environment points to missing Python at: $VENV_HOME"
        NEEDS_RECREATE=true
    fi
    # Check if the venv activate script has the correct path
    if grep -q "VIRTUAL_ENV=" .venv/bin/activate 2>/dev/null; then
        ACTIVATE_PATH=$(grep "^VIRTUAL_ENV=" .venv/bin/activate | cut -d'"' -f2)
        EXPECTED_PATH="$SCRIPT_DIR/.venv"
        if [ "$ACTIVATE_PATH" != "$EXPECTED_PATH" ]; then
            echo "⚠️  Virtual environment was created for a different path."
            echo "   Expected: $EXPECTED_PATH"
            echo "   Found:    $ACTIVATE_PATH"
            NEEDS_RECREATE=true
        fi
    fi
fi

if [ "$NEEDS_RECREATE" = true ]; then
    echo "📦 Creating virtual environment (.venv)..."
    # Remove old broken venv if it exists
    if [ -d ".venv" ]; then
        echo "🗑️  Removing old virtual environment..."
        rm -rf ".venv"
    fi
    python3 -m venv .venv
    if [ $? -ne 0 ]; then
        echo "❌ Failed to create virtual environment!"
        echo "   Make sure python3 is installed on this system."
        echo "--------------------------------------------------------"
        echo "⚠️  Press any key to close this window."
        read -k 1 -s
        exit 1
    fi
    echo "✅ Virtual environment created successfully."
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source .venv/bin/activate

# Verify activation worked
if ! command -v pip &> /dev/null; then
    echo "❌ Failed to activate virtual environment (pip not found)."
    echo "   Trying to use venv pip directly..."
    # Fallback: use the venv python/pip directly
    PYTHON="$SCRIPT_DIR/.venv/bin/python3"
    PIP="$SCRIPT_DIR/.venv/bin/pip"
else
    PYTHON="python3"
    PIP="pip"
fi

# Install/upgrade pip and requirements
echo "📥 Checking dependencies (this may take a while on first run)..."
"$PIP" install --upgrade pip

"$PIP" install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies!"
    echo "--------------------------------------------------------"
    echo "⚠️  Press any key to close this window."
    read -k 1 -s
    exit 1
fi

# fil-finder requires astropy<6 but works fine with astropy>=6 at runtime.
# Install with --no-deps to skip the incompatible version check.
echo "📥 Installing fil-finder (skipping astropy version check)..."
"$PIP" install fil-finder==1.7.2 --no-deps
if [ $? -ne 0 ]; then
    echo "❌ Failed to install fil-finder!"
    echo "--------------------------------------------------------"
    echo "⚠️  Press any key to close this window."
    read -k 1 -s
    exit 1
fi

# Run the app
echo "🌐 Launching app in your browser..."
"$PYTHON" -m streamlit run angioPySegmentation.py

# If the app stops, keep the terminal open so the user can see errors
echo "--------------------------------------------------------"
echo "⚠️  Application stopped. Press any key to close this window."
read -k 1 -s
