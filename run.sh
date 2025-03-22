#!/bin/bash

# This script runs the ComfyUI Launcher Streamlit application

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3 to run this application."
    exit 1
fi

# Check if required packages are installed
if ! python3 -c "import streamlit" &> /dev/null; then
    echo "Installing required packages..."
    pip install -r requirements.txt
fi

# Create necessary directories
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
mkdir -p "$DIR/projects"
mkdir -p "$DIR/models"
mkdir -p "$DIR/templates"

# Check if streamlit_patches.py exists
if [ ! -f "$DIR/streamlit_patches.py" ]; then
    echo "Warning: streamlit_patches.py not found. Some features may not work correctly."
fi

# Run the Streamlit application
echo "Starting ComfyUI Launcher Streamlit application..."
streamlit run "$DIR/app.py" "$@"