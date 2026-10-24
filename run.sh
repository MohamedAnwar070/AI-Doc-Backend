#!/bin/bash

echo "========================================"
echo "PDF OCR Backend - Server Startup"
echo "========================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install/Update dependencies
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements-linuxx.txt

# Check for .env file
if [ ! -f ".env" ]; then
    echo "Warning: .env file not found, using defaults"
fi

# Ensure directories exist
mkdir -p uploads outputs

echo ""
echo "========================================"
echo "Starting server..."
echo "========================================"
echo ""

# Run the server
python main.py
