#!/bin/bash

# Simple AI Image Editor Setup Script
echo "🚀 Setting up AI Image Editor..."

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not found. Please install Python first."
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
if [[ -d "imgapp" ]]; then
    rm -rf imgapp
fi
python3 -m venv imgapp
source imgapp/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies based on architecture
if [[ "$(uname -m)" == "arm64" ]] && [[ -f "requirements-arm64.txt" ]]; then
    echo "🍎 Installing ARM64 dependencies..."
    pip install -r requirements-arm64.txt
else
    echo "🔧 Installing standard dependencies..."
    pip install -r requirements.txt
fi

# Setup Django
echo "🗄️ Setting up Django..."
python manage.py migrate

# Create media directories
mkdir -p media/uploads media/processed

echo "✅ Setup complete!"
echo ""
echo "To start the app:"
echo "1. source imgapp/bin/activate"
echo "2. python manage.py runserver" 