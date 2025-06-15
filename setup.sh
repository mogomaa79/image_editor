#!/bin/bash

# AI Image Editor with RealESRGAN - Setup Script
# This script sets up the complete environment for the AI Image Editor using Python venv

echo "🚀 Setting up AI Image Editor with RealESRGAN..."
echo "=============================================="

# Check if Python 3.10+ is installed
python_version=$(python3 --version 2>/dev/null | cut -d' ' -f2 | cut -d'.' -f1,2)
if [[ -z "$python_version" ]]; then
    echo "❌ Python 3 is not installed. Please install Python 3.10 or higher."
    exit 1
fi

# Convert version to comparable format (e.g., 3.10 -> 310)
version_num=$(echo $python_version | sed 's/\.//')
if [[ $version_num -lt 310 ]]; then
    echo "❌ Python 3.10 or higher is required. Found: $python_version"
    exit 1
fi

echo "✅ Found Python $python_version"

# Remove existing virtual environment if it exists
if [[ -d "imgapp_env" ]]; then
    echo "🗑️ Removing existing virtual environment..."
    rm -rf imgapp_env
fi

# Create virtual environment
echo "📦 Creating virtual environment 'imgapp_env'..."
python3 -m venv imgapp_env

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source imgapp_env/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
python -m pip install --upgrade pip setuptools wheel

# Install PyTorch first (for better dependency resolution)
echo "⚡ Installing PyTorch and torchvision..."
python -m pip install torch>=1.13.0 torchvision>=0.14.0 --index-url https://download.pytorch.org/whl/cpu

# Install BasicSR from source (required for RealESRGAN)
echo "🤖 Installing BasicSR..."
python -m pip install basicsr

# Install other dependencies from requirements.txt
echo "📚 Installing project dependencies..."
python -m pip install -r requirements.txt

# Run Django migrations
echo "🗄️ Setting up database..."
python manage.py makemigrations
python manage.py migrate

# Create media directories
echo "📁 Creating media directories..."
mkdir -p media/uploads
mkdir -p media/processed
mkdir -p static
mkdir -p temp

# Download RealESRGAN model if not present
echo "📥 Checking RealESRGAN model..."
if [ ! -f "realesr-general-x4v3.pth" ]; then
    echo "Downloading RealESRGAN model..."
    wget -O realesr-general-x4v3.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth
else
    echo "RealESRGAN model already exists."
fi

echo ""
echo "🎉 Setup completed successfully!"
echo "=============================================="
echo ""
echo "To activate the environment:"
echo "  source imgapp_env/bin/activate"
echo ""
echo "To start the development server:"
echo "  python manage.py runserver"
echo ""
echo "To deactivate the environment when done:"
echo "  deactivate"
echo ""
echo "🚀 Enjoy professional AI image enhancement!" 