#!/bin/bash

# AI Image Editor with RealESRGAN - Setup Script
# This script sets up the complete environment for the AI Image Editor

echo "🚀 Setting up AI Image Editor with RealESRGAN..."
echo "=============================================="

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed. Please install Anaconda or Miniconda first."
    exit 1
fi

# Create conda environment
echo "📦 Creating conda environment 'imgapp' with Python 3.10..."
conda create -n imgapp python=3.10 -y

# Activate environment
echo "🔧 Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate imgapp

# Install PyTorch and core dependencies via conda
echo "⚡ Installing PyTorch and core dependencies..."
conda install pytorch torchvision torchaudio -c pytorch -y
conda install numpy opencv pillow scikit-image -c conda-forge -y

# Install build tools
echo "🛠️ Installing build tools..."
conda install -c conda-forge cython setuptools wheel -y

# Install BasicSR from source
echo "🤖 Installing BasicSR (RealESRGAN foundation)..."
python -m pip install git+https://github.com/XPixelGroup/BasicSR@master --use-pep517

# Install RealESRGAN and related packages
echo "🎯 Installing RealESRGAN and enhancement libraries..."
python -m pip install facexlib gfpgan realesrgan

# Install Django and web framework dependencies
echo "🌐 Installing Django and web dependencies..."
python -m pip install Django==5.2.1 django-cors-headers==4.7.0

# Install additional utilities
echo "📚 Installing additional utilities..."
python -m pip install tqdm requests pyyaml lmdb

# Run Django migrations
echo "🗄️ Setting up database..."
python manage.py makemigrations
python manage.py migrate

# Create media directories
echo "📁 Creating media directories..."
mkdir -p media/uploads
mkdir -p media/processed
mkdir -p static

# Download RealESRGAN model (optional)
echo "📥 Downloading RealESRGAN model..."
if [ ! -f "realesr-general-x4v3.pth" ]; then
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth
    echo "✅ Model downloaded successfully!"
else
    echo "✅ Model already exists!"
fi

echo ""
echo "🎉 Setup completed successfully!"
echo "=============================================="
echo "To start the AI Image Editor:"
echo "1. conda activate imgapp"
echo "2. python manage.py runserver"
echo "3. Open http://127.0.0.1:8000 in your browser"
echo ""
echo "🚀 Enjoy professional AI image enhancement!" 