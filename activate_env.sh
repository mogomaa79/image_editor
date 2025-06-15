#!/bin/bash

# AI Image Editor - Environment Activation Script
# Use this script to quickly activate the virtual environment

if [ ! -d "imgapp_env" ]; then
    echo "❌ Virtual environment 'imgapp_env' not found!"
    echo "Please run setup.sh first to create the environment."
    exit 1
fi

echo "🔧 Activating AI Image Editor environment..."
source imgapp_env/bin/activate

echo "✅ Environment activated!"
echo "To start the server: python manage.py runserver"
echo "To deactivate: deactivate"

# Keep the shell open in the activated environment
exec $SHELL 