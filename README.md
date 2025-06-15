# AI Image Editor

A Django web application for image processing using Real-ESRGAN and computer vision techniques.

## Features

- **AI Enhancer**: Real-ESRGAN super-resolution with up to 4x upscaling
- **Gamma CLAHE**: Gamma correction with CLAHE
- **Shadow Fight**: Shadow detection and brightness adjustment
- **Grayscale**: Advanced grayscale conversion

## Setup

### Prerequisites

- Python 3.10+
- pip (Python package manager)

### Quick Setup (Recommended)

Run the automated setup script:

```bash
chmod +x setup.sh
./setup.sh
```

### Manual Installation

1. **Create virtual environment:**
   ```bash
   python3 -m venv imgapp_env
   source imgapp_env/bin/activate  # On Windows: imgapp_env\Scripts\activate
   ```

2. **Upgrade pip and install PyTorch:**
   ```bash
   python -m pip install --upgrade pip setuptools wheel
   
   # For CPU only (recommended for most users)
   python -m pip install torch>=1.13.0 torchvision>=0.14.0 --index-url https://download.pytorch.org/whl/cpu
   
   # For CUDA GPU support (if you have compatible GPU)
   python -m pip install torch>=1.13.0 torchvision>=0.14.0 --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Install BasicSR (RealESRGAN foundation):**
   ```bash
   python -m pip install git+https://github.com/XPixelGroup/BasicSR@master --use-pep517
   ```

4. **Install dependencies:**
   ```bash
   python -m pip install -r requirements.txt
   ```

5. **Run migrations:**
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

6. **Create directories:**
   ```bash
   mkdir -p media/uploads media/processed static temp
   ```

7. **Start server:**
   ```bash
   python manage.py runserver
   ```

8. **Open browser:**
   ```
   http://127.0.0.1:8000
   ```

### Activating Environment

After initial setup, activate the virtual environment:

```bash
source imgapp_env/bin/activate  # On Windows: imgapp_env\Scripts\activate
```

To deactivate when done:
```bash
deactivate
```

## Usage

1. Upload an image (drag & drop or click to browse)
2. Select processing mode
3. Click "Process Image"
4. View results and download

## API Endpoints

- `POST /process/` - Process image
- `GET /download/<id>/` - Download processed image
- `GET /modes/` - Get available modes

## Technologies

- Django 5.2.1
- Real-ESRGAN, PyTorch
- OpenCV, scikit-image
- Bootstrap 5