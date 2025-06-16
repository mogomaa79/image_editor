# AI Image Editor

A Django web application for advanced image processing using Real-ESRGAN AI super-resolution and computer vision techniques.

## Features

- **AI Enhancer**: Real-ESRGAN super-resolution with up to 4x upscaling
- **Gamma CLAHE**: Advanced gamma correction with CLAHE (Contrast Limited Adaptive Histogram Equalization)
- **Shadow Fight**: Shadow detection and brightness adjustment with adaptive enhancement
- **Grayscale**: Advanced grayscale conversion with multiple methods

## Requirements

### System Requirements
- Python 3.11+
- macOS (Intel/Apple Silicon), Linux, or Windows
- 4GB+ RAM (8GB+ recommended for AI processing)
- 2GB+ free disk space

### Platform-Specific Installation

#### For Apple Silicon (M1/M2) Macs
```bash
# Create virtual environment
python3.11 -m venv imgapp
source imgapp/bin/activate

# Install M1-optimized dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements-arm64.txt
```

#### For Intel Macs, Linux, and Windows
```bash
# Create virtual environment
python3.11 -m venv imgapp
# On Windows: imgapp\Scripts\activate
source imgapp/bin/activate

# Install standard dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Quick Setup with Script
For automated setup, use the provided script:
```bash
chmod +x setup.sh
./setup.sh
```

## Manual Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd image_editor
   ```

2. **Create and activate virtual environment:**
   ```bash
   python3.11 -m venv imgapp
   source imgapp/bin/activate  # On Windows: imgapp\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   # For Apple Silicon Macs
   pip install -r requirements-arm64.txt
   
   # For other systems
   pip install -r requirements.txt
   ```

4. **Run Django migrations:**
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

5. **Create media directories:**
   ```bash
   mkdir -p media/uploads media/processed
   ```

6. **Download AI model (if not included):**
   ```bash
   # The RealESRGAN model will be downloaded automatically on first use
   # Or manually download to project root:
   wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth
   ```

7. **Start the development server:**
   ```bash
   python manage.py runserver
   ```

8. **Open in browser:**
   ```
   http://127.0.0.1:8000
   ```

## Usage

1. **Upload Image**: Drag & drop or click to browse and select an image
2. **Select Mode**: Choose from AI Enhancer, Gamma CLAHE, Shadow Fight, or Grayscale
3. **Set Scale**: For AI Enhancer, choose scale factor (1x-4x)
4. **Process**: Click "Process Image" and wait for results
5. **Download**: View the enhanced image and download when ready

## API Endpoints

- `POST /process/` - Process uploaded image
- `GET /download/<uuid>/` - Download processed image (expires in 2 hours)
- `GET /modes/` - Get available processing modes

## Technologies

### Backend
- **Django 5.2.1** - Web framework

### AI Models
- **RealESRGAN** - Real-world super-resolution

## Acknowledgments

- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) for the AI super-resolution model
- [BasicSR](https://github.com/XPixelGroup/BasicSR) for the super-resolution toolkit
- [OpenCV](https://opencv.org/) for computer vision capabilities