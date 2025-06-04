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
- Conda (recommended)

### Installation

1. **Create conda environment:**
   ```bash
   conda create -n imgapp python=3.10 -y
   conda activate imgapp
   ```

2. **Install PyTorch:**
   ```bash
   # For CUDA GPU support
   conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
   
   # For CPU only
   conda install pytorch torchvision cpuonly -c pytorch
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run migrations:**
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

5. **Start server:**
   ```bash
   python manage.py runserver
   ```

6. **Open browser:**
   ```
   http://127.0.0.1:8000
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