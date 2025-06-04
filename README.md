# AI Image Editor

A professional Django web application for advanced image processing using **Real-ESRGAN** and cutting-edge computer vision techniques.

## Features

- **🤖 AI Enhancer**: Real-ESRGAN super-resolution with 4x upscaling and advanced neural network enhancement
- **🎨 Gamma CLAHE**: Advanced gamma correction with CLAHE, local contrast enhancement, and morphological operations  
- **☀️ Shadow Fight**: Intelligent shadow detection and local adaptive enhancement for balanced lighting
- **🎭 Grayscale**: Advanced grayscale conversion with luminance weighting and contrast enhancement

## Key Advantages

✅ **Real-ESRGAN Integration** - State-of-the-art AI super-resolution model  
✅ **Professional Quality** - 4x upscaling with neural network enhancement  
✅ **Cached Processing** - Singleton pattern with model persistence for optimal performance  
✅ **Fallback System** - Robust error handling with graceful degradation  
✅ **Production Ready** - Comprehensive logging, monitoring, and deployment tools  
✅ **GPU Acceleration** - Automatic CUDA detection and optimization  

## Screenshots

The application features a beautiful, modern UI with:
- Drag & drop file upload
- Interactive mode selection cards
- Real-time progress indicators
- Side-by-side image comparison
- One-click download functionality

## Setup Instructions

### Prerequisites

- Python 3.10+
- Conda (recommended)
- CUDA-compatible GPU (optional, for faster processing)

### Installation

1. **Create and activate conda environment:**
   ```bash
   conda create -n imgapp python=3.10 -y
   conda activate imgapp
   ```

2. **Install PyTorch with CUDA support (recommended):**
   ```bash
   # For CUDA 11.8
   conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
   
   # For CPU only
   conda install pytorch torchvision cpuonly -c pytorch
   ```

3. **Install remaining dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Real-ESRGAN model:**
   ```bash
   # The model will be automatically downloaded on first use
   # Or manually download:
   wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth
   ```

5. **Run migrations:**
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

6. **Create superuser (optional):**
   ```bash
   python manage.py createsuperuser
   ```

7. **Start the development server:**
   ```bash
   python manage.py runserver
   ```

8. **Open your browser and navigate to:**
   ```
   http://127.0.0.1:8000
   ```

## Usage

1. **Upload an Image**: Drag and drop an image file or click to browse
2. **Select Processing Mode**: Choose from the available enhancement options
3. **Process**: Click the "Process Image" button and watch the AI enhancement
4. **View Results**: Compare original and processed images side-by-side
5. **Download**: Save the enhanced image to your device

## API Endpoints

- `POST /process/` - Process an uploaded image
- `GET /download/<id>/` - Download processed image
- `GET /modes/` - Get available processing modes

## Technical Details

### Real-ESRGAN AI Enhancement

The AI Enhancer now uses **Real-ESRGAN** (Real-World Enhanced Super-Resolution GAN):

- **Model**: SRVGGNetCompact with 4x upscaling
- **Architecture**: 64 features, 32 convolution layers, PReLU activation
- **Performance**: CUDA acceleration when available, CPU fallback
- **Quality**: State-of-the-art super-resolution with artifact reduction
- **Model Size**: ~4.7MB efficient deployment

### Advanced Image Processing Algorithms

1. **🤖 AI Enhancer (Real-ESRGAN)**: 
   - Real-ESRGAN SRVGGNetCompact model
   - 4x super-resolution upscaling
   - CUDA/CPU automatic detection
   - Tile-based processing for large images
   - Edge preservation and artifact reduction

2. **🎨 Gamma CLAHE**: 
   - LAB color space processing
   - Contrast Limited Adaptive Histogram Equalization
   - Gamma correction with lookup tables
   - Morphological top-hat and black-hat operations
   - Local contrast improvement

3. **☀️ Shadow Fight**: 
   - Intelligent shadow region detection
   - Local adaptive enhancement
   - Morphological mask refinement
   - Gaussian blur for smooth transitions
   - Multi-level brightness/contrast adjustment

4. **🎭 Grayscale**: 
   - ITU-R BT.709 luminance weighting
   - Custom weighted averaging options
   - Local contrast enhancement with CLAHE
   - Desaturation method support

### Performance Optimization

- **Model Caching**: Real-ESRGAN model loaded once and reused
- **GPU Acceleration**: Automatic CUDA detection and utilization
- **Memory Management**: Automatic image size limiting and tile processing
- **Singleton Pattern**: Enhancement manager with cached parameters
- **LRU Caching**: Frequently used parameters cached
- **Fallback System**: Graceful degradation on GPU/model failures

### File Structure

```
image_editor/
├── image_editor_app/          # Django project settings
├── image_processor/           # Main app
│   ├── utils.py              # Image processing functions
│   ├── model_manager.py      # Real-ESRGAN enhancement manager
│   ├── models.py             # Database models
│   ├── views.py              # API and web views
│   ├── admin.py              # Admin interface
│   └── urls.py               # URL routing
├── templates/                 # HTML templates
├── media/                     # Uploaded and processed images
├── static/                    # Static files (CSS, JS)
├── realesr-general-x4v3.pth  # Real-ESRGAN model weights
├── requirements.txt           # Python dependencies
└── manage.py                  # Django management script
```

### Technologies Used

- **Backend**: Django 5.2.1
- **AI Enhancement**: Real-ESRGAN, PyTorch
- **Image Processing**: OpenCV, scikit-image, NumPy
- **Frontend**: Bootstrap 5, Font Awesome, vanilla JavaScript
- **Database**: SQLite (development) / PostgreSQL (production)
- **Deployment**: Gunicorn, WhiteNoise
- **Performance**: LRU caching, singleton patterns

## Configuration

### Environment Variables

Create a `.env` file for production:

```bash
DEBUG=False
SECRET_KEY=your-secret-key-here
ALLOWED_HOSTS=your-domain.com,www.your-domain.com
DATABASE_URL=postgresql://user:pass@localhost/dbname

# GPU Settings
CUDA_VISIBLE_DEVICES=0  # Set GPU device
TORCH_HOME=/path/to/torch/models  # Model cache directory
```

### Settings

Key configuration options in `image_editor_app/settings.py`:

```python
# File upload limits
FILE_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024  # 10MB
DATA_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024  # 10MB

# Media files
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

# Real-ESRGAN Model Path
REALESRGAN_MODEL_PATH = BASE_DIR / 'realesr-general-x4v3.pth'

# CORS (for API access)
CORS_ALLOW_ALL_ORIGINS = True  # Only for development

# Logging
LOGGING = {
    'loggers': {
        'image_processor': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
        },
    },
}
```

## Development

### Adding New Processing Modes

1. Add the processing function to `image_processor/model_manager.py`
2. Update the `PROCESSING_MODES` in `image_processor/models.py`
3. Add the mode to `process_image()` function in `utils.py`
4. Update the frontend template with the new mode card

### Performance Monitoring

The application includes comprehensive logging:
- Real-ESRGAN model loading and GPU detection
- Processing times for each algorithm
- Memory usage tracking
- CUDA availability and utilization
- Error reporting with fallback status

### Extending the API

The application provides RESTful API endpoints for external integrations. All endpoints return JSON responses with appropriate HTTP status codes.

## Production Deployment

### Docker Deployment

```dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY . /app
WORKDIR /app

# Download model
RUN python -c "from image_processor.model_manager import enhancement_manager"

# Run application
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "image_editor_app.wsgi:application"]
```

### Production Checklist

1. Set `DEBUG = False` in settings
2. Configure proper database (PostgreSQL recommended)
3. Set up static file serving with WhiteNoise
4. Configure media file storage (AWS S3, etc.)
5. Add proper CORS settings
6. Set up HTTPS with SSL certificates
7. Configure logging to external services
8. Set up monitoring and alerting
9. Download and cache Real-ESRGAN model
10. Configure GPU drivers if using CUDA

## Performance Benchmarks

### Real-ESRGAN Performance

**With GPU (CUDA):**
- Processing time: 2-5 seconds per image
- Memory usage: ~2GB VRAM
- Quality: State-of-the-art super-resolution
- Upscaling: True 4x enhancement

**CPU Fallback:**
- Processing time: 30-60 seconds per image
- Memory usage: ~1GB RAM
- Quality: Same model, slower processing
- Upscaling: True 4x enhancement

### System Requirements

**Recommended (GPU):**
- NVIDIA GPU with 4GB+ VRAM
- 8GB+ RAM
- CUDA 11.8+

**Minimum (CPU):**
- 4GB+ RAM
- Modern CPU (4+ cores recommended)

## Troubleshooting

### Common Issues

1. **CUDA not detected**: Install appropriate CUDA drivers
2. **Model download fails**: Check internet connection and disk space
3. **Out of memory**: Reduce image size or use CPU processing
4. **Slow processing**: Enable GPU acceleration or reduce image resolution

### Performance Tips

1. Use GPU acceleration for best performance
2. Process images in batches for efficiency
3. Monitor memory usage with large images
4. Use appropriate tile sizes for GPU memory

## License

This project is open source and available under the MIT License.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Support

For issues and questions, please create an issue in the repository.

## Acknowledgments

- **Real-ESRGAN**: [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- **BasicSR**: [XPixelGroup/BasicSR](https://github.com/XPixelGroup/BasicSR)
- **PyTorch**: [pytorch.org](https://pytorch.org/)