# AI Image Editor with RealESRGAN

A professional Django web application for advanced image processing using **RealESRGAN** - Real-ESRGAN super-resolution and enhancement technology with **GPU acceleration** and **temporary file storage**.

## 🚀 Features

- **🤖 AI Super-Resolution**: State-of-the-art RealESRGAN for 1x-4x configurable upscaling with exceptional quality
- **🔧 Gamma Fix & CLAHE**: Advanced gamma correction with CLAHE and local contrast enhancement  
- **☀️ Shadow Fight**: Intelligent shadow detection and adaptive enhancement for balanced lighting
- **🎨 Grayscale**: Chroma-based grayscale conversion with adaptive thresholding

## ✨ Key Advantages

✅ **RealESRGAN Technology** - Industry-leading super-resolution AI model  
✅ **GPU Acceleration** - CUDA support for ultra-fast processing  
✅ **Temporary Storage** - No permanent file storage, automatic cleanup after 2 hours  
✅ **Production Ready** - Docker GPU support, Kubernetes deployment, comprehensive monitoring  
✅ **Memory Efficient** - Smart temporary file management saves storage space  
✅ **Secure by Design** - UUID-based download links, automatic expiration, no data retention  

## 🗂️ Storage Architecture

### Temporary File System
- **🔄 No Permanent Storage**: Images are processed temporarily and automatically cleaned up
- **⏰ 2-Hour Expiration**: All temporary files and download links expire after 2 hours
- **🔐 UUID Security**: Secure, non-guessable download links using UUIDs
- **💾 Space Efficient**: No long-term storage requirements for deployment
- **🗑️ Auto Cleanup**: Built-in management command for automated file cleanup

### Benefits for Deployment
- **☁️ Cloud-Friendly**: Perfect for stateless deployments and auto-scaling
- **💰 Cost Effective**: No persistent storage costs for user uploads
- **🛡️ Privacy Focused**: User images are never permanently stored
- **📏 Predictable Storage**: Known, limited temporary storage requirements

## 📸 Screenshots

The application features a beautiful, modern UI with:
- 🎯 Drag & drop file upload with visual feedback
- 🎨 Interactive mode selection cards with animations
- 🎚️ Scale selector (1x-4x) for AI Super-Resolution mode
- 📊 Real-time progress indicators with smooth animations
- 🔄 Side-by-side image comparison viewer
- 📥 Temporary download functionality with expiration notice
- 📱 Fully responsive design for all devices

## 🚀 GPU Deployment

### Quick GPU Deployment

For GPU-enabled production deployment:

```bash
# Clone the repository
git clone <repository-url>
cd image_editor

# Deploy with GPU support
docker-compose up --build -d
```

### GPU Requirements
- **NVIDIA GPU** with CUDA Compute Capability 6.0+
- **8GB+ GPU Memory** (16GB+ recommended for 4K images)
- **Docker with NVIDIA Container Runtime**
- **CUDA 11.8+ drivers**

See [DEPLOYMENT.md](DEPLOYMENT.md) for comprehensive GPU deployment guide.

## 🛠️ Setup Instructions

### Quick Setup (Development)

```bash
# Clone the repository
git clone <repository-url>
cd image_editor

# Run automated setup
chmod +x setup.sh
./setup.sh
```

### Manual Setup

#### Prerequisites

- **Python 3.10** (Required for optimal compatibility)
- **Conda** (Recommended for environment management)
- **CUDA-compatible GPU** (Optional, for accelerated processing)

#### Installation Steps

1. **Create and activate conda environment:**
   ```bash
   conda create -n imgapp python=3.10 -y
   conda activate imgapp
   ```

2. **Install PyTorch and dependencies:**
   ```bash
   conda install pytorch torchvision torchaudio -c pytorch
   conda install numpy opencv pillow scikit-image -c conda-forge
   ```

3. **Install RealESRGAN and related packages:**
   ```bash
   python -m pip install git+https://github.com/XPixelGroup/BasicSR@master --use-pep517
   python -m pip install -r requirements.txt
   ```

4. **Run database migrations:**
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

5. **Start the development server:**
   ```bash
   python manage.py runserver
   ```

## 🐳 Docker Deployment

### GPU-Enabled Production
```bash
# Build and run with GPU support
docker-compose up --build -d

# Check GPU access
docker-compose exec web python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Environment Variables
```env
USE_GPU=true
CUDA_VISIBLE_DEVICES=0
TEMP_FILE_CLEANUP_HOURS=2
SECRET_KEY=your-secret-key-here
DEBUG=False
ALLOWED_HOSTS=yourdomain.com,localhost
```

## 🎯 Usage

1. **Upload an Image**: Drag and drop an image file or click to browse (JPG, PNG, GIF - Max 10MB)
2. **Select Processing Mode**: Choose from AI Super-Resolution, Gamma Fix & CLAHE, Shadow Fight, or Grayscale
3. **Choose Scale** (AI mode only): Select 1x-4x upscaling factor
4. **Process**: Click the "Enhance with AI" button and watch the progress indicator
5. **View Results**: Compare original and processed images side-by-side
6. **Download**: Save the enhanced image (link expires in 2 hours)

## 🔧 Maintenance

### Automatic Cleanup
The system automatically manages temporary files:

```bash
# Manual cleanup
python manage.py cleanup_temp_files --hours=2

# Dry run to see what would be cleaned
python manage.py cleanup_temp_files --hours=2 --dry-run

# Set up automated cleanup (production)
# Add to crontab: 0 */2 * * * python manage.py cleanup_temp_files
```

### Monitoring
```bash
# Check temporary storage usage
du -sh temp/

# Monitor GPU usage
nvidia-smi

# Check application health
curl http://localhost:8000/
```

## 🌐 API Endpoints

- `POST /process/` - Process image with temporary storage (includes scale parameter)
- `GET /download/<uuid>/` - Download processed image by UUID (expires in 2 hours)
- `GET /modes/` - Get available processing modes and descriptions

## 🔬 Technical Details

### Temporary Storage Architecture

1. **Session-Based Processing**: 
   - UUID-based record identification
   - Temporary file path storage in database
   - Automatic expiration tracking
   - Secure download link generation

2. **File Management**: 
   - Organized temporary directory structure
   - Atomic file operations for reliability
   - Automatic cleanup on errors
   - Configurable retention periods

3. **Security Features**: 
   - Non-guessable UUID download links
   - Automatic expiration after 2 hours
   - No permanent user data storage
   - Secure file path handling

### RealESRGAN Integration

The application uses **Real-ESRGAN (Real-Enhanced Super-Resolution Generative Adversarial Networks)** for superior image enhancement:

1. **AI Super-Resolution**: 
   - RealESRGAN model with SRVGGNetCompact architecture
   - Configurable 1x-4x upscaling with exceptional detail preservation
   - GPU acceleration with CUDA support
   - Automatic fallback to CPU processing
   - Pre and post-processing optimization
   - Intelligent size limiting for memory management

2. **Advanced Processing Pipeline**: 
   - LAB color space processing for better quality
   - Bilateral filtering for edge preservation
   - Morphological operations for local contrast
   - CLAHE (Contrast Limited Adaptive Histogram Equalization)
   - Custom gamma correction with lookup tables

3. **Shadow Enhancement**: 
   - Intelligent shadow region detection using L channel analysis
   - Local adaptive enhancement with morphological refinement
   - Gaussian blur for smooth transitions
   - Multi-level brightness/contrast adjustment

4. **Advanced Grayscale**: 
   - Chroma-based conversion methods
   - Adaptive thresholding techniques
   - CLAHE for local contrast enhancement

### Performance & Architecture

- **Singleton Pattern**: Model manager loads once and caches resources
- **GPU Memory Management**: Automatic CUDA detection and optimization  
- **Smart Fallback**: Graceful degradation when RealESRGAN unavailable
- **Temporary File Optimization**: Fast I/O with automatic cleanup
- **Memory-Efficient**: Smart batching and memory cleanup
- **Scalable Design**: Stateless architecture for horizontal scaling

## 🚀 Production Deployment

### Docker Deployment

```