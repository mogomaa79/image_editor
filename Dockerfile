# Multi-stage build for GPU-enabled AI Image Editor
FROM nvidia/cuda:11.8-devel-ubuntu22.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Create directories
RUN mkdir -p /app/temp /app/models /app/static /app/media

# Copy requirements first for better Docker layer caching
COPY requirements.txt /app/
COPY requirements-dev.txt /app/

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app/

# Download RealESRGAN model if not present
RUN if [ ! -f "/app/realesr-general-x4v3.pth" ]; then \
    wget -O /app/realesr-general-x4v3.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth; \
    fi

# Set permissions
RUN chmod +x /app/setup.sh || true

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Run Django migrations and start server
CMD python3 manage.py makemigrations && \
    python3 manage.py migrate && \
    python3 manage.py collectstatic --noinput && \
    python3 manage.py cleanup_temp_files --hours=2 && \
    gunicorn --bind 0.0.0.0:8000 --workers 2 --threads 4 --timeout 300 image_editor_app.wsgi:application 