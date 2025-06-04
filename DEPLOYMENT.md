# GPU Deployment Guide for AI Image Editor

## Overview
This guide covers deploying the AI Image Editor with RealESRGAN on GPU-enabled infrastructure.

## Prerequisites

### Hardware Requirements
- NVIDIA GPU with CUDA Compute Capability 6.0+
- Minimum 8GB GPU memory (16GB+ recommended for 4K images)
- 16GB+ system RAM
- SSD storage for better temporary file I/O

### Software Requirements
- Docker with NVIDIA Container Runtime
- NVIDIA Docker 2.0+
- CUDA 11.8+ drivers
- Docker Compose 3.8+

## Quick Start

### 1. Clone and Setup
```bash
git clone <repository-url>
cd image_editor
```

### 2. Environment Configuration
Create `.env` file:
```env
SECRET_KEY=your-secret-key-here-make-it-long-and-random
DEBUG=False
ALLOWED_HOSTS=yourdomain.com,localhost,127.0.0.1
USE_GPU=true
CUDA_VISIBLE_DEVICES=0
TEMP_FILE_CLEANUP_HOURS=2
```

### 3. GPU Docker Deployment
```bash
# Build with GPU support
docker-compose up --build

# Or run in production mode
docker-compose -f docker-compose.yml up -d
```

### 4. Database Migration
```bash
# Run migrations
docker-compose exec web python3 manage.py makemigrations
docker-compose exec web python3 manage.py migrate

# Create superuser (optional)
docker-compose exec web python3 manage.py createsuperuser
```

## Advanced Deployment

### Cloud Platforms

#### AWS EC2 with GPU
```bash
# Launch GPU instance (p3.2xlarge or p4d.2xlarge)
# Instance type: Deep Learning AMI with CUDA

# Install Docker and NVIDIA Docker
sudo apt-get update
sudo apt-get install docker.io docker-compose
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install nvidia-docker2
sudo systemctl restart docker

# Deploy application
git clone <repository-url>
cd image_editor
docker-compose up -d
```

#### Google Cloud Platform
```bash
# Create GPU-enabled Compute Engine instance
gcloud compute instances create ai-image-editor \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=50GB \
    --maintenance-policy=TERMINATE

# SSH and deploy
gcloud compute ssh ai-image-editor
# ... follow setup steps
```

### Kubernetes Deployment
```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-image-editor
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ai-image-editor
  template:
    metadata:
      labels:
        app: ai-image-editor
    spec:
      containers:
      - name: ai-image-editor
        image: ai-image-editor:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
          requests:
            nvidia.com/gpu: 1
        env:
        - name: USE_GPU
          value: "true"
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
```

## Monitoring and Maintenance

### Health Checks
```bash
# Check GPU availability
docker-compose exec web python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Monitor GPU usage
nvidia-smi

# Check application health
curl http://localhost:8000/
```

### Automatic Cleanup
The system includes automatic cleanup of temporary files:

```bash
# Manual cleanup
docker-compose exec web python3 manage.py cleanup_temp_files --hours=2

# Set up cron job for regular cleanup
# Add to crontab:
0 */2 * * * docker-compose exec web python3 manage.py cleanup_temp_files --hours=2
```

### Scaling Considerations

#### Horizontal Scaling
- Use load balancer (nginx, HAProxy)
- Shared temporary storage (NFS, AWS EFS)
- Redis for session management

#### Vertical Scaling
- Increase GPU memory for larger images
- Add more CPU cores for concurrent processing
- Use NVMe SSD for temporary files

## Security

### Production Security Checklist
- [ ] Set strong SECRET_KEY
- [ ] Enable HTTPS (SSL_REDIRECT=true)
- [ ] Configure firewall (only port 443/80)
- [ ] Use environment variables for secrets
- [ ] Set up proper backup strategy
- [ ] Monitor for security updates

### File Security
- Temporary files are automatically cleaned up
- No permanent storage of user images
- UUID-based download links expire in 2 hours

## Troubleshooting

### Common Issues

#### GPU Not Detected
```bash
# Check NVIDIA drivers
nvidia-smi

# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi

# Check container GPU access
docker-compose exec web python3 -c "import torch; print(torch.cuda.is_available())"
```

#### Out of GPU Memory
```bash
# Monitor GPU memory
watch -n 1 nvidia-smi

# Reduce scale or image size
# Check CUDA_VISIBLE_DEVICES setting
```

#### Performance Issues
```bash
# Check disk I/O
iostat -x 1

# Monitor CPU usage
htop

# Check memory usage
free -h
```

## Performance Optimization

### GPU Optimization
- Use mixed precision (FP16) for faster processing
- Batch process multiple images when possible
- Optimize CUDA memory allocation

### System Optimization
- Use tmpfs for temporary files
- Optimize Docker layer caching
- Use multi-stage builds for smaller images

## Support

For deployment issues:
1. Check logs: `docker-compose logs web`
2. Verify GPU access: `nvidia-smi`
3. Check temporary file cleanup: `docker-compose exec web python3 manage.py cleanup_temp_files --dry-run`

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `DEBUG` | `False` | Django debug mode |
| `USE_GPU` | `true` | Enable GPU acceleration |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device ID |
| `TEMP_FILE_CLEANUP_HOURS` | `2` | Hours before temp file cleanup |
| `SECRET_KEY` | - | Django secret key (required) |
| `ALLOWED_HOSTS` | `localhost` | Allowed hostnames |
| `DATABASE_URL` | - | PostgreSQL connection string (optional) |
| `REDIS_URL` | - | Redis connection string (optional) | 