"""
Production settings for image_editor_app project.
"""

from .settings import *
import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = False

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.environ.get('SECRET_KEY', SECRET_KEY)

# Production hosts
ALLOWED_HOSTS = os.environ.get('ALLOWED_HOSTS', 'localhost,127.0.0.1').split(',')

# Database
# Uncomment to use PostgreSQL in production
# DATABASES = {
#     'default': {
#         'ENGINE': 'django.db.backends.postgresql',
#         'NAME': os.environ.get('DB_NAME', 'image_editor'),
#         'USER': os.environ.get('DB_USER', 'imageuser'),
#         'PASSWORD': os.environ.get('DB_PASSWORD', 'imagepass123'),
#         'HOST': os.environ.get('DB_HOST', 'db'),
#         'PORT': os.environ.get('DB_PORT', '5432'),
#     }
# }

# Static files (CSS, JavaScript, Images)
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')

# WhiteNoise configuration for static files
MIDDLEWARE.insert(1, 'whitenoise.middleware.WhiteNoiseMiddleware')
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'

# Security settings
SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_TYPE_NOSNIFF = True
X_FRAME_OPTIONS = 'DENY'

# HTTPS settings (uncomment for production with HTTPS)
# SECURE_SSL_REDIRECT = True
# SESSION_COOKIE_SECURE = True
# CSRF_COOKIE_SECURE = True
# SECURE_HSTS_SECONDS = 31536000
# SECURE_HSTS_INCLUDE_SUBDOMAINS = True
# SECURE_HSTS_PRELOAD = True

# CORS settings for production
CORS_ALLOW_ALL_ORIGINS = False
CORS_ALLOWED_ORIGINS = os.environ.get('CORS_ALLOWED_ORIGINS', '').split(',')

# Logging
LOGGING['handlers']['file']['filename'] = '/app/logs/image_processor.log'

# Create logs directory
import os
os.makedirs('/app/logs', exist_ok=True)

# Cache (uncomment if using Redis)
# CACHES = {
#     'default': {
#         'BACKEND': 'django.core.cache.backends.redis.RedisCache',
#         'LOCATION': 'redis://redis:6379/1',
#     }
# }

# File upload settings for production
FILE_UPLOAD_MAX_MEMORY_SIZE = 50 * 1024 * 1024  # 50MB
DATA_UPLOAD_MAX_MEMORY_SIZE = 50 * 1024 * 1024  # 50MB

# GPU settings
USE_GPU = os.environ.get('USE_GPU', 'True').lower() == 'true'

# Email settings (configure for production)
# EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
# EMAIL_HOST = os.environ.get('EMAIL_HOST', '')
# EMAIL_PORT = int(os.environ.get('EMAIL_PORT', 587))
# EMAIL_USE_TLS = True
# EMAIL_HOST_USER = os.environ.get('EMAIL_HOST_USER', '')
# EMAIL_HOST_PASSWORD = os.environ.get('EMAIL_HOST_PASSWORD', '')
# DEFAULT_FROM_EMAIL = os.environ.get('DEFAULT_FROM_EMAIL', 'noreply@imageapp.com')

# Temporary files configuration
TEMP_FILES_DIR = os.environ.get('TEMP_FILES_DIR', BASE_DIR / 'temp')
TEMP_FILE_CLEANUP_HOURS = int(os.environ.get('TEMP_FILE_CLEANUP_HOURS', '2'))

# GPU and AI Model settings
MODEL_CACHE_DIR = os.environ.get('MODEL_CACHE_DIR', BASE_DIR / 'models')
REALESRGAN_MODEL_PATH = os.environ.get('REALESRGAN_MODEL_PATH', BASE_DIR / 'realesr-general-x4v3.pth')

# CUDA settings
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '0')

# Cache (Redis in production, in-memory for development)
if os.environ.get('REDIS_URL'):
    CACHES = {
        'default': {
            'BACKEND': 'django_redis.cache.RedisCache',
            'LOCATION': os.environ.get('REDIS_URL'),
            'OPTIONS': {
                'CLIENT_CLASS': 'django_redis.client.DefaultClient',
            }
        }
    }
else:
    CACHES = {
        'default': {
            'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',
            'LOCATION': 'ai-image-editor-cache',
            'OPTIONS': {
                'MAX_ENTRIES': 1000,
            }
        }
    }

# Session settings
SESSION_COOKIE_SECURE = SECURE_SSL_REDIRECT
CSRF_COOKIE_SECURE = SECURE_SSL_REDIRECT
SESSION_COOKIE_HTTPONLY = True
CSRF_COOKIE_HTTPONLY = True

# Performance settings
DATABASES['default']['CONN_MAX_AGE'] = 60

# Create necessary directories
os.makedirs(TEMP_FILES_DIR, exist_ok=True)
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
os.makedirs(BASE_DIR / 'logs', exist_ok=True) 