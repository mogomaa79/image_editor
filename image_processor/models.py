from django.db import models
from django.utils import timezone
import uuid

class TempProcessedImage(models.Model):
    """
    Temporary image processing record - cleaned up automatically
    """
    PROCESSING_MODES = [
        ('ai_enhancer', 'AI Super-Resolution'),
        ('gamma_clahe', 'Gamma Fix & CLAHE'),
        ('shadow_fight', 'Shadow Fight'),
        ('grayscale', 'Grayscale'),
    ]
    
    # Use UUID for secure temporary access
    uuid = models.UUIDField(default=uuid.uuid4, unique=True, editable=False)
    processing_mode = models.CharField(max_length=20, choices=PROCESSING_MODES)
    scale = models.IntegerField(default=2)
    created_at = models.DateTimeField(default=timezone.now)
    
    # Store temporary file paths instead of Django FileField
    temp_original_path = models.CharField(max_length=500, null=True, blank=True)
    temp_processed_path = models.CharField(max_length=500, null=True, blank=True)
    
    # Metadata
    original_filename = models.CharField(max_length=255, null=True, blank=True)
    file_size = models.IntegerField(null=True, blank=True)
    image_width = models.IntegerField(null=True, blank=True)
    image_height = models.IntegerField(null=True, blank=True)
    
    def __str__(self):
        return f"{self.processing_mode} ({self.scale}x) - {self.created_at.strftime('%Y-%m-%d %H:%M')}"
    
    def is_expired(self):
        """Check if record is older than 2 hours"""
        from datetime import timedelta
        from django.utils import timezone
        return timezone.now() - self.created_at > timedelta(hours=2)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['uuid']),
            models.Index(fields=['created_at']),
        ]
