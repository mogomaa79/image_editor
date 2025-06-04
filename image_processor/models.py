from django.db import models
from django.utils import timezone

class ProcessedImage(models.Model):
    PROCESSING_MODES = [
        ('ai_enhancer', 'AI Enhancer'),
        ('gamma_clahe', 'Gamma CLAHE'),
        ('shadow_fight', 'Shadow Fight'),
        ('grayscale', 'Grayscale'),
    ]
    
    original_image = models.ImageField(upload_to='original/')
    processed_image = models.ImageField(upload_to='processed/', null=True, blank=True)
    processing_mode = models.CharField(max_length=20, choices=PROCESSING_MODES)
    created_at = models.DateTimeField(default=timezone.now)
    
    def __str__(self):
        return f"{self.processing_mode} - {self.created_at.strftime('%Y-%m-%d %H:%M')}"
    
    class Meta:
        ordering = ['-created_at']
