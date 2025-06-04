from django.contrib import admin
from django.utils.html import format_html
from .models import ProcessedImage

@admin.register(ProcessedImage)
class ProcessedImageAdmin(admin.ModelAdmin):
    list_display = ['id', 'processing_mode', 'created_at', 'image_preview', 'processed_preview']
    list_filter = ['processing_mode', 'created_at']
    search_fields = ['processing_mode']
    readonly_fields = ['created_at', 'image_preview', 'processed_preview']
    
    def image_preview(self, obj):
        if obj.original_image:
            return format_html(
                '<img src="{}" style="max-width: 100px; max-height: 100px;" />',
                obj.original_image.url
            )
        return "No image"
    image_preview.short_description = "Original Image"
    
    def processed_preview(self, obj):
        if obj.processed_image:
            return format_html(
                '<img src="{}" style="max-width: 100px; max-height: 100px;" />',
                obj.processed_image.url
            )
        return "No processed image"
    processed_preview.short_description = "Processed Image"
    
    fieldsets = (
        ('Image Information', {
            'fields': ('processing_mode', 'created_at')
        }),
        ('Images', {
            'fields': ('original_image', 'image_preview', 'processed_image', 'processed_preview')
        }),
    )
