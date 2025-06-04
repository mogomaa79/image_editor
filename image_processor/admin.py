from django.contrib import admin
from django.utils.html import format_html
from django.utils import timezone
from .models import TempProcessedImage

@admin.register(TempProcessedImage)
class TempProcessedImageAdmin(admin.ModelAdmin):
    list_display = ['uuid_short', 'processing_mode', 'scale', 'original_filename', 'file_size_mb', 'dimensions', 'created_at', 'expires_status']
    list_filter = ['processing_mode', 'scale', 'created_at']
    search_fields = ['uuid', 'original_filename']
    readonly_fields = ['uuid', 'created_at', 'expires_status', 'file_paths']
    ordering = ['-created_at']
    
    def uuid_short(self, obj):
        return str(obj.uuid)[:8] + '...'
    uuid_short.short_description = 'UUID'
    
    def file_size_mb(self, obj):
        if obj.file_size:
            return f"{obj.file_size / (1024 * 1024):.2f} MB"
        return "Unknown"
    file_size_mb.short_description = 'File Size'
    
    def dimensions(self, obj):
        if obj.image_width and obj.image_height:
            return f"{obj.image_width} x {obj.image_height}"
        return "Unknown"
    dimensions.short_description = 'Dimensions'
    
    def expires_status(self, obj):
        if obj.is_expired():
            return format_html('<span style="color: red;">Expired</span>')
        else:
            # Calculate remaining time
            from datetime import timedelta
            expires_at = obj.created_at + timedelta(hours=2)
            remaining = expires_at - timezone.now()
            hours = remaining.total_seconds() / 3600
            if hours > 1:
                return format_html(f'<span style="color: green;">Expires in {hours:.1f}h</span>')
            else:
                minutes = remaining.total_seconds() / 60
                return format_html(f'<span style="color: orange;">Expires in {minutes:.0f}m</span>')
    expires_status.short_description = 'Status'
    
    def file_paths(self, obj):
        paths = []
        if obj.temp_original_path:
            paths.append(f"Original: {obj.temp_original_path}")
        if obj.temp_processed_path:
            paths.append(f"Processed: {obj.temp_processed_path}")
        return '\n'.join(paths) if paths else "No files"
    file_paths.short_description = 'File Paths'
    
    actions = ['cleanup_expired']
    
    def cleanup_expired(self, request, queryset):
        """Admin action to clean up expired records"""
        import os
        deleted_count = 0
        
        for record in queryset:
            if record.is_expired():
                # Delete files
                for path in [record.temp_original_path, record.temp_processed_path]:
                    if path and os.path.exists(path):
                        try:
                            os.remove(path)
                        except Exception:
                            pass
                
                # Delete record
                record.delete()
                deleted_count += 1
        
        self.message_user(request, f'Cleaned up {deleted_count} expired records.')
    cleanup_expired.short_description = 'Clean up expired records'
