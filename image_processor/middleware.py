"""
Middleware for automatic cleanup of temporary files
"""

import os
import logging
from datetime import timedelta
from django.utils import timezone
from django.conf import settings
from .models import TempProcessedImage

logger = logging.getLogger(__name__)

class TempFileCleanupMiddleware:
    """
    Middleware to periodically clean up expired temporary files
    """
    
    def __init__(self, get_response):
        self.get_response = get_response
        self.last_cleanup = timezone.now()
        self.cleanup_interval = timedelta(minutes=15)  # Clean up every 15 minutes
    
    def __call__(self, request):
        # Check if we need to run cleanup
        now = timezone.now()
        if now - self.last_cleanup > self.cleanup_interval:
            self.cleanup_expired_files()
            self.last_cleanup = now
        
        response = self.get_response(request)
        return response
    
    def cleanup_expired_files(self):
        """
        Clean up expired temporary files and records
        """
        try:
            # Find expired records (older than 2 hours)
            cutoff_time = timezone.now() - timedelta(hours=2)
            expired_records = TempProcessedImage.objects.filter(created_at__lt=cutoff_time)
            
            deleted_files = 0
            deleted_records = 0
            
            for record in expired_records:
                try:
                    # Delete temporary files
                    files_to_delete = [
                        record.temp_original_path,
                        record.temp_processed_path
                    ]
                    
                    for file_path in files_to_delete:
                        if file_path and os.path.exists(file_path):
                            os.remove(file_path)
                            deleted_files += 1
                    
                    # Delete database record
                    record.delete()
                    deleted_records += 1
                    
                except Exception as e:
                    logger.error(f'Error cleaning up record {record.uuid}: {e}')
            
            # Clean up orphaned files in temp directory
            temp_dir = getattr(settings, 'TEMP_FILES_DIR', '/tmp/image_processor')
            if os.path.exists(temp_dir):
                try:
                    for filename in os.listdir(temp_dir):
                        file_path = os.path.join(temp_dir, filename)
                        if os.path.isfile(file_path):
                            file_age = timezone.now().timestamp() - os.path.getmtime(file_path)
                            if file_age > (2 * 3600):  # 2 hours in seconds
                                os.remove(file_path)
                                deleted_files += 1
                except Exception as e:
                    logger.error(f'Error cleaning up temp directory: {e}')
            
            if deleted_files > 0 or deleted_records > 0:
                logger.info(f'Automatic cleanup: {deleted_files} files and {deleted_records} records deleted')
                
        except Exception as e:
            logger.error(f'Error in automatic cleanup: {e}')

class ProcessingTimeoutMiddleware:
    """
    Middleware to handle processing timeouts and cleanup stuck processes
    """
    
    def __init__(self, get_response):
        self.get_response = get_response
    
    def __call__(self, request):
        response = self.get_response(request)
        return response
    
    def process_exception(self, request, exception):
        """
        Handle processing exceptions and clean up if needed
        """
        if request.path.startswith('/process/'):
            logger.error(f'Processing exception in {request.path}: {exception}')
            # Could add additional cleanup logic here
        return None 