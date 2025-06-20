"""
Management command to clean up expired temporary images and files
"""

import os
import logging
from datetime import timedelta
from django.core.management.base import BaseCommand
from django.utils import timezone
from django.conf import settings
from image_processor.models import TempProcessedImage

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Clean up expired temporary image files and database records'

    def add_arguments(self, parser):
        parser.add_argument(
            '--hours',
            type=int,
            default=2,
            help='Delete files older than this many hours (default: 2)'
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be deleted without actually deleting'
        )
        parser.add_argument(
            '--aggressive',
            action='store_true',
            help='Delete all temp files regardless of database records'
        )

    def handle(self, *args, **options):
        hours = options['hours']
        dry_run = options['dry_run']
        aggressive = options['aggressive']
        
        cutoff_time = timezone.now() - timedelta(hours=hours)
        
        # Find expired records
        expired_records = TempProcessedImage.objects.filter(created_at__lt=cutoff_time)
        
        deleted_files = 0
        deleted_records = 0
        errors = 0
        
        self.stdout.write(
            self.style.SUCCESS(f'Finding temporary files older than {hours} hours...')
        )
        
        for record in expired_records:
            try:
                # Delete temporary files
                files_to_delete = [
                    record.temp_original_path,
                    record.temp_processed_path
                ]
                
                for file_path in files_to_delete:
                    if file_path and os.path.exists(file_path):
                        if dry_run:
                            self.stdout.write(f'Would delete: {file_path}')
                        else:
                            os.remove(file_path)
                            deleted_files += 1
                            logger.info(f'Deleted temporary file: {file_path}')
                
                # Delete database record
                if dry_run:
                    self.stdout.write(f'Would delete record: {record.uuid}')
                else:
                    record.delete()
                    deleted_records += 1
                    
            except Exception as e:
                errors += 1
                logger.error(f'Error cleaning up record {record.uuid}: {e}')
                self.stdout.write(
                    self.style.ERROR(f'Error cleaning up record {record.uuid}: {e}')
                )
        
        # Also clean up orphaned temporary files in temp directory
        temp_dir = getattr(settings, 'TEMP_FILES_DIR', '/tmp/image_processor')
        if os.path.exists(temp_dir):
            try:
                for filename in os.listdir(temp_dir):
                    file_path = os.path.join(temp_dir, filename)
                    if os.path.isfile(file_path):
                        file_age = timezone.now().timestamp() - os.path.getmtime(file_path)
                        
                        # For aggressive cleanup or old files
                        should_delete = aggressive or file_age > (hours * 3600)
                        
                        if should_delete:
                            if dry_run:
                                self.stdout.write(f'Would delete orphaned file: {file_path}')
                            else:
                                os.remove(file_path)
                                deleted_files += 1
                                logger.info(f'Deleted orphaned temporary file: {file_path}')
            except Exception as e:
                logger.error(f'Error cleaning up temp directory: {e}')
        
        # Clean up broken/incomplete records (no files exist)
        if not dry_run:
            broken_records = TempProcessedImage.objects.all()
            for record in broken_records:
                files_exist = False
                for path in [record.temp_original_path, record.temp_processed_path]:
                    if path and os.path.exists(path):
                        files_exist = True
                        break
                
                if not files_exist:
                    try:
                        record.delete()
                        deleted_records += 1
                        logger.info(f'Deleted broken record {record.uuid} with no files')
                    except Exception as e:
                        errors += 1
                        logger.error(f'Error deleting broken record {record.uuid}: {e}')
        
        # Summary
        if dry_run:
            self.stdout.write(
                self.style.WARNING(f'DRY RUN: Would delete {deleted_files} files and {deleted_records} records')
            )
        else:
            self.stdout.write(
                self.style.SUCCESS(
                    f'Cleanup completed: {deleted_files} files deleted, '
                    f'{deleted_records} records deleted, {errors} errors'
                )
            ) 