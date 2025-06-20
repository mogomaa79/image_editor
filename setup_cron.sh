#!/bin/bash

# Setup script for automatic temp file cleanup

echo "Setting up automatic temp file cleanup..."

# Get the current directory
APP_DIR=$(pwd)

# Create cron job for cleanup every hour
CRON_JOB="0 * * * * cd $APP_DIR && python3 manage.py cleanup_temp_files --hours 2 >> /var/log/image_processor_cleanup.log 2>&1"

# Add cron job
(crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -

echo "Cron job added: $CRON_JOB"

# Create cleanup log file with proper permissions
sudo touch /var/log/image_processor_cleanup.log
sudo chmod 666 /var/log/image_processor_cleanup.log

echo "Automatic cleanup is now set up to run every hour."
echo "Log file: /var/log/image_processor_cleanup.log"

# Show current cron jobs
echo ""
echo "Current cron jobs:"
crontab -l 