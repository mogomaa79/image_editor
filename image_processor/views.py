from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.conf import settings
import os
import tempfile
import uuid
from PIL import Image
from .models import TempProcessedImage
from .utils import process_image, image_to_base64
import logging

logger = logging.getLogger(__name__)

def home(request):
    """
    Main page with image upload and processing interface
    """
    return render(request, 'image_processor/home.html')

@csrf_exempt
@require_http_methods(["POST"])
def process_image_api(request):
    """
    API endpoint to process uploaded images with temporary storage
    """
    try:
        if 'image' not in request.FILES:
            return JsonResponse({'error': 'No image file provided'}, status=400)
        
        if 'mode' not in request.POST:
            return JsonResponse({'error': 'No processing mode specified'}, status=400)
        
        image_file = request.FILES['image']
        mode = request.POST['mode']
        
        # Get scale parameter (default to 2 if not provided)
        scale = int(request.POST.get('scale', 2))
        scale = max(1, min(4, scale))  # Clamp between 1 and 4
        
        # Validate mode
        valid_modes = ['ai_enhancer', 'gamma_clahe', 'shadow_fight', 'grayscale']
        if mode not in valid_modes:
            return JsonResponse({'error': 'Invalid processing mode'}, status=400)
        
        # Create temporary directory
        temp_dir = getattr(settings, 'TEMP_FILES_DIR', tempfile.gettempdir())
        os.makedirs(temp_dir, exist_ok=True)
        
        # Generate unique filename
        unique_id = str(uuid.uuid4())
        original_ext = os.path.splitext(image_file.name)[1].lower()
        
        # Save original file temporarily
        temp_original_path = os.path.join(temp_dir, f"original_{unique_id}{original_ext}")
        with open(temp_original_path, 'wb') as f:
            for chunk in image_file.chunks():
                f.write(chunk)
        
        # Get image metadata
        with Image.open(temp_original_path) as img:
            image_width, image_height = img.size
        
        # Process the image with scale parameter
        processed_image = process_image(image_file, mode, scale=scale)
        
        if processed_image is None:
            # Clean up temporary file
            if os.path.exists(temp_original_path):
                os.remove(temp_original_path)
            
            # Provide specific error message based on mode and scale
            if mode == 'ai_enhancer' and scale >= 3:
                error_msg = f'Image too large for {scale}x AI enhancement. Try using 2x scale or reduce image size to under 1200x1200 pixels.'
            elif mode == 'ai_enhancer':
                error_msg = f'AI enhancement failed. Try reducing the scale factor or using a smaller image.'
            else:
                error_msg = 'Failed to process image. Try using a different image or processing mode.'
                
            return JsonResponse({'error': error_msg}, status=400)
        
        # Save processed image temporarily
        processed_ext = '.jpg'  # Always save as JPEG
        temp_processed_path = os.path.join(temp_dir, f"processed_{unique_id}_{mode}_{scale}x{processed_ext}")
        
        format_type = 'JPEG'  # Always use JPEG format
        # Convert to RGB if it's grayscale for JPEG compatibility
        if processed_image.mode == 'L':
            processed_image = processed_image.convert('RGB')
        
        processed_image.save(temp_processed_path, format=format_type, quality=95)
        
        # Convert processed image to base64 for preview
        processed_base64 = image_to_base64(processed_image)
        
        # Create database record with temporary paths
        processed_record = TempProcessedImage.objects.create(
            processing_mode=mode,
            scale=scale,
            temp_original_path=temp_original_path,
            temp_processed_path=temp_processed_path,
            original_filename=image_file.name,
            file_size=image_file.size,
            image_width=image_width,
            image_height=image_height
        )
        
        logger.info(f"Created temporary processing record {processed_record.uuid} for {mode} mode")
        
        return JsonResponse({
            'success': True,
            'processed_image': processed_base64,
            'record_id': str(processed_record.uuid),
            'scale': scale,
            'temp_download_url': f'/download/{processed_record.uuid}/',
            'expires_in_hours': 2
        })
        
    except Exception as e:
        logger.error(f"Error in process_image_api: {e}")
        return JsonResponse({'error': f'Server error: {str(e)}'}, status=500)

@require_http_methods(["GET"])
def download_image(request, record_uuid):
    """
    Download processed image using UUID
    """
    try:
        # Parse UUID - ensure we're working with string representation
        try:
            if isinstance(record_uuid, str):
                uuid_obj = uuid.UUID(record_uuid)
            else:
                uuid_obj = record_uuid
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid UUID format: {record_uuid}, error: {e}")
            return JsonResponse({'error': 'Invalid download link'}, status=400)
        
        record = TempProcessedImage.objects.get(uuid=uuid_obj)
        
        # Check if expired
        if record.is_expired():
            logger.info(f"Download link {uuid_obj} has expired")
            return JsonResponse({'error': 'Download link has expired'}, status=410)
        
        if not record.temp_processed_path or not os.path.exists(record.temp_processed_path):
            logger.error(f"Processed image file not found: {record.temp_processed_path}")
            return JsonResponse({'error': 'Processed image not found'}, status=404)
        
        # Determine content type
        file_ext = os.path.splitext(record.temp_processed_path)[1].lower()
        content_type = 'image/jpeg'  # Always JPEG now
        
        # Open the file and return as response
        with open(record.temp_processed_path, 'rb') as f:
            response = HttpResponse(f.read(), content_type=content_type)
            
            # Generate safe filename
            try:
                # Ensure original_filename is a string and clean it
                original_name = str(record.original_filename) if record.original_filename else "image"
                # Remove any problematic characters
                safe_name = "".join(c for c in original_name if c.isalnum() or c in "._-")
                if not safe_name:
                    safe_name = "image"
                
                # Remove extension from original name if present
                safe_name = os.path.splitext(safe_name)[0]
                
                # Generate mode name - include scale only for AI mode
                if record.processing_mode == 'ai_enhancer':
                    mode_name = f"ai_enhancer_{record.scale}x"
                else:
                    mode_name = record.processing_mode
                
                # Generate final filename: {original_name}_{mode}.jpg
                filename = f"{safe_name}_{mode_name}.jpg"
                
            except Exception as e:
                logger.error(f"Error generating filename: {e}")
                # Fallback filename
                if record.processing_mode == 'ai_enhancer':
                    filename = f"enhanced_ai_enhancer_{record.scale}x.jpg"
                else:
                    filename = f"enhanced_{record.processing_mode}.jpg"
            
            response['Content-Disposition'] = f'attachment; filename="{filename}"'
            logger.info(f"Successfully serving download for UUID {uuid_obj}")
            return response
            
    except TempProcessedImage.DoesNotExist:
        logger.error(f"Download record not found for UUID: {record_uuid}")
        return JsonResponse({'error': 'Download link not found or expired'}, status=404)
    except Exception as e:
        logger.error(f"Error in download_image: {e}")
        return JsonResponse({'error': f'Server error: {str(e)}'}, status=500)

@require_http_methods(["GET"])
def get_processing_modes(request):
    """
    Get available processing modes
    """
    modes = [
        {
            'key': 'ai_enhancer',
            'name': 'AI Super-Resolution',
            'description': 'RealESRGAN AI model for 1x to 4x upscaling with GPU acceleration'
        },
        {
            'key': 'gamma_clahe',
            'name': 'Gamma Fix & CLAHE',
            'description': 'LAB color space processing with adaptive histogram equalization'
        },
        {
            'key': 'shadow_fight',
            'name': 'Shadow Fight',
            'description': 'Intelligent shadow region detection with morphological operations'
        },
        {
            'key': 'grayscale',
            'name': 'Grayscale',
            'description': 'Chroma-based grayscale conversion with adaptive thresholding'
        }
    ]
    
    return JsonResponse({'modes': modes})
