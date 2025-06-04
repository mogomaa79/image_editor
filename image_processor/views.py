from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
import json
import io
from PIL import Image
from .models import ProcessedImage
from .utils import process_image, image_to_base64

def home(request):
    """
    Main page with image upload and processing interface
    """
    return render(request, 'image_processor/home.html')

@csrf_exempt
@require_http_methods(["POST"])
def process_image_api(request):
    """
    API endpoint to process uploaded images
    """
    try:
        if 'image' not in request.FILES:
            return JsonResponse({'error': 'No image file provided'}, status=400)
        
        if 'mode' not in request.POST:
            return JsonResponse({'error': 'No processing mode specified'}, status=400)
        
        image_file = request.FILES['image']
        mode = request.POST['mode']
        
        # Validate mode
        valid_modes = ['ai_enhancer', 'gamma_clahe', 'shadow_fight', 'grayscale']
        if mode not in valid_modes:
            return JsonResponse({'error': 'Invalid processing mode'}, status=400)
        
        # Process the image
        processed_image = process_image(image_file, mode)
        
        if processed_image is None:
            return JsonResponse({'error': 'Failed to process image'}, status=500)
        
        # Convert processed image to base64 for preview
        processed_base64 = image_to_base64(processed_image)
        
        # Save original image to model
        processed_record = ProcessedImage.objects.create(
            original_image=image_file,
            processing_mode=mode
        )
        
        # Save processed image
        buffer = io.BytesIO()
        format_type = 'PNG' if mode == 'grayscale' else 'JPEG'
        processed_image.save(buffer, format=format_type)
        buffer.seek(0)
        
        # Create filename
        filename = f"processed_{processed_record.id}_{mode}.{'png' if mode == 'grayscale' else 'jpg'}"
        processed_record.processed_image.save(
            filename,
            ContentFile(buffer.read()),
            save=True
        )
        
        return JsonResponse({
            'success': True,
            'processed_image': processed_base64,
            'record_id': processed_record.id,
            'download_url': processed_record.processed_image.url if processed_record.processed_image else None
        })
        
    except Exception as e:
        return JsonResponse({'error': f'Server error: {str(e)}'}, status=500)

@require_http_methods(["GET"])
def download_image(request, record_id):
    """
    Download processed image
    """
    try:
        record = ProcessedImage.objects.get(id=record_id)
        
        if not record.processed_image:
            return JsonResponse({'error': 'Processed image not found'}, status=404)
        
        # Open the file and return as response
        with default_storage.open(record.processed_image.name, 'rb') as f:
            response = HttpResponse(f.read(), content_type='image/jpeg')
            response['Content-Disposition'] = f'attachment; filename="processed_{record.processing_mode}_{record.id}.jpg"'
            return response
            
    except ProcessedImage.DoesNotExist:
        return JsonResponse({'error': 'Record not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': f'Server error: {str(e)}'}, status=500)

@require_http_methods(["GET"])
def get_processing_modes(request):
    """
    Get available processing modes
    """
    modes = [
        {
            'key': 'ai_enhancer',
            'name': 'AI Enhancer',
            'description': 'Enhance image quality using AI upscaling techniques'
        },
        {
            'key': 'gamma_clahe',
            'name': 'Gamma CLAHE',
            'description': 'Apply gamma correction with contrast limited adaptive histogram equalization'
        },
        {
            'key': 'shadow_fight',
            'name': 'Shadow Fight',
            'description': 'Reduce shadows and improve brightness/contrast'
        },
        {
            'key': 'grayscale',
            'name': 'Grayscale',
            'description': 'Convert image to grayscale'
        }
    ]
    
    return JsonResponse({'modes': modes})
