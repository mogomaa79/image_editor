import cv2
import numpy as np
from PIL import Image
import io
import base64
import logging
from .model_manager import enhancement_manager

logger = logging.getLogger(__name__)

def process_image(image_file, mode, scale=2):
    """
    Process image based on selected mode using the model manager
    """
    try:
        logger.info(f"Processing image with mode: {mode}, scale: {scale}")
        
        # Validate scale parameter
        scale = max(1, min(4, scale))
        
        # Open and convert image
        image = Image.open(image_file)
        image_array = np.array(image)
        
        # Process based on mode using model manager directly
        if mode == 'ai_enhancer':
            # Convert to BGR for OpenCV processing
            if len(image_array.shape) == 3:
                img_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = image_array
            
            # Use model manager for AI enhancement
            result = enhancement_manager.ai_super_resolution(img_bgr, scale=scale)
            
        elif mode == 'gamma_clahe':
            result = enhancement_manager.advanced_gamma_clahe(image_array, gamma=1.3, clip_limit=3.0)
            
        elif mode == 'shadow_fight':
            result = enhancement_manager.advanced_shadow_fight(image_array, alpha=1.4, beta=40)
            
        elif mode == 'grayscale':
            result = enhancement_manager.advanced_grayscale(image_array)
            
        else:
            logger.warning(f"Unknown mode: {mode}, returning original image")
            result = image_array
        
        # Convert result to PIL Image
        if len(result.shape) == 2:  # Grayscale
            result_image = Image.fromarray(result, mode='L')
        else:  # RGB
            result_image = Image.fromarray(result, mode='RGB')
        
        logger.info(f"Image processing completed successfully for mode: {mode} with scale: {scale}")
        return result_image
        
    except Exception as e:
        logger.error(f"Error in process_image: {e}")
        return None

def image_to_base64(image):
    """
    Convert PIL Image to base64 string for web display (always use JPEG)
    """
    try:
        buffer = io.BytesIO()
        
        # Always use JPEG format
        # Convert grayscale to RGB for JPEG compatibility
        if image.mode == 'L':
            image = image.convert('RGB')
        
        image.save(buffer, format='JPEG', quality=95)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        # Return JPEG data URL
        return f"data:image/jpeg;base64,{img_str}"
            
    except Exception as e:
        logger.error(f"Error in image_to_base64: {e}")
        return None

# Initialize enhancement manager on module import
def initialize_enhancement_manager():
    """Initialize the enhancement manager"""
    try:
        # This will trigger the singleton initialization
        manager = enhancement_manager
        logger.info("Enhancement manager initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize enhancement manager: {e}")
        return False

# Initialize on import
initialize_enhancement_manager() 