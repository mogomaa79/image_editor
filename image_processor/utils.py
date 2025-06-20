import cv2
import numpy as np
from PIL import Image
import io
import base64
import logging
import traceback
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
        
        # Convert to RGB if needed (handle RGBA, CMYK, etc.)
        if image.mode not in ['RGB', 'L']:
            image = image.convert('RGB')
        
        image_array = np.array(image)
        logger.info(f"Image loaded: {image_array.shape}, mode: {image.mode}")
        
        # Process based on mode using model manager directly
        if mode == 'ai_enhancer':
            try:
                # For AI enhancement, we need to be careful with color conversion
                if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                    # Image is RGB, convert to BGR for RealESRGAN
                    img_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                    logger.info(f"Converted RGB to BGR for processing: {img_bgr.shape}")
                else:
                    img_bgr = image_array
                
                # Use model manager for AI enhancement
                result = enhancement_manager.ai_super_resolution(img_bgr, scale=scale)
                
                # Check if result is valid
                if result is None:
                    logger.error(f"AI enhancement returned None result for scale {scale}")
                    raise Exception("AI enhancement failed - no result returned")
                
                # Ensure result is in RGB format for PIL
                if len(result.shape) == 3 and result.shape[2] == 3:
                    # If the result is in BGR (which it should be from RealESRGAN), convert to RGB
                    if np.array_equal(result, img_bgr):
                        # No processing happened, it's still BGR
                        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                        logger.info("Converted result from BGR to RGB (no processing)")
                    else:
                        # Processing happened, check if we need to convert
                        # RealESRGAN returns BGR, so convert to RGB
                        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                        logger.info("Converted processed result from BGR to RGB")
                
                logger.info(f"AI enhancement completed: {result.shape}")
                
            except Exception as e:
                logger.error(f"AI enhancement failed: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                # Fallback to original image
                result = image_array
                
        elif mode == 'gamma_clahe':
            try:
                result = enhancement_manager.advanced_gamma_clahe(image_array, gamma=1.3, clip_limit=3.0)
                logger.info(f"Gamma CLAHE completed: {result.shape if result is not None else 'None'}")
            except Exception as e:
                logger.error(f"Gamma CLAHE failed: {e}")
                result = image_array
                
        elif mode == 'shadow_fight':
            try:
                result = enhancement_manager.advanced_shadow_fight(image_array, alpha=1.4, beta=40)
                logger.info(f"Shadow fight completed: {result.shape if result is not None else 'None'}")
            except Exception as e:
                logger.error(f"Shadow fight failed: {e}")
                result = image_array
                
        elif mode == 'grayscale':
            try:
                result = enhancement_manager.advanced_grayscale(image_array)
                logger.info(f"Grayscale completed: {result.shape if result is not None else 'None'}")
            except Exception as e:
                logger.error(f"Grayscale failed: {e}")
                result = image_array
                
        else:
            logger.warning(f"Unknown mode: {mode}, returning original image")
            result = image_array
        
        # Validate result
        if result is None:
            logger.error(f"Processing returned None result for mode: {mode}")
            result = image_array  # Fallback to original
            
        # Ensure result is valid numpy array
        if not isinstance(result, np.ndarray):
            logger.error(f"Result is not numpy array: {type(result)}")
            result = image_array
        
        # Convert result to PIL Image with proper mode handling
        try:
            if len(result.shape) == 2:  # Grayscale
                result_image = Image.fromarray(result.astype(np.uint8), mode='L')
                logger.info("Created grayscale PIL image")
            elif len(result.shape) == 3 and result.shape[2] == 3:  # RGB
                # Ensure values are in valid range
                result = np.clip(result, 0, 255).astype(np.uint8)
                result_image = Image.fromarray(result, mode='RGB')
                logger.info("Created RGB PIL image")
            else:
                logger.error(f"Unexpected result shape: {result.shape}")
                result_image = image  # Return original image
        except Exception as e:
            logger.error(f"Error creating PIL image: {e}")
            result_image = image  # Return original image
        
        logger.info(f"Image processing completed successfully for mode: {mode} with scale: {scale}")
        return result_image
        
    except Exception as e:
        logger.error(f"Error in process_image: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
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
        elif image.mode not in ['RGB', 'RGBA']:
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