import cv2
import numpy as np
from PIL import Image
import io
import base64
import logging
from .model_manager import enhancement_manager

logger = logging.getLogger(__name__)

def enhance_image(image_array, scale=2):
    """
    AI Image Enhancement using advanced algorithms without PyTorch
    """
    try:
        logger.info(f"Starting AI enhancement with scale={scale}")
        
        # Convert PIL to OpenCV format if needed
        if len(image_array.shape) == 3:
            img = cv2.cvtColor(np.array(image_array), cv2.COLOR_RGB2BGR)
        else:
            img = image_array
        
        h, w = img.shape[:2]
        logger.info(f"Original image size: {w}x{h}")
        
        # Resize if image is too small for better quality
        if h < 300:
            img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
            logger.info("Upscaled small image for better processing")
        
        # Use the advanced AI super-resolution from model manager
        output = enhancement_manager.ai_super_resolution(img, scale=scale, method='advanced')
        
        # Limit maximum size to prevent memory issues
        h, w = output.shape[:2]
        max_size = 3480
        if h > max_size or w > max_size:
            if h > max_size:
                w = int(w * max_size / h)
                h = max_size
            
            if w > max_size:
                h = int(h * max_size / w)
                w = max_size
            
            output = cv2.resize(output, (w, h), interpolation=cv2.INTER_LANCZOS4)
            logger.info(f"Resized to fit max size: {w}x{h}")
        
        # Convert back to RGB
        if len(output.shape) == 3:
            enhanced_image = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        else:
            enhanced_image = output
        
        logger.info("AI enhancement completed successfully")
        return enhanced_image
        
    except Exception as e:
        logger.error(f"Error in enhance_image: {e}")
        # Fallback to simple upscaling
        if len(image_array.shape) == 3:
            img = cv2.cvtColor(np.array(image_array), cv2.COLOR_RGB2BGR)
            output = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
            return cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        else:
            return cv2.resize(image_array, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)

def gamma_clahe(image_array, gamma=1.3):
    """
    Advanced Gamma correction with CLAHE and local contrast enhancement
    """
    try:
        logger.info(f"Starting advanced gamma CLAHE with gamma={gamma}")
        
        # Convert to OpenCV format if needed
        if len(image_array.shape) == 3:
            image = cv2.cvtColor(np.array(image_array), cv2.COLOR_RGB2BGR)
        else:
            image = image_array
        
        # Use the advanced gamma CLAHE from model manager
        result = enhancement_manager.advanced_gamma_clahe(image, gamma=gamma, clip_limit=3.0)
        
        # Convert back to RGB if needed
        if len(result.shape) == 3:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        
        logger.info("Advanced gamma CLAHE completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Error in gamma_clahe: {e}")
        # Fallback to basic implementation
        return _fallback_gamma_clahe(image_array, gamma)

def _fallback_gamma_clahe(image_array, gamma):
    """Fallback implementation for gamma CLAHE"""
    try:
        if len(image_array.shape) == 3:
            image = cv2.cvtColor(np.array(image_array), cv2.COLOR_RGB2BGR)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            clahe_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            clahe_img = clahe.apply(image_array)
        
        # Apply Gamma Correction
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        final_img = cv2.LUT(clahe_img, table)
        
        if len(final_img.shape) == 3:
            result = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
        else:
            result = final_img
        
        return result
    except Exception as e:
        logger.error(f"Error in fallback gamma CLAHE: {e}")
        return image_array

def shadow_fight(image_array, bc_alpha=1.4, bc_beta=40):
    """
    Advanced shadow fighting using local adaptive enhancement
    """
    try:
        logger.info(f"Starting advanced shadow fight with alpha={bc_alpha}, beta={bc_beta}")
        
        # Convert to OpenCV format if needed
        if len(image_array.shape) == 3:
            original_image = cv2.cvtColor(np.array(image_array), cv2.COLOR_RGB2BGR)
        else:
            original_image = image_array
        
        # Use the advanced shadow fight from model manager
        result = enhancement_manager.advanced_shadow_fight(
            original_image, 
            alpha=bc_alpha, 
            beta=bc_beta, 
            shadow_enhance=True
        )
        
        # Convert back to RGB if needed
        if len(result.shape) == 3:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        
        logger.info("Advanced shadow fight completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Error in shadow_fight: {e}")
        # Fallback to basic implementation
        return _fallback_shadow_fight(image_array, bc_alpha, bc_beta)

def _fallback_shadow_fight(image_array, bc_alpha, bc_beta):
    """Fallback implementation for shadow fight"""
    try:
        if len(image_array.shape) == 3:
            original_image = cv2.cvtColor(np.array(image_array), cv2.COLOR_RGB2BGR)
            enhanced_image = cv2.convertScaleAbs(original_image, alpha=bc_alpha, beta=bc_beta)
            result = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)
        else:
            result = cv2.convertScaleAbs(image_array, alpha=bc_alpha, beta=bc_beta)
        
        return result
    except Exception as e:
        logger.error(f"Error in fallback shadow fight: {e}")
        return image_array

def grayscale(image_array):
    """
    Advanced grayscale conversion with multiple methods and enhancement
    """
    try:
        logger.info("Starting advanced grayscale conversion")
        
        # Use the advanced grayscale from model manager
        result = enhancement_manager.advanced_grayscale(image_array, method='luminance')
        
        logger.info("Advanced grayscale conversion completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Error in grayscale: {e}")
        # Fallback to basic implementation
        return _fallback_grayscale(image_array)

def _fallback_grayscale(image_array):
    """Fallback implementation for grayscale"""
    try:
        if len(image_array.shape) == 3:
            image = cv2.cvtColor(np.array(image_array), cv2.COLOR_RGB2BGR)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image_array
        
        return gray_image
    except Exception as e:
        logger.error(f"Error in fallback grayscale: {e}")
        return image_array

def process_image(image_file, mode, scale=2):
    """
    Process image based on selected mode using advanced algorithms
    """
    try:
        logger.info(f"Processing image with mode: {mode}, scale: {scale}")
        
        # Validate scale parameter
        scale = max(1, min(4, scale))
        
        # Open and convert image
        image = Image.open(image_file)
        image_array = np.array(image)
        
        # Process based on mode
        if mode == 'ai_enhancer':
            result = enhance_image(image_array, scale=scale)
        elif mode == 'gamma_clahe':
            result = gamma_clahe(image_array, gamma=1.3)
        elif mode == 'shadow_fight':
            result = shadow_fight(image_array, bc_alpha=1.4, bc_beta=40)
        elif mode == 'grayscale':
            result = grayscale(image_array)
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