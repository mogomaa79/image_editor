"""
RealESRGAN Image Enhancement Model Manager
Provides AI-style image enhancement using RealESRGAN
"""

import cv2
import numpy as np
import torch
import os
import logging
from basicsr.archs.srvgg_arch import SRVGGNetCompact
from realesrgan.utils import RealESRGANer
from PIL import Image

logger = logging.getLogger(__name__)

class ImageEnhancementManager:
    """
    Singleton class to manage RealESRGAN image enhancement model
    """
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._initialize_models()
            self._initialized = True
    
    def _initialize_models(self):
        """Initialize RealESRGAN model"""
        logger.info("Initializing RealESRGAN Model Manager...")
        
        try:
            # Download model if it doesn't exist
            model_path = 'realesr-general-x4v3.pth'
            if not os.path.exists(model_path):
                logger.info("Downloading RealESRGAN model...")
                import subprocess
                subprocess.run([
                    "wget", 
                    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
                    "-P", "."
                ], check=True)
            
            # Initialize model
            self.model = SRVGGNetCompact(
                num_in_ch=3, 
                num_out_ch=3, 
                num_feat=64, 
                num_conv=32, 
                upscale=4, 
                act_type='prelu'
            )
            
            # Check if CUDA is available
            self.half = torch.cuda.is_available()
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # Initialize RealESRGAN upsampler
            self.upsampler = RealESRGANer(
                scale=2, 
                model_path=model_path, 
                model=self.model, 
                tile=0, 
                tile_pad=10, 
                pre_pad=0, 
                half=self.half
            )
            
            logger.info(f"RealESRGAN Model Manager initialized successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize RealESRGAN: {e}")
            # Fallback initialization without RealESRGAN
            self.upsampler = None
            logger.warning("Falling back to basic OpenCV enhancement")
    
    def ai_super_resolution(self, image, scale=2, method='advanced'):
        """
        AI Super-resolution using RealESRGAN with configurable scale (1-4x)
        """
        try:
            # Validate scale parameter
            if scale < 1 or scale > 4:
                logger.warning(f"Invalid scale {scale}, using default scale 2")
                scale = 2
            
            if self.upsampler is None:
                logger.warning("RealESRGAN not available, using fallback method")
                return self._fallback_enhancement(image, scale)
            
            # Convert PIL Image to numpy array if needed
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            # If scale is 1, return original image with minimal processing
            if scale == 1:
                if isinstance(image, np.ndarray):
                    return image
                return np.array(image)
            
            # Ensure image is in BGR format for OpenCV
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Convert RGB to BGR if needed
                if isinstance(image, np.ndarray) and image.dtype == np.uint8:
                    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                else:
                    img_bgr = image
            else:
                img_bgr = image
            
            # Pre-process: Resize small images for better results
            h, w = img_bgr.shape[:2]
            if h < 200 and scale > 2:
                # For small images with high scale, do initial upscaling
                img_bgr = cv2.resize(img_bgr, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
            
            # Apply RealESRGAN enhancement with specified scale
            output, _ = self.upsampler.enhance(img_bgr, outscale=scale)
            
            # Post-process: Limit max size to prevent memory issues
            h, w = output.shape[:2]
            max_size = 4000  # Increased for higher scale support
            
            if h > max_size or w > max_size:
                if h > w:
                    new_h = max_size
                    new_w = int(w * max_size / h)
                else:
                    new_w = max_size
                    new_h = int(h * max_size / w)
                
                output = cv2.resize(output, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                logger.info(f"Resized output from {w}x{h} to {new_w}x{new_h} to fit memory constraints")
            
            # Convert back to RGB for consistency
            if len(output.shape) == 3:
                output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            
            logger.info(f"Successfully applied RealESRGAN with {scale}x scale")
            return output
            
        except Exception as e:
            logger.error(f"Error in RealESRGAN super-resolution: {e}")
    
    def advanced_gamma_clahe(self, image, gamma=1.3, clip_limit=3.0):
        """
        Advanced Gamma correction with CLAHE
        """
        try:
            if isinstance(image, Image.Image):
                image = np.array(image)
                
            # Convert to BGR if needed for OpenCV operations
            if len(image.shape) == 3:
                # Check if we need to convert RGB to BGR
                img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Convert to LAB color space for better processing
                lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                
                # Apply CLAHE to L channel
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
                l = clahe.apply(l)
                
                # Merge back
                lab = cv2.merge([l, a, b])
                result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                
                # Convert back to RGB
                result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            else:
                # Grayscale processing
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
                result = clahe.apply(image)
            
            # Apply gamma correction
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            result = cv2.LUT(result, table)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in advanced gamma CLAHE: {e}")
            return image
    
    def advanced_shadow_fight(self, image, alpha=1.4, beta=30):
        """
        Advanced shadow fighting with adaptive enhancement
        """
        try:
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            # Basic brightness/contrast adjustment
            result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            return result
            
        except Exception as e:
            logger.error(f"Error in advanced shadow fight: {e}")
            return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    def advanced_grayscale(self, image):
        """
        Advanced grayscale conversion with different methods
        """
        try:
            if isinstance(image, Image.Image):
                image = np.array(image)
                
            if len(image.shape) == 2:
                return image  # Already grayscale
            
            # Convert to BGR for OpenCV operations if needed
            img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.shape[2] == 3 else image
            
            # Convert to grayscale using OpenCV
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            
            return gray
            
        except Exception as e:
            logger.error(f"Error in advanced grayscale: {e}")
            if len(image.shape) == 3:
                return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            return image

# Global instance
enhancement_manager = ImageEnhancementManager() 