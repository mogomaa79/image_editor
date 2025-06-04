"""
Lightweight Image Enhancement Model Manager
Provides AI-style image enhancement without heavy ML dependencies
"""

import cv2
import numpy as np
from skimage import restoration, filters, exposure, morphology
from skimage.transform import rescale
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

class ImageEnhancementManager:
    """
    Singleton class to manage image enhancement models and algorithms
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
        """Initialize enhancement parameters and filters"""
        logger.info("Initializing Image Enhancement Manager...")
        
        # Cache for computed kernels and filters
        self.kernels = {}
        self.filters = {}
        
        # Pre-compute commonly used kernels
        self._precompute_kernels()
        
        logger.info("Image Enhancement Manager initialized successfully")
    
    def _precompute_kernels(self):
        """Pre-compute and cache image processing kernels"""
        
        # Sharpening kernels
        self.kernels['sharpen_light'] = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=np.float32)
        
        self.kernels['sharpen_strong'] = np.array([
            [-1, -1, -1],
            [-1, 9, -1],
            [-1, -1, -1]
        ], dtype=np.float32)
        
        # Unsharp mask kernel
        self.kernels['unsharp_mask'] = np.array([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ], dtype=np.float32)
        
        # Edge enhancement kernel
        self.kernels['edge_enhance'] = np.array([
            [0, 0, 0],
            [-1, 1, 0],
            [0, 0, 0]
        ], dtype=np.float32)
    
    @lru_cache(maxsize=32)
    def get_bilateral_params(self, noise_level='medium'):
        """Get cached bilateral filter parameters based on noise level"""
        params = {
            'low': {'d': 5, 'sigmaColor': 50, 'sigmaSpace': 50},
            'medium': {'d': 9, 'sigmaColor': 75, 'sigmaSpace': 75},
            'high': {'d': 13, 'sigmaColor': 100, 'sigmaSpace': 100}
        }
        return params.get(noise_level, params['medium'])
    
    def ai_super_resolution(self, image, scale=2, method='advanced'):
        """
        Advanced super-resolution using multiple upscaling techniques
        """
        try:
            if len(image.shape) == 3:
                # Multi-channel processing
                channels = cv2.split(image)
                enhanced_channels = []
                
                for channel in channels:
                    enhanced_channel = self._enhance_single_channel(channel, scale, method)
                    enhanced_channels.append(enhanced_channel)
                
                result = cv2.merge(enhanced_channels)
            else:
                # Single channel processing
                result = self._enhance_single_channel(image, scale, method)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in AI super-resolution: {e}")
            # Fallback to basic bicubic interpolation
            return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    def _enhance_single_channel(self, channel, scale, method):
        """Enhance a single channel with advanced algorithms"""
        
        # Step 1: Initial upscaling with edge-preserving interpolation
        h, w = channel.shape
        
        if method == 'advanced':
            # Multi-step upscaling for better quality
            if scale > 2:
                # First upscale by 2
                temp = cv2.resize(channel, (w*2, h*2), interpolation=cv2.INTER_LANCZOS4)
                # Apply enhancement
                temp = self._apply_edge_preserving_enhancement(temp)
                # Final upscale
                result = cv2.resize(temp, (w*scale, h*scale), interpolation=cv2.INTER_LANCZOS4)
            else:
                result = cv2.resize(channel, (w*scale, h*scale), interpolation=cv2.INTER_LANCZOS4)
        else:
            # Simple upscaling
            result = cv2.resize(channel, (w*scale, h*scale), interpolation=cv2.INTER_LANCZOS4)
        
        # Step 2: Apply enhancement filters
        result = self._apply_enhancement_pipeline(result)
        
        return result
    
    def _apply_edge_preserving_enhancement(self, image):
        """Apply edge-preserving enhancement techniques"""
        
        # Bilateral filtering for noise reduction while preserving edges
        params = self.get_bilateral_params('medium')
        filtered = cv2.bilateralFilter(image, **params)
        
        # Unsharp masking for sharpening
        gaussian = cv2.GaussianBlur(filtered, (5, 5), 1.0)
        unsharp = cv2.addWeighted(filtered, 1.5, gaussian, -0.5, 0)
        
        # Clip values to valid range
        return np.clip(unsharp, 0, 255).astype(np.uint8)
    
    def _apply_enhancement_pipeline(self, image):
        """Apply a pipeline of enhancement algorithms"""
        
        # Convert to float for processing
        img_float = image.astype(np.float32) / 255.0
        
        # Step 1: Denoising
        img_float = restoration.denoise_tv_chambolle(img_float, weight=0.1)
        
        # Step 2: Contrast enhancement
        img_float = exposure.rescale_intensity(img_float)
        
        # Step 3: Adaptive histogram equalization
        img_uint8 = (img_float * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_uint8 = clahe.apply(img_uint8)
        
        # Step 4: Light sharpening
        kernel = self.kernels['sharpen_light']
        sharpened = cv2.filter2D(img_uint8, -1, kernel)
        
        # Blend original and sharpened
        result = cv2.addWeighted(img_uint8, 0.7, sharpened, 0.3, 0)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def advanced_gamma_clahe(self, image, gamma=1.3, clip_limit=3.0):
        """
        Advanced Gamma correction with CLAHE and additional enhancements
        """
        try:
            if len(image.shape) == 3:
                # Convert to LAB color space for better processing
                lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                
                # Apply CLAHE to L channel
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
                l = clahe.apply(l)
                
                # Merge back
                lab = cv2.merge([l, a, b])
                result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            else:
                # Grayscale processing
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
                result = clahe.apply(image)
            
            # Apply gamma correction
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            result = cv2.LUT(result, table)
            
            # Additional enhancement: local contrast improvement
            result = self._improve_local_contrast(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in advanced gamma CLAHE: {e}")
            return image
    
    def _improve_local_contrast(self, image):
        """Improve local contrast using morphological operations"""
        
        if len(image.shape) == 3:
            # Process each channel
            channels = cv2.split(image)
            enhanced_channels = []
            
            for channel in channels:
                # Create morphological operations
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                tophat = cv2.morphologyEx(channel, cv2.MORPH_TOPHAT, kernel)
                blackhat = cv2.morphologyEx(channel, cv2.MORPH_BLACKHAT, kernel)
                
                # Enhance the channel
                enhanced = cv2.add(channel, tophat)
                enhanced = cv2.subtract(enhanced, blackhat)
                enhanced_channels.append(enhanced)
            
            return cv2.merge(enhanced_channels)
        else:
            # Grayscale processing
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
            blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
            
            enhanced = cv2.add(image, tophat)
            enhanced = cv2.subtract(enhanced, blackhat)
            return enhanced
    
    def advanced_shadow_fight(self, image, alpha=1.4, beta=40, shadow_enhance=True):
        """
        Advanced shadow fighting with local adaptive enhancement
        """
        try:
            # Basic brightness/contrast adjustment
            basic_enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            
            if not shadow_enhance:
                return basic_enhanced
            
            # Advanced shadow detection and enhancement
            if len(image.shape) == 3:
                # Convert to LAB for better shadow detection
                lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                l_channel = lab[:,:,0]
            else:
                l_channel = image
            
            # Detect shadow regions (dark areas)
            shadow_mask = l_channel < np.mean(l_channel) - np.std(l_channel) * 0.5
            shadow_mask = shadow_mask.astype(np.uint8) * 255
            
            # Morphological operations to refine shadow mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)
            shadow_mask = cv2.GaussianBlur(shadow_mask, (21, 21), 0)
            
            # Apply stronger enhancement to shadow regions
            shadow_enhanced = cv2.convertScaleAbs(image, alpha=alpha*1.3, beta=beta*1.5)
            
            # Blend based on shadow mask
            shadow_mask_norm = shadow_mask.astype(np.float32) / 255.0
            if len(image.shape) == 3:
                shadow_mask_norm = np.stack([shadow_mask_norm] * 3, axis=2)
            
            result = (shadow_enhanced.astype(np.float32) * shadow_mask_norm + 
                     basic_enhanced.astype(np.float32) * (1 - shadow_mask_norm))
            
            return np.clip(result, 0, 255).astype(np.uint8)
            
        except Exception as e:
            logger.error(f"Error in advanced shadow fight: {e}")
            return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    def advanced_grayscale(self, image, method='luminance'):
        """
        Advanced grayscale conversion with different methods
        """
        try:
            if len(image.shape) == 2:
                return image  # Already grayscale
            
            if method == 'luminance':
                # Luminance method (ITU-R BT.709)
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            elif method == 'weighted':
                # Custom weighted average
                gray = np.average(image, axis=2, weights=[0.3, 0.59, 0.11])
                gray = gray.astype(np.uint8)
            elif method == 'desaturation':
                # Desaturation method
                gray = (np.max(image, axis=2) + np.min(image, axis=2)) / 2
                gray = gray.astype(np.uint8)
            else:
                # Default OpenCV method
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Apply local contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced_gray = clahe.apply(gray)
            
            return enhanced_gray
            
        except Exception as e:
            logger.error(f"Error in advanced grayscale: {e}")
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Global instance
enhancement_manager = ImageEnhancementManager() 