import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import albumentations as A
from skimage import exposure, restoration
import logging

logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """Advanced image preprocessing for skin cancer detection with bias mitigation."""
    
    def __init__(self):
        self.target_size = (224, 224)
        self.augmentation_pipeline = self._create_augmentation_pipeline()
        self.bias_mitigation_augmentations = self._create_bias_mitigation_pipeline()
        
    def preprocess_image(self, image, apply_enhancement=True, normalize=True):
        """
        Preprocess a single image for model inference.
        
        Args:
            image: PIL Image or numpy array
            apply_enhancement: Whether to apply image enhancement
            normalize: Whether to normalize pixel values
            
        Returns:
            Preprocessed image as numpy array
        """
        try:
            # Convert to PIL Image if numpy array
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply enhancement techniques
            if apply_enhancement:
                image = self._enhance_image(image)
            
            # Resize to target size
            image = image.resize(self.target_size, Image.LANCZOS)
            
            # Convert to numpy array
            image_array = np.array(image, dtype=np.float32)
            
            # Normalize if requested
            if normalize:
                image_array = image_array / 255.0
            
            return image_array
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            # Return a blank image as fallback
            return np.zeros((*self.target_size, 3), dtype=np.float32)
    
    def _enhance_image(self, image):
        """Apply AI-powered image enhancement techniques."""
        try:
            # Convert to numpy for OpenCV operations
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            lab = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            cv_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # Convert back to RGB PIL Image
            enhanced_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            
            # Apply additional PIL enhancements
            enhanced_image = self._apply_pil_enhancements(enhanced_image)
            
            return enhanced_image
            
        except Exception as e:
            logger.warning(f"Enhancement failed, using original image: {str(e)}")
            return image
    
    def _apply_pil_enhancements(self, image):
        """Apply PIL-based image enhancements."""
        try:
            # Enhance contrast
            contrast_enhancer = ImageEnhance.Contrast(image)
            image = contrast_enhancer.enhance(1.2)
            
            # Enhance sharpness
            sharpness_enhancer = ImageEnhance.Sharpness(image)
            image = sharpness_enhancer.enhance(1.1)
            
            # Reduce noise with gentle blur
            image = image.filter(ImageFilter.GaussianBlur(radius=0.5))
            
            return image
            
        except Exception as e:
            logger.warning(f"PIL enhancement failed: {str(e)}")
            return image
    
    def _create_augmentation_pipeline(self):
        """Create augmentation pipeline for training data."""
        return A.Compose([
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=0.5
            ),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.5
            ),
            A.GaussianBlur(blur_limit=(1, 3), p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.ElasticTransform(
                alpha=1,
                sigma=50,
                alpha_affine=50,
                p=0.3
            )
        ])
    
    def _create_bias_mitigation_pipeline(self):
        """Create specialized augmentations for bias mitigation across skin tones."""
        return A.Compose([
            # Color space augmentations to simulate different skin tones
            A.HueSaturationValue(
                hue_shift_limit=30,
                sat_shift_limit=30,
                val_shift_limit=30,
                p=0.8
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.8
            ),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.7
            ),
            # Lighting variations
            A.RandomShadow(p=0.3),
            A.RandomSunFlare(p=0.2),
            # Texture variations
            A.GaussNoise(var_limit=(5.0, 25.0), p=0.4),
            A.ISONoise(p=0.3),
        ])
    
    def augment_for_fairness(self, image, skin_tone_group):
        """
        Apply skin-tone specific augmentations for fairness.
        
        Args:
            image: Input image as numpy array
            skin_tone_group: Fitzpatrick skin type (1-6)
            
        Returns:
            Augmented image
        """
        try:
            # Apply base bias mitigation augmentations
            augmented = self.bias_mitigation_augmentations(image=image)['image']
            
            # Apply skin-tone specific adjustments
            if skin_tone_group in [1, 2]:  # Very fair to fair skin
                # Increase contrast and brightness for better lesion visibility
                augmented = A.RandomBrightnessContrast(
                    brightness_limit=0.1,
                    contrast_limit=0.2,
                    p=1.0
                )(image=augmented)['image']
                
            elif skin_tone_group in [3, 4]:  # Medium to olive skin
                # Apply moderate adjustments
                augmented = A.HueSaturationValue(
                    hue_shift_limit=15,
                    sat_shift_limit=15,
                    val_shift_limit=15,
                    p=1.0
                )(image=augmented)['image']
                
            elif skin_tone_group in [5, 6]:  # Brown to dark brown/black skin
                # Enhance visibility while preserving natural appearance
                augmented = A.CLAHE(
                    clip_limit=3.0,
                    tile_grid_size=(8, 8),
                    p=1.0
                )(image=augmented)['image']
            
            return augmented
            
        except Exception as e:
            logger.warning(f"Fairness augmentation failed: {str(e)}")
            return image
    
    def preprocess_batch(self, images, labels=None, skin_tones=None, apply_augmentation=False):
        """
        Preprocess a batch of images.
        
        Args:
            images: List of PIL Images or numpy arrays
            labels: Optional labels for the images
            skin_tones: Optional skin tone information for fairness augmentation
            apply_augmentation: Whether to apply data augmentation
            
        Returns:
            Preprocessed batch as numpy array
        """
        preprocessed_images = []
        
        for i, image in enumerate(images):
            # Basic preprocessing
            processed = self.preprocess_image(image)
            
            # Apply augmentation if requested
            if apply_augmentation:
                if skin_tones is not None and i < len(skin_tones):
                    # Apply fairness-aware augmentation
                    processed = self.augment_for_fairness(processed, skin_tones[i])
                else:
                    # Apply standard augmentation
                    processed = self.augmentation_pipeline(image=processed)['image']
            
            preprocessed_images.append(processed)
        
        return np.array(preprocessed_images)
    
    def assess_image_quality(self, image):
        """
        Assess the quality of an input image for clinical analysis.
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            Dictionary with quality metrics
        """
        try:
            if isinstance(image, Image.Image):
                image_array = np.array(image)
            else:
                image_array = image
            
            # Convert to grayscale for some metrics
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            
            # Calculate quality metrics
            quality_metrics = {
                'sharpness': self._calculate_sharpness(gray),
                'contrast': self._calculate_contrast(gray),
                'brightness': self._calculate_brightness(gray),
                'noise_level': self._estimate_noise(gray),
                'overall_quality': 0.0
            }
            
            # Calculate overall quality score
            quality_metrics['overall_quality'] = self._calculate_overall_quality(quality_metrics)
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {str(e)}")
            return {
                'sharpness': 0.0,
                'contrast': 0.0,
                'brightness': 0.0,
                'noise_level': 1.0,
                'overall_quality': 0.0
            }
    
    def _calculate_sharpness(self, gray_image):
        """Calculate image sharpness using Laplacian variance."""
        return cv2.Laplacian(gray_image, cv2.CV_64F).var()
    
    def _calculate_contrast(self, gray_image):
        """Calculate image contrast using standard deviation."""
        return np.std(gray_image)
    
    def _calculate_brightness(self, gray_image):
        """Calculate average brightness."""
        return np.mean(gray_image)
    
    def _estimate_noise(self, gray_image):
        """Estimate noise level in the image."""
        try:
            # Use median absolute deviation for noise estimation
            noise = np.median(np.abs(gray_image - np.median(gray_image)))
            return noise / 128.0  # Normalize to [0, 1]
        except:
            return 0.5
    
    def _calculate_overall_quality(self, metrics):
        """Calculate overall quality score from individual metrics."""
        # Normalize sharpness (typical range: 0-1000)
        sharpness_score = min(metrics['sharpness'] / 100.0, 1.0)
        
        # Normalize contrast (typical range: 0-100)
        contrast_score = min(metrics['contrast'] / 50.0, 1.0)
        
        # Brightness score (optimal around 127.5 for 8-bit images)
        brightness_score = 1.0 - abs(metrics['brightness'] - 127.5) / 127.5
        
        # Noise score (lower is better)
        noise_score = 1.0 - metrics['noise_level']
        
        # Weighted average
        overall_quality = (
            0.3 * sharpness_score +
            0.25 * contrast_score +
            0.2 * brightness_score +
            0.25 * noise_score
        )
        
        return min(max(overall_quality, 0.0), 1.0)
    
    def create_attention_map(self, image, attention_weights):
        """
        Create attention visualization overlay.
        
        Args:
            image: Original image
            attention_weights: Attention weights from model
            
        Returns:
            Image with attention overlay
        """
        try:
            if isinstance(image, Image.Image):
                image_array = np.array(image)
            else:
                image_array = image
            
            # Resize attention weights to match image size
            attention_resized = cv2.resize(
                attention_weights,
                (image_array.shape[1], image_array.shape[0])
            )
            
            # Normalize attention weights
            attention_normalized = (attention_resized - attention_resized.min()) / (
                attention_resized.max() - attention_resized.min()
            )
            
            # Create heatmap
            heatmap = cv2.applyColorMap(
                (attention_normalized * 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            
            # Blend with original image
            overlay = cv2.addWeighted(image_array, 0.6, heatmap, 0.4, 0)
            
            return Image.fromarray(overlay)
            
        except Exception as e:
            logger.error(f"Attention map creation failed: {str(e)}")
            return image if isinstance(image, Image.Image) else Image.fromarray(image)
    
    def denormalize_image(self, normalized_image):
        """Convert normalized image back to [0, 255] range."""
        return (normalized_image * 255).astype(np.uint8)
    
    def resize_with_aspect_ratio(self, image, target_size, fill_color=(255, 255, 255)):
        """Resize image while maintaining aspect ratio."""
        # Calculate scaling factor
        width, height = image.size
        target_width, target_height = target_size
        
        scale = min(target_width / width, target_height / height)
        
        # Calculate new size
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize image
        resized = image.resize((new_width, new_height), Image.LANCZOS)
        
        # Create new image with target size and fill color
        result = Image.new('RGB', target_size, fill_color)
        
        # Paste resized image in center
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2
        result.paste(resized, (x_offset, y_offset))
        
        return result

class DatasetPreprocessor:
    """Preprocessor for handling entire datasets with fairness considerations."""
    
    def __init__(self, image_preprocessor=None):
        self.image_preprocessor = image_preprocessor or ImagePreprocessor()
        
    def preprocess_dataset(self, dataset_path, output_path, fairness_balance=True):
        """
        Preprocess an entire dataset with fairness considerations.
        
        Args:
            dataset_path: Path to original dataset
            output_path: Path to save preprocessed dataset
            fairness_balance: Whether to balance dataset across demographic groups
        """
        # This would implement full dataset preprocessing
        # For now, we'll just document the interface
        pass
    
    def balance_dataset_by_demographics(self, dataset, skin_tone_labels):
        """Balance dataset across different skin tone groups."""
        # Implementation would go here
        pass
    
    def generate_synthetic_samples(self, images, labels, target_group):
        """Generate synthetic samples for underrepresented groups."""
        # Implementation would use techniques like SMOTE, GANs, etc.
        pass
