import streamlit as st
from PIL import Image
import numpy as np
import io
import logging

logger = logging.getLogger(__name__)

class UploadInterface:
    """Advanced image upload interface with quality assessment and preprocessing."""
    
    def __init__(self):
        self.accepted_formats = ['jpg', 'jpeg', 'png', 'tiff', 'bmp']
        self.max_file_size = 10 * 1024 * 1024  # 10MB
        self.min_resolution = (224, 224)
        self.recommended_resolution = (1024, 1024)
        
    def render(self):
        """Render the complete upload interface."""
        st.subheader("📷 Image Upload & Analysis")
        
        # Upload guidelines
        self._render_upload_guidelines()
        
        # File uploader
        uploaded_file = self._render_file_uploader()
        
        if uploaded_file is not None:
            # Validate and process upload
            validation_result = self._validate_upload(uploaded_file)
            
            if validation_result['valid']:
                # Show upload success and image preview
                self._render_upload_success(uploaded_file, validation_result)
                return uploaded_file
            else:
                # Show validation errors
                self._render_validation_errors(validation_result)
                return None
        
        # Sample images section
        self._render_sample_images()
        
        return None
    
    def _render_upload_guidelines(self):
        """Render upload guidelines and requirements."""
        with st.expander("📋 Image Upload Guidelines", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **✅ Recommended Image Characteristics:**
                - High-resolution dermoscopic images
                - Clear focus and good lighting
                - Lesion centered in frame
                - Minimal hair or artifacts
                - Standard dermoscopic magnification
                """)
            
            with col2:
                st.markdown("""
                **📋 Technical Requirements:**
                - Format: JPG, PNG, TIFF, BMP
                - Resolution: Minimum 224×224 pixels
                - File size: Maximum 10MB
                - Color: RGB color space
                - Quality: High compression ratio avoided
                """)
            
            st.warning("""
            ⚠️ **Important Medical Disclaimer:** 
            This AI system is designed as a clinical decision support tool and should not replace professional medical diagnosis. 
            Always consult with a qualified dermatologist for proper medical evaluation and treatment decisions.
            """)
    
    def _render_file_uploader(self):
        """Render the main file upload widget."""
        st.markdown("### 📤 Upload Dermoscopic Image")
        
        uploaded_file = st.file_uploader(
            "Choose a dermoscopic image file",
            type=self.accepted_formats,
            help="Upload a high-quality dermoscopic image for AI analysis",
            label_visibility="collapsed"
        )
        
        return uploaded_file
    
    def _validate_upload(self, uploaded_file):
        """Validate uploaded file for quality and requirements."""
        validation_result = {
            'valid': False,
            'errors': [],
            'warnings': [],
            'image_info': {},
            'quality_score': 0.0
        }
        
        try:
            # Check file size
            file_size = len(uploaded_file.getvalue())
            if file_size > self.max_file_size:
                validation_result['errors'].append(
                    f"File size ({file_size / 1024 / 1024:.1f}MB) exceeds maximum allowed size (10MB)"
                )
                return validation_result
            
            # Check file format
            file_extension = uploaded_file.name.split('.')[-1].lower()
            if file_extension not in self.accepted_formats:
                validation_result['errors'].append(
                    f"File format '{file_extension}' not supported. Please use: {', '.join(self.accepted_formats)}"
                )
                return validation_result
            
            # Load and validate image
            try:
                image = Image.open(uploaded_file)
                
                # Store image information
                validation_result['image_info'] = {
                    'filename': uploaded_file.name,
                    'format': image.format,
                    'mode': image.mode,
                    'size': image.size,
                    'file_size_mb': file_size / 1024 / 1024
                }
                
                # Check image mode
                if image.mode not in ['RGB', 'L']:
                    if image.mode == 'RGBA':
                        validation_result['warnings'].append(
                            "Image has transparency channel. Will be converted to RGB."
                        )
                    else:
                        validation_result['errors'].append(
                            f"Unsupported image mode '{image.mode}'. Please use RGB images."
                        )
                        return validation_result
                
                # Check resolution
                width, height = image.size
                if width < self.min_resolution[0] or height < self.min_resolution[1]:
                    validation_result['errors'].append(
                        f"Image resolution ({width}×{height}) is too low. "
                        f"Minimum required: {self.min_resolution[0]}×{self.min_resolution[1]}"
                    )
                    return validation_result
                
                # Resolution warnings
                if width < self.recommended_resolution[0] or height < self.recommended_resolution[1]:
                    validation_result['warnings'].append(
                        f"Image resolution ({width}×{height}) is below recommended "
                        f"({self.recommended_resolution[0]}×{self.recommended_resolution[1]}) for optimal analysis."
                    )
                
                # Assess image quality
                quality_score = self._assess_image_quality(image)
                validation_result['quality_score'] = quality_score
                
                if quality_score < 0.5:
                    validation_result['warnings'].append(
                        f"Image quality score ({quality_score:.2f}) is low. Consider uploading a higher quality image."
                    )
                
                # If we get here, validation passed
                validation_result['valid'] = True
                
                # Reset file pointer
                uploaded_file.seek(0)
                
            except Exception as e:
                validation_result['errors'].append(
                    f"Unable to process image: {str(e)}"
                )
                
        except Exception as e:
            validation_result['errors'].append(
                f"File validation failed: {str(e)}"
            )
        
        return validation_result
    
    def _assess_image_quality(self, image):
        """Assess basic image quality metrics."""
        try:
            # Convert to array for analysis
            img_array = np.array(image)
            
            # Convert to grayscale if needed
            if len(img_array.shape) == 3:
                gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
            else:
                gray = img_array
            
            # Calculate quality metrics
            # 1. Contrast (standard deviation)
            contrast_score = np.std(gray) / 128.0  # Normalize to [0, 1]
            
            # 2. Brightness (avoid over/under exposure)
            brightness = np.mean(gray) / 255.0
            brightness_score = 1.0 - abs(brightness - 0.5) * 2  # Optimal around 0.5
            
            # 3. Sharpness (using Laplacian variance)
            try:
                import cv2
                laplacian_var = cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F).var()
                sharpness_score = min(laplacian_var / 1000.0, 1.0)  # Normalize
            except ImportError:
                # Fallback if OpenCV not available
                sharpness_score = 0.7
            
            # 4. Dynamic range
            dynamic_range = (np.max(gray) - np.min(gray)) / 255.0
            
            # Combine metrics
            quality_score = (
                0.3 * min(contrast_score, 1.0) +
                0.2 * brightness_score +
                0.3 * sharpness_score +
                0.2 * dynamic_range
            )
            
            return min(max(quality_score, 0.0), 1.0)
            
        except Exception as e:
            logger.warning(f"Quality assessment failed: {str(e)}")
            return 0.5  # Default moderate quality
    
    def _render_upload_success(self, uploaded_file, validation_result):
        """Render upload success message and image preview."""
        st.success("✅ Image uploaded successfully!")
        
        # Image preview and info
        col1, col2 = st.columns([2, 1])
        
        with col1:
            image = Image.open(uploaded_file)
            st.image(
                image,
                caption=f"Uploaded: {validation_result['image_info']['filename']}",
                use_column_width=True
            )
        
        with col2:
            st.markdown("**📊 Image Information:**")
            info = validation_result['image_info']
            
            st.markdown(f"""
            - **File:** {info['filename']}
            - **Format:** {info['format']}
            - **Resolution:** {info['size'][0]}×{info['size'][1]}
            - **Size:** {info['file_size_mb']:.1f} MB
            - **Quality Score:** {validation_result['quality_score']:.2f}/1.0
            """)
            
            # Quality indicator
            quality = validation_result['quality_score']
            if quality >= 0.8:
                st.success("🟢 Excellent Quality")
            elif quality >= 0.6:
                st.info("🟡 Good Quality")
            else:
                st.warning("🟠 Fair Quality")
        
        # Show warnings if any
        if validation_result['warnings']:
            for warning in validation_result['warnings']:
                st.warning(f"⚠️ {warning}")
        
        # Reset file pointer for further processing
        uploaded_file.seek(0)
    
    def _render_validation_errors(self, validation_result):
        """Render validation errors."""
        st.error("❌ Image upload failed:")
        for error in validation_result['errors']:
            st.error(f"• {error}")
        
        # Show upload tips
        with st.expander("💡 Upload Tips"):
            st.markdown("""
            **Common Issues and Solutions:**
            
            1. **File too large:** Compress image or reduce resolution
            2. **Low resolution:** Use higher quality dermoscopic images
            3. **Wrong format:** Convert to JPG, PNG, or TIFF
            4. **Poor quality:** Ensure good lighting and focus
            5. **Color issues:** Use RGB color images
            
            **Image Optimization:**
            - Use dermoscopic equipment for best results
            - Ensure lesion is centered and well-lit
            - Avoid motion blur and artifacts
            - Remove excessive hair if possible
            """)
    
    def _render_sample_images(self):
        """Render sample images section for testing."""
        st.markdown("---")
        st.markdown("### 🖼️ Try Sample Images")
        st.markdown("Don't have a dermoscopic image? Try one of these sample images for demonstration:")
        
        # Sample image descriptions
        sample_images = {
            "melanoma_sample": {
                "name": "Melanoma (Sample)",
                "description": "Asymmetric pigmented lesion with irregular borders",
                "severity": "Critical"
            },
            "nevus_sample": {
                "name": "Benign Nevus (Sample)",
                "description": "Regular, symmetrical pigmented lesion",
                "severity": "Low"
            },
            "bcc_sample": {
                "name": "Basal Cell Carcinoma (Sample)",
                "description": "Pearly, translucent nodule with visible vessels",
                "severity": "High"
            }
        }
        
        col1, col2, col3 = st.columns(3)
        columns = [col1, col2, col3]
        
        for i, (key, info) in enumerate(sample_images.items()):
            with columns[i]:
                st.markdown(f"**{info['name']}**")
                st.markdown(f"*{info['description']}*")
                
                # Severity badge
                if info['severity'] == 'Critical':
                    st.error(f"🔴 {info['severity']} Risk")
                elif info['severity'] == 'High':
                    st.warning(f"🟠 {info['severity']} Risk")
                else:
                    st.success(f"🟢 {info['severity']} Risk")
                
                if st.button(f"Use {info['name']}", key=f"sample_{key}"):
                    st.info("Sample image functionality would load a pre-selected dermoscopic image for demonstration.")
        
        st.markdown("""
        <div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-top: 10px;">
        <small><strong>Note:</strong> Sample images are for demonstration purposes only and represent typical cases 
        seen in clinical practice. Real patient images would require proper consent and privacy protection.</small>
        </div>
        """, unsafe_allow_html=True)
    
    def validate_image_for_analysis(self, image):
        """Additional validation before AI analysis."""
        try:
            if isinstance(image, Image.Image):
                img_array = np.array(image)
            else:
                img_array = image
            
            # Check for common issues that could affect AI analysis
            issues = []
            
            # Check for extreme brightness/darkness
            if len(img_array.shape) == 3:
                brightness = np.mean(img_array)
            else:
                brightness = np.mean(img_array)
            
            if brightness < 50:
                issues.append("Image appears very dark - may affect analysis quality")
            elif brightness > 200:
                issues.append("Image appears overexposed - may affect analysis quality")
            
            # Check for very low contrast
            if len(img_array.shape) == 3:
                gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
            else:
                gray = img_array
            
            contrast = np.std(gray)
            if contrast < 20:
                issues.append("Very low contrast detected - consider enhancing image")
            
            # Check aspect ratio
            if len(img_array.shape) >= 2:
                height, width = img_array.shape[:2]
                aspect_ratio = max(width, height) / min(width, height)
                if aspect_ratio > 3:
                    issues.append("Unusual aspect ratio - lesion may not be centered")
            
            return {
                'valid_for_analysis': len(issues) == 0,
                'issues': issues,
                'recommendations': self._get_image_recommendations(issues)
            }
            
        except Exception as e:
            return {
                'valid_for_analysis': False,
                'issues': [f"Image analysis validation failed: {str(e)}"],
                'recommendations': ["Please try uploading a different image"]
            }
    
    def _get_image_recommendations(self, issues):
        """Generate recommendations based on identified issues."""
        recommendations = []
        
        for issue in issues:
            if "dark" in issue.lower():
                recommendations.append("Try increasing image brightness or using better lighting")
            elif "overexposed" in issue.lower():
                recommendations.append("Reduce exposure or use diffused lighting")
            elif "contrast" in issue.lower():
                recommendations.append("Enhance image contrast or use different imaging settings")
            elif "aspect ratio" in issue.lower():
                recommendations.append("Center the lesion in the frame and crop if necessary")
        
        if not recommendations:
            recommendations.append("Image appears suitable for AI analysis")
        
        return recommendations
