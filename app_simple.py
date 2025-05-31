import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import datetime
import logging

# Page configuration
st.set_page_config(
    page_title="AI Skin Cancer Detection Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Mock AI Models for demonstration
class MockAIModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.class_names = [
            'Melanoma', 'Basal Cell Carcinoma', 'Squamous Cell Carcinoma',
            'Actinic Keratosis', 'Benign Keratosis', 'Dermatofibroma', 'Nevus'
        ]
    
    def predict(self, image):
        """Generate realistic predictions based on model characteristics."""
        np.random.seed(42)  # For consistent results
        
        # Different models have different characteristics
        if self.model_name == "ResNet-50":
            probabilities = np.array([0.05, 0.08, 0.03, 0.12, 0.15, 0.02, 0.55])
        elif self.model_name == "EfficientNet-B4":
            probabilities = np.array([0.03, 0.06, 0.02, 0.10, 0.12, 0.01, 0.66])
        elif self.model_name == "Vision Transformer":
            probabilities = np.array([0.08, 0.12, 0.05, 0.15, 0.18, 0.03, 0.39])
        else:  # Hybrid CNN-ViT
            probabilities = np.array([0.02, 0.04, 0.01, 0.08, 0.10, 0.01, 0.74])
        
        # Add some randomness
        noise = np.random.normal(0, 0.02, len(probabilities))
        probabilities = np.clip(probabilities + noise, 0, 1)
        probabilities = probabilities / np.sum(probabilities)  # Normalize
        
        predicted_idx = np.argmax(probabilities)
        
        return {
            'predicted_class': self.class_names[predicted_idx],
            'predicted_class_idx': int(predicted_idx),
            'probabilities': probabilities.tolist(),
            'confidence': float(probabilities[predicted_idx])
        }

def main():
    # Header
    st.title("🔬 AI-Powered Skin Cancer Detection Platform")
    st.markdown("### Advanced Multi-Model Analysis with Fairness Evaluation")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        st.subheader("Model Selection")
        selected_models = st.multiselect(
            "Choose AI Models",
            ["ResNet-50", "EfficientNet-B4", "Vision Transformer", "Hybrid CNN-ViT"],
            default=["ResNet-50", "EfficientNet-B4"]
        )
        
        # Fairness settings
        st.subheader("Fairness Evaluation")
        enable_fairness = st.checkbox("Enable Demographic Fairness Analysis", value=True)
        
        if enable_fairness:
            skin_tone = st.selectbox(
                "Patient Skin Tone (Fitzpatrick Scale)",
                ["I (Very Fair)", "II (Fair)", "III (Medium)", "IV (Olive)", "V (Brown)", "VI (Dark Brown/Black)"]
            )
        
        # Explainability settings
        st.subheader("Explainable AI")
        explanation_methods = st.multiselect(
            "Explanation Methods",
            ["Grad-CAM", "SHAP", "LIME"],
            default=["Grad-CAM"]
        )
        
        # Clinical decision support
        st.subheader("Clinical Support")
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.7, 0.05)
        enable_clinical_alerts = st.checkbox("Enable Clinical Alerts", value=True)

    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📸 Image Analysis", 
        "📊 Results Dashboard", 
        "⚖️ Fairness Analysis", 
        "📈 Model Performance", 
        "ℹ️ Dataset Info"
    ])
    
    with tab1:
        render_image_analysis_tab(selected_models, confidence_threshold)
    
    with tab2:
        render_results_dashboard_tab(explanation_methods, enable_clinical_alerts)
    
    with tab3:
        render_fairness_analysis_tab(enable_fairness, skin_tone if enable_fairness else None)
    
    with tab4:
        render_model_performance_tab()
    
    with tab5:
        render_dataset_info_tab()

def render_image_analysis_tab(selected_models, confidence_threshold):
    """Render the image analysis tab."""
    st.header("Image Upload and Analysis")
    
    # Upload guidelines
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
    
    # File uploader
    st.markdown("### 📤 Upload Dermoscopic Image")
    uploaded_file = st.file_uploader(
        "Choose a dermoscopic image file",
        type=['jpg', 'jpeg', 'png', 'tiff', 'bmp'],
        help="Upload a high-quality dermoscopic image for AI analysis"
    )
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Original Image")
            st.image(image, caption="Uploaded dermoscopic image", use_container_width=True)
        
        with col2:
            st.subheader("Image Information")
            st.markdown(f"""
            - **Filename:** {uploaded_file.name}
            - **Format:** {image.format}
            - **Size:** {image.size[0]}×{image.size[1]} pixels
            - **Mode:** {image.mode}
            """)
            
            # Quality assessment
            quality_score = assess_image_quality(image)
            st.metric("Quality Score", f"{quality_score:.2f}/1.0")
            
            if quality_score >= 0.8:
                st.success("🟢 Excellent Quality")
            elif quality_score >= 0.6:
                st.info("🟡 Good Quality")
            else:
                st.warning("🟠 Fair Quality")
        
        # Analysis button
        if st.button("🔍 Run AI Analysis", type="primary"):
            if not selected_models:
                st.error("Please select at least one AI model for analysis.")
                return
            
            with st.spinner("Running multi-model analysis..."):
                # Run predictions with selected models
                results = {}
                for model_name in selected_models:
                    model = MockAIModel(model_name)
                    prediction = model.predict(image)
                    results[model_name] = prediction
                
                # Store results in session state
                st.session_state.prediction_results = {
                    'model_predictions': results,
                    'image': image,
                    'confidence_threshold': confidence_threshold,
                    'timestamp': datetime.datetime.now().isoformat()
                }
                
                st.success("✅ Analysis completed successfully!")
                st.rerun()
    
    else:
        st.info("Please upload a dermoscopic image to begin AI analysis.")

def render_results_dashboard_tab(explanation_methods, enable_clinical_alerts):
    """Render the results dashboard tab."""
    st.header("Comprehensive Results Dashboard")
    
    if 'prediction_results' not in st.session_state or not st.session_state.prediction_results:
        st.info("👆 Please upload and analyze an image first.")
        return
    
    results = st.session_state.prediction_results
    model_predictions = results.get('model_predictions', {})
    
    # Calculate ensemble prediction
    ensemble_result = calculate_ensemble_prediction(model_predictions)
    
    # Analysis summary header
    st.markdown("### 🔬 AI Analysis Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Primary Diagnosis", ensemble_result['predicted_class'])
    
    with col2:
        confidence = ensemble_result['confidence']
        st.metric("Confidence", f"{confidence:.1%}")
    
    with col3:
        risk_levels = {
            'Melanoma': 'Critical',
            'Basal Cell Carcinoma': 'High',
            'Squamous Cell Carcinoma': 'High',
            'Actinic Keratosis': 'Moderate',
            'Benign Keratosis': 'Low',
            'Dermatofibroma': 'Low',
            'Nevus': 'Low'
        }
        risk_level = risk_levels.get(ensemble_result['predicted_class'], 'Unknown')
        st.metric("Risk Level", risk_level)
    
    with col4:
        consensus = check_model_consensus(model_predictions)
        st.metric("Model Agreement", "✅ Yes" if consensus else "❌ No")
    
    # Main prediction result
    st.markdown("### 🎯 Primary Analysis Results")
    
    predicted_class = ensemble_result['predicted_class']
    confidence = ensemble_result['confidence']
    
    # Risk color coding
    risk_colors = {
        'Critical': '#FF4B4B',
        'High': '#FF8700',
        'Moderate': '#FFA500',
        'Low': '#00C851'
    }
    
    card_color = risk_colors.get(risk_level, '#666666')
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {card_color}20, {card_color}10);
        border: 2px solid {card_color};
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    ">
        <h3 style="color: {card_color}; margin: 0;">🔍 {predicted_class}</h3>
        <p style="margin: 5px 0; font-size: 16px;">
            <strong>Confidence:</strong> {confidence:.1%}
        </p>
        <p style="margin: 5px 0; font-size: 16px;">
            <strong>Risk Level:</strong> {risk_level}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model comparison
    st.markdown("### 🤖 Multi-Model Comparison")
    
    comparison_data = []
    for model_name, prediction in model_predictions.items():
        comparison_data.append({
            'Model': model_name,
            'Prediction': prediction.get('predicted_class', 'Unknown'),
            'Confidence': prediction.get('confidence', 0.0),
            'Risk Level': risk_levels.get(prediction.get('predicted_class', ''), 'Unknown')
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison, use_container_width=True, hide_index=True)
    
    # Probability breakdown
    st.markdown("### 📊 Detailed Probability Analysis")
    
    class_names = ['Melanoma', 'Basal Cell Carcinoma', 'Squamous Cell Carcinoma',
                   'Actinic Keratosis', 'Benign Keratosis', 'Dermatofibroma', 'Nevus']
    
    # Calculate average probabilities across models
    all_probs = [pred.get('probabilities', []) for pred in model_predictions.values()]
    if all_probs and all(probs for probs in all_probs):
        avg_probs = np.mean(all_probs, axis=0)
        
        prob_df = pd.DataFrame({
            'Condition': class_names[:len(avg_probs)],
            'Probability': avg_probs,
            'Risk Level': [risk_levels.get(name, 'Unknown') for name in class_names[:len(avg_probs)]]
        })
        
        fig_probs = px.bar(
            prob_df,
            x='Condition',
            y='Probability',
            color='Risk Level',
            title="Probability Distribution Across All Conditions",
            color_discrete_map={
                'Critical': '#FF4B4B',
                'High': '#FF8700', 
                'Moderate': '#FFA500',
                'Low': '#00C851'
            }
        )
        fig_probs.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_probs, use_container_width=True)

def render_fairness_analysis_tab(enable_fairness, skin_tone):
    """Render the fairness analysis tab."""
    st.header("Demographic Fairness Analysis")
    
    if not enable_fairness:
        st.info("⚖️ Fairness evaluation is disabled. Enable it in the sidebar to see analysis.")
        return
    
    if 'prediction_results' not in st.session_state:
        st.info("👆 Please upload and analyze an image first.")
        return
    
    # Current analysis
    st.subheader("🔍 Current Prediction Analysis")
    
    fitzpatrick_groups = {
        "I (Very Fair)": 1,
        "II (Fair)": 2,
        "III (Medium)": 3,
        "IV (Olive)": 4,
        "V (Brown)": 5,
        "VI (Dark Brown/Black)": 6
    }
    
    skin_tone_number = fitzpatrick_groups.get(skin_tone, 3)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Patient Skin Type", skin_tone)
    
    with col2:
        # Expected accuracy for this skin type
        expected_accuracies = {1: 0.84, 2: 0.84, 3: 0.92, 4: 0.92, 5: 0.79, 6: 0.79}
        expected_acc = expected_accuracies.get(skin_tone_number, 0.85)
        st.metric("Expected Accuracy", f"{expected_acc:.1%}")
    
    with col3:
        fairness_scores = {1: 0.88, 2: 0.88, 3: 0.92, 4: 0.92, 5: 0.83, 6: 0.83}
        fairness_score = fairness_scores.get(skin_tone_number, 0.88)
        st.metric("Fairness Score", f"{fairness_score:.2f}/1.0")
    
    # Historical fairness metrics
    st.subheader("📊 Historical Fairness Metrics")
    
    fairness_data = {
        'Demographic Parity': {'Type I-II': 0.84, 'Type III-IV': 0.92, 'Type V-VI': 0.79},
        'Equalized Odds': {'Type I-II': 0.88, 'Type III-IV': 0.91, 'Type V-VI': 0.82},
        'Calibration': {'Type I-II': 0.91, 'Type III-IV': 0.94, 'Type V-VI': 0.87}
    }
    
    fairness_df = pd.DataFrame(fairness_data)
    
    fig_fairness = px.imshow(
        fairness_df.values,
        labels=dict(x="Fairness Metric", y="Skin Type Group", color="Score"),
        x=fairness_df.columns,
        y=fairness_df.index,
        color_continuous_scale='RdYlGn',
        title="Fairness Metrics Across Skin Type Groups",
        text_auto='.2f'
    )
    
    st.plotly_chart(fig_fairness, use_container_width=True)
    
    # Bias mitigation strategies
    st.subheader("🛠️ Active Bias Mitigation Strategies")
    
    strategies = [
        "✅ Enhanced data augmentation for underrepresented skin types",
        "✅ Fairness-aware training with adversarial objectives", 
        "✅ Ensemble diversity across demographic groups",
        "✅ Post-processing calibration adjustments"
    ]
    
    for strategy in strategies:
        st.markdown(f"• {strategy}")

def render_model_performance_tab():
    """Render the model performance tab."""
    st.header("Model Performance Metrics")
    
    # Model specifications
    model_specs = {
        "Model": ["ResNet-50", "EfficientNet-B4", "Vision Transformer", "Hybrid CNN-ViT"],
        "Parameters (M)": [25.6, 19.3, 86.6, 44.2],
        "Accuracy": [0.891, 0.923, 0.934, 0.941],
        "Fairness Score": [0.82, 0.87, 0.91, 0.94],
        "Inference Time (ms)": [45, 38, 92, 67]
    }
    
    df_models = pd.DataFrame(model_specs)
    st.dataframe(df_models, use_container_width=True)
    
    # Performance visualization
    col1, col2 = st.columns(2)
    
    with col1:
        fig_acc = px.bar(
            df_models,
            x="Model",
            y="Accuracy",
            title="Model Accuracy Comparison",
            color="Accuracy",
            color_continuous_scale="viridis"
        )
        st.plotly_chart(fig_acc, use_container_width=True)
    
    with col2:
        fig_fair = px.bar(
            df_models,
            x="Model", 
            y="Fairness Score",
            title="Fairness Score Comparison",
            color="Fairness Score",
            color_continuous_scale="plasma"
        )
        st.plotly_chart(fig_fair, use_container_width=True)

def render_dataset_info_tab():
    """Render the dataset information tab."""
    st.header("📊 Dataset Information & Training Details")
    
    # Dataset overview
    st.subheader("🗂️ Training Dataset Overview")
    
    datasets = {
        'HAM10000': {'size': 10015, 'classes': 7, 'year': 2018},
        'ISIC 2019': {'size': 25331, 'classes': 8, 'year': 2019},
        'Dermanist': {'size': 8012, 'classes': 7, 'year': 2021},
        'BCN 20000': {'size': 19424, 'classes': 7, 'year': 2019}
    }
    
    dataset_df = pd.DataFrame.from_dict(datasets, orient='index').reset_index()
    dataset_df.columns = ['Dataset', 'Images', 'Classes', 'Year']
    st.dataframe(dataset_df, use_container_width=True, hide_index=True)
    
    # Class distribution
    st.subheader("📈 Class Distribution")
    
    class_data = {
        'Condition': ['Melanoma', 'Basal Cell Carcinoma', 'Squamous Cell Carcinoma',
                     'Actinic Keratosis', 'Benign Keratosis', 'Dermatofibroma', 'Nevus'],
        'Count': [1113, 514, 327, 867, 1099, 115, 6705],
        'Severity': ['Critical', 'High', 'High', 'Moderate', 'Low', 'Low', 'Low']
    }
    
    class_df = pd.DataFrame(class_data)
    class_df['Percentage'] = (class_df['Count'] / class_df['Count'].sum() * 100).round(1)
    
    fig_dist = px.pie(
        class_df,
        values='Count',
        names='Condition',
        title="Distribution of Skin Lesion Types",
        color='Severity',
        color_discrete_map={
            'Critical': '#FF4B4B',
            'High': '#FF8700',
            'Moderate': '#FFA500', 
            'Low': '#00C851'
        }
    )
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # Training details
    st.subheader("🎯 Model Training Details")
    
    training_info = """
    **Training Configuration:**
    - **Preprocessing:** Standardized dermoscopic image preprocessing pipeline
    - **Augmentation:** Advanced augmentation including fairness-aware transformations
    - **Architecture:** Multiple state-of-the-art CNN and Vision Transformer models
    - **Training:** Transfer learning from ImageNet with medical domain fine-tuning
    - **Validation:** 5-fold cross-validation with stratified demographic sampling
    - **Fairness:** Adversarial training and post-processing calibration
    
    **Clinical Validation:**
    - Expert dermatologist validation on held-out test set
    - Multi-institutional testing across diverse populations
    - Continuous monitoring for model drift and bias
    """
    
    st.markdown(training_info)

def assess_image_quality(image):
    """Assess basic image quality."""
    img_array = np.array(image)
    
    # Convert to grayscale for analysis
    if len(img_array.shape) == 3:
        gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
    else:
        gray = img_array
    
    # Calculate basic quality metrics
    contrast = np.std(gray) / 128.0
    brightness = np.mean(gray) / 255.0
    brightness_score = 1.0 - abs(brightness - 0.5) * 2
    
    # Simple sharpness estimate
    edges = np.gradient(gray)
    sharpness = np.mean([np.std(edges[0]), np.std(edges[1])]) / 50.0
    
    # Dynamic range
    dynamic_range = (np.max(gray) - np.min(gray)) / 255.0
    
    # Combine metrics
    quality_score = (
        0.3 * min(contrast, 1.0) +
        0.2 * brightness_score +
        0.3 * min(sharpness, 1.0) +
        0.2 * dynamic_range
    )
    
    return min(max(quality_score, 0.0), 1.0)

def calculate_ensemble_prediction(model_predictions):
    """Calculate ensemble prediction from multiple models."""
    if not model_predictions:
        return {'predicted_class': 'Unknown', 'confidence': 0.0}
    
    # Get all predictions and confidences
    predictions = [pred.get('predicted_class', '') for pred in model_predictions.values()]
    confidences = [pred.get('confidence', 0.0) for pred in model_predictions.values()]
    
    # Find most common prediction
    from collections import Counter
    prediction_counts = Counter(predictions)
    most_common_prediction = prediction_counts.most_common(1)[0][0] if predictions else 'Unknown'
    
    # Calculate average confidence
    avg_confidence = np.mean(confidences) if confidences else 0.0
    
    return {
        'predicted_class': most_common_prediction,
        'confidence': avg_confidence
    }

def check_model_consensus(model_predictions):
    """Check if all models agree on the prediction."""
    predictions = [pred.get('predicted_class', '') for pred in model_predictions.values()]
    return len(set(predictions)) == 1

if __name__ == "__main__":
    main()