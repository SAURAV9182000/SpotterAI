import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ResultsDashboard:
    """Comprehensive results dashboard for AI skin cancer detection analysis."""
    
    def __init__(self):
        self.class_names = [
            'Melanoma', 'Basal Cell Carcinoma', 'Squamous Cell Carcinoma',
            'Actinic Keratosis', 'Benign Keratosis', 'Dermatofibroma', 'Nevus'
        ]
        
        self.risk_levels = {
            'Melanoma': {'level': 'Critical', 'color': '#FF4B4B', 'urgency': 'Immediate'},
            'Basal Cell Carcinoma': {'level': 'High', 'color': '#FF8700', 'urgency': 'Urgent'},
            'Squamous Cell Carcinoma': {'level': 'High', 'color': '#FF8700', 'urgency': 'Urgent'},
            'Actinic Keratosis': {'level': 'Moderate', 'color': '#FFA500', 'urgency': 'Routine'},
            'Benign Keratosis': {'level': 'Low', 'color': '#00C851', 'urgency': 'Routine'},
            'Dermatofibroma': {'level': 'Low', 'color': '#00C851', 'urgency': 'Routine'},
            'Nevus': {'level': 'Low', 'color': '#00C851', 'urgency': 'Routine'}
        }
    
    def render(self, prediction_results, explanation_methods=None, enable_clinical_alerts=True):
        """Render the complete results dashboard."""
        if not prediction_results:
            st.error("No prediction results available.")
            return
        
        try:
            # Analysis summary header
            self._render_analysis_header(prediction_results)
            
            # Main results section
            self._render_main_results(prediction_results)
            
            # Model ensemble analysis
            self._render_ensemble_analysis(prediction_results)
            
            # Confidence and uncertainty analysis
            self._render_confidence_analysis(prediction_results)
            
            # Explainability section
            if explanation_methods:
                self._render_explainability_section(prediction_results, explanation_methods)
            
            # Clinical decision support
            self._render_clinical_support(prediction_results, enable_clinical_alerts)
            
            # Detailed probability breakdown
            self._render_probability_breakdown(prediction_results)
            
            # Quality assessment
            self._render_quality_assessment(prediction_results)
            
        except Exception as e:
            logger.error(f"Dashboard rendering failed: {str(e)}")
            st.error(f"Dashboard rendering failed: {str(e)}")
    
    def _render_analysis_header(self, prediction_results):
        """Render analysis summary header."""
        st.markdown("### 🔬 AI Analysis Results")
        
        # Extract key information
        model_predictions = prediction_results.get('model_predictions', {})
        if not model_predictions:
            st.error("No model predictions found.")
            return
        
        # Calculate ensemble prediction
        ensemble_result = self._calculate_ensemble_prediction(model_predictions)
        
        # Analysis timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Header metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Primary Diagnosis",
                ensemble_result['predicted_class'],
                help="Most likely diagnosis based on ensemble analysis"
            )
        
        with col2:
            confidence = ensemble_result['confidence']
            confidence_color = "normal" if confidence >= 0.7 else "inverse"
            st.metric(
                "Confidence",
                f"{confidence:.1%}",
                delta=f"{'High' if confidence >= 0.8 else 'Moderate' if confidence >= 0.6 else 'Low'}",
                help="AI confidence in the primary diagnosis"
            )
        
        with col3:
            risk_info = self.risk_levels.get(ensemble_result['predicted_class'], {'level': 'Unknown'})
            st.metric(
                "Risk Level",
                risk_info['level'],
                delta=risk_info.get('urgency', 'Unknown'),
                help="Clinical risk level and urgency"
            )
        
        with col4:
            consensus = self._check_model_consensus(model_predictions)
            st.metric(
                "Model Agreement",
                "✅ Yes" if consensus else "❌ No",
                help="Whether all AI models agree on the diagnosis"
            )
        
        # Analysis details
        st.markdown(f"**Analysis Time:** {timestamp}")
        st.markdown(f"**Models Used:** {', '.join(model_predictions.keys())}")
    
    def _render_main_results(self, prediction_results):
        """Render main prediction results."""
        st.markdown("### 🎯 Primary Analysis Results")
        
        model_predictions = prediction_results.get('model_predictions', {})
        ensemble_result = self._calculate_ensemble_prediction(model_predictions)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Main prediction card
            predicted_class = ensemble_result['predicted_class']
            confidence = ensemble_result['confidence']
            risk_info = self.risk_levels.get(predicted_class, {'level': 'Unknown', 'color': '#666666'})
            
            # Create styled prediction card
            card_color = risk_info['color']
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
                    <strong>Risk Level:</strong> {risk_info['level']}
                </p>
                <p style="margin: 5px 0; font-size: 16px;">
                    <strong>Recommended Action:</strong> {risk_info.get('urgency', 'Unknown')} evaluation
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Clinical interpretation
            interpretation = self._generate_clinical_interpretation(predicted_class, confidence)
            st.markdown(f"**Clinical Interpretation:** {interpretation}")
        
        with col2:
            # Confidence gauge
            self._render_confidence_gauge(confidence)
            
            # Risk level indicator
            self._render_risk_indicator(risk_info)
    
    def _render_ensemble_analysis(self, prediction_results):
        """Render ensemble model analysis."""
        st.markdown("### 🤖 Multi-Model Ensemble Analysis")
        
        model_predictions = prediction_results.get('model_predictions', {})
        
        if len(model_predictions) < 2:
            st.info("Single model analysis - ensemble comparison not available.")
            return
        
        # Create comparison table
        comparison_data = []
        for model_name, prediction in model_predictions.items():
            comparison_data.append({
                'Model': model_name,
                'Prediction': prediction.get('predicted_class', 'Unknown'),
                'Confidence': prediction.get('confidence', 0.0),
                'Risk Level': self.risk_levels.get(
                    prediction.get('predicted_class', ''), {'level': 'Unknown'}
                )['level']
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.dataframe(
                df_comparison,
                use_container_width=True,
                hide_index=True
            )
        
        with col2:
            # Model agreement visualization
            predictions = [pred.get('predicted_class', '') for pred in model_predictions.values()]
            unique_predictions = list(set(predictions))
            
            if len(unique_predictions) == 1:
                st.success("🎯 **Perfect Model Consensus**")
                st.markdown(f"All models agree: **{unique_predictions[0]}**")
            else:
                st.warning("⚠️ **Model Disagreement Detected**")
                for pred in unique_predictions:
                    count = predictions.count(pred)
                    percentage = count / len(predictions) * 100
                    st.markdown(f"• {pred}: {count}/{len(predictions)} models ({percentage:.0f}%)")
        
        # Confidence distribution
        confidences = [pred.get('confidence', 0.0) for pred in model_predictions.values()]
        if confidences:
            fig_conf = px.box(
                y=confidences,
                title="Model Confidence Distribution",
                labels={'y': 'Confidence Score'}
            )
            fig_conf.update_layout(height=300)
            st.plotly_chart(fig_conf, use_container_width=True)
    
    def _render_confidence_analysis(self, prediction_results):
        """Render detailed confidence and uncertainty analysis."""
        st.markdown("### 📊 Confidence & Uncertainty Analysis")
        
        model_predictions = prediction_results.get('model_predictions', {})
        
        # Calculate uncertainty metrics
        all_probabilities = []
        confidences = []
        
        for prediction in model_predictions.values():
            probs = prediction.get('probabilities', [])
            conf = prediction.get('confidence', 0.0)
            
            if probs:
                all_probabilities.append(probs)
                confidences.append(conf)
        
        if not all_probabilities:
            st.error("No probability data available for uncertainty analysis.")
            return
        
        # Calculate ensemble statistics
        mean_probs = np.mean(all_probabilities, axis=0)
        std_probs = np.std(all_probabilities, axis=0)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Uncertainty heatmap
            uncertainty_data = {
                'Class': self.class_names[:len(mean_probs)],
                'Mean_Probability': mean_probs,
                'Uncertainty': std_probs
            }
            df_uncertainty = pd.DataFrame(uncertainty_data)
            
            fig_uncertainty = px.scatter(
                df_uncertainty,
                x='Mean_Probability',
                y='Uncertainty',
                text='Class',
                title="Prediction Uncertainty vs Mean Probability",
                labels={
                    'Mean_Probability': 'Mean Probability',
                    'Uncertainty': 'Standard Deviation (Uncertainty)'
                }
            )
            fig_uncertainty.update_traces(textposition="top center")
            fig_uncertainty.update_layout(height=400)
            st.plotly_chart(fig_uncertainty, use_container_width=True)
        
        with col2:
            # Confidence statistics
            if confidences:
                conf_stats = {
                    'Mean Confidence': np.mean(confidences),
                    'Std Confidence': np.std(confidences),
                    'Min Confidence': np.min(confidences),
                    'Max Confidence': np.max(confidences)
                }
                
                st.markdown("**Confidence Statistics:**")
                for stat, value in conf_stats.items():
                    st.metric(stat, f"{value:.3f}")
            
            # Uncertainty interpretation
            max_uncertainty_idx = np.argmax(std_probs)
            max_uncertainty_class = self.class_names[max_uncertainty_idx] if max_uncertainty_idx < len(self.class_names) else "Unknown"
            
            st.markdown("**Uncertainty Analysis:**")
            st.markdown(f"• Highest uncertainty: **{max_uncertainty_class}**")
            st.markdown(f"• Average uncertainty: **{np.mean(std_probs):.3f}**")
            
            if np.mean(std_probs) > 0.1:
                st.warning("⚠️ High model disagreement detected")
            else:
                st.success("✅ Low uncertainty - consistent predictions")
    
    def _render_explainability_section(self, prediction_results, explanation_methods):
        """Render explainability analysis section."""
        st.markdown("### 🔍 Model Explainability")
        
        # Mock explainability for demonstration (would use actual explainability analyzer)
        st.info("🔧 Explainability analysis would be integrated here with Grad-CAM, SHAP, and LIME visualizations.")
        
        col1, col2, col3 = st.columns(3)
        
        explanations_available = []
        for method in explanation_methods:
            if method == 'grad_cam':
                explanations_available.append('Grad-CAM')
            elif method == 'shap':
                explanations_available.append('SHAP')
            elif method == 'lime':
                explanations_available.append('LIME')
        
        with col1:
            st.markdown("**Available Explanations:**")
            for explanation in explanations_available:
                st.markdown(f"• ✅ {explanation}")
        
        with col2:
            st.markdown("**Key Findings:**")
            st.markdown("• Model focuses on lesion center")
            st.markdown("• Border irregularity detected")
            st.markdown("• Color variation noted")
        
        with col3:
            st.markdown("**Clinical Relevance:**")
            st.markdown("• ABCDE criteria alignment")
            st.markdown("• Dermoscopic pattern recognition")
            st.markdown("• Expert-validated features")
        
        # Placeholder for actual explainability visualizations
        if st.button("🔍 Generate Detailed Explanations"):
            st.info("This would trigger detailed explainability analysis using the selected methods.")
    
    def _render_clinical_support(self, prediction_results, enable_clinical_alerts):
        """Render clinical decision support section."""
        st.markdown("### 🏥 Clinical Decision Support")
        
        if not enable_clinical_alerts:
            st.info("Clinical alerts are disabled.")
            return
        
        # Get ensemble prediction for clinical analysis
        model_predictions = prediction_results.get('model_predictions', {})
        ensemble_result = self._calculate_ensemble_prediction(model_predictions)
        
        predicted_class = ensemble_result['predicted_class']
        confidence = ensemble_result['confidence']
        
        # Generate clinical recommendations
        recommendations = self._generate_clinical_recommendations(predicted_class, confidence)
        
        # Clinical alerts
        alerts = self._generate_clinical_alerts(predicted_class, confidence)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🚨 Clinical Alerts:**")
            for alert in alerts:
                if alert['level'] == 'critical':
                    st.error(f"🔴 **CRITICAL:** {alert['message']}")
                elif alert['level'] == 'warning':
                    st.warning(f"🟠 **WARNING:** {alert['message']}")
                else:
                    st.info(f"🔵 **INFO:** {alert['message']}")
        
        with col2:
            st.markdown("**📋 Recommendations:**")
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"{i}. {rec}")
        
        # Follow-up guidance
        follow_up = self._generate_follow_up_guidance(predicted_class, confidence)
        
        st.markdown("**📅 Follow-up Guidance:**")
        st.markdown(f"• **Timeline:** {follow_up['timeline']}")
        st.markdown(f"• **Provider:** {follow_up['provider']}")
        st.markdown(f"• **Priority:** {follow_up['priority']}")
        
        # Patient communication
        patient_message = self._generate_patient_communication(predicted_class, confidence)
        
        with st.expander("💬 Patient Communication Template"):
            st.markdown(patient_message)
    
    def _render_probability_breakdown(self, prediction_results):
        """Render detailed probability breakdown."""
        st.markdown("### 📈 Detailed Probability Analysis")
        
        model_predictions = prediction_results.get('model_predictions', {})
        
        # Create probability matrix
        prob_data = []
        for model_name, prediction in model_predictions.items():
            probs = prediction.get('probabilities', [])
            if probs:
                for i, prob in enumerate(probs):
                    if i < len(self.class_names):
                        prob_data.append({
                            'Model': model_name,
                            'Class': self.class_names[i],
                            'Probability': prob
                        })
        
        if not prob_data:
            st.error("No probability data available.")
            return
        
        df_probs = pd.DataFrame(prob_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Heatmap of probabilities
            pivot_probs = df_probs.pivot(index='Model', columns='Class', values='Probability')
            
            fig_heatmap = px.imshow(
                pivot_probs.values,
                labels=dict(x="Condition", y="Model", color="Probability"),
                x=pivot_probs.columns,
                y=pivot_probs.index,
                color_continuous_scale='viridis',
                title="Model Probability Heatmap"
            )
            fig_heatmap.update_layout(height=400)
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        with col2:
            # Top predictions across models
            ensemble_probs = df_probs.groupby('Class')['Probability'].mean().sort_values(ascending=False)
            
            fig_ensemble = px.bar(
                x=ensemble_probs.index,
                y=ensemble_probs.values,
                title="Ensemble Average Probabilities",
                labels={'x': 'Condition', 'y': 'Average Probability'}
            )
            fig_ensemble.update_layout(height=400)
            st.plotly_chart(fig_ensemble, use_container_width=True)
        
        # Probability statistics table
        prob_stats = df_probs.groupby('Class').agg({
            'Probability': ['mean', 'std', 'min', 'max']
        }).round(3)
        prob_stats.columns = ['Mean', 'Std Dev', 'Min', 'Max']
        
        st.markdown("**Probability Statistics by Condition:**")
        st.dataframe(prob_stats, use_container_width=True)
    
    def _render_quality_assessment(self, prediction_results):
        """Render analysis quality assessment."""
        st.markdown("### ✅ Analysis Quality Assessment")
        
        model_predictions = prediction_results.get('model_predictions', {})
        
        # Quality metrics
        quality_score = self._calculate_quality_score(model_predictions)
        reliability_score = self._calculate_reliability_score(model_predictions)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Quality Score",
                f"{quality_score:.2f}/1.0",
                delta="Overall analysis quality",
                help="Composite score based on confidence, consensus, and uncertainty"
            )
        
        with col2:
            st.metric(
                "Reliability Score",
                f"{reliability_score:.2f}/1.0",
                delta="Prediction reliability",
                help="Model agreement and confidence consistency"
            )
        
        with col3:
            num_models = len(model_predictions)
            st.metric(
                "Model Coverage",
                f"{num_models}/4",
                delta="Available models",
                help="Number of AI models used in analysis"
            )
        
        # Quality indicators
        quality_indicators = self._assess_quality_indicators(model_predictions)
        
        st.markdown("**Quality Indicators:**")
        for indicator in quality_indicators:
            if indicator['status'] == 'good':
                st.success(f"✅ {indicator['message']}")
            elif indicator['status'] == 'warning':
                st.warning(f"⚠️ {indicator['message']}")
            else:
                st.error(f"❌ {indicator['message']}")
    
    def _calculate_ensemble_prediction(self, model_predictions):
        """Calculate ensemble prediction from multiple models."""
        if not model_predictions:
            return {'predicted_class': 'Unknown', 'confidence': 0.0}
        
        # Get all probabilities
        all_probs = []
        for prediction in model_predictions.values():
            probs = prediction.get('probabilities', [])
            if probs:
                all_probs.append(probs)
        
        if not all_probs:
            # Fallback to first prediction
            first_pred = list(model_predictions.values())[0]
            return {
                'predicted_class': first_pred.get('predicted_class', 'Unknown'),
                'confidence': first_pred.get('confidence', 0.0)
            }
        
        # Calculate ensemble average
        ensemble_probs = np.mean(all_probs, axis=0)
        predicted_idx = np.argmax(ensemble_probs)
        
        predicted_class = self.class_names[predicted_idx] if predicted_idx < len(self.class_names) else 'Unknown'
        confidence = float(ensemble_probs[predicted_idx])
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': ensemble_probs.tolist()
        }
    
    def _check_model_consensus(self, model_predictions):
        """Check if all models agree on the prediction."""
        predictions = [pred.get('predicted_class', '') for pred in model_predictions.values()]
        return len(set(predictions)) == 1
    
    def _render_confidence_gauge(self, confidence):
        """Render confidence gauge visualization."""
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = confidence,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Confidence"},
            delta = {'reference': 0.8},
            gauge = {
                'axis': {'range': [None, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 0.5], 'color': "lightgray"},
                    {'range': [0.5, 0.7], 'color': "yellow"},
                    {'range': [0.7, 1], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.9
                }
            }
        ))
        
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    def _render_risk_indicator(self, risk_info):
        """Render risk level indicator."""
        risk_level = risk_info['level']
        color = risk_info['color']
        
        st.markdown(f"""
        <div style="
            background-color: {color}20;
            border: 2px solid {color};
            border-radius: 10px;
            padding: 15px;
            text-align: center;
            margin: 10px 0;
        ">
            <h4 style="color: {color}; margin: 0;">Risk Level</h4>
            <h2 style="color: {color}; margin: 5px 0;">{risk_level}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    def _generate_clinical_interpretation(self, predicted_class, confidence):
        """Generate clinical interpretation of the prediction."""
        interpretations = {
            'Melanoma': f"Possible melanoma detected with {confidence:.1%} confidence. Immediate dermatological evaluation recommended.",
            'Basal Cell Carcinoma': f"Findings suggest basal cell carcinoma with {confidence:.1%} confidence. Dermatological consultation advised.",
            'Squamous Cell Carcinoma': f"Features consistent with squamous cell carcinoma with {confidence:.1%} confidence. Medical evaluation recommended.",
            'Actinic Keratosis': f"Actinic keratosis suspected with {confidence:.1%} confidence. Consider treatment and monitoring.",
            'Benign Keratosis': f"Benign keratosis likely with {confidence:.1%} confidence. Routine monitoring recommended.",
            'Dermatofibroma': f"Findings suggest dermatofibroma with {confidence:.1%} confidence. Benign lesion, routine follow-up.",
            'Nevus': f"Benign nevus most likely with {confidence:.1%} confidence. Regular self-examination recommended."
        }
        
        return interpretations.get(predicted_class, f"Analysis suggests {predicted_class} with {confidence:.1%} confidence.")
    
    def _generate_clinical_recommendations(self, predicted_class, confidence):
        """Generate clinical recommendations."""
        base_recommendations = {
            'Melanoma': [
                "URGENT: Immediate dermatology/oncology referral required",
                "Avoid sun exposure and use broad-spectrum sunscreen",
                "Document lesion characteristics and changes",
                "Consider dermoscopy and biopsy"
            ],
            'Basal Cell Carcinoma': [
                "Dermatological evaluation within 2-4 weeks",
                "Protect lesion from sun exposure",
                "Monitor for changes in size or appearance",
                "Discuss treatment options with specialist"
            ],
            'Squamous Cell Carcinoma': [
                "Dermatological consultation recommended",
                "Sun protection and wound care if ulcerated",
                "Monitor for rapid growth or changes",
                "Consider staging if large or aggressive features"
            ]
        }
        
        recommendations = base_recommendations.get(predicted_class, [
            "Monitor lesion for changes",
            "Practice sun safety measures",
            "Regular skin self-examinations",
            "Consult healthcare provider if concerned"
        ])
        
        if confidence < 0.7:
            recommendations.insert(0, "Low AI confidence - expert review recommended")
        
        return recommendations
    
    def _generate_clinical_alerts(self, predicted_class, confidence):
        """Generate clinical alerts."""
        alerts = []
        
        if predicted_class == 'Melanoma':
            alerts.append({
                'level': 'critical',
                'message': 'Potential melanoma detected - immediate specialist consultation required'
            })
        
        if confidence < 0.6:
            alerts.append({
                'level': 'warning',
                'message': f'Low AI confidence ({confidence:.1%}) - manual review recommended'
            })
        
        if confidence >= 0.9:
            alerts.append({
                'level': 'info',
                'message': f'High confidence prediction ({confidence:.1%})'
            })
        
        return alerts
    
    def _generate_follow_up_guidance(self, predicted_class, confidence):
        """Generate follow-up guidance."""
        if predicted_class == 'Melanoma':
            return {
                'timeline': 'Immediate (1-2 days)',
                'provider': 'Dermatologist/Oncologist',
                'priority': 'Critical'
            }
        elif predicted_class in ['Basal Cell Carcinoma', 'Squamous Cell Carcinoma']:
            return {
                'timeline': '2-4 weeks',
                'provider': 'Dermatologist',
                'priority': 'High'
            }
        else:
            return {
                'timeline': '3-6 months',
                'provider': 'Primary Care or Dermatologist',
                'priority': 'Routine'
            }
    
    def _generate_patient_communication(self, predicted_class, confidence):
        """Generate patient communication template."""
        return f"""
**AI Skin Analysis Results**

Dear Patient,

Our AI analysis has reviewed your skin lesion image. Here are the key findings:

**Analysis Result:** The AI system suggests this lesion is most consistent with {predicted_class}.

**Confidence Level:** The AI analysis has {confidence:.1%} confidence in this assessment.

**What This Means:** {self._get_patient_friendly_explanation(predicted_class)}

**Next Steps:** {self._get_patient_next_steps(predicted_class)}

**Important Reminder:** This AI analysis is a tool to assist your healthcare provider and should not replace professional medical evaluation. Please discuss these results with your doctor.

**Questions?** Contact your healthcare provider for any concerns or questions about these results.
        """
    
    def _get_patient_friendly_explanation(self, predicted_class):
        """Get patient-friendly explanation of the condition."""
        explanations = {
            'Melanoma': 'This is a type of skin cancer that requires immediate medical attention.',
            'Basal Cell Carcinoma': 'This is a common form of skin cancer that is usually slow-growing.',
            'Squamous Cell Carcinoma': 'This is a type of skin cancer that should be evaluated by a specialist.',
            'Actinic Keratosis': 'This is a precancerous condition caused by sun damage.',
            'Benign Keratosis': 'This appears to be a non-cancerous skin growth.',
            'Dermatofibroma': 'This appears to be a benign (non-cancerous) skin nodule.',
            'Nevus': 'This appears to be a benign mole.'
        }
        return explanations.get(predicted_class, 'Please discuss this finding with your healthcare provider.')
    
    def _get_patient_next_steps(self, predicted_class):
        """Get patient-friendly next steps."""
        if predicted_class == 'Melanoma':
            return 'Schedule an immediate appointment with a dermatologist or oncologist.'
        elif predicted_class in ['Basal Cell Carcinoma', 'Squamous Cell Carcinoma']:
            return 'Schedule an appointment with a dermatologist within the next few weeks.'
        else:
            return 'Discuss these results with your healthcare provider and follow their recommendations for monitoring.'
    
    def _calculate_quality_score(self, model_predictions):
        """Calculate overall quality score."""
        if not model_predictions:
            return 0.0
        
        confidences = [pred.get('confidence', 0.0) for pred in model_predictions.values()]
        consensus = self._check_model_consensus(model_predictions)
        
        avg_confidence = np.mean(confidences)
        consensus_bonus = 0.1 if consensus else 0
        
        return min(avg_confidence + consensus_bonus, 1.0)
    
    def _calculate_reliability_score(self, model_predictions):
        """Calculate reliability score."""
        if not model_predictions:
            return 0.0
        
        confidences = [pred.get('confidence', 0.0) for pred in model_predictions.values()]
        confidence_std = np.std(confidences)
        
        # Lower standard deviation means higher reliability
        reliability = 1.0 - min(confidence_std * 2, 1.0)
        
        return max(reliability, 0.0)
    
    def _assess_quality_indicators(self, model_predictions):
        """Assess various quality indicators."""
        indicators = []
        
        if not model_predictions:
            indicators.append({
                'status': 'error',
                'message': 'No model predictions available'
            })
            return indicators
        
        # Check number of models
        num_models = len(model_predictions)
        if num_models >= 3:
            indicators.append({
                'status': 'good',
                'message': f'Multiple models used ({num_models}) for robust analysis'
            })
        else:
            indicators.append({
                'status': 'warning',
                'message': f'Limited models used ({num_models}) - consider ensemble approach'
            })
        
        # Check consensus
        consensus = self._check_model_consensus(model_predictions)
        if consensus:
            indicators.append({
                'status': 'good',
                'message': 'All models agree on the diagnosis'
            })
        else:
            indicators.append({
                'status': 'warning',
                'message': 'Models show disagreement - requires careful interpretation'
            })
        
        # Check confidence levels
        confidences = [pred.get('confidence', 0.0) for pred in model_predictions.values()]
        avg_confidence = np.mean(confidences)
        
        if avg_confidence >= 0.8:
            indicators.append({
                'status': 'good',
                'message': f'High average confidence ({avg_confidence:.1%})'
            })
        elif avg_confidence >= 0.6:
            indicators.append({
                'status': 'warning',
                'message': f'Moderate confidence ({avg_confidence:.1%}) - consider additional analysis'
            })
        else:
            indicators.append({
                'status': 'error',
                'message': f'Low confidence ({avg_confidence:.1%}) - expert review required'
            })
        
        return indicators
