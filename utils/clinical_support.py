import numpy as np
import pandas as pd
from datetime import datetime
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class ClinicalDecisionSupport:
    """Advanced clinical decision support system for skin cancer detection."""
    
    def __init__(self):
        self.confidence_thresholds = {
            'high_confidence': 0.9,
            'moderate_confidence': 0.7,
            'low_confidence': 0.5
        }
        
        self.risk_categories = {
            'Melanoma': {'risk_level': 'critical', 'urgency': 'immediate'},
            'Basal Cell Carcinoma': {'risk_level': 'high', 'urgency': 'urgent'},
            'Squamous Cell Carcinoma': {'risk_level': 'high', 'urgency': 'urgent'},
            'Actinic Keratosis': {'risk_level': 'moderate', 'urgency': 'routine'},
            'Benign Keratosis': {'risk_level': 'low', 'urgency': 'routine'},
            'Dermatofibroma': {'risk_level': 'low', 'urgency': 'routine'},
            'Nevus': {'risk_level': 'low', 'urgency': 'routine'}
        }
        
        self.demographic_risk_factors = {
            'age_high_risk': 50,
            'skin_type_high_risk': [1, 2],  # Fitzpatrick I-II
            'family_history_multiplier': 1.5
        }
    
    def generate_clinical_report(self, prediction_results: Dict, patient_info: Optional[Dict] = None) -> Dict:
        """
        Generate comprehensive clinical decision support report.
        
        Args:
            prediction_results: Results from AI model predictions
            patient_info: Optional patient demographic information
            
        Returns:
            Comprehensive clinical report with recommendations
        """
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'ai_analysis': self._analyze_ai_predictions(prediction_results),
                'risk_assessment': self._assess_clinical_risk(prediction_results, patient_info),
                'recommendations': self._generate_recommendations(prediction_results, patient_info),
                'follow_up': self._determine_follow_up(prediction_results, patient_info),
                'differential_diagnosis': self._suggest_differential_diagnosis(prediction_results),
                'quality_indicators': self._assess_prediction_quality(prediction_results),
                'alerts': self._generate_clinical_alerts(prediction_results, patient_info)
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Clinical report generation failed: {str(e)}")
            return self._generate_error_report(str(e))
    
    def _analyze_ai_predictions(self, prediction_results: Dict) -> Dict:
        """Analyze AI model predictions for clinical relevance."""
        try:
            model_predictions = prediction_results.get('model_predictions', {})
            
            if not model_predictions:
                return {'error': 'No model predictions available'}
            
            # Aggregate predictions across models
            all_probabilities = []
            all_predicted_classes = []
            confidence_scores = []
            
            for model_name, prediction in model_predictions.items():
                all_probabilities.append(prediction.get('probabilities', []))
                all_predicted_classes.append(prediction.get('predicted_class', ''))
                confidence_scores.append(prediction.get('confidence', 0.0))
            
            # Calculate ensemble statistics
            if all_probabilities:
                avg_probabilities = np.mean(all_probabilities, axis=0)
                std_probabilities = np.std(all_probabilities, axis=0)
                ensemble_prediction_idx = np.argmax(avg_probabilities)
                
                # Determine class names
                class_names = list(self.risk_categories.keys())
                if ensemble_prediction_idx < len(class_names):
                    ensemble_predicted_class = class_names[ensemble_prediction_idx]
                else:
                    ensemble_predicted_class = all_predicted_classes[0] if all_predicted_classes else 'Unknown'
            else:
                ensemble_predicted_class = 'Unknown'
                avg_probabilities = []
                std_probabilities = []
            
            # Model consensus analysis
            unique_predictions = set(all_predicted_classes)
            consensus = len(unique_predictions) == 1
            
            analysis = {
                'ensemble_prediction': ensemble_predicted_class,
                'ensemble_confidence': float(np.mean(confidence_scores)) if confidence_scores else 0.0,
                'model_consensus': consensus,
                'prediction_uncertainty': float(np.mean(std_probabilities)) if len(std_probabilities) > 0 else 0.0,
                'individual_models': {
                    model: {
                        'prediction': pred.get('predicted_class', 'Unknown'),
                        'confidence': pred.get('confidence', 0.0)
                    }
                    for model, pred in model_predictions.items()
                },
                'confidence_distribution': confidence_scores
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"AI prediction analysis failed: {str(e)}")
            return {'error': f'Analysis failed: {str(e)}'}
    
    def _assess_clinical_risk(self, prediction_results: Dict, patient_info: Optional[Dict]) -> Dict:
        """Assess overall clinical risk based on AI predictions and patient factors."""
        try:
            ai_analysis = self._analyze_ai_predictions(prediction_results)
            predicted_class = ai_analysis.get('ensemble_prediction', 'Unknown')
            confidence = ai_analysis.get('ensemble_confidence', 0.0)
            
            # Base risk from AI prediction
            base_risk = self.risk_categories.get(predicted_class, {
                'risk_level': 'unknown',
                'urgency': 'routine'
            })
            
            # Risk modifiers based on patient demographics
            risk_multiplier = 1.0
            risk_factors = []
            
            if patient_info:
                # Age factor
                age = patient_info.get('age', 0)
                if age >= self.demographic_risk_factors['age_high_risk']:
                    risk_multiplier *= 1.2
                    risk_factors.append(f'Age ≥{self.demographic_risk_factors["age_high_risk"]} years')
                
                # Skin type factor
                skin_tone = patient_info.get('skin_tone', 3)
                if skin_tone in self.demographic_risk_factors['skin_type_high_risk']:
                    risk_multiplier *= 1.3
                    risk_factors.append('High-risk skin type (Fitzpatrick I-II)')
                
                # Family history
                family_history = patient_info.get('family_history_melanoma', False)
                if family_history:
                    risk_multiplier *= self.demographic_risk_factors['family_history_multiplier']
                    risk_factors.append('Family history of melanoma')
            
            # Confidence-based risk adjustment
            if confidence < self.confidence_thresholds['low_confidence']:
                risk_factors.append('Low AI confidence - requires expert review')
            
            # Calculate adjusted risk score
            base_risk_scores = {
                'critical': 5,
                'high': 4,
                'moderate': 3,
                'low': 2,
                'unknown': 1
            }
            
            base_score = base_risk_scores.get(base_risk['risk_level'], 1)
            adjusted_score = min(base_score * risk_multiplier, 5)
            
            # Determine final risk level
            if adjusted_score >= 4.5:
                final_risk_level = 'critical'
            elif adjusted_score >= 3.5:
                final_risk_level = 'high'
            elif adjusted_score >= 2.5:
                final_risk_level = 'moderate'
            else:
                final_risk_level = 'low'
            
            risk_assessment = {
                'predicted_condition': predicted_class,
                'base_risk_level': base_risk['risk_level'],
                'final_risk_level': final_risk_level,
                'risk_score': round(adjusted_score, 2),
                'risk_multiplier': round(risk_multiplier, 2),
                'risk_factors': risk_factors,
                'urgency': base_risk['urgency'],
                'confidence_in_assessment': confidence
            }
            
            return risk_assessment
            
        except Exception as e:
            logger.error(f"Risk assessment failed: {str(e)}")
            return {'error': f'Risk assessment failed: {str(e)}'}
    
    def _generate_recommendations(self, prediction_results: Dict, patient_info: Optional[Dict]) -> List[Dict]:
        """Generate specific clinical recommendations."""
        try:
            recommendations = []
            
            ai_analysis = self._analyze_ai_predictions(prediction_results)
            risk_assessment = self._assess_clinical_risk(prediction_results, patient_info)
            
            predicted_class = ai_analysis.get('ensemble_prediction', 'Unknown')
            confidence = ai_analysis.get('ensemble_confidence', 0.0)
            risk_level = risk_assessment.get('final_risk_level', 'unknown')
            
            # Confidence-based recommendations
            if confidence < self.confidence_thresholds['low_confidence']:
                recommendations.append({
                    'type': 'urgent',
                    'category': 'AI Uncertainty',
                    'recommendation': 'Low AI confidence detected. Immediate dermatologist consultation recommended.',
                    'priority': 'high',
                    'rationale': f'AI confidence of {confidence:.1%} is below clinical threshold of {self.confidence_thresholds["low_confidence"]:.1%}'
                })
            
            # Condition-specific recommendations
            if predicted_class == 'Melanoma':
                recommendations.extend([
                    {
                        'type': 'immediate',
                        'category': 'Specialist Referral',
                        'recommendation': 'URGENT: Immediate dermatology/oncology referral required',
                        'priority': 'critical',
                        'rationale': 'Potential melanoma detection requires immediate specialist evaluation'
                    },
                    {
                        'type': 'diagnostic',
                        'category': 'Further Testing',
                        'recommendation': 'Dermoscopy and possible biopsy indicated',
                        'priority': 'high',
                        'rationale': 'Histopathological confirmation required for melanoma diagnosis'
                    }
                ])
            
            elif predicted_class in ['Basal Cell Carcinoma', 'Squamous Cell Carcinoma']:
                recommendations.extend([
                    {
                        'type': 'routine',
                        'category': 'Specialist Referral',
                        'recommendation': 'Dermatology referral within 2-4 weeks',
                        'priority': 'high',
                        'rationale': 'Non-melanoma skin cancer requires specialist evaluation and treatment planning'
                    },
                    {
                        'type': 'diagnostic',
                        'category': 'Monitoring',
                        'recommendation': 'Monitor for changes in size, color, or texture',
                        'priority': 'medium',
                        'rationale': 'Early detection of progression is crucial for optimal outcomes'
                    }
                ])
            
            elif predicted_class == 'Actinic Keratosis':
                recommendations.append({
                    'type': 'routine',
                    'category': 'Treatment',
                    'recommendation': 'Consider topical treatment or dermatology consultation',
                    'priority': 'medium',
                    'rationale': 'Actinic keratosis is a precancerous condition requiring treatment'
                })
            
            else:  # Benign conditions
                recommendations.append({
                    'type': 'routine',
                    'category': 'Monitoring',
                    'recommendation': 'Routine monitoring recommended. Seek evaluation if changes occur.',
                    'priority': 'low',
                    'rationale': 'Benign lesion with low malignant potential'
                })
            
            # Model consensus recommendations
            if not ai_analysis.get('model_consensus', True):
                recommendations.append({
                    'type': 'diagnostic',
                    'category': 'AI Discordance',
                    'recommendation': 'Multiple AI models show disagreement. Expert review recommended.',
                    'priority': 'medium',
                    'rationale': 'Lack of model consensus suggests diagnostic uncertainty'
                })
            
            # Risk-based recommendations
            if risk_level in ['critical', 'high']:
                recommendations.append({
                    'type': 'preventive',
                    'category': 'Risk Management',
                    'recommendation': 'Enhanced skin surveillance and sun protection counseling',
                    'priority': 'medium',
                    'rationale': 'High-risk patient profile requires proactive monitoring'
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {str(e)}")
            return [{'error': f'Recommendation generation failed: {str(e)}'}]
    
    def _determine_follow_up(self, prediction_results: Dict, patient_info: Optional[Dict]) -> Dict:
        """Determine appropriate follow-up schedule."""
        try:
            ai_analysis = self._analyze_ai_predictions(prediction_results)
            risk_assessment = self._assess_clinical_risk(prediction_results, patient_info)
            
            predicted_class = ai_analysis.get('ensemble_prediction', 'Unknown')
            risk_level = risk_assessment.get('final_risk_level', 'low')
            confidence = ai_analysis.get('ensemble_confidence', 0.0)
            
            # Determine follow-up timeline
            if predicted_class == 'Melanoma' or risk_level == 'critical':
                follow_up = {
                    'timeline': 'immediate',
                    'interval': '1-2 days',
                    'provider': 'Dermatologist/Oncologist',
                    'urgency': 'critical'
                }
            elif predicted_class in ['Basal Cell Carcinoma', 'Squamous Cell Carcinoma']:
                follow_up = {
                    'timeline': 'urgent',
                    'interval': '2-4 weeks',
                    'provider': 'Dermatologist',
                    'urgency': 'high'
                }
            elif confidence < self.confidence_thresholds['moderate_confidence']:
                follow_up = {
                    'timeline': 'urgent',
                    'interval': '1-2 weeks',
                    'provider': 'Dermatologist',
                    'urgency': 'high'
                }
            else:
                follow_up = {
                    'timeline': 'routine',
                    'interval': '3-6 months',
                    'provider': 'Primary Care or Dermatologist',
                    'urgency': 'routine'
                }
            
            # Add specific follow-up instructions
            follow_up['instructions'] = self._generate_follow_up_instructions(predicted_class, risk_level)
            
            return follow_up
            
        except Exception as e:
            logger.error(f"Follow-up determination failed: {str(e)}")
            return {'error': f'Follow-up determination failed: {str(e)}'}
    
    def _generate_follow_up_instructions(self, predicted_class: str, risk_level: str) -> List[str]:
        """Generate specific follow-up instructions."""
        instructions = []
        
        # General instructions
        instructions.extend([
            "Monitor lesion for any changes in size, color, shape, or texture",
            "Photograph lesion for comparison at follow-up visits",
            "Seek immediate medical attention if rapid changes occur"
        ])
        
        # Condition-specific instructions
        if predicted_class == 'Melanoma':
            instructions.extend([
                "Avoid sun exposure and use broad-spectrum SPF 30+ sunscreen",
                "Perform monthly self-skin examinations",
                "Keep appointment with specialist - do not delay"
            ])
        elif predicted_class in ['Basal Cell Carcinoma', 'Squamous Cell Carcinoma']:
            instructions.extend([
                "Protect area from sun exposure",
                "Avoid picking or irritating the lesion"
            ])
        elif risk_level in ['high', 'critical']:
            instructions.extend([
                "Enhanced sun protection measures",
                "Consider genetic counseling if family history present"
            ])
        
        return instructions
    
    def _suggest_differential_diagnosis(self, prediction_results: Dict) -> List[Dict]:
        """Suggest differential diagnoses based on AI predictions."""
        try:
            ai_analysis = self._analyze_ai_predictions(prediction_results)
            model_predictions = prediction_results.get('model_predictions', {})
            
            # Collect all predictions and confidences
            all_predictions = []
            for model_name, prediction in model_predictions.items():
                probabilities = prediction.get('probabilities', [])
                if probabilities:
                    class_names = list(self.risk_categories.keys())
                    for i, prob in enumerate(probabilities):
                        if i < len(class_names) and prob > 0.1:  # Only include significant probabilities
                            all_predictions.append({
                                'condition': class_names[i],
                                'probability': prob,
                                'source_model': model_name
                            })
            
            # Group by condition and calculate average probability
            condition_probs = {}
            for pred in all_predictions:
                condition = pred['condition']
                if condition not in condition_probs:
                    condition_probs[condition] = []
                condition_probs[condition].append(pred['probability'])
            
            # Calculate differential diagnosis list
            differential = []
            for condition, probs in condition_probs.items():
                avg_prob = np.mean(probs)
                if avg_prob > 0.05:  # Minimum threshold for inclusion
                    differential.append({
                        'condition': condition,
                        'probability': round(avg_prob, 3),
                        'likelihood': self._categorize_likelihood(avg_prob),
                        'clinical_significance': self.risk_categories.get(condition, {}).get('risk_level', 'unknown')
                    })
            
            # Sort by probability
            differential.sort(key=lambda x: x['probability'], reverse=True)
            
            return differential[:5]  # Top 5 differential diagnoses
            
        except Exception as e:
            logger.error(f"Differential diagnosis generation failed: {str(e)}")
            return []
    
    def _categorize_likelihood(self, probability: float) -> str:
        """Categorize probability into likelihood terms."""
        if probability >= 0.7:
            return 'Highly Likely'
        elif probability >= 0.4:
            return 'Possible'
        elif probability >= 0.2:
            return 'Less Likely'
        else:
            return 'Unlikely'
    
    def _assess_prediction_quality(self, prediction_results: Dict) -> Dict:
        """Assess the quality and reliability of AI predictions."""
        try:
            ai_analysis = self._analyze_ai_predictions(prediction_results)
            
            quality_indicators = {
                'model_agreement': ai_analysis.get('model_consensus', False),
                'prediction_confidence': ai_analysis.get('ensemble_confidence', 0.0),
                'uncertainty_level': ai_analysis.get('prediction_uncertainty', 0.0),
                'reliability_score': 0.0,
                'quality_flags': []
            }
            
            # Calculate reliability score
            confidence = quality_indicators['prediction_confidence']
            uncertainty = quality_indicators['uncertainty_level']
            agreement = quality_indicators['model_agreement']
            
            reliability_score = confidence * (1 - uncertainty) * (1.2 if agreement else 0.8)
            quality_indicators['reliability_score'] = round(min(reliability_score, 1.0), 3)
            
            # Generate quality flags
            if confidence < self.confidence_thresholds['moderate_confidence']:
                quality_indicators['quality_flags'].append('Low confidence prediction')
            
            if uncertainty > 0.3:
                quality_indicators['quality_flags'].append('High prediction uncertainty')
            
            if not agreement:
                quality_indicators['quality_flags'].append('Model disagreement detected')
            
            if not quality_indicators['quality_flags']:
                quality_indicators['quality_flags'].append('Good prediction quality')
            
            return quality_indicators
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {str(e)}")
            return {'error': f'Quality assessment failed: {str(e)}'}
    
    def _generate_clinical_alerts(self, prediction_results: Dict, patient_info: Optional[Dict]) -> List[Dict]:
        """Generate clinical alerts for high-risk conditions or situations."""
        try:
            alerts = []
            
            ai_analysis = self._analyze_ai_predictions(prediction_results)
            risk_assessment = self._assess_clinical_risk(prediction_results, patient_info)
            
            predicted_class = ai_analysis.get('ensemble_prediction', 'Unknown')
            confidence = ai_analysis.get('ensemble_confidence', 0.0)
            risk_level = risk_assessment.get('final_risk_level', 'low')
            
            # Critical condition alerts
            if predicted_class == 'Melanoma':
                alerts.append({
                    'type': 'critical',
                    'title': 'POTENTIAL MELANOMA DETECTED',
                    'message': 'AI analysis suggests possible melanoma. Immediate specialist consultation required.',
                    'action_required': 'URGENT REFERRAL',
                    'priority': 1
                })
            
            # Low confidence alerts
            if confidence < self.confidence_thresholds['low_confidence']:
                alerts.append({
                    'type': 'warning',
                    'title': 'LOW AI CONFIDENCE',
                    'message': f'AI confidence is {confidence:.1%}, below clinical threshold. Expert review recommended.',
                    'action_required': 'MANUAL REVIEW',
                    'priority': 2
                })
            
            # Model disagreement alerts
            if not ai_analysis.get('model_consensus', True):
                alerts.append({
                    'type': 'info',
                    'title': 'MODEL DISAGREEMENT',
                    'message': 'Multiple AI models show different predictions. Consider additional evaluation.',
                    'action_required': 'CLINICAL CORRELATION',
                    'priority': 3
                })
            
            # High-risk patient alerts
            if risk_level in ['critical', 'high'] and patient_info:
                risk_factors = risk_assessment.get('risk_factors', [])
                if risk_factors:
                    alerts.append({
                        'type': 'warning',
                        'title': 'HIGH-RISK PATIENT PROFILE',
                        'message': f'Patient has multiple risk factors: {", ".join(risk_factors)}',
                        'action_required': 'ENHANCED MONITORING',
                        'priority': 2
                    })
            
            # Sort alerts by priority
            alerts.sort(key=lambda x: x['priority'])
            
            return alerts
            
        except Exception as e:
            logger.error(f"Alert generation failed: {str(e)}")
            return []
    
    def _generate_error_report(self, error_message: str) -> Dict:
        """Generate error report when clinical analysis fails."""
        return {
            'timestamp': datetime.now().isoformat(),
            'status': 'error',
            'error_message': error_message,
            'recommendations': [{
                'type': 'urgent',
                'category': 'System Error',
                'recommendation': 'AI analysis failed. Manual clinical evaluation required.',
                'priority': 'critical',
                'rationale': 'System error prevents automated analysis'
            }],
            'follow_up': {
                'timeline': 'immediate',
                'provider': 'Clinical specialist',
                'urgency': 'high'
            }
        }
    
    def validate_patient_info(self, patient_info: Dict) -> Dict:
        """Validate and standardize patient information."""
        try:
            validated = {}
            
            # Age validation
            age = patient_info.get('age')
            if age is not None:
                try:
                    validated['age'] = max(0, min(int(age), 120))
                except (ValueError, TypeError):
                    validated['age'] = None
            
            # Skin tone validation (Fitzpatrick scale 1-6)
            skin_tone = patient_info.get('skin_tone')
            if skin_tone is not None:
                try:
                    validated['skin_tone'] = max(1, min(int(skin_tone), 6))
                except (ValueError, TypeError):
                    validated['skin_tone'] = 3  # Default to medium
            
            # Gender validation
            gender = patient_info.get('gender', '').lower()
            if gender in ['male', 'female', 'other']:
                validated['gender'] = gender
            
            # Family history validation
            family_history = patient_info.get('family_history_melanoma')
            if isinstance(family_history, bool):
                validated['family_history_melanoma'] = family_history
            
            return validated
            
        except Exception as e:
            logger.error(f"Patient info validation failed: {str(e)}")
            return {}
    
    def generate_patient_education(self, prediction_results: Dict) -> Dict:
        """Generate patient education materials based on AI analysis."""
        try:
            ai_analysis = self._analyze_ai_predictions(prediction_results)
            predicted_class = ai_analysis.get('ensemble_prediction', 'Unknown')
            
            education = {
                'condition_overview': self._get_condition_overview(predicted_class),
                'risk_factors': self._get_risk_factors(predicted_class),
                'prevention_tips': self._get_prevention_tips(predicted_class),
                'warning_signs': self._get_warning_signs(),
                'when_to_seek_care': self._get_care_guidelines(predicted_class)
            }
            
            return education
            
        except Exception as e:
            logger.error(f"Patient education generation failed: {str(e)}")
            return {}
    
    def _get_condition_overview(self, condition: str) -> str:
        """Get patient-friendly condition overview."""
        overviews = {
            'Melanoma': 'Melanoma is the most serious type of skin cancer that can spread to other parts of the body if not treated early.',
            'Basal Cell Carcinoma': 'Basal cell carcinoma is the most common type of skin cancer, usually slow-growing and rarely spreads.',
            'Squamous Cell Carcinoma': 'Squamous cell carcinoma is the second most common skin cancer type, can spread if not treated.',
            'Actinic Keratosis': 'Actinic keratosis is a precancerous skin condition caused by sun damage.',
            'Benign Keratosis': 'Seborrheic keratosis is a common, non-cancerous skin growth.',
            'Dermatofibroma': 'Dermatofibroma is a common, benign skin nodule.',
            'Nevus': 'A nevus (mole) is usually a benign skin growth made of melanocytes.'
        }
        return overviews.get(condition, 'Please consult with your healthcare provider for information about your specific condition.')
    
    def _get_risk_factors(self, condition: str) -> List[str]:
        """Get risk factors for the condition."""
        return [
            'Fair skin that burns easily',
            'History of sunburns',
            'Excessive UV exposure',
            'Family history of skin cancer',
            'Multiple moles',
            'Weakened immune system',
            'Age over 50'
        ]
    
    def _get_prevention_tips(self, condition: str) -> List[str]:
        """Get prevention tips."""
        return [
            'Use broad-spectrum sunscreen SPF 30 or higher',
            'Seek shade during peak sun hours (10 AM - 4 PM)',
            'Wear protective clothing and wide-brimmed hats',
            'Avoid tanning beds and deliberate tanning',
            'Perform monthly self-skin examinations',
            'Get regular professional skin checks',
            'Stay hydrated and maintain healthy skin'
        ]
    
    def _get_warning_signs(self) -> List[str]:
        """Get general warning signs to watch for."""
        return [
            'Changes in size, shape, or color of existing moles',
            'New growths or spots that look different',
            'Sores that do not heal within 2-3 weeks',
            'Spots that itch, bleed, or become tender',
            'Asymmetrical moles or irregular borders',
            'Moles with multiple colors or larger than 6mm'
        ]
    
    def _get_care_guidelines(self, condition: str) -> Dict:
        """Get guidelines for when to seek care."""
        if condition == 'Melanoma':
            return {
                'immediate': 'Seek immediate medical attention',
                'reason': 'Potential melanoma requires urgent evaluation'
            }
        elif condition in ['Basal Cell Carcinoma', 'Squamous Cell Carcinoma']:
            return {
                'urgent': 'Schedule appointment within 2-4 weeks',
                'reason': 'Skin cancer requires prompt medical evaluation'
            }
        else:
            return {
                'routine': 'Discuss with healthcare provider at next visit',
                'reason': 'Monitor for changes and seek care if concerned'
            }
