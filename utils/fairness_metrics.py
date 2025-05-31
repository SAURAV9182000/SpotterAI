import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging

logger = logging.getLogger(__name__)

class FairnessEvaluator:
    """Comprehensive fairness evaluation for skin cancer detection models."""
    
    def __init__(self):
        self.fitzpatrick_groups = {
            1: "I (Very Fair)",
            2: "II (Fair)", 
            3: "III (Medium)",
            4: "IV (Olive)",
            5: "V (Brown)",
            6: "VI (Dark Brown/Black)"
        }
        
        self.protected_attributes = ['skin_tone', 'age_group', 'gender']
        
    def evaluate_demographic_fairness(self, y_true, y_pred, y_proba, protected_attrs):
        """
        Comprehensive demographic fairness evaluation.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels  
            y_proba: Prediction probabilities
            protected_attrs: Dictionary with protected attribute values
            
        Returns:
            Dictionary with fairness metrics
        """
        fairness_results = {
            'overall_metrics': self._calculate_overall_metrics(y_true, y_pred, y_proba),
            'demographic_parity': {},
            'equalized_odds': {},
            'calibration': {},
            'statistical_tests': {}
        }
        
        try:
            # Evaluate for each protected attribute
            for attr_name, attr_values in protected_attrs.items():
                if attr_name in self.protected_attributes:
                    fairness_results['demographic_parity'][attr_name] = self._demographic_parity(
                        y_true, y_pred, attr_values
                    )
                    fairness_results['equalized_odds'][attr_name] = self._equalized_odds(
                        y_true, y_pred, attr_values
                    )
                    fairness_results['calibration'][attr_name] = self._calibration_analysis(
                        y_true, y_proba, attr_values
                    )
                    fairness_results['statistical_tests'][attr_name] = self._statistical_significance_tests(
                        y_true, y_pred, attr_values
                    )
            
            # Calculate aggregate fairness scores
            fairness_results['aggregate_scores'] = self._calculate_aggregate_fairness_scores(
                fairness_results
            )
            
        except Exception as e:
            logger.error(f"Fairness evaluation failed: {str(e)}")
            
        return fairness_results
    
    def _calculate_overall_metrics(self, y_true, y_pred, y_proba):
        """Calculate overall model performance metrics."""
        try:
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision_macro': precision_recall_fscore_support(y_true, y_pred, average='macro')[0],
                'recall_macro': precision_recall_fscore_support(y_true, y_pred, average='macro')[1],
                'f1_macro': precision_recall_fscore_support(y_true, y_pred, average='macro')[2],
                'precision_weighted': precision_recall_fscore_support(y_true, y_pred, average='weighted')[0],
                'recall_weighted': precision_recall_fscore_support(y_true, y_pred, average='weighted')[1],
                'f1_weighted': precision_recall_fscore_support(y_true, y_pred, average='weighted')[2]
            }
            
            # Add AUC if binary classification or multi-class with probabilities
            if y_proba is not None:
                if len(np.unique(y_true)) == 2:
                    metrics['auc'] = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    metrics['auc_macro'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
                    metrics['auc_weighted'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
            
            return metrics
            
        except Exception as e:
            logger.error(f"Overall metrics calculation failed: {str(e)}")
            return {}
    
    def _demographic_parity(self, y_true, y_pred, protected_attr):
        """
        Calculate demographic parity metrics.
        Measures whether positive prediction rates are similar across groups.
        """
        try:
            unique_groups = np.unique(protected_attr)
            group_metrics = {}
            positive_rates = []
            
            for group in unique_groups:
                group_mask = protected_attr == group
                group_pred = y_pred[group_mask]
                
                # Calculate positive prediction rate
                positive_rate = np.mean(group_pred == 1) if len(group_pred) > 0 else 0.0
                positive_rates.append(positive_rate)
                
                group_metrics[str(group)] = {
                    'sample_size': np.sum(group_mask),
                    'positive_rate': positive_rate
                }
            
            # Calculate demographic parity difference
            dp_difference = np.max(positive_rates) - np.min(positive_rates)
            dp_ratio = np.min(positive_rates) / np.max(positive_rates) if np.max(positive_rates) > 0 else 0.0
            
            return {
                'group_metrics': group_metrics,
                'demographic_parity_difference': dp_difference,
                'demographic_parity_ratio': dp_ratio,
                'fairness_threshold_80': dp_ratio >= 0.8  # 80% rule
            }
            
        except Exception as e:
            logger.error(f"Demographic parity calculation failed: {str(e)}")
            return {}
    
    def _equalized_odds(self, y_true, y_pred, protected_attr):
        """
        Calculate equalized odds metrics.
        Measures whether true positive and false positive rates are similar across groups.
        """
        try:
            unique_groups = np.unique(protected_attr)
            group_metrics = {}
            tpr_list = []
            fpr_list = []
            
            for group in unique_groups:
                group_mask = protected_attr == group
                group_true = y_true[group_mask]
                group_pred = y_pred[group_mask]
                
                if len(group_true) > 0:
                    # Calculate TPR and FPR
                    tn, fp, fn, tp = confusion_matrix(
                        group_true, group_pred, labels=[0, 1]
                    ).ravel() if len(np.unique(group_true)) > 1 else (0, 0, 0, 0)
                    
                    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                    
                    tpr_list.append(tpr)
                    fpr_list.append(fpr)
                    
                    group_metrics[str(group)] = {
                        'sample_size': np.sum(group_mask),
                        'true_positive_rate': tpr,
                        'false_positive_rate': fpr,
                        'accuracy': accuracy_score(group_true, group_pred)
                    }
            
            # Calculate equalized odds metrics
            eo_tpr_diff = np.max(tpr_list) - np.min(tpr_list) if len(tpr_list) > 0 else 0.0
            eo_fpr_diff = np.max(fpr_list) - np.min(fpr_list) if len(fpr_list) > 0 else 0.0
            
            return {
                'group_metrics': group_metrics,
                'tpr_difference': eo_tpr_diff,
                'fpr_difference': eo_fpr_diff,
                'equalized_odds_difference': max(eo_tpr_diff, eo_fpr_diff)
            }
            
        except Exception as e:
            logger.error(f"Equalized odds calculation failed: {str(e)}")
            return {}
    
    def _calibration_analysis(self, y_true, y_proba, protected_attr):
        """
        Analyze model calibration across demographic groups.
        Measures whether predicted probabilities reflect true outcome rates.
        """
        try:
            unique_groups = np.unique(protected_attr)
            calibration_metrics = {}
            
            for group in unique_groups:
                group_mask = protected_attr == group
                group_true = y_true[group_mask]
                group_proba = y_proba[group_mask]
                
                if len(group_true) > 0 and group_proba is not None:
                    # Calculate calibration curve
                    n_bins = 10
                    bin_boundaries = np.linspace(0, 1, n_bins + 1)
                    bin_lowers = bin_boundaries[:-1]
                    bin_uppers = bin_boundaries[1:]
                    
                    calibration_data = []
                    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                        # Handle multi-class case by using max probability
                        if len(group_proba.shape) > 1:
                            max_proba = np.max(group_proba, axis=1)
                        else:
                            max_proba = group_proba
                            
                        in_bin = (max_proba > bin_lower) & (max_proba <= bin_upper)
                        
                        if np.sum(in_bin) > 0:
                            bin_accuracy = np.mean(group_true[in_bin] == np.argmax(group_proba[in_bin], axis=1))
                            bin_confidence = np.mean(max_proba[in_bin])
                            bin_count = np.sum(in_bin)
                            
                            calibration_data.append({
                                'bin_lower': bin_lower,
                                'bin_upper': bin_upper,
                                'accuracy': bin_accuracy,
                                'confidence': bin_confidence,
                                'count': bin_count
                            })
                    
                    # Calculate Expected Calibration Error (ECE)
                    ece = self._calculate_ece(calibration_data, len(group_true))
                    
                    calibration_metrics[str(group)] = {
                        'calibration_curve': calibration_data,
                        'expected_calibration_error': ece,
                        'sample_size': np.sum(group_mask)
                    }
            
            return calibration_metrics
            
        except Exception as e:
            logger.error(f"Calibration analysis failed: {str(e)}")
            return {}
    
    def _calculate_ece(self, calibration_data, total_samples):
        """Calculate Expected Calibration Error."""
        ece = 0.0
        for bin_data in calibration_data:
            bin_weight = bin_data['count'] / total_samples
            accuracy_diff = abs(bin_data['accuracy'] - bin_data['confidence'])
            ece += bin_weight * accuracy_diff
        return ece
    
    def _statistical_significance_tests(self, y_true, y_pred, protected_attr):
        """Perform statistical significance tests for fairness metrics."""
        try:
            unique_groups = np.unique(protected_attr)
            test_results = {}
            
            if len(unique_groups) >= 2:
                # Chi-square test for independence
                contingency_table = pd.crosstab(protected_attr, y_pred)
                chi2, p_chi2, dof, expected = stats.chi2_contingency(contingency_table)
                
                test_results['chi_square_test'] = {
                    'statistic': chi2,
                    'p_value': p_chi2,
                    'significant': p_chi2 < 0.05
                }
                
                # Pairwise accuracy comparisons
                if len(unique_groups) == 2:
                    group1_mask = protected_attr == unique_groups[0]
                    group2_mask = protected_attr == unique_groups[1]
                    
                    acc1 = accuracy_score(y_true[group1_mask], y_pred[group1_mask])
                    acc2 = accuracy_score(y_true[group2_mask], y_pred[group2_mask])
                    
                    # Z-test for difference in proportions
                    n1, n2 = np.sum(group1_mask), np.sum(group2_mask)
                    p_pooled = (acc1 * n1 + acc2 * n2) / (n1 + n2)
                    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
                    z_stat = (acc1 - acc2) / se if se > 0 else 0
                    p_z = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                    
                    test_results['accuracy_difference_test'] = {
                        'z_statistic': z_stat,
                        'p_value': p_z,
                        'significant': p_z < 0.05,
                        'accuracy_group1': acc1,
                        'accuracy_group2': acc2
                    }
            
            return test_results
            
        except Exception as e:
            logger.error(f"Statistical tests failed: {str(e)}")
            return {}
    
    def _calculate_aggregate_fairness_scores(self, fairness_results):
        """Calculate aggregate fairness scores across all metrics."""
        try:
            aggregate_scores = {}
            
            # Demographic Parity Score
            dp_scores = []
            for attr, metrics in fairness_results['demographic_parity'].items():
                if 'demographic_parity_ratio' in metrics:
                    dp_scores.append(metrics['demographic_parity_ratio'])
            
            # Equalized Odds Score  
            eo_scores = []
            for attr, metrics in fairness_results['equalized_odds'].items():
                if 'equalized_odds_difference' in metrics:
                    # Convert difference to score (1 - difference)
                    eo_scores.append(1.0 - metrics['equalized_odds_difference'])
            
            # Calibration Score
            cal_scores = []
            for attr, metrics in fairness_results['calibration'].items():
                for group, group_metrics in metrics.items():
                    if 'expected_calibration_error' in group_metrics:
                        # Convert ECE to score (1 - ECE)
                        cal_scores.append(1.0 - group_metrics['expected_calibration_error'])
            
            # Calculate aggregate scores
            aggregate_scores['demographic_parity_score'] = np.mean(dp_scores) if dp_scores else 0.0
            aggregate_scores['equalized_odds_score'] = np.mean(eo_scores) if eo_scores else 0.0
            aggregate_scores['calibration_score'] = np.mean(cal_scores) if cal_scores else 0.0
            
            # Overall fairness score (weighted average)
            weights = [0.4, 0.4, 0.2]  # DP, EO, Calibration
            scores = [
                aggregate_scores['demographic_parity_score'],
                aggregate_scores['equalized_odds_score'], 
                aggregate_scores['calibration_score']
            ]
            
            aggregate_scores['overall_fairness_score'] = np.average(scores, weights=weights)
            
            return aggregate_scores
            
        except Exception as e:
            logger.error(f"Aggregate score calculation failed: {str(e)}")
            return {}
    
    def generate_fairness_report(self, fairness_results, save_path=None):
        """Generate a comprehensive fairness report."""
        try:
            report = {
                'summary': self._generate_fairness_summary(fairness_results),
                'detailed_metrics': fairness_results,
                'recommendations': self._generate_fairness_recommendations(fairness_results)
            }
            
            if save_path:
                import json
                with open(save_path, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
            
            return report
            
        except Exception as e:
            logger.error(f"Report generation failed: {str(e)}")
            return {}
    
    def _generate_fairness_summary(self, fairness_results):
        """Generate executive summary of fairness evaluation."""
        summary = {
            'overall_assessment': 'Fair',
            'key_findings': [],
            'risk_level': 'Low'
        }
        
        try:
            if 'aggregate_scores' in fairness_results:
                overall_score = fairness_results['aggregate_scores'].get('overall_fairness_score', 0.0)
                
                if overall_score >= 0.8:
                    summary['overall_assessment'] = 'Fair'
                    summary['risk_level'] = 'Low'
                elif overall_score >= 0.6:
                    summary['overall_assessment'] = 'Moderately Fair'
                    summary['risk_level'] = 'Medium'
                else:
                    summary['overall_assessment'] = 'Unfair'
                    summary['risk_level'] = 'High'
                
                summary['key_findings'].append(f"Overall fairness score: {overall_score:.3f}")
            
            # Add specific findings
            for attr, metrics in fairness_results.get('demographic_parity', {}).items():
                if 'demographic_parity_ratio' in metrics:
                    ratio = metrics['demographic_parity_ratio']
                    if ratio < 0.8:
                        summary['key_findings'].append(
                            f"Demographic parity violation for {attr} (ratio: {ratio:.3f})"
                        )
            
        except Exception as e:
            logger.error(f"Summary generation failed: {str(e)}")
        
        return summary
    
    def _generate_fairness_recommendations(self, fairness_results):
        """Generate actionable recommendations for improving fairness."""
        recommendations = []
        
        try:
            # Check demographic parity
            for attr, metrics in fairness_results.get('demographic_parity', {}).items():
                if 'demographic_parity_ratio' in metrics:
                    ratio = metrics['demographic_parity_ratio']
                    if ratio < 0.8:
                        recommendations.append({
                            'issue': f'Demographic parity violation for {attr}',
                            'recommendation': 'Consider rebalancing training data or applying fairness constraints during training',
                            'priority': 'High'
                        })
            
            # Check equalized odds
            for attr, metrics in fairness_results.get('equalized_odds', {}).items():
                if 'equalized_odds_difference' in metrics:
                    diff = metrics['equalized_odds_difference']
                    if diff > 0.1:
                        recommendations.append({
                            'issue': f'Equalized odds violation for {attr}',
                            'recommendation': 'Apply post-processing calibration or use fairness-aware training objectives',
                            'priority': 'High'
                        })
            
            # Check calibration
            high_ece_groups = []
            for attr, metrics in fairness_results.get('calibration', {}).items():
                for group, group_metrics in metrics.items():
                    if 'expected_calibration_error' in group_metrics:
                        ece = group_metrics['expected_calibration_error']
                        if ece > 0.1:
                            high_ece_groups.append(f"{attr}={group}")
            
            if high_ece_groups:
                recommendations.append({
                    'issue': f'Poor calibration for groups: {", ".join(high_ece_groups)}',
                    'recommendation': 'Apply temperature scaling or Platt scaling for better calibration',
                    'priority': 'Medium'
                })
            
            # General recommendations
            if not recommendations:
                recommendations.append({
                    'issue': 'Model shows good fairness properties',
                    'recommendation': 'Continue monitoring fairness metrics and consider expanding evaluation to additional demographic groups',
                    'priority': 'Low'
                })
            
        except Exception as e:
            logger.error(f"Recommendations generation failed: {str(e)}")
        
        return recommendations
    
    def create_fairness_visualization(self, fairness_results):
        """Create visualizations for fairness metrics."""
        visualizations = {}
        
        try:
            # Create demographic parity visualization
            if 'demographic_parity' in fairness_results:
                for attr, metrics in fairness_results['demographic_parity'].items():
                    if 'group_metrics' in metrics:
                        visualizations[f'demographic_parity_{attr}'] = self._plot_demographic_parity(
                            metrics['group_metrics']
                        )
            
            # Create accuracy comparison visualization
            if 'equalized_odds' in fairness_results:
                for attr, metrics in fairness_results['equalized_odds'].items():
                    if 'group_metrics' in metrics:
                        visualizations[f'accuracy_comparison_{attr}'] = self._plot_accuracy_comparison(
                            metrics['group_metrics']
                        )
        
        except Exception as e:
            logger.error(f"Visualization creation failed: {str(e)}")
        
        return visualizations
    
    def _plot_demographic_parity(self, group_metrics):
        """Create demographic parity visualization."""
        # This would create actual plots - returning data structure for now
        return {
            'type': 'bar_chart',
            'data': {
                'groups': list(group_metrics.keys()),
                'positive_rates': [metrics['positive_rate'] for metrics in group_metrics.values()],
                'sample_sizes': [metrics['sample_size'] for metrics in group_metrics.values()]
            }
        }
    
    def _plot_accuracy_comparison(self, group_metrics):
        """Create accuracy comparison visualization."""
        return {
            'type': 'bar_chart',
            'data': {
                'groups': list(group_metrics.keys()),
                'accuracies': [metrics['accuracy'] for metrics in group_metrics.values()],
                'sample_sizes': [metrics['sample_size'] for metrics in group_metrics.values()]
            }
        }
