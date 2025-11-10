import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from typing import List, Dict, Tuple
from config import RESULTS_DIR

class ResultsEvaluator:
    """Evaluate and visualize experiment results."""
    
    def __init__(self, results_dir: str = RESULTS_DIR):
        self.results_dir = results_dir
        
    def load_results(self, filename: str) -> List[Dict]:
        """Load results from JSON file."""
        filepath = os.path.join(self.results_dir, f"{filename}.json")
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def calculate_metrics(self, results: List[Dict]) -> Dict:
        """Calculate performance metrics."""
        # Filter out results with invalid predictions
        valid_results = [r for r in results if r['predicted_label'] is not None]
        
        if not valid_results:
            return {}
            
        y_true = [r['true_label'] for r in valid_results]
        y_pred = [r['predicted_label'] for r in valid_results]
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'total_samples': len(valid_results),
            'valid_predictions': len(valid_results)
        }
        
        # Calculate per-class metrics
        for label in [0, 1]:
            label_name = 'normal' if label == 0 else 'glaucoma'
            true_positives = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
            false_positives = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
            false_negatives = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            
            metrics[f'{label_name}_precision'] = precision
            metrics[f'{label_name}_recall'] = recall
        
        return metrics
    
    def plot_confusion_matrix(self, results: List[Dict], method_name: str, save_path: str = None):
        """Plot confusion matrix."""
        valid_results = [r for r in results if r['predicted_label'] is not None]
        y_true = [r['true_label'] for r in valid_results]
        y_pred = [r['predicted_label'] for r in valid_results]
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Glaucoma'], 
                   yticklabels=['Normal', 'Glaucoma'])
        plt.title(f'Confusion Matrix - {method_name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confidence_distribution(self, results: List[Dict], method_name: str, save_path: str = None):
        """Plot confidence score distribution."""
        valid_results = [r for r in results if r['confidence'] is not None]
        
        correct_confidences = [r['confidence'] for r in valid_results if r['predicted_label'] == r['true_label']]
        incorrect_confidences = [r['confidence'] for r in valid_results if r['predicted_label'] != r['true_label']]
        
        plt.figure(figsize=(10, 6))
        
        plt.hist(correct_confidences, bins=20, alpha=0.7, label='Correct Predictions', color='green')
        plt.hist(incorrect_confidences, bins=20, alpha=0.7, label='Incorrect Predictions', color='red')
        
        plt.xlabel('Confidence Score (%)')
        plt.ylabel('Frequency')
        plt.title(f'Confidence Score Distribution - {method_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def compare_methods(self, one_shot_results: List[Dict], zero_shot_results: List[Dict]):
        """Compare one-shot vs zero-shot performance."""
        one_shot_metrics = self.calculate_metrics(one_shot_results)
        zero_shot_metrics = self.calculate_metrics(zero_shot_results)
        
        comparison = {
            'Method': ['One-Shot', 'Zero-Shot'],
            'Accuracy': [one_shot_metrics.get('accuracy', 0), zero_shot_metrics.get('accuracy', 0)],
            'Precision': [one_shot_metrics.get('precision', 0), zero_shot_metrics.get('precision', 0)],
            'Recall': [one_shot_metrics.get('recall', 0), zero_shot_metrics.get('recall', 0)],
            'F1-Score': [one_shot_metrics.get('f1_score', 0), zero_shot_metrics.get('f1_score', 0)]
        }
        
        df = pd.DataFrame(comparison)
        
        # Plot comparison
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        colors = ['skyblue', 'lightcoral']
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            bars = ax.bar(df['Method'], df[metric], color=colors)
            ax.set_title(f'{metric} Comparison')
            ax.set_ylabel(metric)
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, df[metric]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'method_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        return df
    
    def generate_report(self, one_shot_results: List[Dict], zero_shot_results: List[Dict]):
        """Generate comprehensive evaluation report."""
        print("=" * 60)
        print("GLAUCOMA DETECTION EXPERIMENT RESULTS")
        print("=" * 60)
        
        # Calculate metrics
        one_shot_metrics = self.calculate_metrics(one_shot_results)
        zero_shot_metrics = self.calculate_metrics(zero_shot_results)
        
        print("\n📊 PERFORMANCE SUMMARY")
        print("-" * 40)
        print(f"{'Metric':<15} {'One-Shot':<12} {'Zero-Shot':<12} {'Improvement':<12}")
        print("-" * 40)
        
        metrics_to_show = ['accuracy', 'precision', 'recall', 'f1_score']
        for metric in metrics_to_show:
            one_shot_val = one_shot_metrics.get(metric, 0)
            zero_shot_val = zero_shot_metrics.get(metric, 0)
            improvement = ((one_shot_val - zero_shot_val) / zero_shot_val * 100) if zero_shot_val > 0 else 0
            
            print(f"{metric.capitalize():<15} {one_shot_val:<12.3f} {zero_shot_val:<12.3f} {improvement:+.1f}%")
        
        print(f"\nSample Size: {one_shot_metrics.get('total_samples', 0)} images")
        
        # Class-specific performance
        print("\n📋 CLASS-SPECIFIC PERFORMANCE")
        print("-" * 40)
        for class_name in ['normal', 'glaucoma']:
            print(f"\n{class_name.upper()}:")
            one_shot_prec = one_shot_metrics.get(f'{class_name}_precision', 0)
            one_shot_rec = one_shot_metrics.get(f'{class_name}_recall', 0)
            zero_shot_prec = zero_shot_metrics.get(f'{class_name}_precision', 0)
            zero_shot_rec = zero_shot_metrics.get(f'{class_name}_recall', 0)
            
            print(f"  Precision: One-Shot {one_shot_prec:.3f}, Zero-Shot {zero_shot_prec:.3f}")
            print(f"  Recall:    One-Shot {one_shot_rec:.3f}, Zero-Shot {zero_shot_rec:.3f}")
        
        # Generate visualizations
        print("\n📈 GENERATING VISUALIZATIONS...")
        
        # Confusion matrices
        self.plot_confusion_matrix(one_shot_results, "One-Shot Prompting", 
                                 os.path.join(self.results_dir, 'confusion_matrix_one_shot.png'))
        self.plot_confusion_matrix(zero_shot_results, "Zero-Shot Prompting",
                                 os.path.join(self.results_dir, 'confusion_matrix_zero_shot.png'))
        
        # Confidence distributions
        self.plot_confidence_distribution(one_shot_results, "One-Shot Prompting",
                                        os.path.join(self.results_dir, 'confidence_dist_one_shot.png'))
        self.plot_confidence_distribution(zero_shot_results, "Zero-Shot Prompting",
                                        os.path.join(self.results_dir, 'confidence_dist_zero_shot.png'))
        
        # Method comparison
        comparison_df = self.compare_methods(one_shot_results, zero_shot_results)
        
        # Save detailed report
        report_path = os.path.join(self.results_dir, 'evaluation_report.txt')
        with open(report_path, 'w') as f:
            f.write("GLAUCOMA DETECTION EXPERIMENT RESULTS\n")
            f.write("=" * 60 + "\n\n")
            f.write("ONE-SHOT PROMPTING METRICS:\n")
            for key, value in one_shot_metrics.items():
                f.write(f"{key}: {value}\n")
            f.write("\nZERO-SHOT PROMPTING METRICS:\n")
            for key, value in zero_shot_metrics.items():
                f.write(f"{key}: {value}\n")
        
        print(f"\n✅ Evaluation complete! Results saved in {self.results_dir}")
        
        return {
            'one_shot_metrics': one_shot_metrics,
            'zero_shot_metrics': zero_shot_metrics,
            'comparison': comparison_df
        }
