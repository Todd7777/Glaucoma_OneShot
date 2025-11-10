# Results Directory

This directory contains the output from glaucoma detection experiments.

## Generated Files

After running experiments, you'll find:

### Data Files
- `one_shot_results.json/csv` - One-shot prompting results
- `zero_shot_results.json/csv` - Zero-shot prompting results  
- `combined_results.json/csv` - Combined dataset for comparison

### Analysis Reports
- `evaluation_report.txt` - Detailed performance metrics
- `method_comparison.png` - Side-by-side performance chart

### Visualizations
- `confusion_matrix_one_shot.png` - One-shot confusion matrix
- `confusion_matrix_zero_shot.png` - Zero-shot confusion matrix
- `confidence_dist_one_shot.png` - One-shot confidence distribution
- `confidence_dist_zero_shot.png` - Zero-shot confidence distribution

## Sample Results Structure

```json
{
  "image_path": "data/images/sample_001.jpg",
  "true_label": 1,
  "predicted_label": 1,
  "confidence": 87.5,
  "explanation": "Increased cup-to-disc ratio suggestive of glaucoma",
  "method": "one_shot"
}
```

## Metrics Included

- **Accuracy**: Overall classification performance
- **Sensitivity**: True positive rate (glaucoma detection)
- **Specificity**: True negative rate (normal classification)
- **Precision/Recall**: Per-class performance metrics
- **F1-Score**: Harmonic mean of precision and recall
- **Confidence Calibration**: AI certainty vs. accuracy correlation
