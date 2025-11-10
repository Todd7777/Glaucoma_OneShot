# Usage Guide

## Quick Start

1. **Clone and setup:**
   ```bash
   git clone https://github.com/Todd7777/Glaucoma_OneShot.git
   cd Glaucoma_OneShot
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure API access:**
   ```bash
   cp .env.example .env
   # Edit .env and add: OPENAI_API_KEY=your_actual_api_key_here
   ```

3. **Test without API:**
   ```bash
   python demo.py
   ```

4. **Run experiment:**
   ```bash
   python main.py --sample-size 50
   ```

## Using Real LAG Dataset

1. **Download the LAG dataset** from the official repository (requires permission)
2. **Extract images** to `data/images/` directory
3. **Create labels file** at `data/labels.csv` with columns:
   - `filename`: Image filename
   - `label`: 0 (normal) or 1 (glaucoma)
   - `path`: Full path to image file

4. **Run the experiment:**
   ```bash
   python main.py --sample-size 100
   ```

## Command Line Options

- `--sample-size N`: Number of test images to evaluate (default: 20)
- `--skip-experiment`: Skip running experiment, only evaluate existing results
- `--create-sample-data`: Create sample dataset for testing

## Output Files

Results are saved in the `results/` directory:

- `one_shot_results.json/csv`: One-shot prompting results
- `zero_shot_results.json/csv`: Zero-shot prompting results
- `combined_results.json/csv`: Combined results for comparison
- `evaluation_report.txt`: Detailed performance metrics
- `*.png`: Visualization plots (confusion matrices, confidence distributions)

## Understanding Results

### Key Metrics
- **Accuracy**: Overall classification accuracy
- **Precision**: Positive predictive value per class
- **Recall**: Sensitivity per class
- **F1-Score**: Harmonic mean of precision and recall

### Expected Outcomes
The project tests whether providing reference images (one-shot prompting) improves ChatGPT's ability to diagnose glaucoma compared to zero-shot prompting without examples.

### Interpreting Confidence Scores
ChatGPT provides confidence scores (0-100%) for each prediction. Higher confidence scores for correct predictions indicate better model certainty.

## Troubleshooting

### Common Issues

1. **"OpenAI API key not found"**
   - Ensure `.env` file exists with `OPENAI_API_KEY=your_key`

2. **"No dataset found"**
   - Run `python main.py --create-sample-data` for testing
   - Or place real LAG dataset in `data/images/`

3. **API Rate Limits**
   - Reduce `--sample-size` parameter
   - Add delays between API calls if needed

4. **Memory Issues**
   - Reduce image resolution in `config.py`
   - Process smaller batches

### Cost Considerations

- Each image analysis costs ~$0.01-0.03 in OpenAI API credits
- One-shot prompting uses more tokens (higher cost) due to reference images
- Estimate: ~$2-5 for 100 test images with both methods

## Customization

### Modifying Prompts
Edit `src/chatgpt_client.py` to customize the prompting strategy:
- Change system messages
- Modify instruction format
- Add domain-specific guidance

### Adjusting Reference Images
Modify `config.py`:
- `REFERENCE_IMAGES_PER_CLASS`: Number of reference images per class
- `MAX_IMAGE_SIZE`: Image resolution for API efficiency

### Adding New Metrics
Extend `src/evaluation.py` to include additional evaluation metrics or visualizations.
