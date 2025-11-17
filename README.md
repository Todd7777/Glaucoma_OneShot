# One-Shot Prompting for Glaucoma Detection with ChatGPT-4 Vision

A novel study investigating whether providing reference images improves large language model performance in automated glaucoma detection using the LAG (Large-scale Attention-based Glaucoma) dataset.

## 🔬 Study Overview

This research evaluates the effectiveness of one-shot prompting versus zero-shot prompting with ChatGPT-4 Vision for glaucoma diagnosis from fundus images. The study addresses a critical question in AI-assisted ophthalmology: **Does providing visual examples enhance diagnostic accuracy?**

### Dataset
- **LAG Database**: 11,760 fundus images from published CVPR 2019 study
- **Composition**: 4,878 suspicious glaucoma + 6,882 negative samples
- **Expert Annotations**: 5,824 images with attention region labels
- **Reference**: [Li et al., CVPR 2019](http://openaccess.thecvf.com/content_CVPR_2019/html/Li_Attention_Based_Glaucoma_Detection_A_Large-Scale_Database_and_CNN_Model_CVPR_2019_paper.html)

## 🏗️ Architecture

```
├── src/                    # Core implementation
│   ├── data_loader.py     # Dataset handling & preprocessing
│   ├── chatgpt_client.py  # OpenAI API integration
│   ├── one_shot_prompt.py # Experiment orchestration
│   └── evaluation.py      # Statistical analysis & visualization
├── config.py              # Experimental parameters
├── main.py                # Execution pipeline
└── demo.py                # Demonstration without API calls
```

## 🚀 Quick Start

### Prerequisites
```bash
git clone https://github.com/Todd7777/Glaucoma_OneShot.git
cd Glaucoma_OneShot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Configuration
```bash
cp .env.example .env
# Add your OpenAI API key to .env file
```

### Demo Run (No API Required)
```bash
python demo.py
```

### Full Experiment
```bash
python main.py --sample-size 100
```

## 📊 Methodology

### Experimental Design
1. **Zero-Shot (Test-Time) Baseline**: ChatGPT analyzes test images without references
2. **Few-/One-Shot (In-Context) Intervention**: ChatGPT receives few reference images from train/val (default 3 per class) before analyzing test images
3. **Controlled Comparison**: Same test split, standardized prompts, identical evaluation metrics

### Prompting Criteria (Unified)
- Look for key glaucoma indicators such as:
  - Optic disc cupping (increased cup-to-disc ratio)
  - Focal neuroretinal rim thinning or notching
  - Retinal nerve fiber layer defects
  - Beta-zone peripapillary atrophy
  - Vessel bayoneting

### Evaluation Metrics
- **Primary**: Diagnostic accuracy, sensitivity, specificity
- **Secondary**: Precision, recall, F1-score, confidence calibration
- **Qualitative**: Clinical reasoning analysis from AI explanations

### Statistical Analysis
- Confusion matrices and ROC curves
- Confidence distribution analysis
- Cost-effectiveness assessment
- Publication-ready visualizations

## 📦 Data Protocols

- **Subset**: Use only the image-only subset of the data
- **Split**: Deterministic 60/10/30 for train/val/test (stratified when labels available)
- **Zero-Shot**: Evaluate only on the 30% test split with unified indicators
- **Few-/One-Shot (In-Context)**: Draw a few references from train+val, evaluate on the test split
- **SFT (Qwen-LLaVA / LLaMA)**: Use train+val for tuning, test for final evaluation

## 🎯 Clinical Significance

This study provides evidence for:
- Optimal prompting strategies for medical AI systems
- Human-AI collaboration frameworks in ophthalmology
- Few-shot learning potential in diagnostic applications
- Implementation guidelines for AI screening programs

## 📈 Expected Outcomes

The research will generate:
- Comparative performance metrics between prompting approaches
- Clinical decision support system design recommendations
- Framework for extending methodology to other ophthalmic conditions
- ARVO-ready abstract and manuscript materials

## 📝 Citation

If you use this work, please cite:
```
@misc{glaucoma_oneshot_2024,
  title={One-Shot Prompting for Glaucoma Detection with ChatGPT-4 Vision},
  author={[Your Name]},
  year={2024},
  url={https://github.com/Todd7777/Glaucoma_OneShot}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

This is a research project. For questions or collaboration opportunities, please open an issue or contact the authors.
