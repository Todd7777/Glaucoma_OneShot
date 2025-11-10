# LAG Glaucoma One-Shot Prompting Study - ARVO Abstract Description

## Project Overview

We have developed and implemented a novel study to evaluate the effectiveness of one-shot prompting with large language models (specifically ChatGPT-4 Vision) for automated glaucoma detection using fundus images from the LAG (Large-scale Attention-based Glaucoma) dataset.

## Study Design & Methodology

### Dataset
- **Source**: LAG database containing 11,760 fundus images
- **Composition**: 4,878 suspicious glaucoma samples and 6,882 negative samples
- **Labels**: Binary classification (0 = normal, 1 = glaucomatous/suspicious)
- **Subset**: 5,824 images with expert-annotated attention regions

### Experimental Framework
We designed a controlled comparison between two prompting approaches:

**1. Zero-Shot Prompting (Baseline)**
- ChatGPT analyzes fundus images without prior examples
- Relies solely on pre-trained medical knowledge
- Standard clinical instruction prompts

**2. One-Shot Prompting (Intervention)**
- Provides 3 reference images per class (normal and glaucomatous)
- Reference images selected from high-quality, representative samples
- ChatGPT analyzes test images after seeing labeled examples

### Technical Implementation
- **API Integration**: OpenAI GPT-4 Vision API for image analysis
- **Image Processing**: Standardized preprocessing (512x512 resolution, JPEG compression)
- **Prompt Engineering**: Structured medical prompts requesting classification, confidence scores, and clinical reasoning
- **Sample Size**: Configurable (100-200 test images for statistical significance)

## Evaluation Metrics

### Primary Outcomes
- **Diagnostic Accuracy**: Overall classification performance
- **Sensitivity**: True positive rate for glaucoma detection
- **Specificity**: True negative rate for normal cases
- **Confidence Calibration**: Correlation between AI confidence and diagnostic accuracy

### Secondary Analyses
- **Class-specific Performance**: Per-class precision and recall
- **Error Analysis**: Characterization of misclassified cases
- **Cost-Benefit Analysis**: API usage costs vs. diagnostic improvement
- **Clinical Reasoning**: Qualitative analysis of AI explanations

## Expected Clinical Significance

This study addresses a critical gap in AI-assisted ophthalmology by investigating whether providing visual examples can enhance large language model performance in glaucoma screening. The findings will inform:

1. **Clinical Implementation**: Optimal prompting strategies for AI diagnostic tools
2. **Human-AI Collaboration**: How to effectively provide context to AI systems
3. **Few-Shot Learning**: Potential for rapid AI adaptation with minimal training data
4. **Screening Programs**: Feasibility of LLM-based glaucoma detection in resource-limited settings

## Preliminary Results Framework

Our analysis pipeline generates:
- **Comparative Performance Tables**: Side-by-side accuracy metrics
- **Confusion Matrices**: Visual representation of classification errors
- **Confidence Distribution Plots**: AI certainty across correct/incorrect predictions
- **ROC Curves**: Threshold-independent performance assessment

## Innovation & Impact

This represents the first systematic evaluation of one-shot prompting for glaucoma detection using a large-scale dataset. The methodology can be extended to other ophthalmic conditions and provides a framework for optimizing AI diagnostic performance through strategic prompt design.

## Timeline & Deliverables

- **Implementation**: Complete (functional codebase with evaluation pipeline)
- **Data Collection**: Ready to execute with API access
- **Analysis**: Automated reporting and visualization system
- **Manuscript Preparation**: Results can be generated within 24-48 hours of execution

The study is designed to produce publication-ready results suitable for ARVO presentation and subsequent peer-reviewed publication.
