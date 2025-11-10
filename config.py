import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4-vision-preview")

# Dataset Configuration
DATA_DIR = "data"
RESULTS_DIR = "results"

# Experiment Configuration
SAMPLE_SIZE = 100  # Number of test images to evaluate
REFERENCE_IMAGES_PER_CLASS = 3  # Number of reference images for one-shot prompting
RANDOM_SEED = 42

# Image Processing
MAX_IMAGE_SIZE = (512, 512)  # Resize images for API efficiency
IMAGE_QUALITY = 85  # JPEG quality for compression
