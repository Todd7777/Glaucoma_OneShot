import os
import pandas as pd
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict
import random
from config import DATA_DIR, MAX_IMAGE_SIZE, RANDOM_SEED

class LAGDataLoader:
    """Data loader for the LAG glaucoma dataset."""
    
    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, "images")
        self.labels_file = os.path.join(data_dir, "labels.csv")
        self.labels_df = None
        random.seed(RANDOM_SEED)
        
    def load_labels(self) -> pd.DataFrame:
        """Load labels from CSV file or create from directory structure."""
        if os.path.exists(self.labels_file):
            self.labels_df = pd.read_csv(self.labels_file)
        else:
            # If no labels file, try to infer from directory structure
            self.labels_df = self._create_labels_from_structure()
        return self.labels_df
    
    def _create_labels_from_structure(self) -> pd.DataFrame:
        """Create labels DataFrame from directory structure."""
        image_data = []
        
        if os.path.exists(self.images_dir):
            for filename in os.listdir(self.images_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Try to infer label from filename or use placeholder
                    # This is a placeholder - actual implementation depends on dataset structure
                    image_data.append({
                        'filename': filename,
                        'label': 0,  # Placeholder - needs actual labels
                        'path': os.path.join(self.images_dir, filename)
                    })
        
        return pd.DataFrame(image_data)
    
    def get_reference_images(self, num_per_class: int = 3) -> Dict[int, List[str]]:
        """Select reference images for each class."""
        if self.labels_df is None:
            self.load_labels()
            
        reference_images = {}
        
        for label in [0, 1]:  # 0: normal, 1: glaucoma
            class_images = self.labels_df[self.labels_df['label'] == label]['path'].tolist()
            if len(class_images) >= num_per_class:
                reference_images[label] = random.sample(class_images, num_per_class)
            else:
                reference_images[label] = class_images
                
        return reference_images
    
    def get_test_images(self, sample_size: int = 100) -> List[Tuple[str, int]]:
        """Get test images with their labels."""
        if self.labels_df is None:
            self.load_labels()
            
        # Sample images for testing
        if len(self.labels_df) > sample_size:
            test_df = self.labels_df.sample(n=sample_size, random_state=RANDOM_SEED)
        else:
            test_df = self.labels_df
            
        return [(row['path'], row['label']) for _, row in test_df.iterrows()]
    
    def preprocess_image(self, image_path: str) -> Image.Image:
        """Preprocess image for API submission."""
        try:
            image = Image.open(image_path)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            # Resize image
            image.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
            
            return image
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None
    
    def create_sample_dataset(self, num_samples: int = 20):
        """Create a sample dataset for testing when real dataset is not available."""
        print("Creating sample dataset for testing...")
        
        # Create sample directory structure
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Create sample images (placeholder colored rectangles)
        sample_data = []
        
        for i in range(num_samples):
            # Alternate between normal (0) and glaucoma (1) labels
            label = i % 2
            filename = f"sample_{i:03d}.jpg"
            filepath = os.path.join(self.images_dir, filename)
            
            # Create a sample image (red for glaucoma, green for normal)
            color = (255, 100, 100) if label == 1 else (100, 255, 100)
            sample_image = Image.new('RGB', (256, 256), color)
            sample_image.save(filepath)
            
            sample_data.append({
                'filename': filename,
                'label': label,
                'path': filepath
            })
        
        # Save labels
        sample_df = pd.DataFrame(sample_data)
        sample_df.to_csv(self.labels_file, index=False)
        
        print(f"Created {num_samples} sample images in {self.images_dir}")
        return sample_df
