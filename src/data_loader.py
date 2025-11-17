import os
import pandas as pd
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict
import random
from config import (
    DATA_DIR,
    MAX_IMAGE_SIZE,
    RANDOM_SEED,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    PSEUDO_DREAM_ONLY,
    FEW_SHOT_REFERENCE_SPLITS,
)

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
        # Normalize columns
        if 'path' not in self.labels_df.columns and 'filepath' in self.labels_df.columns:
            self.labels_df = self.labels_df.rename(columns={'filepath': 'path'})

        # Optional: filter to Pseudo Dream > image subset when possible
        if PSEUDO_DREAM_ONLY:
            self._apply_pseudo_dream_filter()

        # Ensure split column exists
        self._ensure_splits()
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

    def _apply_pseudo_dream_filter(self):
        """Filter rows to Pseudo Dream > image subset when metadata allows.

        Heuristics:
        - If a column named 'modality' or 'type' exists, keep rows where value == 'image'.
        - Else if a column named 'folder' or 'subset' exists, keep rows where value == 'image'.
        - Else if file path contains '/image/' or '\\image\\', keep.
        If none of the above applies, keep all rows.
        """
        df = self.labels_df
        kept = None
        for col in ['modality', 'type', 'folder', 'subset', 'source']:
            if col in df.columns:
                kept = df[df[col].astype(str).str.lower() == 'image']
                break
        if kept is None and 'path' in df.columns:
            mask = df['path'].astype(str).str.contains(r"[\\/]+image[\\/]+", regex=True)
            kept = df[mask]
        if kept is not None and len(kept) > 0:
            self.labels_df = kept.reset_index(drop=True)

    def _ensure_splits(self):
        """Create deterministic train/val/test splits if not present."""
        df = self.labels_df
        if 'split' in df.columns:
            return
        rng = np.random.default_rng(RANDOM_SEED)
        df = df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

        # Stratify by label if available
        if 'label' in df.columns:
            parts = []
            for label_value, g in df.groupby('label', sort=False):
                n = len(g)
                n_train = int(round(n * TRAIN_RATIO))
                n_val = int(round(n * VAL_RATIO))
                # ensure total does not exceed
                n_train = min(n_train, n)
                n_val = min(n_val, max(0, n - n_train))
                n_test = n - n_train - n_val
                g = g.reset_index(drop=True)
                g.loc[: n_train - 1, 'split'] = 'train'
                g.loc[n_train : n_train + n_val - 1, 'split'] = 'val'
                g.loc[n_train + n_val :, 'split'] = 'test'
                parts.append(g)
            df = pd.concat(parts, ignore_index=True)
        else:
            n = len(df)
            n_train = int(round(n * TRAIN_RATIO))
            n_val = int(round(n * VAL_RATIO))
            df.loc[: n_train - 1, 'split'] = 'train'
            df.loc[n_train : n_train + n_val - 1, 'split'] = 'val'
            df.loc[n_train + n_val :, 'split'] = 'test'

        self.labels_df = df
    
    def get_reference_images(self, num_per_class: int = 3, splits: List[str] = None) -> Dict[int, List[str]]:
        """Select reference images for each class from specified splits (defaults to train+val)."""
        if self.labels_df is None:
            self.load_labels()
        if splits is None:
            splits = FEW_SHOT_REFERENCE_SPLITS

        df = self.labels_df
        if 'split' in df.columns and splits:
            df = df[df['split'].isin(splits)]

        reference_images = {}
        for label in [0, 1]:  # 0: normal, 1: glaucoma
            class_images = df[df['label'] == label]['path'].tolist()
            if len(class_images) >= num_per_class:
                reference_images[label] = random.sample(class_images, num_per_class)
            else:
                reference_images[label] = class_images
        return reference_images
    
    def get_test_images(self, sample_size: int = 100) -> List[Tuple[str, int]]:
        """Get test images with their labels strictly from the test split."""
        if self.labels_df is None:
            self.load_labels()
        df = self.labels_df
        if 'split' in df.columns:
            df = df[df['split'] == 'test']
        # Sample images for testing
        if len(df) > sample_size:
            test_df = df.sample(n=sample_size, random_state=RANDOM_SEED)
        else:
            test_df = df
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
