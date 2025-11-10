import os
import json
from typing import List, Dict, Tuple
from tqdm import tqdm
import pandas as pd
from PIL import Image

from src.data_loader import LAGDataLoader
from src.chatgpt_client import ChatGPTVisionClient
from config import RESULTS_DIR, REFERENCE_IMAGES_PER_CLASS, SAMPLE_SIZE

class OneShotExperiment:
    """Main class for running one-shot prompting experiments."""
    
    def __init__(self):
        self.data_loader = LAGDataLoader()
        self.chatgpt_client = ChatGPTVisionClient()
        self.results_dir = RESULTS_DIR
        os.makedirs(self.results_dir, exist_ok=True)
        
    def prepare_reference_images(self) -> Dict[int, List[Image.Image]]:
        """Load and preprocess reference images."""
        print("Preparing reference images...")
        
        reference_paths = self.data_loader.get_reference_images(REFERENCE_IMAGES_PER_CLASS)
        reference_images = {}
        
        for label, paths in reference_paths.items():
            reference_images[label] = []
            for path in paths:
                img = self.data_loader.preprocess_image(path)
                if img is not None:
                    reference_images[label].append(img)
                    
        print(f"Loaded {len(reference_images[0])} normal and {len(reference_images[1])} glaucoma reference images")
        return reference_images
    
    def run_one_shot_experiment(self, sample_size: int = SAMPLE_SIZE) -> List[Dict]:
        """Run one-shot prompting experiment."""
        print("Running one-shot prompting experiment...")
        
        # Prepare reference images
        reference_images = self.prepare_reference_images()
        
        # Get test images
        test_images = self.data_loader.get_test_images(sample_size)
        
        results = []
        
        for i, (image_path, true_label) in enumerate(tqdm(test_images, desc="Processing images")):
            # Load and preprocess test image
            test_image = self.data_loader.preprocess_image(image_path)
            if test_image is None:
                continue
                
            # Create one-shot prompt
            messages = self.chatgpt_client.create_one_shot_prompt(reference_images, test_image)
            
            # Get prediction
            prediction = self.chatgpt_client.get_prediction(messages)
            
            if prediction is not None:
                result = {
                    'image_path': image_path,
                    'true_label': true_label,
                    'predicted_label': prediction['classification'],
                    'confidence': prediction['confidence'],
                    'explanation': prediction['explanation'],
                    'raw_response': prediction['raw_response'],
                    'method': 'one_shot'
                }
                results.append(result)
                
                # Print progress
                if (i + 1) % 10 == 0:
                    correct = sum(1 for r in results if r['predicted_label'] == r['true_label'])
                    accuracy = correct / len(results) * 100
                    print(f"Processed {len(results)} images, Current accuracy: {accuracy:.1f}%")
        
        return results
    
    def run_zero_shot_experiment(self, sample_size: int = SAMPLE_SIZE) -> List[Dict]:
        """Run zero-shot prompting experiment for comparison."""
        print("Running zero-shot prompting experiment...")
        
        # Get test images (same as one-shot for fair comparison)
        test_images = self.data_loader.get_test_images(sample_size)
        
        results = []
        
        for i, (image_path, true_label) in enumerate(tqdm(test_images, desc="Processing images")):
            # Load and preprocess test image
            test_image = self.data_loader.preprocess_image(image_path)
            if test_image is None:
                continue
                
            # Create zero-shot prompt
            messages = self.chatgpt_client.create_zero_shot_prompt(test_image)
            
            # Get prediction
            prediction = self.chatgpt_client.get_prediction(messages)
            
            if prediction is not None:
                result = {
                    'image_path': image_path,
                    'true_label': true_label,
                    'predicted_label': prediction['classification'],
                    'confidence': prediction['confidence'],
                    'explanation': prediction['explanation'],
                    'raw_response': prediction['raw_response'],
                    'method': 'zero_shot'
                }
                results.append(result)
                
                # Print progress
                if (i + 1) % 10 == 0:
                    correct = sum(1 for r in results if r['predicted_label'] == r['true_label'])
                    accuracy = correct / len(results) * 100
                    print(f"Processed {len(results)} images, Current accuracy: {accuracy:.1f}%")
        
        return results
    
    def save_results(self, results: List[Dict], filename: str):
        """Save results to JSON and CSV files."""
        # Save as JSON
        json_path = os.path.join(self.results_dir, f"{filename}.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        # Save as CSV
        csv_path = os.path.join(self.results_dir, f"{filename}.csv")
        df = pd.DataFrame(results)
        df.to_csv(csv_path, index=False)
        
        print(f"Results saved to {json_path} and {csv_path}")
        
    def run_full_experiment(self):
        """Run both one-shot and zero-shot experiments."""
        print("Starting full experiment...")
        
        # Ensure we have data
        self.data_loader.load_labels()
        if self.data_loader.labels_df is None or len(self.data_loader.labels_df) == 0:
            print("No dataset found. Creating sample dataset...")
            self.data_loader.create_sample_dataset()
            self.data_loader.load_labels()
        
        # Run one-shot experiment
        one_shot_results = self.run_one_shot_experiment()
        self.save_results(one_shot_results, "one_shot_results")
        
        # Run zero-shot experiment
        zero_shot_results = self.run_zero_shot_experiment()
        self.save_results(zero_shot_results, "zero_shot_results")
        
        # Combine results for comparison
        all_results = one_shot_results + zero_shot_results
        self.save_results(all_results, "combined_results")
        
        return one_shot_results, zero_shot_results
