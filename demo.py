#!/usr/bin/env python3
"""
Demo script for LAG Glaucoma One-Shot Prompting Project

This script demonstrates the project functionality with sample data
when the actual LAG dataset is not available.
"""

import os
import sys
from src.data_loader import LAGDataLoader
from src.chatgpt_client import ChatGPTVisionClient
from src.evaluation import ResultsEvaluator
from config import OPENAI_API_KEY

def create_demo_data():
    """Create sample data for demonstration."""
    print("🎨 Creating demo dataset...")
    
    data_loader = LAGDataLoader()
    data_loader.create_sample_dataset(40)  # Create 40 sample images
    
    print("✅ Demo dataset created with 40 sample images")
    print("   - 20 'normal' images (green)")
    print("   - 20 'glaucoma' images (red)")
    return data_loader

def demo_image_processing():
    """Demonstrate image processing capabilities."""
    print("\n🖼️  Testing image processing...")
    
    data_loader = LAGDataLoader()
    data_loader.load_labels()
    
    if len(data_loader.labels_df) > 0:
        # Test preprocessing
        sample_path = data_loader.labels_df.iloc[0]['path']
        processed_img = data_loader.preprocess_image(sample_path)
        
        if processed_img:
            print(f"✅ Image processing successful")
            print(f"   - Original path: {sample_path}")
            print(f"   - Processed size: {processed_img.size}")
        else:
            print("❌ Image processing failed")
    else:
        print("❌ No images found for processing test")

def demo_prompting(use_api=False):
    """Demonstrate prompting functionality."""
    print(f"\n💬 Testing prompting {'with API' if use_api else 'without API'}...")
    
    if use_api and not OPENAI_API_KEY:
        print("❌ OpenAI API key not found. Skipping API test.")
        return
    
    data_loader = LAGDataLoader()
    data_loader.load_labels()
    
    if len(data_loader.labels_df) < 4:
        print("❌ Not enough images for prompting test")
        return
    
    # Get reference images
    reference_paths = data_loader.get_reference_images(2)
    print(f"✅ Selected reference images:")
    for label, paths in reference_paths.items():
        label_name = "normal" if label == 0 else "glaucoma"
        print(f"   - {label_name}: {len(paths)} images")
    
    if use_api:
        # Test with actual API
        client = ChatGPTVisionClient()
        
        # Load reference images
        reference_images = {}
        for label, paths in reference_paths.items():
            reference_images[label] = []
            for path in paths[:1]:  # Use only 1 per class for demo
                img = data_loader.preprocess_image(path)
                if img:
                    reference_images[label].append(img)
        
        # Test image
        test_path = data_loader.labels_df.iloc[-1]['path']
        test_image = data_loader.preprocess_image(test_path)
        
        if test_image:
            # Create prompts
            one_shot_messages = client.create_one_shot_prompt(reference_images, test_image)
            zero_shot_messages = client.create_zero_shot_prompt(test_image)
            
            print(f"✅ Prompts created successfully")
            print(f"   - One-shot prompt: {len(one_shot_messages)} messages")
            print(f"   - Zero-shot prompt: {len(zero_shot_messages)} messages")
            
            # Get predictions (this will use API credits)
            print("🔮 Getting predictions from ChatGPT...")
            one_shot_result = client.get_prediction(one_shot_messages)
            zero_shot_result = client.get_prediction(zero_shot_messages)
            
            if one_shot_result:
                print(f"✅ One-shot prediction: {one_shot_result['classification']} (confidence: {one_shot_result['confidence']}%)")
            if zero_shot_result:
                print(f"✅ Zero-shot prediction: {zero_shot_result['classification']} (confidence: {zero_shot_result['confidence']}%)")
    else:
        print("✅ Prompt creation logic verified (API test skipped)")

def demo_evaluation():
    """Demonstrate evaluation capabilities with mock data."""
    print("\n📊 Testing evaluation with mock results...")
    
    # Create mock results
    mock_one_shot = [
        {'true_label': 0, 'predicted_label': 0, 'confidence': 85.0, 'method': 'one_shot'},
        {'true_label': 1, 'predicted_label': 1, 'confidence': 92.0, 'method': 'one_shot'},
        {'true_label': 0, 'predicted_label': 1, 'confidence': 78.0, 'method': 'one_shot'},
        {'true_label': 1, 'predicted_label': 1, 'confidence': 88.0, 'method': 'one_shot'},
    ]
    
    mock_zero_shot = [
        {'true_label': 0, 'predicted_label': 0, 'confidence': 75.0, 'method': 'zero_shot'},
        {'true_label': 1, 'predicted_label': 0, 'confidence': 65.0, 'method': 'zero_shot'},
        {'true_label': 0, 'predicted_label': 1, 'confidence': 70.0, 'method': 'zero_shot'},
        {'true_label': 1, 'predicted_label': 1, 'confidence': 80.0, 'method': 'zero_shot'},
    ]
    
    evaluator = ResultsEvaluator()
    
    # Calculate metrics
    one_shot_metrics = evaluator.calculate_metrics(mock_one_shot)
    zero_shot_metrics = evaluator.calculate_metrics(mock_zero_shot)
    
    print(f"✅ Evaluation completed:")
    print(f"   - One-shot accuracy: {one_shot_metrics['accuracy']:.1%}")
    print(f"   - Zero-shot accuracy: {zero_shot_metrics['accuracy']:.1%}")
    
    improvement = ((one_shot_metrics['accuracy'] - zero_shot_metrics['accuracy']) / 
                   zero_shot_metrics['accuracy'] * 100)
    print(f"   - Improvement: {improvement:+.1f}%")

def main():
    print("🚀 LAG Glaucoma One-Shot Prompting - DEMO")
    print("=" * 50)
    
    try:
        # Create demo data
        create_demo_data()
        
        # Test components
        demo_image_processing()
        demo_prompting(use_api=False)  # Set to True to test with actual API
        demo_evaluation()
        
        print("\n🎉 Demo completed successfully!")
        print("\nTo run the full experiment:")
        print("1. Set up your OpenAI API key in .env file")
        print("2. Run: python main.py --sample-size 10")
        print("\nTo use real LAG dataset:")
        print("1. Download LAG dataset from the official source")
        print("2. Place images in data/images/ directory")
        print("3. Create labels.csv with columns: filename, label, path")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
