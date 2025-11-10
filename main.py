#!/usr/bin/env python3
"""
LAG Glaucoma One-Shot Prompting Experiment

This script runs an experiment to test one-shot prompting with ChatGPT 
for glaucoma diagnosis using the LAG dataset.
"""

import os
import sys
import argparse
from src.one_shot_prompt import OneShotExperiment
from src.evaluation import ResultsEvaluator
from config import OPENAI_API_KEY

def check_requirements():
    """Check if all requirements are met."""
    if not OPENAI_API_KEY:
        print("❌ Error: OpenAI API key not found!")
        print("Please set your API key in a .env file:")
        print("OPENAI_API_KEY=your_api_key_here")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Run LAG Glaucoma One-Shot Prompting Experiment')
    parser.add_argument('--sample-size', type=int, default=20, 
                       help='Number of test images to evaluate (default: 20)')
    parser.add_argument('--skip-experiment', action='store_true',
                       help='Skip running experiment and only evaluate existing results')
    parser.add_argument('--create-sample-data', action='store_true',
                       help='Create sample dataset for testing')
    
    args = parser.parse_args()
    
    print("🔬 LAG Glaucoma One-Shot Prompting Experiment")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        return 1
    
    try:
        # Initialize experiment
        experiment = OneShotExperiment()
        evaluator = ResultsEvaluator()
        
        if args.create_sample_data:
            print("Creating sample dataset...")
            experiment.data_loader.create_sample_dataset(args.sample_size * 2)
            print("✅ Sample dataset created!")
            return 0
        
        if not args.skip_experiment:
            print(f"🚀 Starting experiment with {args.sample_size} test images...")
            
            # Update sample size in config if different
            import config
            config.SAMPLE_SIZE = args.sample_size
            
            # Run experiments
            one_shot_results, zero_shot_results = experiment.run_full_experiment()
            
            print("✅ Experiments completed!")
            
        else:
            print("📊 Loading existing results for evaluation...")
            try:
                one_shot_results = evaluator.load_results("one_shot_results")
                zero_shot_results = evaluator.load_results("zero_shot_results")
            except FileNotFoundError:
                print("❌ No existing results found. Run experiment first.")
                return 1
        
        # Generate evaluation report
        print("\n📈 Generating evaluation report...")
        results = evaluator.generate_report(one_shot_results, zero_shot_results)
        
        # Print key findings
        print("\n🎯 KEY FINDINGS:")
        one_shot_acc = results['one_shot_metrics'].get('accuracy', 0)
        zero_shot_acc = results['zero_shot_metrics'].get('accuracy', 0)
        improvement = ((one_shot_acc - zero_shot_acc) / zero_shot_acc * 100) if zero_shot_acc > 0 else 0
        
        print(f"• One-shot accuracy: {one_shot_acc:.1%}")
        print(f"• Zero-shot accuracy: {zero_shot_acc:.1%}")
        print(f"• Improvement: {improvement:+.1f}%")
        
        if improvement > 0:
            print("✅ One-shot prompting shows improvement over zero-shot!")
        else:
            print("⚠️  One-shot prompting did not improve performance.")
        
        print(f"\n📁 All results saved in: {experiment.results_dir}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️  Experiment interrupted by user.")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
