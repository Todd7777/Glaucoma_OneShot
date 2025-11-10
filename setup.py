#!/usr/bin/env python3
"""
Setup script for LAG Glaucoma One-Shot Prompting Project
"""

import os
import subprocess
import sys

def install_requirements():
    """Install required packages."""
    print("📦 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def setup_env_file():
    """Set up environment file."""
    env_file = ".env"
    env_example = ".env.example"
    
    if not os.path.exists(env_file):
        if os.path.exists(env_example):
            print("📝 Creating .env file from template...")
            with open(env_example, 'r') as f:
                content = f.read()
            with open(env_file, 'w') as f:
                f.write(content)
            print("✅ .env file created! Please add your OpenAI API key.")
        else:
            print("⚠️  Please create a .env file with your OpenAI API key:")
            print("OPENAI_API_KEY=your_api_key_here")
    else:
        print("✅ .env file already exists.")

def create_directories():
    """Create necessary directories."""
    dirs = ["data", "results", "data/images"]
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        print(f"📁 Created directory: {dir_name}")

def main():
    print("🔧 Setting up LAG Glaucoma One-Shot Prompting Project")
    print("=" * 50)
    
    # Install requirements
    if not install_requirements():
        return 1
    
    # Setup environment
    setup_env_file()
    
    # Create directories
    create_directories()
    
    print("\n🎉 Setup complete!")
    print("\nNext steps:")
    print("1. Add your OpenAI API key to the .env file")
    print("2. Place LAG dataset images in the data/images/ directory")
    print("3. Run: python main.py --create-sample-data (for testing)")
    print("4. Run: python main.py (to start the experiment)")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
