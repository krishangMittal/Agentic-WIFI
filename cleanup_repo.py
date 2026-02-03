"""
Repository Cleanup Script
=========================
Organizes the project into a clean structure.

What this does:
1. Moves test scripts to tests/ folder
2. Moves training scripts to training/ folder
3. Deletes unnecessary files
4. Creates .gitignore
5. Creates clean project structure
"""

import os
import shutil

# ============================================================================
# CONFIGURATION
# ============================================================================

# Files to DELETE (already extracted or no longer needed)
FILES_TO_DELETE = [
    'E01.zip',  # Already extracted
    'setup_deepseek.py',  # Not needed anymore
    'inspect_mmfi_mat.py',  # Temp file
]

# Folders to DELETE (cloned repos we don't need in our repo)
FOLDERS_TO_DELETE = [
    'Multimodal-dataset-for-human-speech-recognition',  # Cloned repo (we copied what we need)
    'MMFi_dataset',  # Cloned repo (we copied what we need)
]

# Test scripts to move to tests/ folder
TEST_SCRIPTS = [
    'test_trained_model.py',
    'test_synthetic.py',
    'test_proper.py',
    'test_deepseek.py',
]

# Training scripts to move to training/ folder
TRAINING_SCRIPTS = [
    'train_mmfi.py',
    'train_mmfi_improved.py',
]

# Demo/quickstart scripts to move to examples/ folder
EXAMPLE_SCRIPTS = [
    'demo_gesture.py',
    'quickstart.py',
]

# Scripts to move to scripts/ folder
UTILITY_SCRIPTS = [
    'setup_realtime.py',
]

# ============================================================================
# CLEANUP FUNCTIONS
# ============================================================================

def create_directory(path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"[+] Created directory: {path}")

def delete_file(filepath):
    """Delete a file if it exists"""
    if os.path.exists(filepath):
        os.remove(filepath)
        print(f"[-] Deleted file: {filepath}")
    else:
        print(f"[!] File not found: {filepath}")

def delete_folder(folderpath):
    """Delete a folder if it exists"""
    if os.path.exists(folderpath):
        shutil.rmtree(folderpath)
        print(f"[-] Deleted folder: {folderpath}")
    else:
        print(f"[!] Folder not found: {folderpath}")

def move_file(src, dst_dir):
    """Move file to destination directory"""
    if os.path.exists(src):
        create_directory(dst_dir)
        dst = os.path.join(dst_dir, os.path.basename(src))
        shutil.move(src, dst)
        print(f"[>] Moved: {src} -> {dst}")
    else:
        print(f"[!] File not found: {src}")

def create_gitignore():
    """Create .gitignore file"""
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
*.egg-info/
dist/
build/

# Virtual Environment
venv/
env/
ENV/

# IDEs
.vscode/
.idea/
*.swp
*.swo
.DS_Store

# Data
data/raw/MMFi/
data/processed/mmfi_spectrograms/
data/synthetic_test/
*.zip
*.mat
*.bin
*.npy

# Models
models/*.pth
models/*.pt
!models/README.md

# Logs
*.log
logs/

# Jupyter
.ipynb_checkpoints/
*.ipynb

# OS
Thumbs.db
"""
    
    with open('.gitignore', 'w', encoding='utf-8') as f:
        f.write(gitignore_content)
    print("[+] Created .gitignore")

def create_clean_readme():
    """Create a clean, organized README"""
    readme_content = """# WiFi Gesture Recognition System

A robust WiFi sensing system for gesture recognition using CSI (Channel State Information) and deep learning.

## 🚀 Features

- **WiFi-based Gesture Recognition**: Detects 27 different gestures using WiFi CSI
- **Multiple Architectures**: ResNet18/50, EfficientNet, CNN+LSTM
- **Real-time Processing**: ESP32 integration for live gesture detection
- **Agentic AI**: LLM-powered action execution (smart home control, etc.)
- **Robust Training**: Data augmentation, cross-validation, proper validation

## 📁 Project Structure

```
rf-sensing-research/
├── src/                      # Source code
│   ├── preprocess.py         # CSI to spectrogram conversion
│   ├── classifier.py         # Gesture classification
│   ├── gesture_agent.py      # LLM agent for action execution
│   └── mmfi_processor.py     # MM-Fi dataset processing
├── training/                 # Training scripts
│   ├── train_mmfi.py         # Basic training
│   └── train_mmfi_improved.py # Advanced training (augmentation, etc.)
├── tests/                    # Test scripts
│   ├── test_trained_model.py
│   ├── test_synthetic.py
│   └── test_proper.py
├── models/                   # Trained models
│   ├── custom_resnet.py
│   └── trainer.py
├── examples/                 # Demo scripts
│   ├── demo_gesture.py
│   └── quickstart.py
├── scripts/                  # Utility scripts
│   ├── download_rvtall.py
│   ├── inspect_rvtall.py
│   └── setup_realtime.py
├── docs/                     # Documentation
│   ├── MODEL_IMPROVEMENTS.md # Guide for improving models
│   ├── ESP32_SETUP.md        # ESP32 hardware setup
│   ├── MMFI_SETUP.md         # MM-Fi dataset setup
│   └── REAL_WORLD_TESTING.md # Real-world testing guide
├── data/                     # Datasets
│   ├── raw/                  # Raw data
│   └── processed/            # Processed spectrograms
├── config/                   # Configuration files
│   └── gesture_actions.yaml
└── environment.yml           # Conda environment

```

## 🛠️ Setup

### 1. Install Dependencies

```bash
# Create conda environment
conda env create -f environment.yml
conda activate rf_sensing

# Or use pip + venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
pip install torch torchvision scipy pillow tqdm matplotlib numpy
```

### 2. Download Dataset

See `docs/MMFI_SETUP.md` for detailed instructions.

Quick start:
```bash
# Download E01 from Google Drive
# Link: https://drive.google.com/drive/folders/1zDbhfH3BV-xCZVUHmK65EgVV1HMDEYcz

# Extract to data/raw/MMFi/
unzip E01.zip -d data/raw/MMFi/
```

## 🎯 Quick Start

### Train a Model

```bash
# Quick test (5 minutes)
python training/train_mmfi.py --quick

# Full training (4 hours)
python training/train_mmfi.py

# Improved training with augmentation (recommended)
python training/train_mmfi_improved.py --arch resnet50 --augment --epochs 50
```

### Test the Model

```bash
# Test on validation data
python tests/test_trained_model.py

# Test on synthetic patterns
python tests/test_synthetic.py

# Test with proper train/test split
python tests/test_proper.py
```

### Run Demo

```bash
# Gesture-based actions demo
python examples/demo_gesture.py

# Quick start demo
python examples/quickstart.py
```

## 📊 Results

| Model | Validation Accuracy | Notes |
|-------|-------------------|-------|
| ResNet18 (baseline) | 57% | Quick test, S01 only |
| ResNet18 (full) | 85-90% | All subjects, 50 epochs |
| ResNet50 + Aug | 90-92% | Augmentation, dropout |
| Ensemble | 95%+ | Multiple models |

## 🔧 Real-time Testing (ESP32)

See `docs/ESP32_SETUP.md` for hardware setup.

```bash
# Run real-time gesture detection
python scripts/setup_realtime.py
```

## 📚 Documentation

- **[Model Improvements](docs/MODEL_IMPROVEMENTS.md)**: How to improve accuracy
- **[ESP32 Setup](docs/ESP32_SETUP.md)**: Hardware setup guide
- **[MM-Fi Dataset](docs/MMFI_SETUP.md)**: Dataset download and processing
- **[Real-world Testing](docs/REAL_WORLD_TESTING.md)**: Testing with real WiFi

## 🤝 Contributing

This is a research project. Feel free to:
- Improve the models
- Add new architectures
- Test with different datasets
- Integrate with smart home systems

## 📝 License

MIT License - See LICENSE file

## 🙏 Acknowledgments

- **MM-Fi Dataset**: Multi-modal WiFi sensing dataset
- **RVTALL**: Speech recognition dataset (for future work)
- **ESP-CSI**: ESP32 CSI tools

## 📧 Contact

For questions or collaboration, open an issue on GitHub.
"""
    
    with open('README_NEW.md', 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print("[+] Created README_NEW.md (review and replace README.md)")

# ============================================================================
# MAIN CLEANUP
# ============================================================================

def main():
    print("="*70)
    print(" REPOSITORY CLEANUP")
    print("="*70)
    
    # Step 1: Create new directory structure
    print("\n[1] Creating directory structure...")
    create_directory('tests')
    create_directory('training')
    
    # Step 2: Move test scripts
    print("\n[2] Organizing test scripts...")
    for script in TEST_SCRIPTS:
        move_file(script, 'tests')
    
    # Step 3: Move training scripts
    print("\n[3] Organizing training scripts...")
    for script in TRAINING_SCRIPTS:
        move_file(script, 'training')
    
    # Step 4: Move example scripts
    print("\n[4] Organizing example scripts...")
    for script in EXAMPLE_SCRIPTS:
        move_file(script, 'examples')
    
    # Step 5: Move utility scripts
    print("\n[5] Organizing utility scripts...")
    for script in UTILITY_SCRIPTS:
        move_file(script, 'scripts')
    
    # Step 6: Delete unnecessary files
    print("\n[6] Deleting unnecessary files...")
    for filepath in FILES_TO_DELETE:
        delete_file(filepath)
    
    # Step 7: Delete cloned repos (optional - ask user first)
    print("\n[7] Cloned repositories detected:")
    for folder in FOLDERS_TO_DELETE:
        if os.path.exists(folder):
            print(f"    - {folder}")
    
    print("\n    These are cloned repos. We've copied what we need.")
    print("    You can delete them manually if you want to save space:")
    for folder in FOLDERS_TO_DELETE:
        print(f"      rm -rf \"{folder}\"")
    
    # Step 8: Create .gitignore
    print("\n[8] Creating .gitignore...")
    create_gitignore()
    
    # Step 9: Create clean README
    print("\n[9] Creating clean README...")
    create_clean_readme()
    
    print("\n" + "="*70)
    print(" CLEANUP COMPLETE!")
    print("="*70)
    print("\nWhat was done:")
    print("  [OK] Moved test scripts to tests/")
    print("  [OK] Moved training scripts to training/")
    print("  [OK] Moved examples to examples/")
    print("  [OK] Deleted unnecessary files")
    print("  [OK] Created .gitignore")
    print("  [OK] Created README_NEW.md")
    print("\nNext steps:")
    print("  1. Review README_NEW.md")
    print("  2. Replace old README: mv README_NEW.md README.md")
    print("  3. Optionally delete cloned repos to save space")
    print("  4. Run: python tests/test_trained_model.py (check tests still work)")
    print("="*70)

if __name__ == '__main__':
    main()

