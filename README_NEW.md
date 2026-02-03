# WiFi Gesture Recognition System

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
source venv/bin/activate  # On Windows: venv\Scripts\activate
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
