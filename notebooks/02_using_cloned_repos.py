"""
Example: Using the Best Components from Cloned RVTALL Repositories

This notebook shows how to leverage the pre-trained models and utilities
from the cloned repositories for your RF sensing research.
"""

import sys
from pathlib import Path

# Add repository paths to Python path
repo_root = Path(__file__).parent.parent
sys.path.append(str(repo_root / "RVTALL-Preprocess" / "classification"))
sys.path.append(str(repo_root / "Multimodal-dataset-for-human-speech-recognition" / "network"))

# ==============================================================================
# OPTION 1: Pre-trained ResNet from RVTALL-Preprocess (Recommended!)
# ==============================================================================

print("="*70)
print("OPTION 1: Using Pre-trained ResNet from RVTALL-Preprocess")
print("="*70 + "\n")

try:
    from models import CustomResNet, MultiInResNet
    from train import Trainer
    
    # Single modality (e.g., UWB only)
    model_single = CustomResNet(
        in_channels=3,      # 3 channels for RGB spectrograms
        num_classes=15,     # 15 word commands
        pre_trained=True,   # Load ImageNet pre-trained weights!
        model='resnet18'    # resnet18, resnet34, resnet101, resnet152
    )
    
    print("[OK] Single-input ResNet created with pre-trained weights")
    print(f"  - Input channels: 3 (RGB spectrograms)")
    print(f"  - Output classes: 15 (word commands)")
    print(f"  - Pre-trained: Yes (ImageNet)")
    print(f"  - Architecture: ResNet18\n")
    
    # Multi-modal (e.g., UWB + FMCW)
    model_multi = MultiInResNet(
        num_inputs=2,           # 2 modalities
        num_classes=15,         # 15 word commands
        num_in_convs=[1, 1],   # 1 conv layer per input
        in_channels=[3, 3],     # 3 channels each
        out1_channels=[3, 3],   # Output 3 channels each
        model='resnet18'        # Pre-trained ResNet18
    )
    
    print("[OK] Multi-input ResNet created for multi-modal fusion")
    print(f"  - Number of inputs: 2 (e.g., UWB + FMCW)")
    print(f"  - Input channels: [3, 3]")
    print(f"  - Output classes: 15")
    print(f"  - Pre-trained backbone: ResNet18\n")
    
    print("Key advantages:")
    print("  [+] Pre-trained on ImageNet (millions of images)")
    print("  [+] Robust feature extraction")
    print("  [+] Less data needed for training")
    print("  [+] Better generalization\n")
    
except ImportError as e:
    print(f"[WARNING] Could not import from RVTALL-Preprocess: {e}")
    print("  Make sure PyTorch is installed: pip install torch torchvision")
    print("  Or create conda environment: conda env create -f environment.yml\n")

# ==============================================================================
# OPTION 2: Dataset Loaders from Multimodal Repository
# ==============================================================================

print("="*70)
print("OPTION 2: Using Dataset Loaders from Multimodal Repository")
print("="*70 + "\n")

try:
    import loaddataset
    
    print("Available dataset loaders:")
    print("  - uwbDataset: Loads UWB spectrograms (.npy -> 256x256)")
    print("  - mmwaveDataset: Loads mmWave spectrograms")
    print("  - laserDataset: Loads laser signals")
    print("  - videoDataset: Loads video sequences")
    print("  - uwbaudDataset: Multi-modal (UWB + audio)")
    print("  - uwbvidDataset: Multi-modal (UWB + video)\n")
    
    print("Example usage:")
    print("""
    from loaddataset import uwbDataset
    
    dataset = uwbDataset(
        data_dir='data/RVTALL/train',
        data_list=file_list,
        num_classes=15,
        norm=True,          # z-score normalization
        abs=True,           # absolute value
        img_size=(256, 256) # Resize to 256x256
    )
    """)
    
except ImportError as e:
    print(f"[WARNING] Could not import from Multimodal repository: {e}")
    print("  Make sure the repository is cloned correctly.\n")

# ==============================================================================
# OPTION 3: Combining Both - Best of Both Worlds!
# ==============================================================================

print("="*70)
print("OPTION 3: Combining Both Repositories (Recommended Approach)")
print("="*70 + "\n")

print("Strategy: Use pre-trained model from Repo 2 with data loaders from Repo 1\n")

print("Example workflow:")
print("""
import torch
from torch.utils.data import DataLoader

# Model from RVTALL-Preprocess (pre-trained)
from models import CustomResNet

# Data loader from Multimodal repository
from loaddataset import uwbDataset

# Create model with pre-trained weights
model = CustomResNet(
    in_channels=3,      # Adjust based on your data
    num_classes=15,
    pre_trained=True,
    model='resnet18'
)

# Load your data
train_dataset = uwbDataset(
    data_dir='data/RVTALL/train',
    data_list=train_files,
    num_classes=15,
    norm=True,
    abs=True
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Train
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for data, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
""")

# ==============================================================================
# Data Preprocessing
# ==============================================================================

print("\n" + "="*70)
print("DATA PREPROCESSING")
print("="*70 + "\n")

print("If you have raw RVTALL data (.mat files):\n")

print("Step 1: Extract spectrograms using sensor_proc.py")
print("""
from sensor_proc import UWBProcessor

processor = UWBProcessor(root_dir='data/RVTALL/raw')
processor._segment_one_exp('uwb_file.mat')
# Output: sample1.npy, sample2.npy, ...
""")

print("\nStep 2: Convert to images (optional, for visualization)")
print("""
from data_preprocess import mat2img

mat2img('data/RVTALL/processed')
# Converts .npy → .png with jet colormap
# Output: sample1.png, sample2.png, ...
""")

print("\nStep 3: Use the dataset loaders to load for training")

# ==============================================================================
# Summary
# ==============================================================================

print("\n" + "="*70)
print("SUMMARY: What You Have Available")
print("="*70 + "\n")

print("From RVTALL-Preprocess/ (Repo 2):")
print("  [+] Pre-trained ResNet models (CustomResNet, MultiInResNet)")
print("  [+] Trainer class for easy training")
print("  [+] Sensor preprocessing (UWB, mmWave, Laser, Kinect)")
print("  [+] Complete training pipeline for 15 words")
print("  [+] Confusion matrix visualization\n")

print("From Multimodal-dataset-for-human-speech-recognition/ (Repo 1):")
print("  [+] Multiple dataset loaders (uwb, mmwave, laser, video)")
print("  [+] Custom ResNet implementations (from scratch)")
print("  [+] MATLAB processing code")
print("  [+] Speech separation (NMF)\n")

print("Recommended approach:")
print("  1. Use pre-trained models from Repo 2 (RVTALL-Preprocess)")
print("  2. Use dataset loaders from Repo 1 or Repo 2 (your choice)")
print("  3. Combine with your existing preprocess.py and model.py")
print("  4. Train on RVTALL dataset (15 word commands)\n")

print("See docs/both_repos_comparison.md for detailed analysis!")

