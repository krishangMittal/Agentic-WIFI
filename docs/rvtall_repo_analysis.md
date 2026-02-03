# RVTALL Repository Analysis

## Repository: Multimodal-dataset-for-human-speech-recognition

This repository contains **extremely valuable** code and resources for your RF sensing research! Here's what you can use:

## 🎯 Key Findings

### 1. **Training Pipeline** (`network/train.py`)
- ✅ Complete training script with ResNet models
- ✅ **15 classes confirmed** (`--nclass 15`)
- ✅ Data loading, training loop, validation
- ✅ Loss tracking and visualization
- ✅ Ready to use with your dataset!

**Key parameters:**
```python
--nclass 15          # 15 command words
--net resnet18       # Model architecture
--batch 8            # Batch size
--epoch 70           # Training epochs
--lr 0.0001          # Learning rate
```

### 2. **Dataset Loaders** (`network/loaddataset.py` and `src/MMSdataset/dataset_load.py`)

**Multiple dataset classes available:**

#### `uwbDataset` - For UWB spectrograms (⭐ Most relevant!)
- Loads **`.npy` files** (preprocessed spectrograms)
- Resizes to **256x256** (perfect for ResNet!)
- Label extraction: `filename.split('_')[2]` (3rd part = word ID)
- Normalization: z-score (`stats.zscore`)
- Takes absolute value of spectrogram
- **Perfect for your spectrogram approach!**

```python
# Example usage:
from network.loaddataset import uwbDataset

dataset = uwbDataset(
    data_dir='path/to/data',
    data_list=file_list,
    num_classes=15,
    norm=True,      # z-score normalization
    abs=True,       # absolute value
    img_size=(256, 256)  # ResNet input size
)
```

#### `TimeDataset` - For raw `.mat` files
- Loads `.mat` files with spectrogram data
- Column: `"dop_spec_ToF"` (Doppler spectrogram)
- Supports normalization and downsampling
- Use if you have raw MATLAB files

#### `TimeDataset3C` - For multi-channel (3 channels)
- Loads 3-channel spectrograms
- Useful for multi-modal fusion

#### `multimodal_uwb` - For video-like sequences
- Processes data as sequences (time-series of images)
- Uses Resize transforms included

#### Other dataset classes:
- `mmwaveDataset` - For mmWave radar data
- `laserDataset` - For laser signals
- `videoDataset` - For video sequences
- `uwbaudDataset`, `uwbvidDataset` - Multi-modal (UWB + audio/video)

### 3. **Pre-trained Models** (`network/models/`)

**Available architectures:**
- ✅ `resnet18()` - Lightweight, fast training
- ✅ `resnet34()` - Medium size
- ✅ `resnet50()` - Larger, more capacity
- ✅ `resnet101()`, `resnet152()` - Very large
- ✅ `dualresnet18()` - Dual input (2 spectrograms)
- ✅ `dualresnet3d()` - 3D convolutions for sequences
- ✅ `r3d_18()` - 3D ResNet for video

**Important:** These are **custom implementations** (not pre-trained on ImageNet), but the architecture is solid and you can:
- Use them as-is for training from scratch
- Or adapt your `CommandClassifierResNet` to use ImageNet pre-trained weights

### 4. **Preprocessing Code** (`RVTALL-Preprocess-main/`)

**UWB Processing** (`uwb_cutting.py`):
- ✅ Segments UWB radar data using timestamps
- ✅ Extracts spectrograms from `.mat` files
- ✅ Saves as `.npy` files ready for training
- ✅ Handles synchronization with Kinect timestamps

**Key function:**
```python
uwb_sample = uwbmat['Data_MicroDop_2'][922:1127, start_idx:end_idx+1]
# Extracts frequency range 922:1127, time range from timestamps
```

**Other processors:**
- `mmWave_cutting.py` - mmWave radar processing
- `laser_cutting.py` - Laser signal processing
- `kinect_cutting.py` - Kinect face/lip point processing

### 5. **Data Format Understanding**

**Two data formats supported:**

#### Format 1: Preprocessed `.npy` files (Recommended)
- **File type:** `.npy` (NumPy arrays)
- **Content:** Preprocessed spectrograms (already extracted)
- **Shape:** `[frequency_bins, time_frames]` → resized to `[256, 256]`
- **Label extraction:** `filename.split('_')[2] - 1` (3rd part = word ID, 0-indexed)
- **Used by:** `uwbDataset`, `mmwaveDataset`, `laserDataset`

**Example filename pattern:**
```
EXPID_TASKID_WORDID_RADARID_sample.npy
# Label = int(filename.split('_')[2]) - 1
# Example: "1_2_5_1_sample.npy" → word ID = 5, label = 4 (0-indexed)
```

#### Format 2: Raw `.mat` files
- **File type:** `.mat` (MATLAB format)
- **Key column:** `"dop_spec_ToF"` - Doppler spectrogram
- **Shape:** `[frequency_bins, time_frames]`
- **Label extraction:** `filename.split('-')[1]` (word ID)
- **Used by:** `TimeDataset`

**Example filename pattern:**
```
EXPID-TASKID-PERSONID-RADARID_sample.mat
# Label = int(filename.split('-')[1]) - 1
```

**Processing pipeline:**
1. Raw data → Preprocessing (UWB cutting) → `.npy` files
2. `.npy` files → Dataset loader → Resize to 256x256 → Training

### 6. **Configuration** (`network/conf.py`)

- Model selection system
- GPU support
- Easy to extend with your models

## 🚀 How to Use This Repository

### Option 1: Use Their Training Code Directly

1. **Prepare your data** in their format:
   ```
   data/
   ├── train/
   │   ├── 1-word1-sample1.mat
   │   ├── 1-word2-sample1.mat
   │   └── ...
   └── test/
       └── ...
   ```

2. **Run training:**
   ```bash
   cd Multimodal-dataset-for-human-speech-recognition/network
   python train.py --nclass 15 --net resnet18 --data_dir ../data/RVTALL
   ```

### Option 2: Adapt Their Code to Your Project

**Integrate their dataset loader into your project:**

1. **Copy dataset loader:**
   ```python
   # Adapt src/MMSdataset/dataset_load.py
   # Use TimeDataset for spectrogram classification
   ```

2. **Use their preprocessing:**
   ```python
   # Adapt RVTALL-Preprocess-main/uwb_cutting.py
   # To convert your CSI data to their format
   ```

3. **Combine with your pre-trained ResNet:**
   - Use their `TimeDataset` loader
   - Use your `CommandClassifierResNet` (ImageNet pre-trained)
   - Best of both worlds!

### Option 3: Extract Valuable Patterns

**Key insights from their code:**

1. **Data normalization:**
   ```python
   radar_mean = (3.4269)
   radar_std = (72.9885)
   # They normalize spectrograms
   ```

2. **Spectrogram extraction:**
   - They use `dop_spec_ToF` column (Doppler spectrogram)
   - Frequency range: 922:1127 (specific to their radar)
   - You'll need to adapt to your CSI data

3. **Label extraction:**
   ```python
   label = int(data_file_name.split('-')[1]) - 1
   # Word ID is in filename
   ```

## 📊 What You Can Learn

1. **Data structure:** How RVTALL organizes 15 words
2. **Preprocessing:** How to segment and align RF signals
3. **Model architecture:** ResNet variants for spectrograms
4. **Training pipeline:** Complete training/validation loop
5. **Evaluation:** Confusion matrix, accuracy tracking

## ⚠️ Important Notes

1. **Data format:** Their code expects `.mat` files with specific structure
2. **Preprocessing needed:** You may need to convert your CSI data to their format
3. **Model weights:** Their ResNet models are NOT pre-trained (from scratch)
4. **Adaptation required:** Their code is specific to their dataset structure

## 🎯 Recommended Next Steps

1. **Inspect the dataset structure:**
   ```bash
   python scripts/inspect_rvtall.py
   # See how RVTALL organizes the 15 words
   ```

2. **Check if dataset is in the repo:**
   - Look in `dataset/multimodal_uwb/`
   - May need to download separately

3. **Adapt their dataset loader:**
   - Copy `TimeDataset` class
   - Modify to work with your data format
   - Use with your pre-trained ResNet model

4. **Use their preprocessing as reference:**
   - Understand how they extract spectrograms
   - Adapt to your CSI preprocessing pipeline

## 💡 Best Approach

**Combine their dataset loader with your pre-trained ResNet:**

```python
# Use their dataset loader
from Multimodal-dataset-for-human-speech-recognition.src.MMSdataset.dataset_load import TimeDataset

# Use your pre-trained model
from src.model import CommandClassifierResNet

# Best of both worlds!
dataset = TimeDataset(...)  # Their loader
model = CommandClassifierResNet(num_classes=15)  # Your pre-trained model
```

This gives you:
- ✅ Their proven data loading pipeline
- ✅ Your ImageNet pre-trained ResNet (more robust)
- ✅ Faster training with less data

## 📁 Repository Structure Summary

```
Multimodal-dataset-for-human-speech-recognition/
├── network/
│   ├── train.py              # ⭐ Training script
│   ├── conf.py               # Model configuration
│   ├── loaddataset.py        # Dataset loading utilities
│   └── models/                # ⭐ ResNet implementations
│       ├── resnet.py         # ResNet18/34/50/101/152
│       ├── dual_resnet.py    # Dual input ResNet
│       └── ...
├── src/MMSdataset/
│   └── dataset_load.py       # ⭐ Dataset classes
├── RVTALL-Preprocess-main/   # ⭐ Preprocessing code
│   ├── uwb_cutting.py        # UWB spectrogram extraction
│   ├── mmWave_cutting.py    # mmWave processing
│   └── ...
└── dataset/
    └── multimodal_uwb/      # Dataset location (may be empty)
```

## 🎉 Conclusion

This repository is **extremely valuable**! It provides:
- ✅ Complete training pipeline
- ✅ Proven dataset loaders
- ✅ Working ResNet implementations
- ✅ Preprocessing code for RF signals
- ✅ Confirmation of 15 classes

**You can use this as a reference or directly adapt their code to your project!**

