# Complete RVTALL Repositories Analysis

You have **TWO valuable repositories** with complementary features! Here's what each provides:

---

## Repository 1: `Multimodal-dataset-for-human-speech-recognition`

### Location
`Multimodal-dataset-for-human-speech-recognition/`

### Key Components

#### 1. Training Pipeline (`network/train.py`)
- ✅ Complete CNN training script
- ✅ ResNet-based architecture
- ✅ 15 classes confirmed
- ✅ Training/validation loop
- ✅ Real-time plotting of losses and accuracy

#### 2. Dataset Loaders (`network/loaddataset.py`)
- ✅ `uwbDataset` - UWB spectrograms (.npy files)
- ✅ `mmwaveDataset` - mmWave radar data
- ✅ `laserDataset` - Laser signals
- ✅ `videoDataset` - Video sequences
- ✅ Multi-modal datasets (UWB+audio, UWB+video)

#### 3. Model Architectures (`network/models/`)
- ✅ Custom ResNet (18, 34, 50, 101, 152)
- ✅ Dual ResNet for multi-modal fusion
- ✅ 3D ResNet for video sequences
- ✅ From scratch implementations

#### 4. Preprocessing (`RVTALL-Preprocess-main/`)
- Basic sensor processing utilities
- Similar to Repo 2 but older version

#### 5. Speech Separation (`Speech-separation-NMF/`)
- ✅ Non-negative Matrix Factorization (NMF) for audio separation
- ✅ MATLAB code for speech separation

#### 6. MATLAB Processing (`processing_code_mat/`)
- ✅ FMCW Radar processing
- ✅ UWB radar processing
- ✅ Plotting utilities

---

## Repository 2: `RVTALL-Preprocess` ⭐ (More Advanced!)

### Location
`RVTALL-Preprocess/`

### Key Components

#### 1. **Classification Module** (`classification/`) ⭐ UNIQUE & POWERFUL!

**Main Training Script** (`main.py`):
- ✅ **Multi-modal fusion** - combines UWB + FMCW radar
- ✅ **Pre-trained ResNet** from PyTorch Hub
- ✅ Trains on **15 word commands** (not vowels/sentences)
- ✅ Complete data loading from multiple subjects
- ✅ Train/test split (80/20)
- ✅ Confusion matrix visualization

**Key features:**
```python
# Uses 2 inputs: UWB + FMCW spectrograms
classifier = MultiInResNet(
    num_inputs=2,
    num_classes=15,        # 15 word commands!
    num_in_convs=[1, 1],
    in_channels=[3, 3],    # 3-channel images
    out1_channels=[3, 3],
    model='resnet18'       # Pre-trained ResNet18!
)
```

**Model Architecture** (`models.py`):
- ✅ `CustomResNet` - Single input with pre-trained ResNet
- ✅ `MultiInResNet` - **Multi-modal** with pre-trained ResNet
- ✅ Uses **PyTorch Hub** for pre-trained weights!
- ✅ Flexible input channels (adapts first conv layer)
- ✅ Custom fully connected layer for 15 classes

**Training Class** (`train.py`):
- ✅ Clean, modular `Trainer` class
- ✅ Learning rate scheduler
- ✅ Separate train/test iterations
- ✅ Accuracy and loss tracking
- ✅ GPU support with DataParallel

**Data Preprocessing** (`data_preprocess.py`):
- ✅ Converts `.npy` → `.png` with color map (`jet`)
- ✅ Loads images from directories
- ✅ Resize to 256x256
- ✅ ToTensor transformation

#### 2. **Sensor Processing** (`sensor_proc.py`)
- ✅ `UWBProcessor` - Process UWB radar data
- ✅ `mmWaveProcessor` - Process mmWave radar data
- ✅ `LaserProcessor` - Process laser signals
- ✅ `KinectProcessor` - Process Kinect face/lip data
- ✅ `BVHReader` - 3D motion capture data
- ✅ Timestamp synchronization across modalities
- ✅ Audio/video extraction and alignment

#### 3. **Basic Processing** (`basic_proc.py`)
- ✅ Timestamp matching utilities
- ✅ Datetime conversions
- ✅ JSON timestamp loading

---

## 🎯 Comparison Matrix

| Feature | Repo 1 (Multimodal) | Repo 2 (Preprocess) |
|---------|---------------------|---------------------|
| **Training Script** | ✅ Basic | ✅ **Advanced** (modular) |
| **Pre-trained Models** | ❌ No | ✅ **Yes (PyTorch Hub)** |
| **Multi-modal Fusion** | ✅ Dual ResNet | ✅ **MultiInResNet (better)** |
| **Data Loaders** | ✅ Many types | ✅ Image-based |
| **Preprocessing** | ✅ Basic | ✅ **Advanced (all sensors)** |
| **Confusion Matrix** | ✅ Basic | ✅ **With visualization** |
| **Code Quality** | Good | **Excellent (more modular)** |
| **Documentation** | Limited | Better structured |

---

## 🔥 Most Valuable Components (Priority Order)

### 1. **Classification Module from Repo 2** ⭐⭐⭐
**Location:** `RVTALL-Preprocess/classification/`

**Why it's amazing:**
- Uses **pre-trained ResNet** (exactly what you need!)
- Multi-modal fusion (UWB + FMCW)
- Clean, modular code
- 15 word classification (your exact use case)
- Confusion matrix with heatmap

**How to use:**
```python
from RVTALL-Preprocess.classification.models import CustomResNet, MultiInResNet
from RVTALL-Preprocess.classification.train import Trainer

# For single modality (UWB only)
model = CustomResNet(
    in_channels=3,      # RGB spectrograms
    num_classes=15,     # 15 words
    pre_trained=True,   # Pre-trained on ImageNet!
    model='resnet18'
)

# For multi-modal (UWB + FMCW)
model = MultiInResNet(
    num_inputs=2,
    num_classes=15,
    num_in_convs=[1, 1],
    in_channels=[3, 3],
    out1_channels=[3, 3],
    model='resnet18'
)
```

### 2. **Sensor Processing from Repo 2** ⭐⭐⭐
**Location:** `RVTALL-Preprocess/sensor_proc.py`

**What it does:**
- Extracts spectrograms from raw `.mat` files
- Synchronizes timestamps across sensors
- Saves as `.npy` files ready for training

**Usage:**
```python
from RVTALL-Preprocess.sensor_proc import UWBProcessor

processor = UWBProcessor(root_dir='path/to/RVTALL')
processor._segment_one_exp('uwbmat_file.mat')
# Outputs: sample1.npy, sample2.npy, ...
```

### 3. **Data Preprocessing from Repo 2** ⭐⭐
**Location:** `RVTALL-Preprocess/classification/data_preprocess.py`

**Key function:**
```python
def mat2img(npy_path):
    """Convert .npy spectrograms to .png images with jet colormap"""
    for npy in glob.glob(npy_path+'/*.npy'):
        arr = np.load(npy)
        # 20*log10 to dB scale, jet colormap
        im = image.imsave(npy.replace('npy', 'png'), 
                          20*np.log10(abs(arr)), 
                          cmap='jet')
```

This converts spectrograms to colorful images that work well with pre-trained vision models!

### 4. **Dataset Loaders from Repo 1** ⭐⭐
**Location:** `Multimodal-dataset-for-human-speech-recognition/network/loaddataset.py`

**When to use:** If you prefer `.npy` files directly (skip .png conversion)

### 5. **Training Pipeline from Repo 1** ⭐
**Location:** `Multimodal-dataset-for-human-speech-recognition/network/train.py`

**When to use:** If you want real-time plotting during training

---

## 🚀 Recommended Workflow

### Option 1: Use Repo 2 Classification (Recommended!)

This is the **easiest and most powerful** approach:

1. **Preprocess your data** (if you have raw RVTALL)
   ```bash
   # Use sensor_proc.py to convert .mat → .npy
   python RVTALL-Preprocess/sensor_proc.py
   ```

2. **Convert to images** (optional, for visualization)
   ```python
   from RVTALL-Preprocess.classification.data_preprocess import mat2img
   mat2img('path/to/npy/files')
   ```

3. **Train using their script**
   ```bash
   cd RVTALL-Preprocess/classification
   python main.py
   ```

4. **Or use their models in your code**
   ```python
   from RVTALL-Preprocess.classification.models import CustomResNet
   from RVTALL-Preprocess.classification.train import Trainer
   
   model = CustomResNet(in_channels=3, num_classes=15, pre_trained=True)
   # Your training code here
   ```

### Option 2: Adapt to Your Project

**Integrate the best parts:**

1. **Copy their model class:**
   ```bash
   cp RVTALL-Preprocess/classification/models.py src/pretrained_models.py
   ```

2. **Copy their trainer:**
   ```bash
   cp RVTALL-Preprocess/classification/train.py src/trainer.py
   ```

3. **Use with your data loader:**
   ```python
   from src.pretrained_models import CustomResNet
   from src.trainer import Trainer
   # Your existing data loading code
   ```

### Option 3: Hybrid Approach

Use Repo 2's models with Repo 1's data loaders:

```python
# Model from Repo 2 (pre-trained)
from RVTALL-Preprocess.classification.models import CustomResNet

# Data loader from Repo 1 (handles .npy directly)
from Multimodal-dataset-for-human-speech-recognition.network.loaddataset import uwbDataset

model = CustomResNet(in_channels=1, num_classes=15, pre_trained=True)
dataset = uwbDataset(data_dir='...', data_list=..., num_classes=15)
# Train!
```

---

## 📊 Dataset Structure from Both Repos

### Expected RVTALL structure:

```
RVTALL/
├── UWB_Person_1/
│   ├── word1/
│   │   ├── sample1.npy
│   │   ├── sample2.npy
│   │   └── ...
│   ├── word2/
│   └── ... (15 words total)
├── UWB_Person_2/
│   └── ... (same structure)
└── ... (20 subjects total)
```

### 15 Word Commands:
From the code in `main.py`:
```python
words = ['word_1', 'word_2', ..., 'word_15']
# Also available: vowels (1-5), sentences (1-10)
```

---

## 💡 Key Insights

### 1. Spectrogram → Image Conversion

Repo 2 does something clever:
```python
20*np.log10(abs(arr))  # Convert to dB scale
cmap='jet'             # Colorful visualization
```

This makes spectrograms look like natural images, which works better with pre-trained vision models!

### 2. Multi-modal is Better

Repo 2's `MultiInResNet` combines multiple sensors:
- UWB spectrogram (close range, detailed)
- FMCW spectrogram (longer range)
- Better accuracy than single modality

### 3. Pre-trained Weights

Repo 2 uses **PyTorch Hub** to load pre-trained weights:
```python
self.res = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
```

This is exactly what you wanted - pre-trained vision models for spectrograms!

---

## 🎯 What to Do Next

1. **Start with Repo 2's classification:**
   ```bash
   cd RVTALL-Preprocess/classification
   # Review main.py, models.py, train.py
   ```

2. **Understand the data flow:**
   - Raw `.mat` files → `sensor_proc.py` → `.npy` files
   - `.npy` files → `data_preprocess.py` → `.png` images
   - `.png` images → `main.py` → Training

3. **Test with sample data:**
   - If you have RVTALL dataset, run preprocessing
   - If not, create dummy spectrograms to test

4. **Adapt to your project:**
   - Copy `models.py` and `train.py` to your `src/`
   - Integrate with your existing code
   - Use pre-trained ResNet for your RF sensing

---

## 🔥 Bottom Line

**Repository 2 (`RVTALL-Preprocess`) is MORE VALUABLE** because:
- ✅ Pre-trained ResNet (PyTorch Hub)
- ✅ Clean, modular code structure
- ✅ Multi-modal fusion
- ✅ Complete training pipeline
- ✅ Exactly matches your use case (15 words)
- ✅ Better documentation

**Use Repository 1 for:**
- Reference implementations
- MATLAB processing code
- Alternative data loaders
- Speech separation techniques

**BEST APPROACH:** Use Repo 2's classification module as your starting point!

