# How to Get the RVTALL Dataset

## Current Status ❌

**The actual dataset is NOT in the cloned repositories.**

The repos contain:
- ✅ Preprocessing code
- ✅ Training scripts  
- ✅ Model architectures
- ❌ **NO actual data files**

---

## Where to Download RVTALL

### Dataset Details (from Nature Paper)
- **Paper DOI:** 10.1038/s41597-023-02793-w
- **Published:** December 13, 2023
- **Size:** ~400 minutes of multimodal recordings
- **Participants:** 20 people
- **Vocabulary:** 5 vowels + 15 words + 16 sentences

### Official Download Sources

#### 1. Nature Scientific Data Page
**URL:** https://www.nature.com/articles/s41597-023-02793-w

Look for:
- "Data Records" section (usually has repository link)
- "Data Availability" statement
- "Supplementary Information"

#### 2. ResearchGate
The authors often upload datasets here:
- **Search:** "RVTALL multimodal dataset"
- **Author:** Yao Ge (University of Glasgow)
- **Link pattern:** researchgate.net/publication/[ID]

#### 3. University of Glasgow Repository
**Likely URL:** https://researchdata.gla.ac.uk

Search for:
- "RVTALL"
- "Multimodal speech dataset"
- Author: "Yao Ge"

#### 4. figshare / Zenodo
Scientific datasets often hosted on:
- https://figshare.com (search "RVTALL")
- https://zenodo.org (search "RVTALL")

#### 5. Direct Contact (If Above Fail)
**Lead Authors:**
- Yao Ge: yao.ge@glasgow.ac.uk
- Qammer H. Abbasi: qammer.abbasi@glasgow.ac.uk

James Watt School of Engineering  
University of Glasgow  
Glasgow, G12 8QQ, UK

---

## What the Dataset Contains

### Modalities (6 total):
1. **UWB Radar** - 7.5 GHz Channel Impulse Response (.mat files)
2. **mmWave Radar** - 77 GHz FMCW data (.mat files)
3. **Video** - Kinect depth camera (RGB-D)
4. **Audio** - Voice recordings (.wav)
5. **Laser** - Laser speckle data
6. **Lip Landmarks** - 3D coordinates from Kinect

### Corpus:
- **5 Vowels:** /a/, /e/, /i/, /o/, /u/
- **15 Words:** (Command vocabulary - extract from Corpus folder)
- **16 Sentences:** Full sentence recordings
- **20 Subjects:** Different speakers

### Expected File Structure:
```
RVTALL/
├── Corpus/                 # Text of words/sentences
├── UWB_Radar/
│   ├── EXP1_TASK2_PERSON1_RADAR1_xethru.mat
│   ├── EXP1_TASK2_PERSON2_RADAR1_xethru.mat
│   └── ...
├── mmWave_Radar/
│   ├── EXP1_TASK2_PERSON1_RADAR2_TI.mat
│   └── ...
├── Kinect/
│   ├── Video/
│   ├── Landmarks/
│   └── ...
├── Audio/
│   ├── EXP1_TASK2_PERSON1_mic.wav
│   └── ...
└── Laser/
    └── ...
```

---

## Once You Download

### Step 1: Extract to `data/raw/`
```bash
cd "c:\Users\Krish\OneDrive\Desktop\rf sensing research"
# Extract RVTALL dataset here:
# data/raw/RVTALL/...
```

### Step 2: Inspect the Dataset
```bash
python scripts/inspect_rvtall.py
```

This will:
- Find the Corpus folder
- Extract the 15 command words
- Show dataset statistics
- Verify file structure

### Step 3: Generate Spectrograms
```python
# Use the preprocessing code from cloned repos
from sensor_proc import UWBProcessor

processor = UWBProcessor(root_dir='data/raw/RVTALL')
processor._segment_one_exp('EXP1_TASK2_PERSON1_RADAR1_xethru.mat')
# Generates .npy spectrograms in data/spectrograms/
```

### Step 4: Convert to Images (for ResNet)
```python
from data_preprocess import mat2img

mat2img(
    input_dir='data/spectrograms',
    output_dir='data/images',
    size=(256, 256)
)
# Generates .png images for training
```

### Step 5: Test End-to-End
```bash
python quickstart.py
```

Or:
```python
from src.siri_for_wifi import SiriForWiFi

system = SiriForWiFi(api_key=os.getenv('DEEPSEEK_API_KEY'))  # Set via environment variable
result = system.run_pipeline(
    raw_rf_file_path='data/raw/RVTALL/UWB_Radar/subject1_help.mat',
    command_to_extract='help',
    subject_id='subject1'
)
```

---

## Alternative: Use Sample Data

If you can't get RVTALL immediately, you can:

### 1. Generate Synthetic RF Data
```python
import numpy as np
from src.preprocess import csi_to_spectrogram

# Simulate RF signal
sample_rate = 1000  # Hz
duration = 1.0      # seconds
t = np.linspace(0, duration, int(sample_rate * duration))

# Synthetic signal (sum of frequencies)
signal = np.sin(2 * np.pi * 100 * t) + 0.5 * np.sin(2 * np.pi * 200 * t)

# Convert to spectrogram
freq, time, spectrogram = csi_to_spectrogram(signal, sample_rate=sample_rate)

# Save as image
import matplotlib.pyplot as plt
plt.imsave('data/images/synthetic_sample.png', spectrogram, cmap='jet')
```

### 2. Test with Dummy Data
```python
# The system already has test code
python test_deepseek.py  # Tests agent fuzzy matching
python src/classifier.py # Tests classifier with dummy image
python src/agent_simple.py # Full demo with simulated data
```

---

## Checklist

- [ ] Check Nature paper "Data Availability" section
- [ ] Search ResearchGate for "RVTALL Yao Ge"
- [ ] Check University of Glasgow repository
- [ ] Search figshare/Zenodo
- [ ] Email authors if needed
- [ ] Extract to `data/raw/RVTALL/`
- [ ] Run `scripts/inspect_rvtall.py`
- [ ] Generate spectrograms with `sensor_proc.py`
- [ ] Convert to images with `mat2img`
- [ ] Test end-to-end with `quickstart.py`

---

## What You Can Do NOW (Without Dataset)

Your system is **already functional** with the test scripts:

1. **Test DeepSeek agent:**
   ```bash
   python test_deepseek.py
   ```
   ✅ Working! (fuzzy matching tested)

2. **Test classifier:**
   ```bash
   python src/classifier.py
   ```
   ✅ Creates dummy image and tests

3. **Review preprocessing code:**
   ```bash
   # The code is ready in:
   RVTALL-Preprocess/sensor_proc.py
   RVTALL-Preprocess/classification/data_preprocess.py
   ```

4. **Study the papers:**
   - Read the Nature paper for methodology
   - Understand the multimodal approach
   - Review the preprocessing steps

---

## Summary

| What | Status |
|------|--------|
| **Dataset in repos** | ❌ No |
| **Preprocessing code** | ✅ Yes |
| **Training code** | ✅ Yes |
| **Your system (Stage 1-3)** | ✅ Complete |
| **DeepSeek agent** | ✅ Working |
| **Classifier** | ✅ Ready |
| **Need:** | Dataset files |

**Next step:** Find the dataset download link from one of the sources above!

