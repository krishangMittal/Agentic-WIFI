# MM-Fi Dataset Setup Guide

## 📥 Download MM-Fi Dataset

### Option 1: Google Drive (Recommended)
**Link:** https://drive.google.com/drive/folders/1zDbhfH3BV-xCZVUHmK65EgVV1HMDEYcz?usp=sharing

### Option 2: Baidu Netdisk  
**Link:** https://pan.baidu.com/s/1IU9okQzdeCIaF7xCr1X_pw?pwd=t316

---

## 📦 What to Download

**For Quick Start (Recommended):**
Download only: `E01/S01/` (Environment 1, Subject 1)

This gives you:
- A01-A27: All 27 activities from one subject
- `wifi-csi/` folder: WiFi CSI data (.mat files)
- ~1-2 GB instead of 100+ GB

**Full Dataset:**
- 4 environments (E01-E04)
- 40 subjects total
- ~320k frames
- 100+ GB

---

## 📂 Expected Structure

After downloading, extract to `data/raw/MMFi/`:

```
data/raw/MMFi/
├── E01/
│   ├── S01/
│   │   ├── A01/                    # Stretching
│   │   │   ├── wifi-csi/
│   │   │   │   ├── csi.mat        # WiFi CSI data
│   │   │   ├── mmwave/
│   │   │   ├── rgb/
│   │   │   └── ...
│   │   ├── A02/                    # Chest expansion
│   │   ├── ...
│   │   └── A27/                    # Bowing
│   ├── S02/
│   └── ...
├── E02/
└── ...
```

---

## 🎬 27 Activities Reference

| Code | Activity | Type | Emergency? |
|------|----------|------|------------|
| A01 | Stretching | Rehab | No |
| A02 | Chest expansion (H) | Daily | No |
| A03 | Chest expansion (V) | Daily | No |
| A04 | Twist (left) | Daily | No |
| A05 | Twist (right) | Daily | No |
| A06 | Mark time | Rehab | No |
| A07 | Limb extension (L) | Rehab | No |
| A08 | Limb extension (R) | Rehab | No |
| A09 | Lunge (left-front) | Rehab | No |
| A10 | Lunge (right-front) | Rehab | No |
| A11 | Limb extension (both) | Rehab | No |
| A12 | Squat | Rehab | No |
| A13 | Raising hand (L) | Daily | No |
| A14 | Raising hand (R) | Daily | No |
| A15 | Lunge (left side) | Rehab | No |
| A16 | Lunge (right side) | Rehab | No |
| A17 | Waving hand (L) | Daily | No |
| A18 | Waving hand (R) | Daily | No |
| A19 | Picking up things | Daily | **⚠️ Possible fall** |
| A20 | Throwing (left) | Daily | No |
| A21 | Throwing (right) | Daily | No |
| A22 | Kicking (left) | Daily | No |
| A23 | Kicking (right) | Daily | No |
| A24 | Body extension (L) | Rehab | No |
| A25 | Body extension (R) | Rehab | No |
| A26 | Jumping up | Rehab | No |
| A27 | Bowing | Daily | **⚠️ Possible fall** |

---

## ✅ Installation

Already done! The MMFi_dataset toolbox is cloned.

### Verify Dependencies:
```bash
pip install scipy pyyaml opencv-python
```

(You should already have torch, numpy from environment.yml)

---

## 🧪 Quick Test (After Download)

Once you have `data/raw/MMFi/E01/S01/A01/wifi-csi/csi.mat`:

```bash
python notebooks/03_mmfi_quickstart.py
```

This will:
1. Load WiFi CSI from A01 (Stretching)
2. Generate spectrogram
3. Classify with ResNet
4. Send to DeepSeek agent
5. Get action recommendation

---

## 🎯 Emergency Detection Use Case

**Example workflow:**
```
Person bends down (A19 or A27)
↓
WiFi signal disrupted
↓
Spectrogram shows "bowing" pattern
↓
Classifier: "A27" (85% confidence)
↓
DeepSeek Agent: "Elderly context + sudden downward motion 
                  → Check if person is OK"
↓
Action: Alert caregiver
```

---

## 📊 Dataset Stats

- **Subjects:** 40 people (11 females, 29 males, age 23-40)
- **Actions:** 27 activities
- **Environments:** 4 different rooms
- **Frames:** 320,000+ synchronized frames
- **Modalities:** 7 (WiFi CSI, mmWave, RGB, Depth, Lidar, etc.)
- **Published:** NeurIPS 2023 Datasets Track

---

## 🔗 Links

- **Paper:** https://arxiv.org/abs/2305.10345
- **Project Page:** https://ntu-aiot-lab.github.io/mm-fi
- **Google Drive:** https://drive.google.com/drive/folders/1zDbhfH3BV-xCZVUHmK65EgVV1HMDEYcz
- **Citation:** Yang et al., NeurIPS 2023

---

## 🚀 Next After Download

1. **Extract to:** `data/raw/MMFi/`
2. **Run test:** `python notebooks/03_mmfi_quickstart.py`
3. **Generate spectrograms:** See `src/mmfi_processor.py`
4. **Train classifier:** Adapt for 27 classes
5. **Test agent:** Use emergency scenarios

---

## ⚡ Quick Download Command (Linux/Mac)

```bash
cd "data/raw"
mkdir MMFi
cd MMFi

# You'll need to download manually from Google Drive
# Then extract:
unzip ~/Downloads/E01.zip
```

---

## 🎓 Why MM-Fi is Perfect for Your System

1. ✅ **Available NOW** (unlike RVTALL)
2. ✅ **WiFi CSI data** (same as RF signals)
3. ✅ **Real-world activities** (fall detection, elderly care)
4. ✅ **Same processing** (spectrograms → ResNet → Agent)
5. ✅ **Emergency use cases** (bending, falling movements)
6. ✅ **Well-documented** (NeurIPS 2023 paper)

Start with just **E01/S01/** (~1-2 GB) to test the full pipeline!

