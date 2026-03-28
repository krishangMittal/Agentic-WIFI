# Agentic WiFi Sensing — Full Progress Report

## Project: "Siri for WiFi" — RF-Based Gesture & Command Recognition

---

## What This Project Does

Uses WiFi signals (CSI — Channel State Information) to detect human gestures and voice commands without cameras or microphones. When someone moves near a WiFi signal, they disrupt it. An ESP32 captures these disruptions, a neural network classifies what gesture/command it was, and an LLM agent decides what action to take.

```
WiFi signal disrupted by movement
    → ESP32 captures CSI data
    → Preprocessing (filter, denoise)
    → Neural network classifies gesture
    → DeepSeek LLM decides action
    → Smart home executes (lights, alerts, etc.)
```

---

## What We Have Built

### 1. Preprocessing Pipeline (`src/preprocessing.py`) ✅
- **Butterworth low-pass filter** — removes high-frequency noise (cutoff 30Hz)
- **Hampel filter** — removes outlier spikes
- **PCA denoising** — removes static background, keeps motion-related components
- **Normalization** — scales to [0,1]
- **Sliding window segmentation** — splits time series into overlapping windows
- **ESP32 parser** — converts live I/Q pairs to amplitude + phase
- Based on: SenseFi benchmark, CSI-DeepNet paper, arxiv 2307.12126

### 2. CNN-LSTM-Attention Model (`src/csi_model.py`) ✅
Two architectures built:

| Model | Type | Parameters | Input | Best For |
|-------|------|-----------|-------|----------|
| **CSINet** | CNN + BiLSTM + Attention | 1.85M | (batch, 228, window) | Highest accuracy |
| **CSINetLite** | 2D CNN only | 625K | (batch, 228, window) | Fast training |

- CNN extracts spatial features across 114 subcarriers
- BiLSTM captures temporal motion patterns
- Attention weights important time steps
- Input: 228 features (114 amplitude + 114 phase)
- Output: 27 activity classes

### 3. Training Pipeline (`src/train.py`) ✅
- Cross-subject evaluation (train on subjects 1-7, test on 8-10)
- Adam optimizer with ReduceLROnPlateau scheduler
- Per-class classification report
- Best model checkpointing
- Supports both CSINet and CSINetLite

### 4. LLM Agent (`src/agent_simple.py`) ✅
- Direct DeepSeek API calls (no LangChain needed)
- 15 command-to-action mappings (help, ambulance, police, fire, etc.)
- Fuzzy matching for unclear predictions
- Context-aware boosting (emergency context → boost emergency commands)
- Rule-based fallback when LLM unavailable
- Also: `src/agent_ai.py` (LangChain version), `src/gesture_agent.py` (gesture-specific)

### 5. ResNet Classifier (`src/classifier.py`) ✅
- Pre-trained ResNet18 for spectrogram image classification
- 15 RVTALL word commands
- ImageNet transfer learning
- Batch prediction support

### 6. Live ESP32 Demo (`src/live_agentic_demo.py`) ✅
- Real-time CSI capture from ESP32 over serial (COM5)
- Motion detection: STILL / MOVEMENT / WAVE / WALKING
- DeepSeek integration — LLM decides smart home action on gesture detection
- Session logging with timestamps

### 7. Full Integration (`src/siri_for_wifi.py`) ✅
- Complete 3-stage pipeline: RF → Classifier → Agent → Action
- Works with both spectrogram images and raw CSI

### 8. ESP32 Firmware ✅
- **Passive mode** (`ESP32-CSI-Tool/passive/`) — sniffs all WiFi packets
- **Active STA mode** (`ESP32-CSI-Tool/active_sta/`) — connects to router, sends packets at 100/sec
- Fixed for ESP-IDF v5.5.3 (priority bug, header renames, gateway IP)
- Compiled and flashed successfully

---

## Dataset: MM-Fi (What We Have)

**Source:** NeurIPS 2023 — "MM-Fi: Multi-Modal Non-Intrusive 4D Human Dataset"

### Structure
```
data/raw/E01/E01/
├── S01/ through S10/     ← 10 subjects (different people)
│   ├── A01/ through A27/ ← 27 activities each
│   │   ├── wifi-csi/     ← WiFi CSI data (.mat files)
│   │   │   ├── frame001.mat
│   │   │   ├── frame002.mat
│   │   │   └── ... (~297 frames per activity)
│   │   ├── depth/        ← depth camera (not used)
│   │   ├── rgb/          ← RGB camera (not used)
│   │   ├── mmwave/       ← mmWave radar (not used)
│   │   └── lidar/        ← LiDAR (not used)
```

### Data Format
Each `.mat` file contains:
- **CSIamp**: shape (3, 114, 10) — 3 antennas × 114 subcarriers × 10 samples
- **CSIphase**: shape (3, 114, 10) — same dimensions

### Statistics
- **10 subjects** × **27 activities** × **~297 frames** = **~80,190 total frames**
- Each frame: 6,840 numbers (3420 amplitude + 3420 phase)
- Perfectly balanced: 2,970 frames per activity

### 27 Activities
| Code | Activity | Category | Smart Home Action |
|------|----------|----------|-------------------|
| A01 | Stretching | Rehab | Start workout mode |
| A02 | Chest expansion (H) | Daily | — |
| A03 | Chest expansion (V) | Daily | — |
| A04 | Twist left | Daily | — |
| A05 | Twist right | Daily | — |
| A06 | Mark time | Rehab | — |
| A07 | Limb extension (L) | Rehab | — |
| A08 | Limb extension (R) | Rehab | — |
| A09 | Lunge (left front) | Rehab | — |
| A10 | Lunge (right front) | Rehab | — |
| A11 | Limb extension (both) | Rehab | — |
| A12 | Squat | Rehab | ⚠️ Possible fall |
| A13 | Raise hand (L) | Daily | Volume up |
| A14 | Raise hand (R) | Daily | Volume down |
| A15 | Lunge (left side) | Rehab | — |
| A16 | Lunge (right side) | Rehab | — |
| A17 | Wave left | Daily | Turn on lights |
| A18 | Wave right | Daily | Turn off lights |
| A19 | Picking up | Daily | ⚠️ Possible fall |
| A20 | Throw left | Daily | Previous track |
| A21 | Throw right | Daily | Next track |
| A22 | Kick left | Daily | Dismiss notification |
| A23 | Kick right | Daily | Snooze alarm |
| A24 | Body extension (L) | Rehab | — |
| A25 | Body extension (R) | Rehab | — |
| A26 | Jumping | Rehab | Log exercise |
| A27 | Bowing | Daily | ⚠️ Possible fall |

---

## ESP32 Hardware Setup

### Hardware
- **Board:** HiLetgo ESP-WROOM-32 ($10)
- **USB Chip:** CP2102 (driver installed)
- **COM Port:** COM5
- **Firmware:** ESP32-CSI-Tool (active_sta mode)
- **ESP-IDF:** v5.5.3

### What the ESP32 Outputs
```
CSI_DATA,STA,22:03:5E:82:2C:67,-45,10,0,0,...,128,[124 64 7 0 -23 16 -23 18 ...]
```
- 128 raw I/Q values (64 subcarrier pairs)
- RSSI signal strength
- MAC address of source device
- Timestamp

### Connection
- ESP32 connects to iPhone hotspot ("Hotspot") on 2.4 GHz channel 6
- Sends UDP packets to gateway at 100/sec
- Each packet generates a CSI reading
- Gateway IP fixed from 192.168.4.1 → 172.20.10.1

### Issues Encountered & Fixed
1. CP2102 driver error → reinstalled from Silicon Labs
2. ESP-IDF v5.5 incompatibility → fixed `esp_spi_flash.h` → `spi_flash_mmap.h`
3. FreeRTOS priority crash → reduced task priority from 100 to 5
4. Wrong gateway IP → changed to 172.20.10.1 for iPhone hotspot
5. WiFi SSID case sensitivity → exact match required
6. Main WiFi is 5 GHz only → ESP32 needs 2.4 GHz → use iPhone hotspot
7. Passive mode too few packets → switched to active_sta mode

---

## Research Findings

### What the literature says (from web search):

**Best model architectures for WiFi CSI:**
| Model | Accuracy | Notes |
|-------|----------|-------|
| CNN + LSTM + Attention | ~98.5% | State of the art |
| ResNet on raw CSI | ~98.1% | Needs lots of data |
| Residual Network | ~98.6% | Best single model |
| CNN alone | ~95% | Fast, good baseline |
| BiLSTM | Best on some datasets | Good for temporal patterns |
| ResNet on spectrogram images | ~85-90% | NOT recommended (lossy) |

**Key preprocessing steps:**
1. Butterworth low-pass filter (cutoff ~30 Hz)
2. Hampel filter for outlier removal
3. PCA — drop 1st component (static), keep 2-5 (motion)
4. Normalization to [0,1]

**Key references:**
- SenseFi benchmark (Chen et al., 2023) — standard benchmark library
- CSI-DeepNet (2022) — ESP32-specific, 96.3% accuracy
- MM-Fi (Yang et al., NeurIPS 2023) — our dataset

### What we learned:
- Converting CSI to spectrogram images is a hack, NOT the standard approach
- Raw numerical CSI fed directly to CNN/LSTM is better
- Both amplitude AND phase should be used (doubles features)
- Cross-subject evaluation is critical (don't just memorize one person)
- More WiFi traffic = more CSI samples = better detection

---

## Training Results So Far

### Run 1: CSINetLite, 5 subjects, window=100, stride=50
- **Result:** 14% accuracy (barely above random 3.7%)
- **Problem:** Only ~4 windows per activity per subject = ~270 total samples
- **Fix:** Smaller windows, more overlap

### Run 2: CSINetLite, window=50, stride=10
- **Result:** Building dataset gives ~6,750 samples (25x more)
- **Status:** Needs full training run with all 10 subjects

### Current saved model: `models/csi_model.pth`
- CSINetLite architecture
- 624,603 parameters
- Needs retraining with proper window settings

---

## Live Demo Results

### What works:
- ESP32 captures CSI data from iPhone hotspot at ~100 packets/sec
- Motion detection responds to hand movement (score increases)
- Wave detection triggers on rhythmic back-and-forth motion
- DeepSeek agent responds with smart home actions
- 3 waves detected in one session

### What doesn't work well:
- False positives when still (baseline calibration sensitivity)
- Detection delay (improved with reset_input_buffer but still present)
- Requires 2.4 GHz WiFi source (main router is 5 GHz only)
- Low packet rate in passive mode (fixed by switching to active_sta)

---

## Generated Visualizations

### data/visualizations/gesture_comparison.png
- 4 gestures compared side by side (Wave, Jump, Bow, Stretch)
- Top row: raw CSI amplitude heatmaps
- Bottom row: spectrograms
- Shows visually distinct patterns per gesture

### data/mmfi_images/ (54 spectrograms)
- 2 subjects × 27 activities
- 256×256 PNG images with jet colormap
- Ready for ResNet classification

---

## File Inventory

### Source Code (src/)
| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| preprocessing.py | 366 | CSI preprocessing pipeline | ✅ |
| csi_model.py | 245 | CNN-LSTM-Attention models | ✅ |
| train.py | 274 | Training pipeline | ✅ |
| agent_simple.py | 377 | DeepSeek LLM agent | ✅ |
| agent_ai.py | 392 | LangChain agent (alternative) | ✅ |
| classifier.py | 231 | ResNet spectrogram classifier | ✅ |
| gesture_agent.py | 330 | Gesture-to-action agent | ✅ |
| siri_for_wifi.py | 222 | Full pipeline integration | ✅ |
| mmfi_processor.py | 309 | MM-Fi data processor | ✅ |
| live_agentic_demo.py | 227 | Live ESP32 + LLM demo | ✅ |
| live_demo.py | 157 | Live ESP32 motion detection | ✅ |
| model.py | 257 | Alternative model architectures | ✅ |
| preprocess.py | 74 | Basic CSI→spectrogram | ✅ |
| agent.py | 127 | Base agent framework | Partial |
| **TOTAL** | **3,588** | | |

### Config & Data
| Item | Size | Status |
|------|------|--------|
| MM-Fi dataset (E01) | 21 GB | ✅ 10 subjects extracted |
| Spectrogram images | 22 MB (54 files) | ✅ Generated |
| Gesture config YAML | 141 lines | ✅ 27 mappings |
| Saved model weights | 2.4 MB | ⚠️ Needs retraining |
| ESP32 firmware | Compiled | ✅ Active STA mode |

---

## What Needs To Be Done

### Critical (for demo/presentation)
1. **Retrain model properly** — window=50, stride=10, all 10 subjects, 30+ epochs
2. **Fix live demo reliability** — better baseline calibration, higher thresholds
3. **Enable 2.4 GHz on main router permanently** — or use dedicated 2.4 GHz AP
4. **Generate confusion matrix & accuracy plots** — visual results for presentation

### Important (for complete system)
5. **Bridge ESP32 format to model** — ESP32 outputs 64 I/Q pairs, model expects 114 amp + 114 phase
6. **Run live inference with trained model** — not just motion detection, actual gesture classification
7. **Download RVTALL dataset** — for voice command recognition (15 spoken words)
8. **Train speech command model** — spectrograms from speech-affected CSI

### Nice to have
9. **Real smart home integration** — connect to Philips Hue, Home Assistant
10. **Mobile app** — remote monitoring dashboard
11. **Multi-environment testing** — train on different rooms
12. **Edge deployment** — run inference on ESP32 itself (TinyML)

---

## How to Run Everything

### Setup
```bash
cd ~/Agentic-WIFI
# Install dependencies
pip install torch torchvision scipy matplotlib h5py requests pyserial

# Set API key
export DEEPSEEK_API_KEY='sk-...'
```

### Demo (no hardware)
```bash
# Gesture demo with DeepSeek
python examples/demo_gesture.py

# Agent demo (simulated predictions → LLM → actions)
python -c "from src.agent_simple import demo_agent; demo_agent()"

# Full pipeline demo
python src/siri_for_wifi.py
```

### Train model
```bash
# Quick test (2 subjects, 5 epochs)
python src/train.py --model lite --epochs 5 --subjects 2

# Full training (all subjects, 30 epochs)
python src/train.py --model lite --epochs 30
```

### Live ESP32 demo
```bash
# Flash ESP32 first (in ESP-IDF CMD terminal):
# cd ESP32-CSI-Tool/active_sta && idf.py build && idf.py -p COM5 flash

# Then run live detection:
python src/live_agentic_demo.py

# Or simple motion detection:
python src/live_demo.py
```
