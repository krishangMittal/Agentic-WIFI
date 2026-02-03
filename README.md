# RF Sensing Research

Research project on RF sensing for command classification, lip reading, and gesture recognition using Channel State Information (CSI) and spectrogram analysis.

## Project Structure

```
rf-sensing-research/
├── data/            # RVTALL or MM-Fi samples
├── notebooks/       # Exploratory CSI analysis (STFT, Doppler)
├── src/
│   ├── preprocess.py # CSI to Spectrogram conversion
│   ├── model.py      # Command classifier
│   └── agent.py      # LangChain/Claude tool-use logic
├── environment.yml   # Conda environment configuration
└── lit_review.md     # Literature review tracking
```

## Setup

1. Create the conda environment:
```bash
conda env create -f environment.yml
conda activate rf-sensing-research
```

2. Download the RVTALL dataset:
```bash
# See detailed instructions in docs/download_rvtall_instructions.md
# Or use the download script:
python scripts/download_rvtall.py

# After downloading, extract and inspect:
python scripts/download_rvtall.py --extract <path_to_archive>
python scripts/inspect_rvtall.py
```

3. Install additional dependencies if needed:
```bash
pip install -r requirements.txt  # If you create one
```

## "Siri for WiFi" - Three-Stage Workflow

### Complete Pipeline
```python
from src.siri_for_wifi import SiriForWiFi

# Initialize the complete system
system = SiriForWiFi(use_llm_agent=True)

# Process RF signal: Spectrogram → Classification → Action
result = system.process_rf_signal('data/images/sample.png')
```

### Stage 1: RF "Ear" (Preprocessing)
```python
# Convert raw RF → spectrograms (use cloned repo tools)
from sensor_proc import UWBProcessor
processor = UWBProcessor(root_dir='data/raw')
processor._segment_one_exp('uwb_file.mat')
```

### Stage 2: Command Classifier
```python
from src.classifier import RFCommandClassifier

classifier = RFCommandClassifier(use_pretrained=True)
predictions = classifier.predict('data/images/sample.png')
# Output: [('ambulance', 0.95), ('help', 0.03), ...]
```

### Stage 3: Agentic Logic (Simple Version - No LangChain!)
```python
from src.agent_simple import RFCommandAgent

# Direct DeepSeek API calls - no LangChain needed!
agent = RFCommandAgent(
    use_llm=True,
    api_key=os.getenv('DEEPSEEK_API_KEY')  # Set via environment variable
)
interpretation = agent.interpret_command(predictions, context="emergency")
result = agent.execute_action(interpretation)
```

**Why no LangChain?** Direct API calls are simpler, faster, and easier to debug.  
See `docs/no_langchain.md` for explanation.

See `docs/siri_for_wifi_workflow.md` for complete documentation.

## Dataset

### RVTALL Dataset

⚠️ **Important:** The cloned repositories contain preprocessing code but **NOT** the actual dataset files.

- **DOI:** 10.1038/s41597-023-02793-w  
- **Paper:** [Nature Scientific Data](https://www.nature.com/articles/s41597-023-02793-w)
- **Commands:** 5 vowels + 15 words + 16 sentences
- **Participants:** 20 speakers
- **Duration:** ~400 minutes multimodal recordings

**📥 Download Guide:** See **`docs/GET_DATASET.md`** for:
- Where to download (ResearchGate, University repository, etc.)
- What the dataset contains
- How to set it up
- Alternative: Use synthetic data for testing

After downloading and extracting to `data/raw/RVTALL/`:
```bash
python scripts/inspect_rvtall.py  # Extract the 15 command words
```

## 🚀 Real-Life Testing (Active!)

**Status:** ✅ System ready for real-world deployment!

### 🎮 **Test Your System in 3 Ways:**

| Method | Cost | Time | Best For |
|--------|------|------|----------|
| **Simulate (MM-Fi data)** | FREE | 5 min | Testing now |
| **ESP32 Hardware** | $10 | 1 hour | Real deployment |
| **Software-Only (Intel WiFi)** | FREE | 2 hours | Research |

### ⚡ **Quick Start:**

```bash
# 1. Test gesture detection
python demo_gesture.py

# 2. Test real-time processing
python setup_realtime.py

# 3. Connect to your smart home!
```

**See:** `REAL_LIFE_TESTING_GUIDE.md` for complete setup

---

## 📊 MM-Fi Dataset

**Status:** ✅ Cloned and ready!

- 27 activities (wave, raise hand, fall detection, etc.)
- WiFi CSI data → Perfect for training
- NeurIPS 2023 dataset
- **Download:** `docs/MMFI_SETUP.md`

## Cloned Repositories

You have **three valuable repositories** cloned:

1. **`Multimodal-dataset-for-human-speech-recognition/`**
   - Complete training pipeline
   - Multiple dataset loaders
   - Custom ResNet implementations
   - MATLAB processing code

2. **`RVTALL-Preprocess/`** ⭐ (Recommended)
   - **Pre-trained ResNet** models (PyTorch Hub)
   - **Multi-modal fusion** architecture
   - Advanced sensor preprocessing
   - 15-word classification pipeline

See `docs/both_repos_comparison.md` for detailed analysis.

### Quick Start with Pre-trained Models

```python
# Use the pre-trained ResNet from RVTALL-Preprocess
from RVTALL-Preprocess.classification.models import CustomResNet

model = CustomResNet(
    in_channels=3,      # RGB spectrograms
    num_classes=15,     # 15 word commands
    pre_trained=True,   # ImageNet pre-trained!
    model='resnet18'
)
```

## Research Notes

See `lit_review.md` for literature review and research tracking.

