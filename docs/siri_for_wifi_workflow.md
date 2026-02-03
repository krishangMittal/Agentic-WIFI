# "Siri for WiFi" Integrated Workflow

Complete guide to building an RF-based voice command system with LLM intelligence.

## Overview

**RF Signals → Spectrogram → Classifier → LLM Agent → Action**

This system combines:
- **RF sensing** (WiFi/radar signals)
- **Pre-trained vision models** (ResNet on spectrograms)
- **LLM reasoning** (Claude for fuzzy matching and context)

## Three-Stage Architecture

### Stage 1: The RF "Ear" (Data Preprocessing)

**Components:**
- `RVTALL-Preprocess/sensor_proc.py` - Extract spectrograms from raw RF
- `RVTALL-Preprocess/classification/data_preprocess.py` - Convert to images

**What it does:**
1. Takes raw RF radar `.mat` files
2. Segments them using timestamps
3. Converts to `.npy` spectrograms
4. Generates colorized `.png` images (jet colormap)

**Usage:**
```python
from sensor_proc import UWBProcessor

processor = UWBProcessor(root_dir='data/raw')
processor._segment_one_exp('subject1_word2_sample.mat')
# Output: data/spectrograms/sample1.npy, sample2.npy, ...
```

Then convert to images:
```python
from data_preprocess import mat2img

mat2img('data/spectrograms')
# Output: data/images/sample1.png, sample2.png, ...
```

**Key insight:** The jet colormap makes spectrograms look like natural images, perfect for pre-trained vision models!

---

### Stage 2: The "Command Classifier" (Inference)

**Component:** `src/classifier.py`

**What it does:**
1. Loads pre-trained ResNet18 (ImageNet weights)
2. Takes spectrogram image as input
3. Outputs top-k command predictions with confidence scores

**Why pre-trained works:**
- ResNet learned to recognize **patterns, textures, shapes** from millions of images
- RF spectrograms have **spatial patterns** (Doppler shifts, frequency bands)
- The model can distinguish "ambulance" vs "help" by their unique Doppler signatures
- **10-100x less training data needed** vs. training from scratch

**Usage:**
```python
from src.classifier import RFCommandClassifier

classifier = RFCommandClassifier(use_pretrained=True)
predictions = classifier.predict('data/images/sample1.png', top_k=3)

# Output: [('ambulance', 0.95), ('help', 0.03), ('emergency', 0.02)]
```

**15 RVTALL Commands:**
```python
["help", "ambulance", "police", "fire", "emergency",
 "stop", "yes", "no", "left", "right",
 "forward", "backward", "up", "down", "home"]
```

---

### Stage 3: The "Agentic Logic" (The Brain)

**Component:** `src/agent_ai.py`

**What it does:**
1. Takes classifier predictions (possibly noisy/ambiguous)
2. Uses LLM reasoning to:
   - Handle fuzzy matches ("amblance" → "ambulance")
   - Consider context (emergency, location, history)
   - Resolve ambiguity
3. Executes appropriate action

**Why LLM is crucial:**

Traditional system:
```
Classifier: 70% "amblance", 20% "ambulance"
System: Unknown command (fails)
```

With LLM agent:
```
Classifier: 70% "amblance", 20% "ambulance"
Agent: "In medical context, 'amblance' is likely a mispronunciation
        of 'ambulance'. High confidence → calling emergency services."
System: ✓ Success!
```

**Usage:**
```python
from src.agent_ai import RFCommandAgent

agent = RFCommandAgent(use_llm=True, confidence_threshold=0.7)

# Low confidence prediction
predictions = [("help", 0.52), ("home", 0.48)]
interpretation = agent.interpret_command(
    predictions,
    context="User pressed emergency button"
)

# Agent output:
{
    "action": "trigger_help_alert",
    "command": "help",
    "confidence": 0.85,  # Boosted by context!
    "reasoning": "Emergency context suggests help over home"
}
```

**Fuzzy Matching Examples:**
- "amblance" → "ambulance"
- "polic" → "police"
- "hel" → "help"
- "emergancy" → "emergency"

---

## Complete Integration

**Component:** `src/siri_for_wifi.py`

Combines all three stages:

```python
from src.siri_for_wifi import SiriForWiFi

# Initialize the complete system
system = SiriForWiFi(
    use_llm_agent=True,
    confidence_threshold=0.7
)

# Process a single RF spectrogram
result = system.process_rf_signal(
    'data/images/sample_ambulance.png',
    context="User fell and activated emergency button"
)

# Output:
# [Stage 2] Classification: ambulance (95%)
# [Stage 3] Agent decides: trigger_emergency_services
# [Execution] 🚨 EMERGENCY: Calling ambulance
```

---

## Directory Structure

```
/rf-sensing-research
├── /data
│   ├── /raw              # Raw .mat files from RVTALL
│   ├── /spectrograms     # .npy files from sensor_proc.py
│   └── /images           # .png files for ResNet training
├── /models
│   ├── custom_resnet.py  # Pre-trained ResNet models
│   └── trainer.py        # Training utilities
├── /src
│   ├── classifier.py     # Stage 2: Command classifier
│   ├── agent_ai.py       # Stage 3: Agentic logic
│   └── siri_for_wifi.py  # Complete integration
└── README.md
```

---

## Quick Start Guide

### 1. Setup Environment

```bash
# Create conda environment
conda env create -f environment.yml
conda activate rf-sensing-research

# Set API key for LLM (optional, for Stage 3)
export ANTHROPIC_API_KEY="your-key-here"
```

### 2. Download RVTALL Dataset

See `docs/download_rvtall_instructions.md`

### 3. Stage 1: Preprocessing (Extract One Word)

```python
from RVTALL-Preprocess.sensor_proc import UWBProcessor

# Extract "help" word from Subject 1
processor = UWBProcessor(root_dir='data/RVTALL/raw')
processor._segment_one_exp('1_2_1_uwb.mat')  # Subject 1, word task, ID 1
```

### 4. Stage 2: Train/Test Classifier

```python
from src.classifier import RFCommandClassifier

# Initialize with pre-trained weights
classifier = RFCommandClassifier(use_pretrained=True)

# Test on a single image
predictions = classifier.predict('data/images/help_sample1.png')
print(predictions)  # [('help', 0.92), ('home', 0.05), ...]
```

### 5. Stage 3: Test Agent

```python
from src.agent_ai import RFCommandAgent

agent = RFCommandAgent(use_llm=True)

# Test with ambiguous predictions
predictions = [("help", 0.55), ("home", 0.45)]
interpretation = agent.interpret_command(predictions, context="Medical alert")
print(interpretation)
```

### 6. Run Complete Pipeline

```python
from src.siri_for_wifi import SiriForWiFi

system = SiriForWiFi(use_llm_agent=True)
result = system.process_rf_signal('data/images/sample.png')
```

---

## Training the Classifier

### Fine-tune on RVTALL Data

```python
from models.custom_resnet import CustomResNet
from models.trainer import Trainer
import torch

# Load model with pre-trained weights
model = CustomResNet(
    in_channels=3,
    num_classes=15,
    pre_trained=True,  # ImageNet weights
    model='resnet18'
)

# Prepare data (use RVTALL-Preprocess loaders)
# ... data loading code ...

# Train
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

trainer = Trainer(
    num_inputs=1,
    classifier=model,
    optimizer=optimizer,
    criterion=criterion,
    print_every=10,
    device='cuda'
)

trainer.train(trainloader, testloader, epochs=50)

# Save fine-tuned weights
torch.save(model.state_dict(), 'models/resnet18_rvtall_finetuned.pth')
```

---

## Real-World Deployment Example

### Emergency Alert System

```python
system = SiriForWiFi(
    classifier_model_path='models/resnet18_rvtall_finetuned.pth',
    use_llm_agent=True,
    confidence_threshold=0.6  # Lower for emergency scenarios
)

# Continuous monitoring
while True:
    # Capture RF signal
    spectrogram = capture_rf_signal()  # Your RF hardware integration
    
    # Save as image
    save_spectrogram(spectrogram, 'temp/current.png')
    
    # Process
    result = system.process_rf_signal(
        'temp/current.png',
        context=get_user_context()  # Location, time, recent activity
    )
    
    # Check for emergency
    if result['interpretation']['action'].startswith('trigger_'):
        execute_emergency_protocol(result)
```

---

## Advantages Over Traditional Systems

| Feature | Traditional RF System | Siri for WiFi |
|---------|----------------------|---------------|
| **Training Data** | 10,000+ samples | 100-1,000 samples |
| **Robustness** | Brittle | Robust (pre-trained) |
| **Fuzzy Matching** | None | LLM-powered |
| **Context Awareness** | Rule-based | Intelligent reasoning |
| **Adaptation** | Retraining required | Prompt engineering |
| **Error Handling** | Fail | Graceful degradation |

---

## Troubleshooting

### Issue: Low classification accuracy
**Solution:** Fine-tune on more RVTALL data, adjust preprocessing

### Issue: Agent makes wrong decisions
**Solution:** Improve prompts, add more context, adjust confidence threshold

### Issue: Slow inference
**Solution:** Use ResNet18 instead of ResNet50, batch processing

---

## Next Steps

1. **Extract RVTALL data** - Get the 15 word commands
2. **Generate spectrograms** - Run Stage 1 preprocessing
3. **Fine-tune classifier** - Train on your spectrograms
4. **Test agent** - Verify intelligent decision making
5. **Deploy** - Integrate with RF hardware

See `docs/both_repos_comparison.md` for detailed repository analysis.

