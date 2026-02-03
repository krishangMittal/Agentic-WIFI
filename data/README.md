# Data Directory

This directory contains the RVTALL and MM-Fi datasets for RF sensing research.

## RVTALL Dataset

**DOI:** 10.1038/s41597-023-02793-w  
**Paper:** Nature Scientific Data, 2023  
**Commands:** 15 voice commands for RF lip reading

### Download Instructions

1. **Manual Download (Recommended):**
   - Visit ResearchGate and search for "RVTALL dataset"
   - Or access via DOI: https://www.nature.com/articles/s41597-023-02793-w
   - Download the dataset archive (usually .zip or .tar.gz)
   - Place it in this directory (`data/`)

2. **Extract the Dataset:**
   ```bash
   # If you have the archive file:
   python scripts/download_rvtall.py --extract <path_to_archive>
   
   # Or manually extract to: data/RVTALL/
   ```

3. **Inspect the Dataset:**
   ```bash
   python scripts/inspect_rvtall.py
   ```
   This will:
   - Locate the Corpus folder
   - Extract the 15 command words
   - Save them to `data/rvtall_commands.txt`
   - Display dataset statistics

### Expected Structure

```
data/
├── RVTALL/
│   ├── Corpus/
│   │   ├── word1/
│   │   │   ├── sample1.mat (or .csv, .npy, etc.)
│   │   │   ├── sample2.mat
│   │   │   └── ...
│   │   ├── word2/
│   │   └── ... (15 words total)
│   └── ... (other dataset files)
├── rvtall_commands.txt  # Generated after inspection
└── rvtall_commands.json  # Generated after inspection
```

### Command Words

After running `inspect_rvtall.py`, the 15 command words will be saved to:
- `data/rvtall_commands.txt` (one word per line)
- `data/rvtall_commands.json` (structured format)

## MM-Fi Dataset (Optional)

If you also download MM-Fi dataset, place it in:
```
data/MM-Fi/
```

## Notes

- Keep original dataset archives for backup
- The Corpus folder contains the labeled command samples
- Each command should have multiple samples for training/validation
- Use `src/preprocess.py` to convert CSI samples to spectrograms

