# Downloading the RVTALL Dataset

## Quick Start

The RVTALL dataset is available through ResearchGate or via the Nature Scientific Data paper.

**DOI:** 10.1038/s41597-023-02793-w  
**Paper:** "RVTALL: A Large-Scale Dataset for Remote Voice Recognition via RF Sensing"  
**Nature Scientific Data, 2023**

## Download Methods

### Method 1: ResearchGate (Recommended)

1. Go to [ResearchGate](https://www.researchgate.net/)
2. Search for "RVTALL dataset" or the DOI `10.1038/s41597-023-02793-w`
3. Request access from the authors (usually granted quickly)
4. Download the dataset archive (typically a `.zip` or `.tar.gz` file)
5. Place the downloaded file in the `data/` directory

### Method 2: Nature Scientific Data

1. Visit: https://www.nature.com/articles/s41597-023-02793-w
2. Look for the "Data availability" section
3. Follow the link to the dataset repository
4. Download the dataset archive
5. Place it in the `data/` directory

## Setting Up the Dataset

Once you have downloaded the archive:

### Option A: Using the Script

```bash
# Extract the downloaded archive
python scripts/download_rvtall.py --extract data/RVTALL_dataset.zip

# Or if it's a tar.gz file
python scripts/download_rvtall.py --extract data/RVTALL_dataset.tar.gz
```

### Option B: Manual Extraction

1. Extract the archive manually
2. Place the extracted `RVTALL` folder in the `data/` directory
3. Expected structure: `data/RVTALL/Corpus/...`

## Inspecting the Dataset

After extraction, inspect the dataset to extract the 15 command words:

```bash
python scripts/inspect_rvtall.py
```

This will:
- ✅ Locate the Corpus folder
- ✅ Extract the 15 command words
- ✅ Save them to `data/rvtall_commands.txt`
- ✅ Display dataset statistics

## Expected Dataset Structure

```
data/
└── RVTALL/
    ├── Corpus/
    │   ├── word1/          # First command word
    │   │   ├── sample1.mat
    │   │   ├── sample2.mat
    │   │   └── ...
    │   ├── word2/          # Second command word
    │   └── ...             # (15 words total)
    ├── README.txt          # Dataset documentation
    └── ...                 # Other dataset files
```

## The 15 Command Words

After running `inspect_rvtall.py`, the command words will be saved to:
- `data/rvtall_commands.txt` - Simple text file (one word per line)
- `data/rvtall_commands.json` - Structured JSON format

These words will be your initial "supported commands" for the RF sensing classifier.

## Troubleshooting

### "Corpus folder not found"
- Check that the dataset was extracted correctly
- The Corpus folder might be nested deeper in the directory structure
- Run `inspect_rvtall.py` - it will search recursively

### "Dataset not found"
- Ensure the RVTALL folder is in `data/RVTALL/`
- Check that you extracted the archive correctly
- Verify the folder structure matches the expected format

### Download Issues
- ResearchGate may require you to create a free account
- Some datasets require email request to authors
- Check the paper's "Data availability" section for alternative links

## Next Steps

Once the dataset is set up:

1. ✅ **Verify the 15 commands** in `data/rvtall_commands.txt`
2. ✅ **Update your model** - Use these commands as your classification classes
3. ✅ **Preprocess data** - Use `src/preprocess.py` to convert CSI to spectrograms
4. ✅ **Start training** - Use `src/model.py` with `CommandClassifierResNet`

## Alternative: CSI-Bench Dataset

If RVTALL is difficult to access, consider **CSI-Bench** as an alternative:
- Large-scale WiFi sensing dataset (461+ hours)
- Available on Kaggle
- More recent (2025) and includes multiple sensing tasks
- See: https://arxiv.org/html/2505.21866v1

