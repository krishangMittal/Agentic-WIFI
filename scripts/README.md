# Scripts Directory

Utility scripts for dataset management and inspection.

## Available Scripts

### `download_rvtall.py`

Download and set up the RVTALL dataset.

**Usage:**
```bash
# Show manual download instructions
python scripts/download_rvtall.py

# Download from direct URL (if available)
python scripts/download_rvtall.py --url <direct_download_url>

# Extract a downloaded archive
python scripts/download_rvtall.py --extract <path_to_archive>

# Inspect after setup
python scripts/download_rvtall.py --inspect
```

### `inspect_rvtall.py`

Inspect the RVTALL dataset structure and extract command words.

**Usage:**
```bash
python scripts/inspect_rvtall.py
```

**What it does:**
- Locates the Corpus folder
- Extracts the 15 command words
- Displays dataset statistics
- Saves command list to `data/rvtall_commands.txt` and `data/rvtall_commands.json`

## Requirements

These scripts require:
- `requests` (for downloading)
- Standard library: `pathlib`, `os`, `json`, `zipfile`, `tarfile`

Install with:
```bash
pip install requests
```

Or use the conda environment:
```bash
conda activate rf-sensing-research
```

