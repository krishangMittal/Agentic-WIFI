"""
Script to inspect the RVTALL dataset structure and extract command words.

This script will:
1. Locate the Corpus folder
2. Extract the 15 command words
3. Display dataset statistics
4. Save command list for use in training
"""

import os
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RVTALL_DIR = DATA_DIR / "RVTALL"


def find_corpus_folder(root_dir: Path) -> Path:
    """
    Recursively search for the Corpus folder.
    
    Args:
        root_dir: Root directory to search in
        
    Returns:
        Path to Corpus folder if found, None otherwise
    """
    for root, dirs, files in os.walk(root_dir):
        if "Corpus" in dirs:
            return Path(root) / "Corpus"
        # Also check case-insensitive
        for d in dirs:
            if d.lower() == "corpus":
                return Path(root) / d
    return None


def extract_command_words(corpus_path: Path) -> list:
    """
    Extract command words from the Corpus folder.
    
    Args:
        corpus_path: Path to Corpus folder
        
    Returns:
        List of command words
    """
    words = []
    
    if not corpus_path.exists():
        print(f"Error: Corpus folder not found at {corpus_path}")
        return words
    
    # Method 1: Check subdirectories (most common structure)
    subdirs = [d for d in corpus_path.iterdir() if d.is_dir()]
    if subdirs:
        words = [d.name for d in subdirs if not d.name.startswith('.')]
        print(f"Found {len(words)} command words from subdirectories:")
        for word in sorted(words):
            # Count samples in each directory
            sample_count = len(list((corpus_path / word).glob("*")))
            print(f"  - {word}: {sample_count} samples")
    
    # Method 2: Check files if no subdirectories
    elif not words:
        files = [f for f in corpus_path.iterdir() if f.is_file()]
        # Try to extract words from filenames
        for file in files:
            # Common patterns: word_001.mat, word_1.csv, etc.
            name_parts = file.stem.split('_')
            if name_parts:
                word = name_parts[0]
                if word not in words:
                    words.append(word)
    
    return sorted(words)


def get_dataset_stats(rvtall_dir: Path) -> dict:
    """
    Get statistics about the RVTALL dataset.
    
    Args:
        rvtall_dir: Path to RVTALL dataset root
        
    Returns:
        Dictionary with dataset statistics
    """
    stats = {
        'total_files': 0,
        'total_dirs': 0,
        'file_types': {},
        'structure': {}
    }
    
    for root, dirs, files in os.walk(rvtall_dir):
        stats['total_dirs'] += len(dirs)
        stats['total_files'] += len(files)
        
        for file in files:
            ext = Path(file).suffix.lower()
            stats['file_types'][ext] = stats['file_types'].get(ext, 0) + 1
    
    return stats


def main():
    """Main inspection function."""
    print("="*70)
    print("RVTALL DATASET INSPECTION")
    print("="*70)
    
    # Check if dataset exists
    if not RVTALL_DIR.exists():
        print(f"\n⚠ RVTALL dataset not found at: {RVTALL_DIR}")
        print("\nPlease download the dataset first using:")
        print("  python scripts/download_rvtall.py")
        return
    
    print(f"\n✓ Dataset found at: {RVTALL_DIR}")
    
    # Find Corpus folder
    corpus_path = find_corpus_folder(RVTALL_DIR)
    
    if not corpus_path:
        print("\n⚠ Corpus folder not found!")
        print("\nDataset structure:")
        for item in sorted(RVTALL_DIR.iterdir()):
            if item.is_dir():
                print(f"  📁 {item.name}/")
            else:
                print(f"  📄 {item.name}")
        return
    
    print(f"\n✓ Corpus folder found at: {corpus_path}")
    
    # Extract command words
    words = extract_command_words(corpus_path)
    
    if not words:
        print("\n⚠ Could not extract command words automatically.")
        print("Please manually inspect the Corpus folder structure.")
        return
    
    print(f"\n{'='*70}")
    print(f"FOUND {len(words)} COMMAND WORDS:")
    print("="*70)
    for i, word in enumerate(words, 1):
        print(f"{i:2d}. {word}")
    
    # Save command words
    commands_file = DATA_DIR / "rvtall_commands.txt"
    commands_json = DATA_DIR / "rvtall_commands.json"
    
    with open(commands_file, 'w') as f:
        for word in words:
            f.write(f"{word}\n")
    
    with open(commands_json, 'w') as f:
        json.dump({
            'dataset': 'RVTALL',
            'num_commands': len(words),
            'commands': words,
            'corpus_path': str(corpus_path.relative_to(PROJECT_ROOT))
        }, f, indent=2)
    
    print(f"\n✓ Saved command list to:")
    print(f"  - {commands_file}")
    print(f"  - {commands_json}")
    
    # Get dataset statistics
    stats = get_dataset_stats(RVTALL_DIR)
    print(f"\n{'='*70}")
    print("DATASET STATISTICS")
    print("="*70)
    print(f"Total directories: {stats['total_dirs']}")
    print(f"Total files: {stats['total_files']}")
    print(f"\nFile types:")
    for ext, count in sorted(stats['file_types'].items()):
        print(f"  {ext or '(no extension)'}: {count}")
    
    print(f"\n{'='*70}")
    print("NEXT STEPS:")
    print("="*70)
    print("1. Review the command words in data/rvtall_commands.txt")
    print("2. Update your model configuration with these 15 commands")
    print("3. Use preprocess.py to convert CSI data to spectrograms")
    print("4. Train your classifier using model.py with CommandClassifierResNet")
    print("="*70)


if __name__ == "__main__":
    main()

