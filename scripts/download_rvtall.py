"""
Script to download and set up the RVTALL dataset.

RVTALL Dataset Information:
- DOI: 10.1038/s41597-023-02793-w
- Nature Scientific Data, 2023
- Contains 15 voice commands for RF sensing research
"""

import os
import sys
import requests
from pathlib import Path
from typing import Optional
import zipfile
import tarfile

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RVTALL_DIR = DATA_DIR / "RVTALL"


def download_file(url: str, output_path: Path, chunk_size: int = 8192) -> bool:
    """
    Download a file from URL with progress tracking.
    
    Args:
        url: URL to download from
        output_path: Path to save the file
        chunk_size: Chunk size for streaming download
        
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"Downloading from: {url}")
        print(f"Saving to: {output_path}")
        
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rProgress: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='')
        
        print("\nDownload complete!")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"\nError downloading file: {e}")
        return False


def extract_archive(archive_path: Path, extract_to: Path) -> bool:
    """
    Extract a zip or tar archive.
    
    Args:
        archive_path: Path to the archive file
        extract_to: Directory to extract to
        
    Returns:
        True if successful, False otherwise
    """
    try:
        extract_to.mkdir(parents=True, exist_ok=True)
        
        if archive_path.suffix == '.zip':
            print(f"Extracting ZIP archive: {archive_path.name}")
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
                
        elif archive_path.suffix in ['.tar', '.gz']:
            print(f"Extracting TAR archive: {archive_path.name}")
            mode = 'r:gz' if archive_path.suffix == '.gz' else 'r'
            with tarfile.open(archive_path, mode) as tar_ref:
                tar_ref.extractall(extract_to)
        else:
            print(f"Unsupported archive format: {archive_path.suffix}")
            return False
        
        print("Extraction complete!")
        return True
        
    except Exception as e:
        print(f"Error extracting archive: {e}")
        return False


def download_rvtall_manual_instructions():
    """
    Print manual download instructions for RVTALL.
    """
    print("\n" + "="*70)
    print("MANUAL DOWNLOAD INSTRUCTIONS FOR RVTALL DATASET")
    print("="*70)
    print("\nSince ResearchGate requires manual download, follow these steps:\n")
    print("1. Visit one of these locations:")
    print("   - ResearchGate: Search for 'RVTALL dataset' or DOI 10.1038/s41597-023-02793-w")
    print("   - Nature Scientific Data: https://www.nature.com/articles/s41597-023-02793-w")
    print("   - Direct repository link (if available from paper)")
    print("\n2. Download the dataset archive (usually .zip or .tar.gz)")
    print("\n3. Place the downloaded file in the data/ directory:")
    print(f"   {DATA_DIR}")
    print("\n4. Run this script again with the archive path:")
    print("   python scripts/download_rvtall.py --extract <path_to_archive>")
    print("\n5. Or manually extract the archive to:")
    print(f"   {RVTALL_DIR}")
    print("\n" + "="*70 + "\n")


def inspect_corpus(corpus_path: Path) -> Optional[list]:
    """
    Inspect the Corpus folder to find the 15 command words.
    
    Args:
        corpus_path: Path to the Corpus folder
        
    Returns:
        List of command words if found, None otherwise
    """
    if not corpus_path.exists():
        print(f"Corpus folder not found at: {corpus_path}")
        return None
    
    print(f"\nInspecting Corpus folder: {corpus_path}")
    
    # Look for subdirectories or files that might contain the words
    words = []
    
    # Check if there are subdirectories (common structure)
    if corpus_path.is_dir():
        subdirs = [d for d in corpus_path.iterdir() if d.is_dir()]
        files = [f for f in corpus_path.iterdir() if f.is_file()]
        
        if subdirs:
            print(f"\nFound {len(subdirs)} subdirectories:")
            for subdir in sorted(subdirs):
                word = subdir.name
                words.append(word)
                print(f"  - {word}")
        
        elif files:
            print(f"\nFound {len(files)} files:")
            for file in sorted(files):
                # Try to extract word from filename
                word = file.stem  # filename without extension
                if word not in words:
                    words.append(word)
                    print(f"  - {word}")
    
    if words:
        print(f"\n✓ Found {len(words)} command words!")
        return sorted(words)
    else:
        print("\n⚠ Could not automatically detect command words.")
        print("Please manually inspect the Corpus folder structure.")
        return None


def main():
    """Main function to download and set up RVTALL dataset."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download and set up RVTALL dataset')
    parser.add_argument('--url', type=str, help='Direct download URL (if available)')
    parser.add_argument('--extract', type=str, help='Path to downloaded archive to extract')
    parser.add_argument('--inspect', action='store_true', help='Inspect Corpus folder after setup')
    
    args = parser.parse_args()
    
    # Create data directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # If URL provided, try to download
    if args.url:
        archive_name = "RVTALL_dataset.zip"  # Default name
        archive_path = DATA_DIR / archive_name
        
        if download_file(args.url, archive_path):
            if extract_archive(archive_path, RVTALL_DIR):
                print(f"\n✓ Dataset extracted to: {RVTALL_DIR}")
                # Optionally remove archive
                # archive_path.unlink()
    
    # If extract path provided, extract it
    elif args.extract:
        archive_path = Path(args.extract)
        if not archive_path.exists():
            print(f"Error: Archive not found at {archive_path}")
            return
        
        if extract_archive(archive_path, RVTALL_DIR):
            print(f"\n✓ Dataset extracted to: {RVTALL_DIR}")
    
    # Otherwise, show manual instructions
    else:
        download_rvtall_manual_instructions()
        
        # Check if dataset already exists
        if RVTALL_DIR.exists():
            print(f"\n✓ RVTALL directory found at: {RVTALL_DIR}")
            print("Checking for Corpus folder...")
            
            corpus_path = RVTALL_DIR / "Corpus"
            if not corpus_path.exists():
                # Try to find Corpus folder in subdirectories
                for root, dirs, files in os.walk(RVTALL_DIR):
                    if "Corpus" in dirs:
                        corpus_path = Path(root) / "Corpus"
                        break
            
            if corpus_path.exists():
                words = inspect_corpus(corpus_path)
                if words:
                    # Save words to a file
                    words_file = PROJECT_ROOT / "data" / "rvtall_commands.txt"
                    with open(words_file, 'w') as f:
                        for word in words:
                            f.write(f"{word}\n")
                    print(f"\n✓ Saved command words to: {words_file}")
            else:
                print(f"\n⚠ Corpus folder not found. Please check the dataset structure.")
    
    # Inspect if requested
    if args.inspect or (RVTALL_DIR.exists() and not args.url and not args.extract):
        corpus_path = RVTALL_DIR / "Corpus"
        if corpus_path.exists():
            inspect_corpus(corpus_path)


if __name__ == "__main__":
    main()

