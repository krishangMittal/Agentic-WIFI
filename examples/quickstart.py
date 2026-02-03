#!/usr/bin/env python3
"""
Quick Start Script for Siri for WiFi

Run this script to test the complete workflow with demo data.
"""

import sys
from pathlib import Path

def check_requirements():
    """Check if required dependencies are installed."""
    print("Checking requirements...")
    
    missing = []
    
    try:
        import torch
        print("✓ PyTorch installed")
    except ImportError:
        missing.append("torch")
    
    try:
        import torchvision
        print("✓ TorchVision installed")
    except ImportError:
        missing.append("torchvision")
    
    try:
        from PIL import Image
        print("✓ Pillow installed")
    except ImportError:
        missing.append("Pillow")
    
    if missing:
        print(f"\n⚠ Missing dependencies: {', '.join(missing)}")
        print("\nInstall with:")
        print(f"  pip install {' '.join(missing)}")
        print("\nOr use conda environment:")
        print("  conda env create -f environment.yml")
        print("  conda activate rf-sensing-research")
        return False
    
    print("✓ All required packages installed\n")
    return True


def check_structure():
    """Check if project structure is set up correctly."""
    print("Checking project structure...")
    
    required = [
        "data/raw",
        "data/spectrograms",
        "data/images",
        "models",
        "src/classifier.py",
        "src/agent_ai.py",
        "src/siri_for_wifi.py"
    ]
    
    missing = []
    for path in required:
        if not Path(path).exists():
            missing.append(path)
        else:
            print(f"✓ {path}")
    
    if missing:
        print(f"\n⚠ Missing: {', '.join(missing)}")
        return False
    
    print("✓ Project structure OK\n")
    return True


def check_models():
    """Check if model files are copied from cloned repos."""
    print("Checking model files...")
    
    if not Path("models/custom_resnet.py").exists():
        print("⚠ models/custom_resnet.py not found")
        print("\nCopy from cloned repo:")
        print("  cp RVTALL-Preprocess/classification/models.py models/custom_resnet.py")
        return False
    
    print("✓ Model files ready\n")
    return True


def run_demos():
    """Run demo scripts."""
    print("="*70)
    print("RUNNING DEMOS")
    print("="*70 + "\n")
    
    demos = [
        ("src/classifier.py", "Command Classifier"),
        ("src/agent_ai.py", "Agentic Logic"),
        ("src/siri_for_wifi.py", "Complete Workflow")
    ]
    
    for script, name in demos:
        print(f"\n{'='*70}")
        print(f"Demo: {name}")
        print("="*70)
        print(f"Running: python {script}\n")
        
        try:
            import subprocess
            result = subprocess.run(
                [sys.executable, script],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                print(result.stdout)
                print(f"✓ {name} demo completed successfully")
            else:
                print(f"⚠ {name} demo had errors:")
                print(result.stderr)
        
        except subprocess.TimeoutExpired:
            print(f"⚠ {name} demo timed out")
        except Exception as e:
            print(f"⚠ Error running {name}: {e}")
        
        input("\nPress Enter to continue to next demo...")


def main():
    """Main entry point."""
    print("="*70)
    print("SIRI FOR WIFI - QUICK START")
    print("="*70 + "\n")
    
    # Check requirements
    if not check_requirements():
        print("\nPlease install missing dependencies first.")
        return
    
    # Check structure
    if not check_structure():
        print("\nPlease ensure project structure is set up correctly.")
        return
    
    # Check models
    if not check_models():
        print("\nPlease copy model files from cloned repositories.")
        return
    
    print("="*70)
    print("SYSTEM READY!")
    print("="*70 + "\n")
    
    print("What would you like to do?")
    print("1. Run all demos (classifier, agent, complete workflow)")
    print("2. Test classifier only")
    print("3. Test agent only")
    print("4. Run complete workflow")
    print("5. Exit")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == "1":
        run_demos()
    elif choice == "2":
        print("\nRunning classifier demo...")
        subprocess.run([sys.executable, "src/classifier.py"])
    elif choice == "3":
        print("\nRunning agent demo...")
        subprocess.run([sys.executable, "src/agent_ai.py"])
    elif choice == "4":
        print("\nRunning complete workflow...")
        subprocess.run([sys.executable, "src/siri_for_wifi.py"])
    else:
        print("Exiting...")
    
    print("\n" + "="*70)
    print("Next Steps:")
    print("  1. Download RVTALL dataset (see docs/download_rvtall_instructions.md)")
    print("  2. Run Stage 1 preprocessing to generate spectrograms")
    print("  3. Fine-tune classifier on RVTALL data")
    print("  4. Deploy in your RF sensing application")
    print("\nSee docs/siri_for_wifi_workflow.md for detailed guide.")
    print("="*70)


if __name__ == "__main__":
    main()

