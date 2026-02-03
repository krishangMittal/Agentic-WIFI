"""
Train ResNet on MM-Fi WiFi CSI Spectrograms
============================================
This script:
1. Processes WiFi CSI .mat files → spectrograms
2. Fine-tunes ResNet on 27 gestures
3. Saves trained model

Usage:
    python train_mmfi.py --quick    # Train on S01 only (fast test)
    python train_mmfi.py            # Train on all subjects (4 hours)
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import scipy.io as scio
from scipy import signal
import matplotlib
matplotlib.use('Agg')  # No GUI
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

# ============================================================================
# STEP 1: WiFi CSI → Spectrogram Processor
# ============================================================================

class CSIProcessor:
    """Convert WiFi CSI .mat files to spectrogram images"""
    
    def __init__(self, output_dir='data/processed/mmfi_spectrograms'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def load_csi_mat(self, mat_path):
        """Load WiFi CSI from .mat file"""
        try:
            data = scio.loadmat(mat_path)
            # MM-Fi CSI format: complex-valued matrix
            if 'csi' in data:
                csi = data['csi']
            elif 'wifi_data' in data:
                csi = data['wifi_data']
            elif 'CSIamp' in data:
                csi = data['CSIamp']
            elif 'CSIphase' in data:
                csi = data['CSIphase']
            else:
                # Try first non-metadata key
                keys = [k for k in data.keys() if not k.startswith('__')]
                if len(keys) > 0:
                    csi = data[keys[0]]
                else:
                    return None
            
            # Ensure CSI is 2D
            if len(csi.shape) == 1:
                csi = csi.reshape(-1, 1)
            
            return csi
        except Exception as e:
            print(f"Error loading {mat_path}: {e}")
            return None
    
    def csi_to_spectrogram(self, csi, nperseg=64, noverlap=32):
        """Convert CSI to spectrogram using STFT
        
        MM-Fi CSI format: (antennas, subcarriers, time_samples)
        We create a 2D representation: subcarriers × time
        """
        # Handle complex CSI or amplitude directly
        if np.iscomplexobj(csi):
            amplitude = np.abs(csi)
        else:
            amplitude = csi
        
        # MM-Fi format: (3, 114, 10) = (antennas, subcarriers, time)
        if len(amplitude.shape) == 3:
            # Average across antennas: (114, 10)
            amplitude = np.mean(amplitude, axis=0)
        elif len(amplitude.shape) == 2:
            # Already 2D, use as-is
            pass
        else:
            # 1D signal - reshape to 2D
            amplitude = amplitude.reshape(-1, 1)
        
        # Now amplitude is (subcarriers, time_samples)
        # This IS our spectrogram! (frequency × time)
        # Just normalize and apply log scale
        
        # Normalize
        amplitude = (amplitude - amplitude.min()) / (amplitude.max() - amplitude.min() + 1e-10)
        
        # Log scale for better visualization
        Sxx_log = 10 * np.log10(amplitude + 1e-10)
        
        return Sxx_log
    
    def save_spectrogram_image(self, spectrogram, output_path):
        """Save spectrogram as image (256x256 Jet colormap)"""
        plt.figure(figsize=(2.56, 2.56), dpi=100)
        plt.imshow(spectrogram, aspect='auto', cmap='jet', origin='lower')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    
    def process_dataset(self, data_root='data/raw/MMFi/E01', subjects=['S01'], max_per_action=None):
        """Process entire MM-Fi dataset"""
        print("\n[*] Processing MM-Fi WiFi CSI Data...")
        
        action_map = {}  # Map action folders to labels
        processed_count = 0
        
        for subject in subjects:
            subject_path = os.path.join(data_root, subject)
            if not os.path.exists(subject_path):
                print(f"[!] Subject {subject} not found, skipping...")
                continue
            
            # Get all action folders (A01-A27)
            actions = sorted([d for d in os.listdir(subject_path) if d.startswith('A')])
            
            for action_idx, action in enumerate(actions):
                action_path = os.path.join(subject_path, action)
                wifi_path = os.path.join(action_path, 'wifi-csi')
                
                if not os.path.exists(wifi_path):
                    continue
                
                # Create output directory for this action
                output_action_dir = os.path.join(self.output_dir, action)
                os.makedirs(output_action_dir, exist_ok=True)
                
                # Store action mapping
                if action not in action_map:
                    action_map[action] = action_idx
                
                # Process all .mat files in this action
                mat_files = [f for f in os.listdir(wifi_path) if f.endswith('.mat')]
                if max_per_action:
                    mat_files = mat_files[:max_per_action]
                
                print(f"  {subject}/{action}: {len(mat_files)} samples")
                
                for mat_file in tqdm(mat_files, desc=f"  Processing {subject}/{action}", leave=False):
                    mat_path = os.path.join(wifi_path, mat_file)
                    
                    # Load CSI
                    csi = self.load_csi_mat(mat_path)
                    if csi is None:
                        continue
                    
                    # Convert to spectrogram
                    spectrogram = self.csi_to_spectrogram(csi)
                    
                    # Save as image
                    output_filename = f"{subject}_{mat_file.replace('.mat', '.png')}"
                    output_path = os.path.join(output_action_dir, output_filename)
                    self.save_spectrogram_image(spectrogram, output_path)
                    
                    processed_count += 1
        
        # Save action mapping
        mapping_path = os.path.join(self.output_dir, 'action_mapping.json')
        with open(mapping_path, 'w') as f:
            json.dump(action_map, f, indent=2)
        
        print(f"\n[OK] Processed {processed_count} spectrograms")
        print(f"[OK] Saved to: {self.output_dir}")
        print(f"[OK] Action mapping: {mapping_path}")
        
        return action_map


# ============================================================================
# STEP 2: Dataset Loader
# ============================================================================

class MMFiSpectrogramDataset(Dataset):
    """PyTorch Dataset for MM-Fi spectrograms"""
    
    def __init__(self, data_dir='data/processed/mmfi_spectrograms', transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        # Load action mapping
        mapping_path = os.path.join(data_dir, 'action_mapping.json')
        with open(mapping_path, 'r') as f:
            self.action_map = json.load(f)
        
        # Build file list
        self.samples = []
        for action, label in self.action_map.items():
            action_dir = os.path.join(data_dir, action)
            if not os.path.exists(action_dir):
                continue
            
            for img_file in os.listdir(action_dir):
                if img_file.endswith('.png'):
                    img_path = os.path.join(action_dir, img_file)
                    self.samples.append((img_path, label))
        
        print(f"[*] Loaded {len(self.samples)} samples, {len(self.action_map)} classes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


# ============================================================================
# STEP 3: Training
# ============================================================================

def train_model(data_dir='data/processed/mmfi_spectrograms', 
                num_epochs=50, 
                batch_size=32,
                learning_rate=0.001,
                output_model='models/resnet_mmfi_trained.pth'):
    """Train ResNet on MM-Fi spectrograms"""
    
    print("\n[*] Setting up training...")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Using device: {device}")
    
    # Data transforms (same as ImageNet)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    dataset = MMFiSpectrogramDataset(data_dir=data_dir, transform=transform)
    
    # Split train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"[*] Train: {train_size}, Val: {val_size}")
    
    # Load pre-trained ResNet
    num_classes = len(dataset.action_map)
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    
    print(f"[*] Model: ResNet18 (pre-trained), {num_classes} classes")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    
    # Training loop
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f"\n[Epoch {epoch+1}/{num_epochs}]")
        
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100.0 * train_correct / train_total
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100.0 * val_correct / val_total
        
        print(f"  Train Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss/len(val_loader):.4f}, Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(output_model), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'action_map': dataset.action_map
            }, output_model)
            print(f"  [OK] Saved best model (val_acc: {val_acc:.2f}%)")
        
        scheduler.step()
    
    print(f"\n[OK] Training complete!")
    print(f"[OK] Best validation accuracy: {best_val_acc:.2f}%")
    print(f"[OK] Model saved: {output_model}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train ResNet on MM-Fi WiFi CSI')
    parser.add_argument('--quick', action='store_true', help='Quick test (S01 only, 10 samples/action)')
    parser.add_argument('--process-only', action='store_true', help='Only process spectrograms, skip training')
    parser.add_argument('--train-only', action='store_true', help='Only train (spectrograms already processed)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    
    args = parser.parse_args()
    
    print("="*70)
    print(" MM-Fi WiFi CSI Gesture Recognition - Training Pipeline")
    print("="*70)
    
    # Step 1: Process WiFi CSI → Spectrograms
    if not args.train_only:
        processor = CSIProcessor()
        
        if args.quick:
            print("\n[MODE] Quick test (S01, 10 samples/action)")
            subjects = ['S01']
            max_per_action = 10
            epochs = 5
        else:
            print("\n[MODE] Full training (all subjects)")
            subjects = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10']
            max_per_action = None
            epochs = args.epochs
        
        processor.process_dataset(
            data_root='data/raw/MMFi/E01',
            subjects=subjects,
            max_per_action=max_per_action
        )
    
    if args.process_only:
        print("\n[*] Process-only mode, skipping training.")
        return
    
    # Step 2: Train model
    if not args.quick:
        epochs = args.epochs
    else:
        epochs = 5
    
    train_model(
        data_dir='data/processed/mmfi_spectrograms',
        num_epochs=epochs,
        batch_size=args.batch_size,
        output_model='models/resnet_mmfi_trained.pth'
    )
    
    print("\n" + "="*70)
    print(" Training Complete!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Test the model: python test_trained_model.py")
    print("  2. Run real-time: python setup_realtime.py")
    print("="*70)


if __name__ == '__main__':
    main()

