"""
Visualize WiFi CSI Data Pipeline
=================================
Shows what the data looks like at each stage of processing.
"""

import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================================
# STEP 1: Load Raw WiFi CSI Data
# ============================================================================

def visualize_raw_csi():
    """Show what raw WiFi CSI looks like"""
    
    # Load one sample
    mat_path = 'data/raw/MMFi/E01/S01/A06/wifi-csi/frame001.mat'
    data = scio.loadmat(mat_path)
    
    csi_amp = data['CSIamp']    # (3, 114, 10) = (antennas, subcarriers, time)
    csi_phase = data['CSIphase']
    
    print("="*70)
    print(" RAW WiFi CSI DATA")
    print("="*70)
    print(f"\nFile: {mat_path}")
    print(f"Action: A06 = Draw Circle (Clockwise)\n")
    
    print("Data Structure:")
    print(f"  CSIamp shape:   {csi_amp.shape}")
    print(f"  CSIphase shape: {csi_phase.shape}\n")
    
    print("What this means:")
    print("  - 3 WiFi antennas capturing signal")
    print("  - 114 frequency subcarriers (WiFi channels)")
    print("  - 10 time snapshots during gesture\n")
    
    print("Raw CSI Amplitude (Antenna 1, first 5 subcarriers, all time steps):")
    print(csi_amp[0, :5, :])
    print("\nThese numbers represent WiFi signal strength!")
    print("When you move your hand -> these numbers change!")
    
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('RAW WiFi CSI Data: A06 (Draw Circle CW)', fontsize=16, fontweight='bold')
    
    # Plot amplitude for each antenna
    for i in range(3):
        ax = axes[0, i]
        im = ax.imshow(csi_amp[i, :, :], aspect='auto', cmap='jet', origin='lower')
        ax.set_title(f'Antenna {i+1} - Amplitude')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Frequency Subcarriers (0-113)')
        plt.colorbar(im, ax=ax, label='Signal Strength')
    
    # Plot phase for each antenna
    for i in range(3):
        ax = axes[1, i]
        im = ax.imshow(csi_phase[i, :, :], aspect='auto', cmap='jet', origin='lower')
        ax.set_title(f'Antenna {i+1} - Phase')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Frequency Subcarriers (0-113)')
        plt.colorbar(im, ax=ax, label='Phase Shift')
    
    plt.tight_layout()
    os.makedirs('data/visualizations', exist_ok=True)
    plt.savefig('data/visualizations/1_raw_csi.png', dpi=150, bbox_inches='tight')
    print("\n[SAVED] data/visualizations/1_raw_csi.png")
    plt.close()
    
    return csi_amp


# ============================================================================
# STEP 2: Process CSI -> Spectrogram
# ============================================================================

def visualize_processing(csi_amp):
    """Show how CSI is converted to spectrogram"""
    
    print("\n" + "="*70)
    print(" PROCESSING: CSI -> SPECTROGRAM")
    print("="*70)
    
    # Step 1: Average across antennas
    print("\nStep 1: Average across 3 antennas")
    print(f"  Before: {csi_amp.shape} (3 antennas)")
    avg_csi = np.mean(csi_amp, axis=0)
    print(f"  After:  {avg_csi.shape} (averaged)")
    print("  Why? Combine information from all antennas")
    
    # Step 2: Normalize
    print("\nStep 2: Normalize values")
    normalized = (avg_csi - avg_csi.min()) / (avg_csi.max() - avg_csi.min() + 1e-10)
    print(f"  Before: Range [{avg_csi.min():.2f}, {avg_csi.max():.2f}]")
    print(f"  After:  Range [0.00, 1.00]")
    print("  Why? Neural networks prefer normalized inputs")
    
    # Step 3: Log scale
    print("\nStep 3: Apply log scale")
    spectrogram = 10 * np.log10(normalized + 1e-10)
    print(f"  Range: [{spectrogram.min():.2f}, {spectrogram.max():.2f}] dB")
    print("  Why? Compress dynamic range, emphasize patterns")
    
    # Visualize transformation
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('CSI Processing Pipeline', fontsize=16, fontweight='bold')
    
    # Raw (antenna 1)
    im1 = axes[0].imshow(csi_amp[0, :, :], aspect='auto', cmap='jet', origin='lower')
    axes[0].set_title('Raw CSI (Antenna 1)')
    axes[0].set_xlabel('Time Steps')
    axes[0].set_ylabel('Frequency Subcarriers')
    plt.colorbar(im1, ax=axes[0])
    
    # Averaged
    im2 = axes[1].imshow(avg_csi, aspect='auto', cmap='jet', origin='lower')
    axes[1].set_title('Averaged (3 antennas)')
    axes[1].set_xlabel('Time Steps')
    axes[1].set_ylabel('Frequency Subcarriers')
    plt.colorbar(im2, ax=axes[1])
    
    # Final spectrogram
    im3 = axes[2].imshow(spectrogram, aspect='auto', cmap='jet', origin='lower')
    axes[2].set_title('Final Spectrogram (log scale)')
    axes[2].set_xlabel('Time Steps')
    axes[2].set_ylabel('Frequency Subcarriers')
    plt.colorbar(im3, ax=axes[2], label='dB')
    
    plt.tight_layout()
    plt.savefig('data/visualizations/2_processing_steps.png', dpi=150, bbox_inches='tight')
    print("\n[SAVED] data/visualizations/2_processing_steps.png")
    plt.close()
    
    return spectrogram


# ============================================================================
# STEP 3: Compare Different Gestures
# ============================================================================

def compare_gestures():
    """Show how different gestures create different spectrograms"""
    
    print("\n" + "="*70)
    print(" COMPARING DIFFERENT GESTURES")
    print("="*70)
    
    # Load 6 different gestures
    gestures = [
        ('A01', 'Push & Pull'),
        ('A03', 'Clap'),
        ('A06', 'Draw Circle CW'),
        ('A10', 'Sit Down'),
        ('A15', 'Jump'),
        ('A20', 'Walk')
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('WiFi Spectrograms: Different Gestures Create Different Patterns!', 
                 fontsize=16, fontweight='bold')
    
    for idx, (action, name) in enumerate(gestures):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        # Load CSI
        mat_path = f'data/raw/MMFi/E01/S01/{action}/wifi-csi/frame001.mat'
        if os.path.exists(mat_path):
            data = scio.loadmat(mat_path)
            csi_amp = data['CSIamp']
            
            # Process to spectrogram
            avg_csi = np.mean(csi_amp, axis=0)
            normalized = (avg_csi - avg_csi.min()) / (avg_csi.max() - avg_csi.min() + 1e-10)
            spectrogram = 10 * np.log10(normalized + 1e-10)
            
            # Plot
            im = ax.imshow(spectrogram, aspect='auto', cmap='jet', origin='lower')
            ax.set_title(f'{action}: {name}', fontweight='bold')
            ax.set_xlabel('Time')
            ax.set_ylabel('Frequency')
            plt.colorbar(im, ax=ax)
            
            print(f"  {action} ({name}): Pattern shape = {spectrogram.shape}")
        else:
            ax.text(0.5, 0.5, 'Data not found', ha='center', va='center')
            ax.set_title(f'{action}: {name}')
    
    plt.tight_layout()
    plt.savefig('data/visualizations/3_gesture_comparison.png', dpi=150, bbox_inches='tight')
    print("\n[SAVED] data/visualizations/3_gesture_comparison.png")
    plt.close()


# ============================================================================
# STEP 4: Show Training Data (Spectrogram Images)
# ============================================================================

def show_training_images():
    """Show the actual .png images fed to ResNet"""
    
    print("\n" + "="*70)
    print(" TRAINING DATA: What ResNet Sees")
    print("="*70)
    
    from PIL import Image
    
    # Load some processed spectrogram images
    actions = ['A01', 'A06', 'A10', 'A15', 'A20', 'A27']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Training Images Fed to ResNet (256x256 pixels)', 
                 fontsize=16, fontweight='bold')
    
    for idx, action in enumerate(actions):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        # Find first image for this action
        img_dir = f'data/processed/mmfi_spectrograms/{action}'
        if os.path.exists(img_dir):
            img_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
            if img_files:
                img_path = os.path.join(img_dir, img_files[0])
                img = Image.open(img_path)
                
                ax.imshow(img)
                ax.set_title(f'{action} - {img_files[0]}')
                ax.axis('off')
                
                print(f"  {action}: {img.size} pixels, mode={img.mode}")
            else:
                ax.text(0.5, 0.5, 'No images', ha='center', va='center')
                ax.set_title(action)
                ax.axis('off')
        else:
            ax.text(0.5, 0.5, 'Not processed yet', ha='center', va='center')
            ax.set_title(action)
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('data/visualizations/4_training_images.png', dpi=150, bbox_inches='tight')
    print("\n[SAVED] data/visualizations/4_training_images.png")
    plt.close()


# ============================================================================
# STEP 5: Explain What Model Learns
# ============================================================================

def explain_model_learning():
    """Explain what the model learns from these patterns"""
    
    print("\n" + "="*70)
    print(" WHAT THE MODEL LEARNS")
    print("="*70)
    
    print("""
ResNet looks at these spectrograms and learns to recognize PATTERNS:

1. TEMPORAL PATTERNS (time axis):
   - Fast motions (jump) = rapid changes left-to-right
   - Slow motions (walk) = gradual changes
   - Repeated motions (clap) = periodic patterns

2. FREQUENCY PATTERNS (vertical axis):
   - Large motions (arms) = affect wide frequency range (many rows)
   - Small motions (fingers) = affect narrow frequency range (few rows)
   - Close to WiFi = stronger signal (brighter colors)

3. SPECIFIC GESTURE SIGNATURES:
   - Circle: Curved pattern (smooth transitions)
   - Jump: Sudden vertical spike (all frequencies change at once)
   - Walk: Repetitive horizontal bands (step, step, step...)
   - Clap: Sharp pulse (brief spike then fade)

ResNet learns: "If I see THIS pattern -> It's gesture A06!"
Just like it learned: "If I see THIS pattern -> It's a cat!" (on ImageNet)

TRANSFER LEARNING MAGIC:
- ResNet knows: edges, curves, textures (from ImageNet cats/dogs)
- We teach it: "These edges/curves in WiFi = specific gestures"
- Much faster than learning from scratch!
""")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print(" VISUALIZING WiFi CSI DATA PIPELINE")
    print("="*70)
    print("\nThis will show you exactly what data we're using to train the model!\n")
    
    # Step 1: Raw CSI
    csi_amp = visualize_raw_csi()
    
    # Step 2: Processing
    spectrogram = visualize_processing(csi_amp)
    
    # Step 3: Compare gestures
    compare_gestures()
    
    # Step 4: Training images
    show_training_images()
    
    # Step 5: Explain
    explain_model_learning()
    
    print("\n" + "="*70)
    print(" VISUALIZATION COMPLETE!")
    print("="*70)
    print("\nGenerated visualizations:")
    print("  1. data/visualizations/1_raw_csi.png")
    print("  2. data/visualizations/2_processing_steps.png")
    print("  3. data/visualizations/3_gesture_comparison.png")
    print("  4. data/visualizations/4_training_images.png")
    print("\nOpen these images to see the full data pipeline!")
    print("="*70)


if __name__ == '__main__':
    main()

