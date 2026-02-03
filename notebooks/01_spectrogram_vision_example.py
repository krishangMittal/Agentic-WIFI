"""
Example: Why Spectrograms Enable Pre-trained Vision Models

This notebook demonstrates the key insight that converting RF signals to
spectrograms allows using pre-trained vision models.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Simulate RF signal (e.g., from lip movement)
def generate_rf_signal(duration=2.0, sample_rate=1000):
    """Generate a simulated RF signal with temporal patterns."""
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Simulate different frequency components over time (like lip movements)
    signal_data = np.zeros_like(t)
    
    # Add time-varying frequency components
    for i, time_point in enumerate(t):
        if 0.3 < time_point < 0.7:
            # "Speaking" phase - multiple frequency components
            signal_data[i] = np.sin(2 * np.pi * 50 * time_point) + \
                           0.5 * np.sin(2 * np.pi * 100 * time_point) + \
                           0.3 * np.sin(2 * np.pi * 150 * time_point)
        elif 0.8 < time_point < 1.2:
            # Different "word" - different frequency pattern
            signal_data[i] = np.sin(2 * np.pi * 75 * time_point) + \
                           0.6 * np.sin(2 * np.pi * 125 * time_point)
        else:
            # Silence or background
            signal_data[i] = 0.1 * np.random.randn()
    
    # Add noise
    signal_data += 0.05 * np.random.randn(len(signal_data))
    
    return t, signal_data


def demonstrate_spectrogram_approach():
    """
    Demonstrate why spectrograms work with vision models.
    
    Key points:
    1. Raw RF signal is 1D time series - hard for vision models
    2. Spectrogram is 2D image - perfect for vision models
    3. Vision models see patterns, textures, spatial relationships
    """
    
    # Generate example RF signal
    t, rf_signal = generate_rf_signal()
    
    # Convert to spectrogram
    frequencies, time_segments, spectrogram = signal.stft(
        rf_signal,
        fs=1000,
        nperseg=256,
        noverlap=128,
        window='hann'
    )
    spectrogram_mag = np.abs(spectrogram)
    
    # Visualize
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Raw RF signal (1D)
    axes[0].plot(t, rf_signal)
    axes[0].set_title('Raw RF Signal (1D Time Series)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    axes[0].text(0.02, 0.95, 
                'Problem: Vision models need 2D images!\n'
                'This is just a 1D signal - no spatial patterns visible.',
                transform=axes[0].transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Spectrogram (2D image)
    im = axes[1].pcolormesh(time_segments, frequencies, spectrogram_mag, 
                            shading='gouraud', cmap='viridis')
    axes[1].set_title('Spectrogram (2D Image) - Ready for Vision Models!', 
                     fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Frequency (Hz)')
    axes[1].set_ylim([0, 200])  # Focus on lower frequencies
    plt.colorbar(im, ax=axes[1], label='Magnitude')
    axes[1].text(0.02, 0.95,
                'Solution: Spectrogram is a 2D image!\n'
                'Vision models can see:\n'
                '• Patterns (frequency bands)\n'
                '• Textures (signal characteristics)\n'
                '• Spatial relationships (time-frequency)\n'
                '• Temporal evolution (vertical patterns)',
                transform=axes[1].transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('spectrogram_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to 'spectrogram_comparison.png'")
    
    return rf_signal, spectrogram_mag


def explain_vision_model_advantages():
    """
    Explain why pre-trained vision models are better.
    """
    print("\n" + "="*70)
    print("WHY PRE-TRAINED VISION MODELS ARE BETTER")
    print("="*70)
    
    print("\n1. TRANSFER LEARNING:")
    print("   • ResNet/CLIP trained on MILLIONS of images (ImageNet, etc.)")
    print("   • Learned universal features: edges, textures, patterns")
    print("   • These features work on spectrograms too!")
    
    print("\n2. ROBUSTNESS:")
    print("   • Pre-trained models are robust to:")
    print("     - Noise variations")
    print("     - Scale changes")
    print("     - Translation/shifts")
    print("     - Different lighting conditions (in images)")
    print("     - → Translates to robustness in RF signals!")
    
    print("\n3. DATA EFFICIENCY:")
    print("   • Custom model: Need 10,000+ RF samples")
    print("   • Pre-trained model: Need 100-1000 samples (fine-tuning)")
    print("   • 10-100x less data required!")
    
    print("\n4. FEATURE QUALITY:")
    print("   • Custom model: Learns from scratch, limited by dataset")
    print("   • Pre-trained: Rich hierarchical features from billions of images")
    print("   • Better generalization to new scenarios")
    
    print("\n5. DEVELOPMENT SPEED:")
    print("   • Custom model: Weeks of training")
    print("   • Pre-trained: Hours of fine-tuning")
    print("   • Faster iteration and experimentation")
    
    print("\n" + "="*70)
    print("\nCONCLUSION:")
    print("Spectrograms = 2D images → Vision models can understand them!")
    print("Pre-trained vision models = Better, faster, more robust")
    print("="*70 + "\n")


if __name__ == "__main__":
    print("Demonstrating why spectrograms enable pre-trained vision models...\n")
    
    # Generate example
    rf_signal, spectrogram = demonstrate_spectrogram_approach()
    
    # Explain advantages
    explain_vision_model_advantages()
    
    print("\nNext steps:")
    print("1. Use preprocess.py to convert your RF data to spectrograms")
    print("2. Use model.py with CommandClassifierResNet for classification")
    print("3. Fine-tune on your specific RF sensing dataset")

