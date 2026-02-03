"""
Synthetic Spectrogram Test
===========================
Generate artificial WiFi spectrograms to test if the model
learned REAL patterns vs just memorizing training data.

If the model can classify synthetic patterns, it proves it
learned the underlying WiFi physics!
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

def load_trained_model(model_path='models/resnet_mmfi_trained.pth'):
    """Load the trained ResNet model"""
    print(f"[*] Loading model from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu')
    action_map = checkpoint['action_map']
    num_classes = len(action_map)
    
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    label_to_action = {v: k for k, v in action_map.items()}
    
    print(f"[OK] Model loaded")
    return model, label_to_action


def generate_synthetic_spectrogram(pattern_type='wave', size=(114, 10), noise_level=0.1):
    """
    Generate synthetic WiFi CSI spectrogram
    
    Pattern types:
    - 'wave': Sinusoidal pattern (like hand waving)
    - 'rise': Sharp rise (like raising hand)
    - 'fall': Sharp fall (like falling)
    - 'flat': No movement (background)
    - 'pulse': Sharp pulse (like clap)
    - 'sweep': Linear sweep (like sweeping gesture)
    """
    freq_bins, time_samples = size
    spectrogram = np.zeros((freq_bins, time_samples))
    
    if pattern_type == 'wave':
        # Sinusoidal pattern across time
        for t in range(time_samples):
            wave = np.sin(2 * np.pi * t / time_samples * 3)  # 3 cycles
            # Apply wave to middle frequencies
            center = freq_bins // 2
            width = freq_bins // 4
            spectrogram[center-width:center+width, t] = wave
    
    elif pattern_type == 'rise':
        # Sharp rise pattern (energy increases over time)
        for t in range(time_samples):
            intensity = t / time_samples  # Linear rise
            # Higher frequencies light up as gesture rises
            active_freqs = int(intensity * freq_bins)
            spectrogram[:active_freqs, t] = intensity
    
    elif pattern_type == 'fall':
        # Sharp fall pattern (energy decreases)
        for t in range(time_samples):
            intensity = 1.0 - (t / time_samples)  # Linear fall
            active_freqs = int(intensity * freq_bins)
            spectrogram[:active_freqs, t] = intensity
    
    elif pattern_type == 'flat':
        # Flat background (no movement)
        spectrogram[:, :] = 0.3  # Low constant energy
    
    elif pattern_type == 'pulse':
        # Sharp pulse in middle (like clap)
        center_time = time_samples // 2
        spectrogram[:, center_time] = 1.0
        # Decay around pulse
        for offset in range(1, 3):
            if center_time - offset >= 0:
                spectrogram[:, center_time - offset] = 0.5 / offset
            if center_time + offset < time_samples:
                spectrogram[:, center_time + offset] = 0.5 / offset
    
    elif pattern_type == 'sweep':
        # Linear sweep across frequencies
        for t in range(time_samples):
            freq_pos = int((t / time_samples) * freq_bins)
            # Activate frequency band that sweeps upward
            if freq_pos < freq_bins:
                spectrogram[freq_pos, t] = 1.0
                # Add some width
                for offset in range(-5, 6):
                    f = freq_pos + offset
                    if 0 <= f < freq_bins:
                        spectrogram[f, t] = 0.5
    
    # Add noise
    noise = np.random.randn(*spectrogram.shape) * noise_level
    spectrogram += noise
    
    # Normalize to [0, 1]
    spectrogram = np.clip(spectrogram, 0, 1)
    
    # Convert to log scale (like real spectrograms)
    spectrogram = 10 * np.log10(spectrogram + 1e-10)
    
    return spectrogram


def save_spectrogram_image(spectrogram, output_path):
    """Save spectrogram as image"""
    plt.figure(figsize=(2.56, 2.56), dpi=100)
    plt.imshow(spectrogram, aspect='auto', cmap='jet', origin='lower')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def predict_spectrogram(model, image_path, label_to_action):
    """Predict gesture from spectrogram image"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        
        # Get top 3 predictions
        top_probs, top_indices = probabilities[0].topk(3)
        
        predictions = []
        for prob, idx in zip(top_probs, top_indices):
            action = label_to_action[idx.item()]
            confidence = prob.item() * 100
            predictions.append((action, confidence))
    
    return predictions


def test_synthetic_patterns():
    """Generate and test synthetic patterns"""
    print("="*70)
    print(" SYNTHETIC PATTERN TEST")
    print("="*70)
    print("\nGenerating artificial WiFi spectrograms to test if model")
    print("learned REAL patterns vs memorizing training data...\n")
    
    # Load model
    model, label_to_action = load_trained_model()
    
    # Create output directory
    os.makedirs('data/synthetic_test', exist_ok=True)
    
    # Test different synthetic patterns
    patterns = {
        'wave': 'Waving motion (sinusoidal)',
        'rise': 'Rising motion (like raising hand)',
        'fall': 'Falling motion',
        'flat': 'No movement (background)',
        'pulse': 'Sharp pulse (like clap)',
        'sweep': 'Sweeping motion (frequency sweep)'
    }
    
    print("[*] Generating synthetic spectrograms...\n")
    
    for pattern_type, description in patterns.items():
        # Generate spectrogram
        spectrogram = generate_synthetic_spectrogram(pattern_type)
        
        # Save as image
        img_path = f'data/synthetic_test/synthetic_{pattern_type}.png'
        save_spectrogram_image(spectrogram, img_path)
        
        # Predict
        predictions = predict_spectrogram(model, img_path, label_to_action)
        
        # Display results
        print(f"Pattern: {pattern_type.upper()} ({description})")
        print(f"  Saved: {img_path}")
        print(f"  Top predictions:")
        for i, (action, conf) in enumerate(predictions, 1):
            print(f"    {i}. {action}: {conf:.1f}%")
        print()
    
    print("="*70)
    print(" ANALYSIS")
    print("="*70)
    print("\nHow to interpret results:")
    print("  - If model gives CONSISTENT predictions for each pattern type:")
    print("    -> Model learned meaningful WiFi features! (GOOD)")
    print("\n  - If model gives RANDOM predictions:")
    print("    -> Model just memorized training data (BAD)")
    print("\n  - If wave/pulse/sweep get different predictions:")
    print("    -> Model can distinguish motion types! (EXCELLENT)")
    print("\nVisual inspection:")
    print(f"  Check images in: data/synthetic_test/")
    print("  Compare to real spectrograms in: data/processed/mmfi_spectrograms/")
    print("="*70)


def test_noise_robustness():
    """Test how model handles increasing noise levels"""
    print("\n" + "="*70)
    print(" NOISE ROBUSTNESS TEST")
    print("="*70)
    print("\nTesting if model can handle noisy WiFi signals...\n")
    
    model, label_to_action = load_trained_model()
    
    # Test wave pattern with increasing noise
    noise_levels = [0.0, 0.1, 0.3, 0.5, 0.8, 1.0]
    
    for noise in noise_levels:
        spectrogram = generate_synthetic_spectrogram('wave', noise_level=noise)
        img_path = f'data/synthetic_test/wave_noise_{noise:.1f}.png'
        save_spectrogram_image(spectrogram, img_path)
        
        predictions = predict_spectrogram(model, img_path, label_to_action)
        top_action, top_conf = predictions[0]
        
        print(f"Noise level: {noise:.1f} -> Prediction: {top_action} ({top_conf:.1f}%)")
    
    print("\nIf confidence drops smoothly with noise, model is robust!")
    print("="*70)


def test_interpolation():
    """Test if model can handle interpolated patterns"""
    print("\n" + "="*70)
    print(" INTERPOLATION TEST")
    print("="*70)
    print("\nTesting morphing from one pattern to another...\n")
    
    model, label_to_action = load_trained_model()
    
    # Morph from wave to rise
    print("[*] Morphing: Wave -> Rise")
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        # Generate both patterns
        wave = generate_synthetic_spectrogram('wave', noise_level=0.0)
        rise = generate_synthetic_spectrogram('rise', noise_level=0.0)
        
        # Interpolate
        morphed = (1 - alpha) * wave + alpha * rise
        
        img_path = f'data/synthetic_test/morph_wave_rise_{alpha:.2f}.png'
        save_spectrogram_image(morphed, img_path)
        
        predictions = predict_spectrogram(model, img_path, label_to_action)
        top_action, top_conf = predictions[0]
        
        print(f"  Alpha {alpha:.2f} (wave={1-alpha:.2f}, rise={alpha:.2f}) -> {top_action} ({top_conf:.1f}%)")
    
    print("\nIf predictions transition smoothly, model learned continuous features!")
    print("="*70)


if __name__ == '__main__':
    # Run all tests
    test_synthetic_patterns()
    test_noise_robustness()
    test_interpolation()
    
    print("\n" + "="*70)
    print(" SYNTHETIC TEST COMPLETE!")
    print("="*70)
    print("\nCheck generated images in: data/synthetic_test/")
    print("\nThis test shows if your model learned REAL WiFi patterns")
    print("or just memorized the training data.")
    print("="*70)

