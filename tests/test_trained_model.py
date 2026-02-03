"""
Test Trained MM-Fi Gesture Model
================================
Load the trained ResNet model and test on sample spectrograms.

Usage:
    python test_trained_model.py
"""

import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
import random

def load_trained_model(model_path='models/resnet_mmfi_trained.pth'):
    """Load the trained ResNet model"""
    print(f"[*] Loading model from: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    action_map = checkpoint['action_map']
    num_classes = len(action_map)
    
    # Create model
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Reverse action map (label -> action name)
    label_to_action = {v: k for k, v in action_map.items()}
    
    print(f"[OK] Model loaded (val_acc: {checkpoint['val_acc']:.2f}%)")
    print(f"[OK] Classes: {num_classes}")
    
    return model, label_to_action


def predict_spectrogram(model, image_path, label_to_action):
    """Predict gesture from spectrogram image"""
    # Image transforms (same as training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = probabilities.max(1)
    
    predicted_label = predicted.item()
    predicted_action = label_to_action[predicted_label]
    confidence_pct = confidence.item() * 100
    
    return predicted_action, confidence_pct


def test_random_samples(model, label_to_action, spectrogram_dir='data/processed/mmfi_spectrograms', num_samples=10):
    """Test on random samples"""
    print(f"\n[*] Testing on {num_samples} random samples...")
    print("="*70)
    
    # Get all spectrogram images
    all_images = []
    for action in os.listdir(spectrogram_dir):
        action_dir = os.path.join(spectrogram_dir, action)
        if not os.path.isdir(action_dir) or action == 'action_mapping.json':
            continue
        
        for img_file in os.listdir(action_dir):
            if img_file.endswith('.png'):
                all_images.append((os.path.join(action_dir, img_file), action))
    
    # Sample random images
    samples = random.sample(all_images, min(num_samples, len(all_images)))
    
    correct = 0
    for img_path, true_action in samples:
        pred_action, confidence = predict_spectrogram(model, img_path, label_to_action)
        
        is_correct = (pred_action == true_action)
        if is_correct:
            correct += 1
        
        status = "[OK]" if is_correct else "[X]"
        print(f"{status} True: {true_action} | Pred: {pred_action} ({confidence:.1f}%)")
    
    accuracy = 100.0 * correct / len(samples)
    print("="*70)
    print(f"Accuracy: {correct}/{len(samples)} = {accuracy:.1f}%")


def test_specific_gesture(model, label_to_action):
    """Interactively test specific gestures"""
    print("\n[*] Interactive Test Mode")
    print("Enter path to spectrogram image (or 'quit' to exit):")
    
    while True:
        img_path = input("\nImage path: ").strip()
        if img_path.lower() == 'quit':
            break
        
        if not os.path.exists(img_path):
            print("[!] File not found")
            continue
        
        try:
            pred_action, confidence = predict_spectrogram(model, img_path, label_to_action)
            print(f"[OK] Prediction: {pred_action} ({confidence:.1f}% confidence)")
        except Exception as e:
            print(f"[!] Error: {e}")


def main():
    print("="*70)
    print(" Test Trained MM-Fi Gesture Model")
    print("="*70)
    
    # Check if model exists
    if not os.path.exists('models/resnet_mmfi_trained.pth'):
        print("\n[!] Error: Trained model not found!")
        print("[!] Please run: python train_mmfi.py --quick")
        return
    
    # Load model
    model, label_to_action = load_trained_model()
    
    # Show action mapping
    print("\nAction Mapping:")
    for label, action in sorted(label_to_action.items()):
        print(f"  {label:2d} -> {action}")
    
    # Test on random samples
    test_random_samples(model, label_to_action, num_samples=20)
    
    # Interactive mode
    test_specific_gesture(model, label_to_action)


if __name__ == '__main__':
    main()

