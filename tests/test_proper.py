"""
Proper Test Script - Uses Separate Test Set
============================================
This script tests on data NOT used during training.

We'll use S02-S10 data for testing (model only trained on S01).
"""

import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
import glob

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
    
    print(f"[OK] Model loaded (val_acc: {checkpoint['val_acc']:.2f}%)")
    print(f"[OK] Classes: {num_classes}")
    
    return model, label_to_action


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
        confidence, predicted = probabilities.max(1)
    
    predicted_label = predicted.item()
    predicted_action = label_to_action[predicted_label]
    confidence_pct = confidence.item() * 100
    
    return predicted_action, confidence_pct


def test_on_unseen_subjects():
    """Test on subjects NOT used in training (S02-S10)"""
    print("="*70)
    print(" PROPER TEST: Using Subjects S02-S10 (NOT SEEN DURING TRAINING)")
    print("="*70)
    
    # Load model (trained on S01 only)
    model, label_to_action = load_trained_model()
    
    # Check if S02+ data exists
    spectrogram_dir = 'data/processed/mmfi_spectrograms'
    
    # Find images from S02, S03, etc. (not S01)
    test_images = []
    for action in os.listdir(spectrogram_dir):
        action_dir = os.path.join(spectrogram_dir, action)
        if not os.path.isdir(action_dir):
            continue
        
        for img_file in os.listdir(action_dir):
            # Only include S02-S10 images, exclude S01
            if img_file.startswith('S0') and not img_file.startswith('S01'):
                img_path = os.path.join(action_dir, img_file)
                test_images.append((img_path, action))
    
    if len(test_images) == 0:
        print("\n[!] No test data found!")
        print("[!] The model was only trained on S01 (quick test mode)")
        print("[!] To properly test, run: python train_mmfi.py (full training)")
        print("\n[*] For now, let's test on VALIDATION subset from S01...")
        print("[*] Note: This is still somewhat valid since validation data")
        print("[*] was held out during training.\n")
        return test_validation_subset()
    
    print(f"\n[*] Found {len(test_images)} test images from unseen subjects")
    print(f"[*] Testing on random 50 samples...\n")
    
    import random
    samples = random.sample(test_images, min(50, len(test_images)))
    
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
    print(f"PROPER TEST Accuracy: {correct}/{len(samples)} = {accuracy:.1f}%")
    print("="*70)


def test_validation_subset():
    """
    Estimate validation performance by testing ONLY on later samples
    (assumes model saw first 8 samples per action, test on last 2)
    """
    print("\n[*] Testing on estimated validation subset (last 20% of S01 data)...\n")
    
    model, label_to_action = load_trained_model()
    
    spectrogram_dir = 'data/processed/mmfi_spectrograms'
    test_images = []
    
    for action in os.listdir(spectrogram_dir):
        action_dir = os.path.join(spectrogram_dir, action)
        if not os.path.isdir(action_dir):
            continue
        
        # Get S01 images only
        s01_images = sorted([
            os.path.join(action_dir, f) 
            for f in os.listdir(action_dir) 
            if f.startswith('S01') and f.endswith('.png')
        ])
        
        # Take last 20% as "validation" (frames 9-10 out of 10)
        val_count = max(1, len(s01_images) // 5)
        val_images = s01_images[-val_count:]
        
        for img_path in val_images:
            test_images.append((img_path, action))
    
    print(f"[*] Testing on {len(test_images)} estimated validation images\n")
    
    correct = 0
    for img_path, true_action in test_images:
        pred_action, confidence = predict_spectrogram(model, img_path, label_to_action)
        is_correct = (pred_action == true_action)
        if is_correct:
            correct += 1
        
        status = "[OK]" if is_correct else "[X]"
        print(f"{status} True: {true_action} | Pred: {pred_action} ({confidence:.1f}%)")
    
    accuracy = 100.0 * correct / len(test_images)
    print("="*70)
    print(f"Estimated Validation Accuracy: {correct}/{len(test_images)} = {accuracy:.1f}%")
    print("="*70)
    print("\nNote: This is an estimate. For true validation, run full training")
    print("with proper train/test split on multiple subjects.")


if __name__ == '__main__':
    test_on_unseen_subjects()

