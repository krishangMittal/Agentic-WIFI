"""
Stage 2: The "Command Classifier" (Inference)

This module loads the pre-trained ResNet model and performs inference
on RF spectrograms to predict word commands.
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from typing import Tuple, List, Optional

# Add models directory to path
sys.path.append(str(Path(__file__).parent.parent / "models"))

try:
    from custom_resnet import CustomResNet, MultiInResNet
except ImportError:
    print("Error: Could not import models. Make sure models/custom_resnet.py exists.")
    print("Run: cp RVTALL-Preprocess/classification/models.py models/custom_resnet.py")
    sys.exit(1)


class RFCommandClassifier:
    """
    RF Command Classifier using pre-trained ResNet.
    
    This classifier takes RF spectrograms and outputs command predictions
    with confidence scores.
    """
    
    # 15 RVTALL word commands (update based on actual corpus inspection)
    WORD_COMMANDS = [
        "help", "ambulance", "police", "fire", "emergency",
        "stop", "yes", "no", "left", "right",
        "forward", "backward", "up", "down", "home"
    ]
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        num_classes: int = 15,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_pretrained: bool = True
    ):
        """
        Initialize the classifier.
        
        Args:
            model_path: Path to saved model weights (if fine-tuned)
            num_classes: Number of command classes (15 for RVTALL words)
            device: Device to run inference on
            use_pretrained: Use ImageNet pre-trained weights
        """
        self.device = torch.device(device)
        self.num_classes = num_classes
        
        # Initialize model
        self.model = CustomResNet(
            in_channels=3,          # RGB spectrograms
            num_classes=num_classes,
            pre_trained=use_pretrained,
            model='resnet18'
        )
        
        # Load fine-tuned weights if available
        if model_path and Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded fine-tuned weights from {model_path}")
        elif use_pretrained:
            print("Using ImageNet pre-trained ResNet18")
        
        self.model.to(self.device)
        self.model.eval()
        
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocess spectrogram image for inference.
        
        Args:
            image_path: Path to spectrogram image (.png)
            
        Returns:
            Preprocessed tensor ready for model input
        """
        from torchvision import transforms
        
        # Standard ImageNet preprocessing
        preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            # Optional: normalize with ImageNet stats
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225])
        ])
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Transform
        tensor = preprocess(image)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def predict(
        self,
        image_path: str,
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Predict command from spectrogram image.
        
        Args:
            image_path: Path to spectrogram image
            top_k: Return top-k predictions
            
        Returns:
            List of (command, confidence) tuples
        """
        # Preprocess
        input_tensor = self.preprocess_image(image_path)
        input_tensor = input_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = F.softmax(logits, dim=1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)
        
        # Format results
        results = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            command = self.WORD_COMMANDS[idx] if idx < len(self.WORD_COMMANDS) else f"class_{idx}"
            confidence = prob.item()
            results.append((command, confidence))
        
        return results
    
    def predict_batch(
        self,
        image_paths: List[str],
        batch_size: int = 8
    ) -> List[List[Tuple[str, float]]]:
        """
        Predict commands for multiple spectrograms.
        
        Args:
            image_paths: List of paths to spectrogram images
            batch_size: Batch size for inference
            
        Returns:
            List of predictions for each image
        """
        all_results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            
            # Preprocess batch
            batch_tensors = [self.preprocess_image(path) for path in batch_paths]
            batch_tensor = torch.cat(batch_tensors, dim=0).to(self.device)
            
            # Inference
            with torch.no_grad():
                logits = self.model(batch_tensor)
                probabilities = F.softmax(logits, dim=1)
            
            # Format results
            for prob_dist in probabilities:
                top_probs, top_indices = torch.topk(prob_dist, 3)
                results = [
                    (self.WORD_COMMANDS[idx] if idx < len(self.WORD_COMMANDS) else f"class_{idx}",
                     prob.item())
                    for prob, idx in zip(top_probs, top_indices)
                ]
                all_results.append(results)
        
        return all_results


def test_classifier():
    """Test the classifier with a sample spectrogram."""
    print("="*70)
    print("TESTING RF COMMAND CLASSIFIER")
    print("="*70 + "\n")
    
    # Initialize classifier
    print("Initializing classifier with pre-trained ResNet18...")
    classifier = RFCommandClassifier(use_pretrained=True)
    print(f"Device: {classifier.device}\n")
    
    # Check for sample images
    sample_dir = Path("data/images")
    if sample_dir.exists():
        sample_images = list(sample_dir.glob("*.png"))
        
        if sample_images:
            print(f"Found {len(sample_images)} sample images")
            
            # Test on first image
            test_image = sample_images[0]
            print(f"\nTesting on: {test_image.name}")
            
            predictions = classifier.predict(str(test_image), top_k=3)
            
            print("\nTop-3 Predictions:")
            for i, (command, confidence) in enumerate(predictions, 1):
                print(f"  {i}. {command:15s} {confidence*100:5.2f}%")
        else:
            print("No sample images found in data/images/")
            print("Generate spectrograms first using Stage 1 preprocessing.")
    else:
        print("data/images/ directory not found")
        print("Run Stage 1 preprocessing to generate spectrograms.")
    
    print("\n" + "="*70)
    print("Classifier ready for integration with agent!")
    print("="*70)


if __name__ == "__main__":
    test_classifier()

