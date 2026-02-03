"""
Command classifier model for RF sensing.

This module implements the neural network model for classifying commands
from RF sensing spectrograms (e.g., lip reading, gesture recognition).

Key Insight: Converting RF signals to spectrograms (2D images) allows us to
leverage pre-trained vision models (ResNet, CLIP, etc.) which are much more
robust than training custom signal models from scratch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional


class CommandClassifierResNet(nn.Module):
    """
    Command classifier using pre-trained ResNet backbone.
    
    Why pre-trained ResNet?
    - Trained on millions of images (ImageNet)
    - Learned rich visual features (edges, textures, patterns)
    - These features transfer well to spectrograms (2D images)
    - Much more robust than training from scratch
    - Requires less data for fine-tuning
    
    Architecture:
    - Pre-trained ResNet50 (or ResNet18) for feature extraction
    - Custom classifier head for command classification
    """
    
    def __init__(
        self,
        num_classes: int = 10,
        backbone: str = 'resnet50',
        pretrained: bool = True,
        dropout: float = 0.5,
        freeze_backbone: bool = False
    ):
        """
        Initialize the command classifier with pre-trained ResNet.
        
        Args:
            num_classes: Number of command classes to classify
            backbone: ResNet variant ('resnet18', 'resnet50', etc.)
            pretrained: Use ImageNet pre-trained weights
            dropout: Dropout probability in classifier head
            freeze_backbone: Freeze ResNet weights (only train classifier)
        """
        super(CommandClassifierResNet, self).__init__()
        
        # Load pre-trained ResNet
        if backbone == 'resnet18':
            resnet = models.resnet18(pretrained=pretrained)
            feature_dim = 512
        elif backbone == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze backbone if specified (for faster training)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input spectrogram tensor [batch, channels, height, width]
               Note: ResNet expects 3 channels, so grayscale spectrograms
               should be repeated to 3 channels or converted to RGB
            
        Returns:
            Logits for each class
        """
        # Extract features using pre-trained ResNet
        features = self.backbone(x)
        features = features.view(features.size(0), -1)  # Flatten
        
        # Classify using custom head
        logits = self.classifier(features)
        
        return logits


class CommandClassifier(nn.Module):
    """
    Simple CNN classifier (from scratch) - kept for comparison.
    
    Note: Pre-trained models (CommandClassifierResNet) are recommended
    as they are more robust and require less data.
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 10,
        dropout: float = 0.5
    ):
        """
        Initialize the command classifier.
        
        Args:
            input_channels: Number of input channels (1 for grayscale spectrogram)
            num_classes: Number of command classes to classify
            dropout: Dropout probability
        """
        super(CommandClassifier, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        # Note: Input size will depend on spectrogram dimensions
        self.fc1 = nn.Linear(128 * 32 * 32, 512)  # Adjust based on input size
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input spectrogram tensor [batch, channels, height, width]
            
        Returns:
            Logits for each class
        """
        # Convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


def load_model(
    model_path: str,
    model_type: str = 'resnet',
    device: Optional[torch.device] = None
) -> nn.Module:
    """
    Load a trained model from disk.
    
    Args:
        model_path: Path to saved model checkpoint
        model_type: Type of model ('resnet' or 'custom')
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # TODO: Implement model loading with proper checkpoint handling
    if model_type == 'resnet':
        model = CommandClassifierResNet(num_classes=10)
    else:
        model = CommandClassifier(input_channels=1, num_classes=10)
    
    # model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model


def prepare_spectrogram_for_resnet(spectrogram: torch.Tensor) -> torch.Tensor:
    """
    Prepare spectrogram for ResNet input.
    
    ResNet expects:
    - 3 channels (RGB)
    - Shape: [batch, 3, height, width]
    - Values normalized to [0, 1] or ImageNet stats
    
    Args:
        spectrogram: Grayscale spectrogram [batch, 1, height, width] or [batch, height, width]
        
    Returns:
        RGB spectrogram [batch, 3, height, width]
    """
    # Handle different input shapes
    if spectrogram.dim() == 3:
        spectrogram = spectrogram.unsqueeze(1)  # Add channel dimension
    
    # Convert grayscale to RGB by repeating channels
    if spectrogram.shape[1] == 1:
        spectrogram = spectrogram.repeat(1, 3, 1, 1)
    
    # Normalize to [0, 1] if needed
    if spectrogram.max() > 1.0:
        spectrogram = spectrogram / spectrogram.max()
    
    return spectrogram


if __name__ == "__main__":
    # Example usage - compare models
    print("=== Model Comparison ===\n")
    
    # Custom model (from scratch)
    custom_model = CommandClassifier(input_channels=1, num_classes=10)
    custom_params = sum(p.numel() for p in custom_model.parameters())
    print(f"Custom CNN: {custom_params:,} parameters")
    print("  - Trained from scratch")
    print("  - Requires large dataset")
    print("  - Less robust to variations\n")
    
    # Pre-trained ResNet model
    resnet_model = CommandClassifierResNet(num_classes=10, backbone='resnet50')
    resnet_params = sum(p.numel() for p in resnet_model.parameters())
    print(f"Pre-trained ResNet50: {resnet_params:,} parameters")
    print("  - Uses ImageNet pre-trained weights")
    print("  - Requires less data for fine-tuning")
    print("  - More robust and generalizable")
    print("  - Recommended approach!")
