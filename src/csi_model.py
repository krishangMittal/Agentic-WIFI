"""
WiFi CSI Activity Recognition Model

CNN-LSTM with Attention — based on state-of-the-art research:
- CNN extracts spatial features across subcarriers
- LSTM captures temporal motion patterns
- Attention focuses on the most important time steps

References:
- SenseFi benchmark (Chen et al., 2023) — CNN+LSTM best combo
- CSI-DeepNet (2022) — depthwise separable conv + attention
- Motion Pattern Recognition via CNN-LSTM-Attention (2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CSIAttention(nn.Module):
    """Attention layer for weighting important time steps."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, lstm_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lstm_output: (batch, seq_len, hidden_size)

        Returns:
            context: (batch, hidden_size) — weighted sum of time steps
        """
        # Compute attention weights
        weights = self.attention(lstm_output)        # (batch, seq_len, 1)
        weights = F.softmax(weights, dim=1)          # normalize over time

        # Weighted sum
        context = torch.sum(weights * lstm_output, dim=1)  # (batch, hidden_size)
        return context


class CSINet(nn.Module):
    """
    CNN-LSTM-Attention model for WiFi CSI activity recognition.

    Input: (batch, n_features, window_size)
        - n_features: 228 (114 amp + 114 phase) or 114 (amp only)
        - window_size: number of time steps (e.g., 100)

    Output: (batch, n_classes)
    """

    def __init__(
        self,
        n_features: int = 228,
        n_classes: int = 27,
        cnn_channels: list = None,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()

        if cnn_channels is None:
            cnn_channels = [64, 128, 256]

        # CNN feature extractor (operates on subcarrier dimension)
        cnn_layers = []
        in_ch = 1  # treat as single-channel 2D input
        for out_ch in cnn_channels:
            cnn_layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
                nn.Dropout(dropout * 0.5),
            ])
            in_ch = out_ch

        self.cnn = nn.Sequential(*cnn_layers)

        # Calculate CNN output size
        # After 3 MaxPool1d(2): n_features // 8
        cnn_out_features = n_features // 8
        cnn_out_size = cnn_channels[-1] * cnn_out_features

        # Projection from CNN output to LSTM input
        self.cnn_proj = nn.Sequential(
            nn.Linear(cnn_out_size, lstm_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # LSTM temporal model
        self.lstm = nn.LSTM(
            input_size=lstm_hidden,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True
        )

        # Attention over LSTM outputs
        self.attention = CSIAttention(lstm_hidden * 2)  # *2 for bidirectional

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_features, window_size)

        Returns:
            logits: (batch, n_classes)
        """
        batch_size = x.size(0)
        window_size = x.size(2)

        # Process each time step through CNN
        # Reshape: (batch, window_size, n_features) for iteration
        x = x.permute(0, 2, 1)  # (batch, window_size, n_features)

        # Apply CNN to each time step
        # Add channel dim: (batch * window_size, 1, n_features)
        x_flat = x.reshape(batch_size * window_size, 1, -1)
        cnn_out = self.cnn(x_flat)  # (batch * window_size, channels, reduced_features)
        cnn_out = cnn_out.reshape(batch_size * window_size, -1)  # flatten

        # Project to LSTM size
        cnn_out = self.cnn_proj(cnn_out)  # (batch * window_size, lstm_hidden)
        cnn_out = cnn_out.reshape(batch_size, window_size, -1)  # (batch, window_size, lstm_hidden)

        # LSTM
        lstm_out, _ = self.lstm(cnn_out)  # (batch, window_size, lstm_hidden * 2)

        # Attention
        context = self.attention(lstm_out)  # (batch, lstm_hidden * 2)

        # Classify
        logits = self.classifier(context)  # (batch, n_classes)

        return logits


class CSINetLite(nn.Module):
    """
    Lightweight version — just CNN + FC.
    Faster to train, good baseline.

    Input: (batch, n_features, window_size)
    Output: (batch, n_classes)
    """

    def __init__(
        self,
        n_features: int = 228,
        window_size: int = 100,
        n_classes: int = 27,
        dropout: float = 0.3
    ):
        super().__init__()

        # 2D CNN: treat (n_features, window_size) as a 2D image with 1 channel
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_features, window_size)

        Returns:
            logits: (batch, n_classes)
        """
        x = x.unsqueeze(1)  # (batch, 1, n_features, window_size)
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    # Test both models
    batch = 4
    n_features = 228
    window_size = 100
    n_classes = 27

    x = torch.randn(batch, n_features, window_size)

    # Test lite model
    print('=== CSINetLite ===')
    model_lite = CSINetLite(n_features, window_size, n_classes)
    out = model_lite(x)
    print(f'Input:  {x.shape}')
    print(f'Output: {out.shape}')
    params = sum(p.numel() for p in model_lite.parameters())
    print(f'Parameters: {params:,}')

    # Test full model
    print('\n=== CSINet (CNN-LSTM-Attention) ===')
    model = CSINet(n_features, n_classes)
    out = model(x)
    print(f'Input:  {x.shape}')
    print(f'Output: {out.shape}')
    params = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {params:,}')
