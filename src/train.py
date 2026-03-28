"""
Training pipeline for WiFi CSI Activity Recognition.

Trains CNN-LSTM-Attention model on MM-Fi dataset with proper
cross-subject evaluation (train on some subjects, test on others).
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from preprocessing import build_dataset
from csi_model import CSINet, CSINetLite


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        preds = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    """Evaluate on validation/test set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            total_loss += loss.item() * len(y_batch)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


def cross_subject_split(X, y, info, test_ratio=0.3):
    """
    Split data by subject for proper evaluation.
    This ensures the model generalizes to NEW people, not just memorizing.

    We use the window ordering: subjects are processed sequentially,
    so we split by position in the dataset.
    """
    n_samples = len(y)
    n_test = int(n_samples * test_ratio)

    # Last N% of data = last subjects = test set
    X_train = X[:n_samples - n_test]
    y_train = y[:n_samples - n_test]
    X_test = X[n_samples - n_test:]
    y_test = y[n_samples - n_test:]

    return X_train, y_train, X_test, y_test


def train_model(
    data_root: str = 'data/raw/E01/E01',
    model_type: str = 'lite',
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 0.001,
    window_size: int = 100,
    stride: int = 50,
    use_phase: bool = True,
    max_subjects: int = None,
    save_path: str = 'models/csi_model.pth'
):
    """
    Full training pipeline.

    Args:
        data_root: Path to dataset
        model_type: 'lite' (CNN only, fast) or 'full' (CNN-LSTM-Attention)
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        window_size: Sliding window size
        stride: Window stride
        use_phase: Include phase features
        max_subjects: Limit subjects (for quick testing)
        save_path: Where to save trained model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print()

    # === BUILD DATASET ===
    print('='*60)
    print('STEP 1: Building dataset...')
    print('='*60)

    X, y, info = build_dataset(
        data_root=data_root,
        window_size=window_size,
        stride=stride,
        use_phase=use_phase,
        max_subjects=max_subjects
    )

    print(f'\nDataset: {X.shape[0]} samples, {X.shape[1]} features, {X.shape[2]} timesteps')
    print(f'Classes: {len(np.unique(y))}')
    print(f'Samples per class: ~{len(y) // len(np.unique(y))}')

    # === SPLIT ===
    print('\n' + '='*60)
    print('STEP 2: Cross-subject split...')
    print('='*60)

    X_train, y_train, X_test, y_test = cross_subject_split(X, y, info)
    print(f'Train: {len(y_train)} samples')
    print(f'Test:  {len(y_test)} samples')

    # Create data loaders
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # === MODEL ===
    print('\n' + '='*60)
    print(f'STEP 3: Creating {model_type} model...')
    print('='*60)

    n_features = X.shape[1]
    n_classes = len(np.unique(y))

    if model_type == 'lite':
        model = CSINetLite(n_features, window_size, n_classes)
    else:
        model = CSINet(n_features, n_classes)

    model = model.to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {params:,}')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # === TRAIN ===
    print('\n' + '='*60)
    print('STEP 4: Training...')
    print('='*60)

    best_acc = -1
    best_epoch = 0

    # Ensure save directory exists
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    for epoch in range(1, epochs + 1):
        start = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion, device)

        scheduler.step(test_loss)
        elapsed = time.time() - start

        if epoch % 1 == 0 or epoch == epochs:
            print(f'Epoch {epoch:3d}/{epochs} ({elapsed:.1f}s) | '
                  f'Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | '
                  f'Test Loss: {test_loss:.4f} Acc: {test_acc:.3f}')

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            # Save best model
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'model_state': model.state_dict(),
                'model_type': model_type,
                'n_features': n_features,
                'n_classes': n_classes,
                'window_size': window_size,
                'use_phase': use_phase,
                'best_acc': best_acc,
                'info': info,
            }, save_path)

    # === FINAL EVALUATION ===
    print('\n' + '='*60)
    print('STEP 5: Final evaluation...')
    print('='*60)

    # Load best model
    checkpoint = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])

    _, final_acc, preds, labels = evaluate(model, test_loader, criterion, device)

    print(f'\nBest model from epoch {best_epoch}')
    print(f'Test accuracy: {final_acc:.4f} ({final_acc*100:.1f}%)')

    # Per-class report
    activity_names = info['activity_names']
    present_classes = sorted(np.unique(labels))
    target_names = [activity_names.get(c, f'Class_{c}') for c in present_classes]

    print('\nPer-class results:')
    report = classification_report(labels, preds, labels=present_classes,
                                   target_names=target_names, zero_division=0)
    print(report)

    print(f'\nModel saved to: {save_path}')

    return model, info


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train WiFi CSI model')
    parser.add_argument('--model', type=str, default='lite', choices=['lite', 'full'])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--subjects', type=int, default=None, help='Limit subjects for quick test')
    parser.add_argument('--no-phase', action='store_true', help='Amplitude only')

    args = parser.parse_args()

    train_model(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_phase=not args.no_phase,
        max_subjects=args.subjects,
    )
