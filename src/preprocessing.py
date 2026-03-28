"""
WiFi CSI Preprocessing Pipeline

Based on research best practices:
- Butterworth low-pass filter (noise removal)
- PCA denoising (background removal)
- Sliding window segmentation
- Normalization

References:
- SenseFi benchmark (Chen et al., 2023)
- CSI-DeepNet (2022)
- Optimal CSI preprocessing (arxiv 2307.12126)
"""

import numpy as np
import scipy.io as scio
from scipy import signal as sig
from pathlib import Path
from typing import Tuple, List, Optional


def butterworth_filter(data: np.ndarray, cutoff: float = 30.0, fs: float = 100.0, order: int = 4) -> np.ndarray:
    """
    Apply Butterworth low-pass filter to remove high-frequency noise.

    Args:
        data: Input signal (n_features, n_timesteps)
        cutoff: Cutoff frequency in Hz
        fs: Sampling frequency in Hz
        order: Filter order

    Returns:
        Filtered signal
    """
    nyquist = fs / 2.0
    if cutoff >= nyquist:
        cutoff = nyquist - 1.0

    b, a = sig.butter(order, cutoff / nyquist, btype='low')

    # Apply along time axis
    if data.ndim == 1:
        return sig.filtfilt(b, a, data)
    else:
        filtered = np.zeros_like(data)
        for i in range(data.shape[0]):
            filtered[i] = sig.filtfilt(b, a, data[i])
        return filtered


def hampel_filter(data: np.ndarray, window_size: int = 5, threshold: float = 3.0) -> np.ndarray:
    """
    Hampel filter to remove outlier spikes.

    Args:
        data: Input signal (n_features, n_timesteps)
        window_size: Half-window size
        threshold: Number of MADs to consider outlier

    Returns:
        Filtered signal with outliers replaced by median
    """
    result = data.copy()

    if data.ndim == 1:
        data = data.reshape(1, -1)
        result = result.reshape(1, -1)

    for feat in range(data.shape[0]):
        for i in range(window_size, data.shape[1] - window_size):
            window = data[feat, i - window_size:i + window_size + 1]
            median = np.median(window)
            mad = 1.4826 * np.median(np.abs(window - median))
            if mad > 0 and np.abs(data[feat, i] - median) / mad > threshold:
                result[feat, i] = median

    return result.squeeze()


def pca_denoise(data: np.ndarray, n_components: int = 5, skip_first: bool = True) -> np.ndarray:
    """
    PCA-based denoising. Removes static background (1st component)
    and keeps motion-related components.

    Args:
        data: Input signal (n_features, n_timesteps)
        n_components: Number of PCA components to keep
        skip_first: Skip first component (static/LoS)

    Returns:
        Denoised signal
    """
    # Center the data
    mean = data.mean(axis=1, keepdims=True)
    centered = data - mean

    # SVD
    try:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return data  # Return original if SVD fails

    # Select components (skip first = background)
    start = 1 if skip_first else 0
    end = min(start + n_components, len(S))

    # Reconstruct
    reconstructed = U[:, start:end] @ np.diag(S[start:end]) @ Vt[start:end, :]

    return reconstructed


def normalize(data: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]."""
    dmin = data.min()
    dmax = data.max()
    if dmax - dmin < 1e-8:
        return np.zeros_like(data)
    return (data - dmin) / (dmax - dmin)


def load_activity_frames(csi_dir: str, use_phase: bool = True) -> Optional[np.ndarray]:
    """
    Load all frames for one activity and stack into time series.

    Args:
        csi_dir: Path to wifi-csi folder
        use_phase: Include phase data alongside amplitude

    Returns:
        Array of shape (n_features, n_timesteps) or None
        If use_phase: n_features = 114 * 2 = 228 (amp + phase, averaged across antennas)
        If not: n_features = 114
    """
    csi_path = Path(csi_dir)
    frames = sorted(csi_path.glob('frame*.mat'))

    if len(frames) < 10:
        return None

    all_data = []
    for f in frames:
        try:
            mat = scio.loadmat(str(f))
            # Average across antennas (axis=0) and samples (axis=2)
            amp = mat['CSIamp'].mean(axis=0).mean(axis=1)  # (114,)

            if use_phase:
                phase = mat['CSIphase'].mean(axis=0).mean(axis=1)  # (114,)
                frame_data = np.concatenate([amp, phase])  # (228,)
            else:
                frame_data = amp  # (114,)

            all_data.append(frame_data)
        except Exception:
            continue

    if len(all_data) < 10:
        return None

    # Stack: (n_timesteps, n_features) -> transpose to (n_features, n_timesteps)
    return np.array(all_data).T


def preprocess_activity(csi_dir: str, use_phase: bool = True, apply_pca: bool = True) -> Optional[np.ndarray]:
    """
    Full preprocessing pipeline for one activity.

    Args:
        csi_dir: Path to wifi-csi folder
        use_phase: Include phase data
        apply_pca: Apply PCA denoising

    Returns:
        Preprocessed array (n_features, n_timesteps) or None
    """
    # Load raw data
    data = load_activity_frames(csi_dir, use_phase=use_phase)
    if data is None:
        return None

    # Step 1: Hampel filter (remove outlier spikes)
    data = hampel_filter(data, window_size=5, threshold=3.0)

    # Step 2: Butterworth low-pass filter (remove high-freq noise)
    data = butterworth_filter(data, cutoff=30.0, fs=100.0, order=4)

    # Step 3: PCA denoise (remove static background)
    if apply_pca and data.shape[1] > 20:
        data = pca_denoise(data, n_components=5, skip_first=True)

    # Step 4: Replace any NaN/Inf with 0
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    # Step 5: Normalize
    data = normalize(data)

    return data


def segment_windows(data: np.ndarray, window_size: int = 100, stride: int = 50) -> List[np.ndarray]:
    """
    Segment time series into overlapping windows.

    Args:
        data: (n_features, n_timesteps)
        window_size: Window length in frames
        stride: Step between windows

    Returns:
        List of (n_features, window_size) arrays
    """
    n_timesteps = data.shape[1]
    windows = []

    for start in range(0, n_timesteps - window_size + 1, stride):
        window = data[:, start:start + window_size]
        windows.append(window)

    # If we got no windows (data too short), pad and use whole thing
    if not windows and n_timesteps > 0:
        padded = np.zeros((data.shape[0], window_size))
        padded[:, :n_timesteps] = data
        windows.append(padded)

    return windows


def parse_esp32_csi(line: str) -> Optional[np.ndarray]:
    """
    Parse live ESP32 CSI data line into amplitude + phase.

    ESP32 outputs raw I/Q pairs: [i0 q0 i1 q1 ...]
    We convert to amplitude and phase per subcarrier.

    Args:
        line: Raw CSI line from ESP32 serial

    Returns:
        Array of (n_features,) — amplitudes and phases concatenated
    """
    try:
        # Extract the data array from the CSV line
        if 'CSI_DATA' not in line:
            return None

        # Find the bracket-enclosed data
        bracket_start = line.index('[')
        bracket_end = line.index(']')
        data_str = line[bracket_start + 1:bracket_end].strip()

        values = [int(x) for x in data_str.split() if x.lstrip('-').isdigit()]

        if len(values) < 4:
            return None

        # Convert I/Q pairs to amplitude and phase
        amplitudes = []
        phases = []
        for i in range(0, len(values) - 1, 2):
            real = values[i]
            imag = values[i + 1]
            amp = np.sqrt(real ** 2 + imag ** 2)
            phase = np.arctan2(imag, real)
            amplitudes.append(amp)
            phases.append(phase)

        return np.concatenate([amplitudes, phases]).astype(np.float32)

    except (ValueError, IndexError):
        return None


def build_dataset(
    data_root: str = 'data/raw/E01/E01',
    window_size: int = 100,
    stride: int = 50,
    use_phase: bool = True,
    apply_pca: bool = True,
    max_subjects: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Build full dataset from MM-Fi data.

    Args:
        data_root: Path to E01 data
        window_size: Sliding window size
        stride: Window stride
        use_phase: Include phase features
        apply_pca: Apply PCA denoising
        max_subjects: Limit number of subjects (for quick testing)

    Returns:
        X: (n_samples, n_features, window_size)
        y: (n_samples,) activity labels 0-26
        info: dict with metadata
    """
    base = Path(data_root)
    subjects = sorted([d.name for d in base.iterdir() if d.is_dir() and d.name.startswith('S')])

    if max_subjects:
        subjects = subjects[:max_subjects]

    activities = [f'A{i:02d}' for i in range(1, 28)]

    all_X = []
    all_y = []

    for subj in subjects:
        for act_idx, act_code in enumerate(activities):
            csi_dir = base / subj / act_code / 'wifi-csi'
            if not csi_dir.exists():
                continue

            # Preprocess
            data = preprocess_activity(str(csi_dir), use_phase=use_phase, apply_pca=apply_pca)
            if data is None:
                continue

            # Segment into windows
            windows = segment_windows(data, window_size=window_size, stride=stride)

            for w in windows:
                all_X.append(w)
                all_y.append(act_idx)

        print(f'  Processed {subj}: {len(all_X)} total windows so far')

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.int64)

    activity_names = {
        0: 'Stretching', 1: 'Chest_exp_H', 2: 'Chest_exp_V',
        3: 'Twist_left', 4: 'Twist_right', 5: 'Mark_time',
        6: 'Limb_ext_L', 7: 'Limb_ext_R', 8: 'Lunge_LF',
        9: 'Lunge_RF', 10: 'Limb_ext_both', 11: 'Squat',
        12: 'Raise_hand_L', 13: 'Raise_hand_R', 14: 'Lunge_LS',
        15: 'Lunge_RS', 16: 'Wave_left', 17: 'Wave_right',
        18: 'Picking_up', 19: 'Throw_left', 20: 'Throw_right',
        21: 'Kick_left', 22: 'Kick_right', 23: 'Body_ext_L',
        24: 'Body_ext_R', 25: 'Jumping', 26: 'Bowing'
    }

    info = {
        'n_subjects': len(subjects),
        'n_activities': len(activities),
        'window_size': window_size,
        'stride': stride,
        'n_features': X.shape[1] if len(X) > 0 else 0,
        'use_phase': use_phase,
        'activity_names': activity_names
    }

    return X, y, info


if __name__ == '__main__':
    print('Building dataset...')
    X, y, info = build_dataset(max_subjects=2)
    print(f'\nDataset built:')
    print(f'  X shape: {X.shape}')
    print(f'  y shape: {y.shape}')
    print(f'  Classes: {len(np.unique(y))}')
    print(f'  Info: {info}')
