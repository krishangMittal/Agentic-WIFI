"""
MM-Fi WiFi CSI Processor
Converts WiFi CSI data to spectrograms for the "Siri for WiFi" system
"""

import os
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path
from typing import Tuple, Optional


class MMFiProcessor:
    """Process MM-Fi WiFi CSI data into spectrograms."""
    
    def __init__(self, mmfi_root: str = 'data/raw/MMFi'):
        """
        Initialize MM-Fi processor.
        
        Args:
            mmfi_root: Root directory of MM-Fi dataset
        """
        self.mmfi_root = Path(mmfi_root)
        self.output_dir = Path('data/mmfi_spectrograms')
        self.image_dir = Path('data/mmfi_images')
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        
        # Activity mapping (27 activities)
        self.activities = {
            'A01': 'Stretching', 'A02': 'Chest_expansion_H', 'A03': 'Chest_expansion_V',
            'A04': 'Twist_left', 'A05': 'Twist_right', 'A06': 'Mark_time',
            'A07': 'Limb_ext_left', 'A08': 'Limb_ext_right', 'A09': 'Lunge_left_front',
            'A10': 'Lunge_right_front', 'A11': 'Limb_ext_both', 'A12': 'Squat',
            'A13': 'Raise_hand_left', 'A14': 'Raise_hand_right', 'A15': 'Lunge_left_side',
            'A16': 'Lunge_right_side', 'A17': 'Wave_left', 'A18': 'Wave_right',
            'A19': 'Picking_up', 'A20': 'Throw_left', 'A21': 'Throw_right',
            'A22': 'Kick_left', 'A23': 'Kick_right', 'A24': 'Body_ext_left',
            'A25': 'Body_ext_right', 'A26': 'Jumping', 'A27': 'Bowing'
        }
        
        # Emergency-relevant activities
        self.emergency_activities = {
            'A19': 'sudden_bend',  # Picking up - could be falling
            'A27': 'bowing',       # Bowing - could be falling
            'A12': 'squat'         # Sudden squat - could be collapse
        }
    
    def load_csi(self, csi_file: str) -> np.ndarray:
        """
        Load WiFi CSI data from .mat file.
        
        Args:
            csi_file: Path to CSI .mat file
            
        Returns:
            CSI data array
        """
        try:
            mat_data = scio.loadmat(csi_file)
            
            # Try common CSI data keys
            for key in ['csi', 'CSI', 'csi_data', 'data']:
                if key in mat_data:
                    csi_data = mat_data[key]
                    print(f"[OK] Loaded CSI data from '{key}': shape {csi_data.shape}")
                    return csi_data
            
            # If no standard key, use the first non-metadata key
            for key, value in mat_data.items():
                if not key.startswith('__'):
                    print(f"[INFO] Using key '{key}' as CSI data: shape {value.shape}")
                    return value
            
            raise ValueError("No CSI data found in .mat file")
        
        except Exception as e:
            print(f"[ERROR] Failed to load CSI: {e}")
            return None
    
    def csi_to_spectrogram(
        self,
        csi_data: np.ndarray,
        sample_rate: float = 1000.0,
        nperseg: int = 256,
        noverlap: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert CSI data to spectrogram using STFT.
        
        Args:
            csi_data: Complex CSI data
            sample_rate: Sampling rate in Hz
            nperseg: Length of each segment
            noverlap: Number of points to overlap
            
        Returns:
            (frequencies, time, spectrogram) tuple
        """
        if noverlap is None:
            noverlap = nperseg // 2
        
        # Take amplitude if complex
        if np.iscomplexobj(csi_data):
            amplitude = np.abs(csi_data)
        else:
            amplitude = csi_data
        
        # If multi-dimensional, average or take first dimension
        if amplitude.ndim > 1:
            # Average across subcarriers/antennas
            amplitude = np.mean(amplitude, axis=0)
        
        # Compute STFT
        frequencies, time, Zxx = signal.stft(
            amplitude,
            fs=sample_rate,
            window='hann',
            nperseg=nperseg,
            noverlap=noverlap
        )
        
        # Get magnitude spectrogram
        spectrogram = np.abs(Zxx)
        
        return frequencies, time, spectrogram
    
    def save_spectrogram_image(
        self,
        spectrogram: np.ndarray,
        output_path: str,
        size: Tuple[int, int] = (256, 256),
        cmap: str = 'jet'
    ):
        """
        Save spectrogram as image file.
        
        Args:
            spectrogram: 2D spectrogram array
            output_path: Output image path
            size: Image size (width, height)
            cmap: Colormap
        """
        plt.figure(figsize=(size[0]/100, size[1]/100), dpi=100)
        plt.imshow(
            spectrogram,
            aspect='auto',
            origin='lower',
            cmap=cmap,
            interpolation='bilinear'
        )
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        print(f"[OK] Saved spectrogram image: {output_path}")
    
    def process_activity(
        self,
        environment: str,
        subject: str,
        activity: str
    ) -> Optional[str]:
        """
        Process one activity's WiFi CSI data.
        
        Args:
            environment: E01, E02, etc.
            subject: S01, S02, etc.
            activity: A01, A02, etc.
            
        Returns:
            Path to generated spectrogram image, or None if failed
        """
        # Locate CSI file
        csi_dir = self.mmfi_root / environment / subject / activity / 'wifi-csi'
        
        if not csi_dir.exists():
            print(f"[ERROR] WiFi CSI directory not found: {csi_dir}")
            return None
        
        # Find .mat file
        mat_files = list(csi_dir.glob('*.mat'))
        if not mat_files:
            print(f"[ERROR] No .mat files found in {csi_dir}")
            return None
        
        csi_file = mat_files[0]
        print(f"\n[PROCESSING] {environment}/{subject}/{activity}")
        print(f"  CSI file: {csi_file.name}")
        
        # Load CSI
        csi_data = self.load_csi(str(csi_file))
        if csi_data is None:
            return None
        
        # Convert to spectrogram
        frequencies, time, spectrogram = self.csi_to_spectrogram(csi_data)
        print(f"  Spectrogram shape: {spectrogram.shape}")
        
        # Save as .npy
        npy_filename = f"{environment}_{subject}_{activity}.npy"
        npy_path = self.output_dir / npy_filename
        np.save(npy_path, spectrogram)
        print(f"  [OK] Saved .npy: {npy_path}")
        
        # Save as image
        img_filename = f"{environment}_{subject}_{activity}.png"
        img_path = self.image_dir / img_filename
        self.save_spectrogram_image(spectrogram, str(img_path))
        
        # Check if emergency-relevant
        if activity in self.emergency_activities:
            print(f"  [!] EMERGENCY-RELEVANT: {self.emergency_activities[activity]}")
        
        return str(img_path)
    
    def process_subject(
        self,
        environment: str = 'E01',
        subject: str = 'S01',
        activities: Optional[list] = None
    ):
        """
        Process all activities for one subject.
        
        Args:
            environment: E01, E02, etc.
            subject: S01, S02, etc.
            activities: List of activities to process (e.g., ['A01', 'A19'])
                       If None, processes all available
        """
        subject_dir = self.mmfi_root / environment / subject
        
        if not subject_dir.exists():
            print(f"[ERROR] Subject directory not found: {subject_dir}")
            return
        
        # Get all activities if not specified
        if activities is None:
            activities = sorted([d.name for d in subject_dir.iterdir() 
                               if d.is_dir() and d.name.startswith('A')])
        
        print(f"\n{'='*70}")
        print(f"PROCESSING MM-Fi: {environment}/{subject}")
        print(f"Activities: {len(activities)}")
        print(f"{'='*70}")
        
        results = {}
        for activity in activities:
            img_path = self.process_activity(environment, subject, activity)
            if img_path:
                results[activity] = img_path
        
        print(f"\n{'='*70}")
        print(f"COMPLETE: Processed {len(results)}/{len(activities)} activities")
        print(f"Spectrograms: {self.output_dir}")
        print(f"Images: {self.image_dir}")
        print(f"{'='*70}\n")
        
        return results


def demo_mmfi_processor():
    """Demo the MM-Fi processor."""
    print("="*70)
    print("MM-Fi WiFi CSI Processor Demo")
    print("="*70 + "\n")
    
    # Initialize
    processor = MMFiProcessor(mmfi_root='data/raw/MMFi')
    
    # Check if data exists
    if not processor.mmfi_root.exists():
        print("[ERROR] MM-Fi dataset not found!")
        print(f"Expected location: {processor.mmfi_root}")
        print("\nDownload from:")
        print("  https://drive.google.com/drive/folders/1zDbhfH3BV-xCZVUHmK65EgVV1HMDEYcz")
        print("\nSee docs/MMFI_SETUP.md for instructions")
        return
    
    # Process a few key activities
    print("Processing emergency-relevant activities...")
    emergency_activities = ['A19', 'A27', 'A12']  # Picking, Bowing, Squat
    
    results = processor.process_subject(
        environment='E01',
        subject='S01',
        activities=emergency_activities
    )
    
    if results:
        print("\n[SUCCESS] Ready for classifier!")
        print(f"Image files: {len(results)}")
        for activity, img_path in results.items():
            print(f"  {activity}: {img_path}")
        
        print("\nNext step:")
        print("  python src/classifier.py data/mmfi_images/E01_S01_A19.png")


if __name__ == '__main__':
    demo_mmfi_processor()

