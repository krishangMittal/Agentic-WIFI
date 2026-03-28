"""
Live WiFi CSI Demo — Motion & Wave Detection via ESP32
"""

import serial
import numpy as np
import time
import sys
from collections import deque


def parse_csi_line(line: str):
    """Parse ESP32 CSI output into amplitude array."""
    try:
        if 'CSI_DATA' not in line or '[' not in line:
            return None, None, None

        parts = line.split(',')
        mac = parts[2] if len(parts) > 2 else 'unknown'
        rssi = int(parts[3]) if len(parts) > 3 else 0

        bracket_start = line.index('[')
        bracket_end = line.index(']')
        data_str = line[bracket_start + 1:bracket_end].strip()

        values = [int(x) for x in data_str.split() if x.lstrip('-').isdigit()]
        if len(values) < 4:
            return None, None, None

        amplitudes = []
        for i in range(0, len(values) - 1, 2):
            amp = np.sqrt(values[i] ** 2 + values[i + 1] ** 2)
            amplitudes.append(amp)

        return np.array(amplitudes), mac, rssi

    except (ValueError, IndexError):
        return None, None, None


def run_demo(port='COM5', baud=115200):
    print("=" * 60)
    print("  LIVE WiFi CSI GESTURE DETECTION")
    print("=" * 60)
    print(f"\nConnecting to {port}...")

    try:
        ser = serial.Serial(port, baud, timeout=1)
    except serial.SerialException as e:
        print(f"Error: {e}")
        return

    print("Waiting for ESP32...")

    # Just collect ALL packets, no phases, no waiting
    all_frames = deque(maxlen=100)  # rolling buffer of last 100 frames
    motion_scores = deque(maxlen=50)
    frame_count = 0
    total_waves = 0
    wave_cooldown = 0
    calibrated = False
    baseline_std = 1.0
    last_print = time.time()
    started = False

    print("Listening for CSI data (play YouTube/ping over hotspot)...\n")

    try:
        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            amps, mac, rssi = parse_csi_line(line)

            if amps is None:
                continue

            if not started:
                started = True
                print("CSI packets coming in! Collecting baseline (don't move)...")

            frame_count += 1
            current = amps[:54]
            # Use ALL devices — don't filter by MAC
            all_frames.append(current)

            # After 10 frames, calibrate baseline automatically
            if frame_count == 10 and not calibrated:
                baseline = np.array(list(all_frames))
                baseline_std = max(np.std(baseline, axis=0).mean(), 0.5)
                calibrated = True
                print(f"Baseline set from {len(all_frames)} frames (std={baseline_std:.2f})")
                print()
                print("=" * 60)
                print("  NOW MOVE! Wave your hand near ESP32/phone!")
                print("=" * 60)
                print()
                continue

            if not calibrated:
                if frame_count % 2 == 0:
                    print(f"  Calibrating... {frame_count}/10 frames")
                continue

            # Motion detection: compare to previous frame (instant)
            if len(all_frames) >= 2:
                prev = list(all_frames)[-2]
                diff = np.abs(current - prev).mean()
                motion_score = diff / max(baseline_std, 0.1)
            else:
                motion_score = 0.0

            motion_scores.append(motion_score)

            # Wave detection: rhythmic motion pattern
            is_wave = False
            if len(motion_scores) >= 8:
                recent_scores = list(motion_scores)[-12:]
                mean_s = np.mean(recent_scores)
                if mean_s > 1.0:
                    crossings = 0
                    for j in range(1, len(recent_scores)):
                        if (recent_scores[j] > mean_s) != (recent_scores[j-1] > mean_s):
                            crossings += 1
                    is_wave = crossings >= 3 and mean_s > 1.2

            # Display every packet (no delay)
            now = time.time()
            if now - last_print > 0.05:
                last_print = now
                avg_score = np.mean(list(motion_scores)[-2:]) if motion_scores else 0
                bar_len = min(int(avg_score * 3), 20)
                bar = "#" * bar_len + "." * (20 - bar_len)

                if is_wave and wave_cooldown <= 0:
                    status = ">>> WAVE DETECTED! <<<"
                    wave_cooldown = 8
                    total_waves += 1
                elif avg_score > 3.0:
                    status = "STRONG MOTION"
                elif avg_score > 1.5:
                    status = "motion"
                else:
                    status = "still"

                wave_cooldown = max(0, wave_cooldown - 1)

                pps = frame_count / max(time.time() - ser.timeout, 1)
                print(f"\r  [{bar}] {avg_score:4.1f} rssi={rssi}dBm pkts={frame_count} | {status:<25} waves={total_waves}", end='', flush=True)

    except KeyboardInterrupt:
        print(f"\n\nDone! {total_waves} waves detected from {frame_count} packets.")
    finally:
        ser.close()


if __name__ == '__main__':
    port = sys.argv[1] if len(sys.argv) > 1 else 'COM5'
    run_demo(port=port)
