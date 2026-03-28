"""
Agentic WiFi CSI Demo — RSSI + CSI combined detection
"""

import serial
import numpy as np
import time
import sys
import os
import requests
from collections import deque


def parse_csi_line(line):
    try:
        if 'CSI_DATA' not in line or '[' not in line:
            return None, None, None
        parts = line.split(',')
        mac = parts[2]
        rssi = int(parts[3])
        bs = line.index('[')
        be = line.index(']')
        vals = [int(x) for x in line[bs+1:be].split() if x.lstrip('-').isdigit()]
        if len(vals) < 4:
            return None, None, None
        amps = [np.sqrt(vals[i]**2 + vals[i+1]**2) for i in range(0, len(vals)-1, 2)]
        return np.array(amps[:54]), mac, rssi
    except:
        return None, None, None


def call_deepseek(gesture, score, rssi, api_key):
    try:
        r = requests.post("https://api.deepseek.com/v1/chat/completions", headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }, json={
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": f"""You are a smart home AI. WiFi sensing detected: {gesture} (confidence: {score:.0f}%, signal: {rssi}dBm, time: {time.strftime('%H:%M:%S')}).
What action should the smart home take? Reply in 1-2 short lines:
ACTION: <action>
REASON: <why>"""}],
            "temperature": 0.3,
            "max_tokens": 100
        }, timeout=10)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"].strip()
    except:
        pass
    return None


def run(port='COM5'):
    api_key = os.getenv("DEEPSEEK_API_KEY")
    has_llm = bool(api_key)

    print("=" * 60)
    print("  AGENTIC WiFi SENSING — 'Siri for WiFi'")
    print("=" * 60)
    print(f"  LLM: {'DeepSeek' if has_llm else 'Rule-based (set DEEPSEEK_API_KEY)'}")
    print()

    ser = serial.Serial(port, 115200, timeout=0.1)
    ser.reset_input_buffer()

    print("Waiting for ESP32...")
    while True:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if 'CSI_DATA' in line and '[' in line:
            break

    # Find target MAC
    print("Finding signal source (3 sec)...")
    macs = {}
    t0 = time.time()
    while time.time() - t0 < 3:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        a, m, r = parse_csi_line(line)
        if a is not None:
            if m not in macs:
                macs[m] = 0
            macs[m] += 1

    if not macs:
        print("No data!")
        ser.close()
        return

    target = max(macs, key=lambda m: macs[m])
    print(f"Tracking: {target} ({macs[target]} pkts)")

    # Calibrate — collect raw CSI values when still
    print("\n>>> STAY STILL FOR 5 SECONDS <<<")
    baseline_amps = []
    baseline_rssi = []
    t0 = time.time()
    while time.time() - t0 < 5:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        a, m, r = parse_csi_line(line)
        if a is not None and m == target:
            baseline_amps.append(a)
            baseline_rssi.append(r)

    n_base = len(baseline_amps)
    print(f"Collected {n_base} baseline frames")

    if n_base < 5:
        print("Not enough data!")
        ser.close()
        return

    # Baseline = average CSI pattern when still
    base_arr = np.array(baseline_amps)
    base_mean = np.mean(base_arr, axis=0)
    base_rssi_mean = np.mean(baseline_rssi)
    base_rssi_std = max(np.std(baseline_rssi), 0.5)

    # Natural variation when still
    still_diffs = []
    for i in range(1, len(baseline_amps)):
        d = np.abs(np.array(baseline_amps[i]) - np.array(baseline_amps[i-1])).mean()
        still_diffs.append(d)
    still_noise = np.mean(still_diffs)
    still_noise_max = np.percentile(still_diffs, 95)  # 95th percentile = threshold

    print(f"Still noise: mean={still_noise:.2f}, p95={still_noise_max:.2f}")
    print(f"RSSI baseline: {base_rssi_mean:.0f} +/- {base_rssi_std:.1f} dBm")

    print(f"\n{'='*60}")
    print(f"  SYSTEM ACTIVE — wave or walk near the ESP32!")
    print(f"  Anything above {still_noise_max:.1f} = motion")
    print(f"{'='*60}\n")

    scores = deque(maxlen=30)
    rssi_history = deque(maxlen=30)
    prev = None
    pkt = 0
    last_event = 0
    events = []
    last_gesture = "STILL"

    actions = {
        "WAVE": "Toggle smart lights",
        "WALKING": "Someone moving through room",
        "MOVEMENT": "Activity detected nearby",
    }

    try:
        while True:
            ser.reset_input_buffer()
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            a, m, r = parse_csi_line(line)
            if a is None or m != target:
                continue

            pkt += 1
            rssi_history.append(r)

            # Motion = how different is this frame from the previous one, relative to natural noise
            if prev is not None:
                raw_diff = np.abs(a - prev).mean()
                # Score: how many times above the still noise level
                score = raw_diff / max(still_noise_max, 0.1)
            else:
                score = 0
            prev = a.copy()
            scores.append(score)

            # Smooth over last few
            avg = np.mean(list(scores)[-4:]) if len(scores) >= 4 else score

            # Classify
            gesture = "STILL"
            if avg > 3.0:
                gesture = "WALKING"
            elif avg > 1.5:
                if len(scores) >= 8:
                    s = list(scores)[-10:]
                    ms = np.mean(s)
                    crosses = sum(1 for j in range(1, len(s)) if (s[j] > ms) != (s[j-1] > ms))
                    gesture = "WAVE" if crosses >= 4 else "MOVEMENT"
                else:
                    gesture = "MOVEMENT"

            # Display
            bar = "#" * min(int(avg * 4), 20) + "." * max(20 - int(avg * 4), 0)
            icon = {"STILL": " ", "MOVEMENT": "!", "WAVE": "~", "WALKING": ">"}
            print(f"\r  {icon.get(gesture, ' ')} [{bar}] {avg:4.1f}x noise | {gesture:<12} rssi={r}dBm pkt={pkt}", end='', flush=True)

            # Trigger on gesture
            now = time.time()
            if gesture != "STILL" and gesture != last_gesture and now - last_event > 3:
                last_event = now
                ts = time.strftime("%H:%M:%S")
                print(f"\n\n  {'~'*50}")
                print(f"  [{ts}] {gesture} detected! ({avg:.1f}x above noise)")

                if has_llm and gesture in ("WAVE", "WALKING"):
                    print(f"  Asking DeepSeek...")
                    resp = call_deepseek(gesture, min(avg * 30, 99), r, api_key)
                    if resp:
                        for ln in resp.split('\n'):
                            if ln.strip():
                                print(f"    {ln.strip()}")
                else:
                    print(f"    -> {actions.get(gesture, 'Logged')}")

                print(f"  {'~'*50}\n")
                events.append({'time': ts, 'gesture': gesture, 'score': avg})

            last_gesture = gesture if gesture != "STILL" else last_gesture
            if gesture == "STILL" and len(scores) > 5 and all(s < 1.2 for s in list(scores)[-5:]):
                last_gesture = "STILL"

    except KeyboardInterrupt:
        print(f"\n\n{'='*60}")
        print(f"  SESSION: {pkt} packets, {len(events)} events")
        for e in events:
            print(f"    [{e['time']}] {e['gesture']} ({e['score']:.1f}x noise)")
        print(f"{'='*60}")
    finally:
        ser.close()


if __name__ == '__main__':
    port = sys.argv[1] if len(sys.argv) > 1 else 'COM5'
    run(port)
