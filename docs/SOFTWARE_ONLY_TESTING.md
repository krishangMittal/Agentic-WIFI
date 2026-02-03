# Software-Only WiFi CSI Capture (No Hardware Needed!)

## 🎯 Goal
Capture WiFi CSI using **your existing laptop** - no extra hardware needed!

**Works if you have:**
- ✅ Linux laptop (Ubuntu, Debian, etc.)
- ✅ Intel WiFi card (most Intel laptops)
- ✅ OR: Qualcomm Atheros WiFi card

---

## Option 1: Intel WiFi Card (IWL5300)

### Step 1: Check Your WiFi Card

```bash
lspci | grep -i wireless
```

Look for:
- Intel Wireless 5300
- Intel Wireless 5100
- Intel Centrino Ultimate-N 6300

### Step 2: Install Linux CSI Tool

```bash
# Install dependencies
sudo apt-get update
sudo apt-get install build-essential linux-headers-$(uname -r) git

# Clone Intel CSI tool
git clone https://github.com/dhalperi/linux-80211n-csitool.git
cd linux-80211n-csitool

# Build and install
make -C /lib/modules/$(uname -r)/build M=$(pwd)/drivers/net/wireless/iwlwifi modules
sudo make -C /lib/modules/$(uname -r)/build M=$(pwd)/drivers/net/wireless/iwlwifi INSTALL_MOD_DIR=updates modules_install
sudo depmod
```

### Step 3: Load Modified Driver

```bash
# Remove existing driver
sudo modprobe -r iwlwifi mac80211

# Load CSI-enabled driver
sudo modprobe iwlwifi connector_log=0x1
```

### Step 4: Capture CSI

```bash
# Start monitor mode
sudo ip link set wlan0 down
sudo iw dev wlan0 set type monitor
sudo ip link set wlan0 up

# Set channel
sudo iw dev wlan0 set channel 64 HT20

# Capture CSI
sudo tcpdump -i wlan0 -w csi_capture.pcap
```

### Step 5: Parse CSI Data

```python
import numpy as np
from scapy.all import rdpcap

# Read captured packets
packets = rdpcap('csi_capture.pcap')

for pkt in packets:
    if pkt.haslayer('RadioTap'):
        # Extract CSI from radiotap header
        csi_data = parse_radiotap_csi(pkt)
        print(f"CSI: {len(csi_data)} subcarriers")
```

---

## Option 2: Atheros WiFi Card (Simpler!)

### Step 1: Check for Atheros

```bash
lspci | grep -i atheros
```

Look for:
- Qualcomm Atheros AR9XXX
- Atheros AR928X

### Step 2: Use Atheros CSI Tool

```bash
# Clone Atheros CSI tool
git clone https://github.com/xieyaxiongfly/Atheros-CSI-Tool.git
cd Atheros-CSI-Tool

# Build
make

# Load module
sudo insmod ath9k_csi.ko
```

### Step 3: Capture CSI

```bash
# Start capture
sudo ./recvCSI

# You should see CSI data streaming
```

---

## Option 3: Use Your Phone! (Android)

### WiSee Android App

1. Install "Nexmon CSI" app from GitHub
2. Root your Android phone (OnePlus, Pixel work best)
3. Capture CSI directly on phone
4. Send to PC via WiFi

**GitHub:** https://github.com/seemoo-lab/nexmon_csi

---

## 🐍 Python Integration (Software-Only)

### Create `capture_intel_csi.py`:

```python
import subprocess
import numpy as np
import struct

def capture_csi_intel(interface='wlan0', duration=10):
    """Capture CSI from Intel WiFi card."""
    print(f"Capturing CSI on {interface} for {duration} seconds...")
    
    # Start tcpdump
    cmd = f"sudo timeout {duration} tcpdump -i {interface} -w /tmp/csi.pcap"
    subprocess.run(cmd, shell=True)
    
    # Parse captured data
    csi_data = parse_pcap_csi('/tmp/csi.pcap')
    
    return csi_data

def parse_pcap_csi(pcap_file):
    """Parse CSI from pcap file."""
    from scapy.all import rdpcap
    
    csi_list = []
    
    packets = rdpcap(pcap_file)
    
    for pkt in packets:
        try:
            # Extract CSI from packet
            # (Format depends on your WiFi card)
            csi = extract_csi_from_packet(pkt)
            if csi is not None:
                csi_list.append(csi)
        except:
            pass
    
    return np.array(csi_list)

def extract_csi_from_packet(pkt):
    """Extract CSI from individual packet."""
    # This is hardware-specific
    # Intel 5300: 30 subcarriers
    # See: https://github.com/dhalperi/linux-80211n-csitool
    pass

if __name__ == '__main__':
    csi = capture_csi_intel(duration=5)
    print(f"Captured {len(csi)} CSI samples")
```

---

## 🚀 Integrate with Your System

### Update `setup_realtime.py`:

```python
def capture_from_intel(self, interface: str = 'wlan0'):
    """Capture from Intel WiFi card."""
    print(f"\n[CAPTURE] Capturing from Intel WiFi: {interface}")
    
    import subprocess
    
    # Start tcpdump in background
    process = subprocess.Popen(
        f"sudo tcpdump -i {interface} -U -w - | tee /tmp/csi_stream.pcap",
        shell=True,
        stdout=subprocess.PIPE
    )
    
    while self.is_running:
        # Read packets in real-time
        # Parse CSI
        # Add to queue
        pass
```

---

## 📊 Comparison

| Method | Cost | Difficulty | Latency | Best For |
|--------|------|------------|---------|----------|
| **Intel CSI Tool** | $0 | Medium | ~100ms | Linux users, research |
| **Atheros Tool** | $0 | Medium | ~100ms | Linux users |
| **ESP32** | $10 | Easy | ~500ms | Everyone, production |
| **Nexmon (Android)** | $0 | Hard | ~200ms | Mobile testing |

---

## 🎯 Recommended Approach

### For Testing NOW (Easiest):
1. **Use MM-Fi dataset** (simulated real-time in `setup_realtime.py`)
2. Proves your system works
3. No hardware needed

### For Real Deployment (Best):
1. **Get ESP32** (~$10)
2. Flash CSI firmware
3. Capture live CSI
4. Full production-ready

### For Research (Most Control):
1. **Intel WiFi card** (if you have it)
2. Linux CSI tool
3. Raw CSI access
4. Best for papers

---

## 🧪 Quick Test

### Option 1: Simulate with MM-Fi

```bash
# Download one MM-Fi file
# Run:
python setup_realtime.py
```

### Option 2: Intel WiFi (Linux)

```bash
# Check if you have Intel WiFi
lspci | grep -i intel | grep -i wireless

# If yes:
git clone https://github.com/dhalperi/linux-80211n-csitool.git
# Follow Intel setup above
```

### Option 3: Order ESP32 ($10)

**Amazon:** Search "ESP32 DevKit"
**Delivery:** 1-2 days
**Setup:** 30 minutes

---

## 💡 Bottom Line

**Easiest path to real-life testing:**

1. **Today:** Use `setup_realtime.py` with MM-Fi data
2. **This week:** Order ESP32 ($10)
3. **Next week:** Test with live WiFi CSI
4. **After fine-tuning:** Deploy in your home!

**Your system is software-ready. Hardware is just $10 away!** 🚀



