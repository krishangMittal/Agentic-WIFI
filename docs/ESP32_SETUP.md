# ESP32 WiFi CSI Capture - Real-Time Setup

## 🎯 Goal
Capture live WiFi CSI data using **ESP32** microcontroller for real-time gesture detection.

---

## 💰 Cost: ~$10-15

**Hardware needed:**
- ESP32 DevKit (~$5-8) - [Amazon](https://www.amazon.com/s?k=esp32+devkit)
- USB cable (usually included)

That's it! No other hardware needed.

---

## 🔧 Hardware Setup

### Step 1: Get ESP32
**Buy:** ESP32-WROOM-32 DevKit (most common)
- Amazon: Search "ESP32 DevKit"
- AliExpress: ~$3-5 (longer shipping)

### Step 2: Install Drivers
**Windows:**
- Install CP210x driver: https://www.silabs.com/developers/usb-to-uart-bridge-vcp-drivers

**Linux:**
- Usually works out of the box

**Mac:**
- Install driver from above link

### Step 3: Connect ESP32
- Plug ESP32 into USB port
- Check device manager (Windows) or `ls /dev/ttyUSB*` (Linux)
- Note the COM port (e.g., COM3 or /dev/ttyUSB0)

---

## 💻 Software Setup

### Step 1: Install ESP-IDF (ESP32 Toolchain)

**Option A: PlatformIO (Easiest)**
```bash
# Install PlatformIO
pip install platformio

# Create project
mkdir esp32-csi && cd esp32-csi
pio init --board esp32dev
```

**Option B: Arduino IDE**
1. Install Arduino IDE
2. Add ESP32 board: https://github.com/espressif/arduino-esp32
3. Select Board: "ESP32 Dev Module"

### Step 2: Flash CSI Firmware

Use this pre-made ESP32 CSI firmware:

**GitHub:** https://github.com/StevenMHernandez/ESP32-CSI-Tool

```bash
# Clone the repo
git clone https://github.com/StevenMHernandez/ESP32-CSI-Tool
cd ESP32-CSI-Tool

# Flash to ESP32
pio run -t upload
```

Or manually:

```bash
# Compile and flash
esptool.py --port /dev/ttyUSB0 write_flash 0x10000 firmware.bin
```

### Step 3: Verify CSI Output

```bash
# Connect to ESP32 serial
screen /dev/ttyUSB0 115200

# Or on Windows:
# Use PuTTY to connect to COM3 at 115200 baud
```

You should see CSI data streaming:
```
CSI_DATA,type=CSI,id=0,mac=XX:XX:XX:XX:XX:XX,rssi=-45,rate=11,sig_mode=1,mcs=7,bandwidth=0,smoothing=1,not_sounding=1,aggregation=1,stbc=0,fec_coding=0,sgi=0,noise_floor=-95,ampdu_cnt=1,channel=1,secondary_channel=0,local_timestamp=1234567890,ant=0,sig_len=100,rx_state=0,len=384,first_word=1,data=[1,2,3,4,...]
```

---

## 🐍 Python Integration

### Step 1: Install PySerial

```bash
pip install pyserial
```

### Step 2: Capture CSI in Python

Create `capture_esp32_csi.py`:

```python
import serial
import numpy as np
import time

# Connect to ESP32
ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)  # Change port on Windows: 'COM3'

print("Connected to ESP32. Capturing CSI...")

while True:
    line = ser.readline().decode('utf-8', errors='ignore').strip()
    
    if line.startswith("CSI_DATA"):
        # Parse CSI data
        parts = line.split(',')
        
        # Extract relevant fields
        csi_dict = {}
        for part in parts:
            if '=' in part:
                key, value = part.split('=', 1)
                csi_dict[key] = value
        
        # Extract CSI amplitude data
        if 'data' in csi_dict:
            data_str = csi_dict['data'].strip('[]')
            csi_values = [int(x) for x in data_str.split(',') if x]
            
            # Convert to numpy
            csi_array = np.array(csi_values)
            
            print(f"Captured CSI: {len(csi_array)} subcarriers, RSSI: {csi_dict.get('rssi', 'N/A')}")
            
            # TODO: Send to your gesture detection pipeline
            # detector.process_csi(csi_array)
```

---

## 🚀 Integrate with Your System

Update `setup_realtime.py` to use ESP32:

```python
# In RealTimeGestureDetector class:

def capture_from_esp32(self, serial_port: str = '/dev/ttyUSB0'):
    """Capture from ESP32 via serial."""
    import serial
    
    print(f"\n[CAPTURE] Connecting to ESP32 on {serial_port}")
    
    try:
        ser = serial.Serial(serial_port, 115200, timeout=1)
        print("[OK] Connected to ESP32")
        
        while self.is_running:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            
            if line.startswith("CSI_DATA"):
                # Parse CSI data
                csi_values = self.parse_esp32_csi(line)
                if csi_values is not None:
                    self.data_queue.put(csi_values)
    
    except Exception as e:
        print(f"[ERROR] ESP32 capture failed: {e}")

def parse_esp32_csi(self, line: str) -> np.ndarray:
    """Parse CSI data from ESP32 output format."""
    try:
        # ESP32 format: "CSI_DATA,...,data=[1,2,3,...]"
        data_part = line.split('data=')[1]
        data_str = data_part.strip('[]').split(']')[0]
        csi_values = [int(x) for x in data_str.split(',') if x]
        return np.array(csi_values)
    except:
        return None
```

Then run:

```python
# Use ESP32 source
detector = RealTimeGestureDetector(source='esp32')
detector.start(source_path='/dev/ttyUSB0')  # Or 'COM3' on Windows
```

---

## 📡 Router Setup (For Best Results)

### Configure WiFi Router:
1. **Set channel:** Channel 1, 6, or 11 (2.4 GHz)
2. **Bandwidth:** 20 MHz (not 40 MHz)
3. **Mode:** 802.11n
4. **Place ESP32:** 2-5 meters from router

### Optimal Placement:
```
Router -----> [2-5m] -----> ESP32
              |
              v
         Your Gestures Here
```

Person moves between router and ESP32 → Maximum signal disruption

---

## 🧪 Testing

### Step 1: Verify CSI Capture

```bash
python capture_esp32_csi.py
```

You should see:
```
Connected to ESP32. Capturing CSI...
Captured CSI: 64 subcarriers, RSSI: -45
Captured CSI: 64 subcarriers, RSSI: -46
Captured CSI: 64 subcarriers, RSSI: -44
```

### Step 2: Test Gesture Detection

```bash
python setup_realtime.py
```

Perform gestures:
1. **Wave left hand** → Should detect and turn on lights
2. **Wave right hand** → Should detect and turn off lights
3. **Raise hand** → Should detect volume control

---

## 📊 Expected Performance

| Metric | Value |
|--------|-------|
| **Sampling Rate** | ~100 Hz |
| **Latency** | 0.5-1 second |
| **Detection Accuracy** | 80-90% (after fine-tuning) |
| **Range** | 2-10 meters |
| **Cost** | ~$10 |

---

## 🐛 Troubleshooting

### Problem: No CSI data
**Solution:**
- Check USB connection
- Verify COM port
- Reflash firmware
- Check WiFi traffic (ESP32 needs active WiFi)

### Problem: Poor detection
**Solution:**
- Move closer to router
- Reduce distance between router and ESP32
- Increase gesture size
- Fine-tune classifier on YOUR environment

### Problem: Interference
**Solution:**
- Change WiFi channel
- Reduce other 2.4 GHz devices
- Use 5 GHz router (requires ESP32-C3 or newer)

---

## 🔗 Resources

- **ESP32 CSI Tool:** https://github.com/StevenMHernandez/ESP32-CSI-Tool
- **ESP-IDF Documentation:** https://docs.espressif.com/projects/esp-idf/
- **CSI Toolkit:** https://github.com/espressif/esp-csi
- **Research Paper:** "ESP32 CSI Toolkit for WiFi Sensing"

---

## 🎯 Next Steps

1. ✅ Get ESP32 (~$10)
2. ✅ Flash CSI firmware
3. ✅ Test CSI capture
4. ✅ Integrate with your system
5. ✅ Fine-tune on your gestures
6. ✅ Deploy in your home!

**You'll have a working system for under $15!** 🎉

