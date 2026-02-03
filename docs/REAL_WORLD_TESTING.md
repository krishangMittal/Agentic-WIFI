"""
Real-Time WiFi Gesture Detection System
Captures live WiFi CSI → Detects gestures → Executes actions
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
from collections import deque
import threading
import queue

sys.path.insert(0, str(Path(__file__).parent))

from mmfi_processor import MMFiProcessor
from classifier import RFCommandClassifier
from agent_simple import RFCommandAgent


class RealTimeGestureDetector:
    """
    Real-time gesture detection system.
    
    Pipeline:
    1. Capture WiFi CSI (from ESP32, Intel tool, or file)
    2. Buffer data (1-2 seconds)
    3. Convert to spectrogram
    4. Classify gesture
    5. Execute action via LLM agent
    """
    
    def __init__(
        self,
        source: str = 'file',  # 'file', 'esp32', 'intel'
        buffer_size: int = 100,
        detection_threshold: float = 0.7,
        api_key: str = None
    ):
        """
        Initialize real-time detector.
        
        Args:
            source: Data source ('file', 'esp32', 'intel')
            buffer_size: Number of CSI samples to buffer
            detection_threshold: Minimum confidence for detection
            api_key: DeepSeek API key
        """
        self.source = source
        self.buffer_size = buffer_size
        self.threshold = detection_threshold
        
        # Data buffer (rolling window)
        self.csi_buffer = deque(maxlen=buffer_size)
        
        # Processing queue
        self.data_queue = queue.Queue()
        
        # Initialize components
        self.processor = MMFiProcessor()
        self.classifier = RFCommandClassifier(num_classes=27, use_pretrained=True)
        self.agent = RFCommandAgent(use_llm=True, api_key=api_key)
        
        # State tracking
        self.is_running = False
        self.last_detection_time = 0
        self.cooldown_period = 2.0  # seconds between detections
        
        print("[OK] Real-Time Gesture Detector initialized")
        print(f"    Source: {source}")
        print(f"    Buffer size: {buffer_size}")
        print(f"    Threshold: {detection_threshold}")
    
    def capture_from_file(self, file_path: str):
        """Simulate real-time capture from file (for testing)."""
        print(f"\n[CAPTURE] Reading from file: {file_path}")
        
        # Load CSI data
        csi_data = self.processor.load_csi(file_path)
        
        if csi_data is None:
            print("[ERROR] Failed to load CSI data")
            return
        
        # Simulate streaming (chunk by chunk)
        chunk_size = 10
        num_chunks = len(csi_data) // chunk_size
        
        print(f"[CAPTURE] Streaming {num_chunks} chunks...")
        
        for i in range(num_chunks):
            if not self.is_running:
                break
            
            # Get chunk
            chunk = csi_data[i*chunk_size:(i+1)*chunk_size]
            
            # Add to queue
            self.data_queue.put(chunk)
            
            # Simulate real-time delay
            time.sleep(0.05)  # 50ms per chunk
    
    def capture_from_esp32(self, serial_port: str = '/dev/ttyUSB0'):
        """Capture from ESP32 via serial."""
        import serial
        
        print(f"\n[CAPTURE] Connecting to ESP32 on {serial_port}")
        
        try:
            ser = serial.Serial(serial_port, 115200, timeout=1)
            print("[OK] Connected to ESP32")
            
            while self.is_running:
                line = ser.readline().decode('utf-8').strip()
                
                if line.startswith("CSI_DATA"):
                    # Parse CSI data from ESP32 format
                    csi_values = self.parse_esp32_csi(line)
                    self.data_queue.put(csi_values)
        
        except Exception as e:
            print(f"[ERROR] ESP32 capture failed: {e}")
    
    def parse_esp32_csi(self, line: str) -> np.ndarray:
        """Parse CSI data from ESP32 output format."""
        # ESP32 format: "CSI_DATA,subcarrier1,subcarrier2,..."
        parts = line.split(',')[1:]  # Skip "CSI_DATA"
        csi_values = [float(x) for x in parts if x]
        return np.array(csi_values)
    
    def process_buffer(self):
        """Process buffered CSI data to detect gestures."""
        if len(self.csi_buffer) < self.buffer_size // 2:
            return None  # Not enough data yet
        
        # Convert buffer to array
        csi_data = np.array(list(self.csi_buffer))
        
        # Convert to spectrogram
        try:
            frequencies, time, spectrogram = self.processor.csi_to_spectrogram(csi_data)
            
            # Save as temporary image
            temp_img = Path('data/temp_realtime_spec.png')
            self.processor.save_spectrogram_image(spectrogram, str(temp_img))
            
            # Classify
            label, confidence = self.classifier.predict(str(temp_img))
            
            # Clean up
            temp_img.unlink()
            
            return (label, confidence)
        
        except Exception as e:
            print(f"[ERROR] Processing failed: {e}")
            return None
    
    def detection_loop(self):
        """Main detection loop."""
        print("\n[DETECTOR] Starting detection loop...")
        
        while self.is_running:
            try:
                # Get data from queue (with timeout)
                chunk = self.data_queue.get(timeout=0.1)
                
                # Add to buffer
                if isinstance(chunk, np.ndarray):
                    if chunk.ndim == 1:
                        self.csi_buffer.append(chunk)
                    else:
                        # Multiple samples
                        for sample in chunk:
                            self.csi_buffer.append(sample)
                
                # Process buffer periodically
                current_time = time.time()
                if current_time - self.last_detection_time > self.cooldown_period:
                    result = self.process_buffer()
                    
                    if result:
                        label, confidence = result
                        
                        if confidence >= self.threshold:
                            self.last_detection_time = current_time
                            self.handle_detection(label, confidence)
            
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[ERROR] Detection loop error: {e}")
    
    def handle_detection(self, label: str, confidence: float):
        """Handle detected gesture."""
        print(f"\n{'='*60}")
        print(f"[DETECTED] Gesture: {label} ({confidence*100:.1f}%)")
        print(f"{'='*60}")
        
        # Map label to activity code (for demo)
        activity_map = {
            'word_17': 'A17',  # wave_left
            'word_18': 'A18',  # wave_right
            'word_13': 'A13',  # raise_left
            'word_14': 'A14',  # raise_right
            'word_19': 'A19',  # picking_up
            'word_27': 'A27',  # bowing
        }
        
        # Get context
        context = self.get_current_context()
        
        # Use agent to decide action
        interpretation = self.agent.interpret_command(
            predictions=[(label, confidence)],
            context=context
        )
        
        print(f"\nAgent Decision:")
        print(f"  Action: {interpretation['action']}")
        print(f"  Reasoning: {interpretation['reasoning']}")
        
        # Execute action
        result = self.execute_action(interpretation)
        print(f"\nExecution: {result}\n")
    
    def get_current_context(self) -> str:
        """Get current context for agent."""
        from datetime import datetime
        
        now = datetime.now()
        hour = now.hour
        
        if 6 <= hour < 12:
            time_of_day = "morning"
        elif 12 <= hour < 18:
            time_of_day = "afternoon"
        elif 18 <= hour < 23:
            time_of_day = "evening"
        else:
            time_of_day = "night"
        
        return f"Time: {time_of_day}, Location: home"
    
    def execute_action(self, interpretation: dict) -> str:
        """Execute the interpreted action."""
        action = interpretation['action']
        
        # Map actions to actual commands
        if action in ['left', 'turn_on_lights']:
            return self.turn_on_lights()
        elif action in ['right', 'turn_off_lights']:
            return self.turn_off_lights()
        elif action == 'help':
            return self.send_alert()
        else:
            return f"[ACTION] {action}"
    
    def turn_on_lights(self) -> str:
        """Turn on smart lights (integrate with your smart home API)."""
        # Example: HTTP request to smart home hub
        # requests.post('http://192.168.1.100/api/lights/on')
        print("[SMART HOME] Turning on lights...")
        return "[OK] Lights turned ON"
    
    def turn_off_lights(self) -> str:
        """Turn off smart lights."""
        print("[SMART HOME] Turning off lights...")
        return "[OK] Lights turned OFF"
    
    def send_alert(self) -> str:
        """Send emergency alert."""
        print("[ALERT] Sending emergency notification...")
        return "[OK] Alert sent"
    
    def start(self, source_path: str = None):
        """Start real-time detection."""
        self.is_running = True
        
        # Start detection thread
        detection_thread = threading.Thread(target=self.detection_loop)
        detection_thread.start()
        
        # Start capture thread
        if self.source == 'file':
            if source_path:
                capture_thread = threading.Thread(
                    target=self.capture_from_file,
                    args=(source_path,)
                )
                capture_thread.start()
        
        elif self.source == 'esp32':
            capture_thread = threading.Thread(
                target=self.capture_from_esp32,
                args=(source_path or '/dev/ttyUSB0',)
            )
            capture_thread.start()
        
        print("\n[OK] Real-time detection started!")
        print("Perform gestures and watch them get detected...")
        print("Press Ctrl+C to stop\n")
        
        try:
            # Keep main thread alive
            while self.is_running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n[STOP] Stopping detection...")
            self.is_running = False
            detection_thread.join()
    
    def stop(self):
        """Stop detection."""
        self.is_running = False


def demo_realtime():
    """Demo real-time detection with file."""
    print("="*60)
    print("REAL-TIME WIFI GESTURE DETECTION DEMO")
    print("="*60)
    
    # Initialize detector
    detector = RealTimeGestureDetector(
        source='file',
        buffer_size=100,
        detection_threshold=0.7,
        api_key=os.getenv('DEEPSEEK_API_KEY')  # Set via environment variable
    )
    
    # Test with MM-Fi data (if available)
    test_file = 'data/raw/MMFi/E01/S01/A17/wifi-csi/csi.mat'
    
    if Path(test_file).exists():
        print(f"\nUsing MM-Fi data: {test_file}")
        detector.start(source_path=test_file)
    else:
        print("\n[INFO] No MM-Fi data found")
        print("Download MM-Fi dataset to test with real WiFi data")
        print("See docs/MMFI_SETUP.md for instructions")


if __name__ == '__main__':
    demo_realtime()

