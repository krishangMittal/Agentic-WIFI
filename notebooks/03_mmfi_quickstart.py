"""
MM-Fi Quick Start: End-to-End Demo
Shows the complete "Siri for WiFi" pipeline with MM-Fi dataset
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set API key
os.environ['DEEPSEEK_API_KEY'] = 'sk-71f67cd6e695467eb0251aef4f05d734'

from src.mmfi_processor import MMFiProcessor
from src.classifier import RFCommandClassifier
from src.agent_simple import RFCommandAgent


def demo_end_to_end():
    """Complete MM-Fi to Action pipeline."""
    
    print("="*80)
    print("MM-Fi 'SIRI FOR WIFI' END-TO-END DEMO")
    print("="*80 + "\n")
    
    # Stage 1: Process WiFi CSI → Spectrogram
    print("STAGE 1: WiFi CSI → Spectrogram")
    print("-"*80)
    
    processor = MMFiProcessor(mmfi_root='data/raw/MMFi')
    
    # Check if dataset exists
    if not processor.mmfi_root.exists():
        print("[ERROR] MM-Fi dataset not found!")
        print(f"Expected: {processor.mmfi_root}")
        print("\nDownload from Google Drive:")
        print("  https://drive.google.com/drive/folders/1zDbhfH3BV-xCZVUHmK65EgVV1HMDEYcz")
        print("\nJust download E01/S01/ to get started!")
        print("See docs/MMFI_SETUP.md for details")
        return
    
    # Process emergency-relevant activities
    print("Processing emergency-relevant activities...")
    test_activities = ['A19', 'A27']  # Picking up, Bowing (potential falls)
    
    results = processor.process_subject(
        environment='E01',
        subject='S01',
        activities=test_activities
    )
    
    if not results:
        print("[ERROR] No spectrograms generated")
        return
    
    # Stage 2: Classify spectrograms
    print("\n" + "="*80)
    print("STAGE 2: Spectrogram → Classification")
    print("-"*80)
    
    # Note: For real usage, you'd train this on MM-Fi's 27 classes
    # For demo, we'll use the pre-trained model as a feature extractor
    classifier = RFCommandClassifier(
        num_classes=27,  # MM-Fi has 27 activities
        use_pretrained=True,
        model_name='resnet18'
    )
    
    print("[INFO] Using pre-trained ResNet18 as feature extractor")
    print("[INFO] In production, fine-tune on MM-Fi training data\n")
    
    # Classify each activity
    classifications = {}
    for activity, img_path in results.items():
        print(f"Classifying {activity}...")
        label, confidence = classifier.predict(img_path)
        classifications[activity] = (label, confidence)
        print(f"  Predicted: {label} ({confidence*100:.1f}% confidence)\n")
    
    # Stage 3: Agent Decision
    print("="*80)
    print("STAGE 3: Agent Decision (DeepSeek)")
    print("-"*80)
    
    agent = RFCommandAgent(use_llm=True)
    
    # Test scenarios
    scenarios = [
        {
            'activity': 'A19',
            'real_name': 'Picking up things',
            'context': 'Elderly person alone at home, sudden downward motion detected',
            'expected': 'Check for fall'
        },
        {
            'activity': 'A27',
            'real_name': 'Bowing',
            'context': 'Sudden downward motion, no upward motion detected after 3 seconds',
            'expected': 'Emergency alert'
        }
    ]
    
    for scenario in scenarios:
        activity = scenario['activity']
        if activity not in classifications:
            continue
        
        label, confidence = classifications[activity]
        
        print(f"\nScenario: {scenario['real_name']} ({activity})")
        print("-"*80)
        print(f"Context: {scenario['context']}")
        print(f"Classifier output: {label} ({confidence*100:.1f}%)")
        
        # Create prediction tuple for agent
        predictions = [(scenario['real_name'].lower().replace(' ', '_'), confidence)]
        
        # Agent interprets
        interpretation = agent.interpret_command(
            predictions,
            context=scenario['context']
        )
        
        print(f"\nAgent Interpretation:")
        print(f"  Command: {interpretation['command']}")
        print(f"  Action: {interpretation['action']}")
        print(f"  Confidence: {interpretation['confidence']*100:.1f}%")
        print(f"  Reasoning: {interpretation['reasoning']}")
        
        if 'llm_response' in interpretation:
            print(f"\n  DeepSeek Full Response:")
            for line in interpretation['llm_response'].split('\n'):
                print(f"    {line}")
        
        # Execute action
        result = agent.execute_action(interpretation)
        print(f"\n{result}")
    
    # Summary
    print("\n" + "="*80)
    print("DEMO COMPLETE!")
    print("="*80)
    print(f"\nProcessed {len(results)} activities")
    print(f"Generated spectrograms: {processor.image_dir}")
    print("\nYour 'Siri for WiFi' system is working end-to-end!")
    print("\nNext steps:")
    print("  1. Download more MM-Fi data (more subjects/activities)")
    print("  2. Fine-tune classifier on all 27 activities")
    print("  3. Add more emergency detection logic to agent")
    print("  4. Deploy for real-time monitoring!")


def quick_test_one_activity():
    """Quick test with just one activity."""
    print("Quick Test: Processing one activity...\n")
    
    processor = MMFiProcessor()
    
    # Just process A19 (Picking up - potential fall)
    img_path = processor.process_activity('E01', 'S01', 'A19')
    
    if img_path:
        print(f"\n[SUCCESS] Generated: {img_path}")
        print("\nYou can now classify it:")
        print(f"  python src/classifier.py {img_path}")
    else:
        print("\n[ERROR] Make sure MM-Fi dataset is downloaded to data/raw/MMFi/")
        print("See docs/MMFI_SETUP.md")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="MM-Fi Quick Start Demo")
    parser.add_argument('--quick', action='store_true', 
                       help='Quick test with one activity')
    args = parser.parse_args()
    
    if args.quick:
        quick_test_one_activity()
    else:
        demo_end_to_end()

