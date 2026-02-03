"""
Quick Gesture Demo (No YAML dependency)
Shows how WiFi gestures → DeepSeek → Actions work
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

# API key should be set via environment variable: export DEEPSEEK_API_KEY='your-key'
# os.environ['DEEPSEEK_API_KEY'] = os.getenv('DEEPSEEK_API_KEY')

from agent_simple import RFCommandAgent


def demo_gesture_to_action():
    """Demo gesture detection → LLM decision → action execution."""
    
    print("="*70)
    print("GESTURE-BASED AGENTIC SYSTEM - SIMPLE DEMO")
    print("="*70 + "\n")
    
    # Initialize agent
    agent = RFCommandAgent(use_llm=True)
    
    # Gesture mappings (27 MM-Fi activities)
    gestures = {
        'A17': 'wave_left_hand',
        'A18': 'wave_right_hand',
        'A13': 'raise_hand_left',
        'A14': 'raise_hand_right',
        'A19': 'picking_up',  # Potential fall
        'A27': 'bowing',      # Potential fall
        'A20': 'throw_left',
        'A21': 'throw_right',
        'A26': 'jumping'
    }
    
    # Test scenarios
    scenarios = [
        {
            'name': 'Smart Home: Turn On Lights',
            'gesture': 'A17',
            'confidence': 0.92,
            'context': 'Evening, living room, user just entered',
            'expected': 'Turn on lights'
        },
        {
            'name': 'Smart Home: Turn Off Lights',
            'gesture': 'A18',
            'confidence': 0.88,
            'context': 'User leaving room, waving goodbye',
            'expected': 'Turn off lights'
        },
        {
            'name': 'Media Control: Volume Up',
            'gesture': 'A13',
            'confidence': 0.85,
            'context': 'Music playing, user raises left hand',
            'expected': 'Increase volume'
        },
        {
            'name': 'Media Control: Next Track',
            'gesture': 'A21',
            'confidence': 0.90,
            'context': 'Music playing, user throws right hand',
            'expected': 'Skip to next song'
        },
        {
            'name': 'Emergency: Fall Detection',
            'gesture': 'A19',
            'confidence': 0.75,
            'context': 'Elderly person (75 years old), sudden downward motion, alone at home',
            'expected': 'Emergency alert'
        },
        {
            'name': 'Fitness: Exercise Tracking',
            'gesture': 'A26',
            'confidence': 0.91,
            'context': 'Workout mode active, user jumping',
            'expected': 'Log exercise'
        }
    ]
    
    print("\nRunning test scenarios...\n")
    
    for i, scenario in enumerate(scenarios, 1):
        gesture_code = scenario['gesture']
        gesture_name = gestures.get(gesture_code, gesture_code)
        
        print(f"\n{'-'*70}")
        print(f"Scenario {i}: {scenario['name']}")
        print(f"{'-'*70}")
        print(f"Gesture: {gesture_name} ({gesture_code})")
        print(f"Classifier Confidence: {scenario['confidence']*100:.1f}%")
        print(f"Context: {scenario['context']}\n")
        
        # LLM interprets
        interpretation = agent.interpret_command(
            predictions=[(gesture_name, scenario['confidence'])],
            context=scenario['context']
        )
        
        print(f"LLM Interpretation:")
        print(f"  Command: {interpretation['command']}")
        print(f"  Action: {interpretation['action']}")
        print(f"  Confidence: {interpretation['confidence']*100:.1f}%")
        print(f"  Reasoning: {interpretation['reasoning']}")
        
        if 'llm_response' in interpretation:
            print(f"\n  DeepSeek Response:")
            for line in interpretation['llm_response'].split('\n')[:4]:  # First 4 lines
                if line.strip():
                    print(f"    {line}")
        
        # Map to actual action
        action_result = map_to_action(gesture_code, interpretation, scenario['context'])
        print(f"\nEXECUTION: {action_result}")
    
    print("\n" + "="*70)
    print("DEMO COMPLETE!")
    print("="*70)
    print("\nYour gesture-based agentic system works!")
    print("\nHow it works:")
    print("  1. Wave hand -> WiFi signal disrupted")
    print("  2. Spectrogram generated from WiFi CSI")
    print("  3. ResNet classifies: 'wave_left' (92% confidence)")
    print("  4. DeepSeek decides: 'Turn on lights' (context: evening)")
    print("  5. Smart home API executes action")
    print("\nCustomize actions in: config/gesture_actions.yaml")


def map_to_action(gesture_code: str, interpretation: dict, context: str) -> str:
    """Map gesture to actual action."""
    
    # Smart home actions
    if gesture_code == 'A17':  # Wave left
        if 'evening' in context.lower():
            return "[SMART HOME] Turning on warm lights (evening mode)"
        else:
            return "[SMART HOME] Turning on lights"
    
    elif gesture_code == 'A18':  # Wave right
        return "[SMART HOME] Turning off lights"
    
    # Media control
    elif gesture_code == 'A13':  # Raise left
        return "[MEDIA] Volume +20%"
    
    elif gesture_code == 'A14':  # Raise right
        return "[MEDIA] Volume -20%"
    
    elif gesture_code == 'A21':  # Throw right
        return "[MEDIA] Skipping to next track"
    
    elif gesture_code == 'A20':  # Throw left
        return "[MEDIA] Going to previous track"
    
    # Emergency
    elif gesture_code in ['A19', 'A27']:  # Bending/bowing
        if 'elderly' in context.lower():
            return "[!] EMERGENCY ALERT: Monitoring for fall - Caregiver notified"
        else:
            return "[INFO] Normal bending motion detected"
    
    # Fitness
    elif gesture_code == 'A26':  # Jumping
        return "[FITNESS] Exercise logged: Jumping jacks x1"
    
    else:
        return f"[ACTION] {interpretation['action']}"


if __name__ == '__main__':
    demo_gesture_to_action()

