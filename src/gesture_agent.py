"""
Gesture-Based Agentic System
Maps physical gestures (detected via WiFi) to smart actions using DeepSeek LLM
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from agent_simple import RFCommandAgent


class GestureAgent:
    """
    Intelligent gesture recognition agent with context-aware actions.
    
    Uses DeepSeek LLM to make smart decisions based on:
    - Detected gesture
    - Current context (time, location, user state)
    - User preferences
    - Safety rules
    """
    
    def __init__(
        self,
        config_path: str = 'config/gesture_actions.yaml',
        api_key: Optional[str] = None
    ):
        """
        Initialize gesture agent.
        
        Args:
            config_path: Path to gesture action configuration
            api_key: DeepSeek API key
        """
        self.api_key = api_key or os.getenv('DEEPSEEK_API_KEY')
        self.llm_agent = RFCommandAgent(use_llm=True, api_key=self.api_key)
        
        # Load configuration
        self.config = self._load_config(config_path)
        self.gestures = self.config.get('gestures', {})
        self.contexts = self.config.get('contexts', {})
        
        # Activity name mapping
        self.activity_names = {
            'A01': 'stretching', 'A17': 'wave_left', 'A18': 'wave_right',
            'A13': 'raise_hand_left', 'A14': 'raise_hand_right',
            'A19': 'picking_up', 'A27': 'bowing', 'A12': 'squat',
            'A20': 'throw_left', 'A21': 'throw_right',
            'A22': 'kick_left', 'A23': 'kick_right', 'A26': 'jumping'
        }
        
        print(f"[OK] Gesture Agent initialized")
        print(f"    Gestures configured: {len(self.gestures)}")
        print(f"    Contexts: {list(self.contexts.keys())}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load gesture action configuration."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"[OK] Loaded config: {config_path}")
            return config
        except FileNotFoundError:
            print(f"[WARNING] Config not found: {config_path}")
            return {'gestures': {}, 'contexts': {}}
    
    def get_context(self) -> Dict[str, Any]:
        """
        Get current context information.
        
        Returns:
            Dict with context data (time, location, user state, etc.)
        """
        now = datetime.now()
        hour = now.hour
        
        # Determine time of day
        if 6 <= hour < 12:
            time_of_day = "morning"
        elif 12 <= hour < 18:
            time_of_day = "afternoon"
        elif 18 <= hour < 23:
            time_of_day = "evening"
        else:
            time_of_day = "night"
        
        # Build context
        context = {
            'time_of_day': time_of_day,
            'hour': hour,
            'day_of_week': now.strftime('%A'),
            'elderly_care_mode': self.contexts.get('elderly_care_mode', {}).get('enabled', False),
            'smart_home_mode': self.contexts.get('smart_home_mode', {}).get('enabled', False),
            'workout_mode': self.contexts.get('workout_mode', {}).get('enabled', False)
        }
        
        return context
    
    def interpret_gesture(
        self,
        activity_code: str,
        confidence: float,
        additional_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Interpret a detected gesture and decide action using LLM.
        
        Args:
            activity_code: MM-Fi activity code (e.g., 'A17')
            confidence: Classifier confidence (0-1)
            additional_context: Extra context information
            
        Returns:
            Dict with action decision
        """
        # Get gesture configuration
        gesture_config = self.gestures.get(activity_code, {})
        gesture_name = self.activity_names.get(activity_code, activity_code)
        
        # Get current context
        context = self.get_context()
        if additional_context:
            context.update(additional_context)
        
        # Build context string for LLM
        context_str = f"""
Gesture: {gesture_name} ({activity_code})
Confidence: {confidence*100:.1f}%
Time: {context['time_of_day']} ({context['hour']}:00)
Day: {context['day_of_week']}
Modes: elderly_care={context['elderly_care_mode']}, smart_home={context['smart_home_mode']}, workout={context['workout_mode']}
"""
        
        if gesture_config:
            context_str += f"\nConfigured action: {gesture_config.get('action', 'none')}"
            context_str += f"\nDescription: {gesture_config.get('description', 'N/A')}"
        
        if additional_context:
            context_str += f"\nAdditional: {additional_context}"
        
        # Use LLM to interpret
        interpretation = self.llm_agent.interpret_command(
            predictions=[(gesture_name, confidence)],
            context=context_str
        )
        
        # Add configured action if available
        interpretation['configured_action'] = gesture_config.get('action', 'none')
        interpretation['gesture_config'] = gesture_config
        interpretation['context'] = context
        
        return interpretation
    
    def execute_gesture_action(
        self,
        activity_code: str,
        confidence: float,
        additional_context: Optional[Dict] = None
    ) -> str:
        """
        Complete gesture → action pipeline.
        
        Args:
            activity_code: MM-Fi activity code
            confidence: Classifier confidence
            additional_context: Extra context
            
        Returns:
            Execution result string
        """
        print(f"\n{'='*70}")
        print(f"GESTURE DETECTED: {activity_code}")
        print(f"Confidence: {confidence*100:.1f}%")
        print(f"{'='*70}\n")
        
        # Interpret gesture
        interpretation = self.interpret_gesture(
            activity_code,
            confidence,
            additional_context
        )
        
        print(f"LLM Interpretation:")
        print(f"  Command: {interpretation['command']}")
        print(f"  Action: {interpretation['action']}")
        print(f"  Configured: {interpretation['configured_action']}")
        print(f"  Reasoning: {interpretation['reasoning']}\n")
        
        # Execute action
        result = self._execute_action(interpretation)
        
        print(f"Execution: {result}\n")
        print(f"{'='*70}\n")
        
        return result
    
    def _execute_action(self, interpretation: Dict[str, Any]) -> str:
        """Execute the interpreted action."""
        action = interpretation.get('configured_action', 'none')
        gesture_name = interpretation['command']
        confidence = interpretation['confidence']
        context = interpretation['context']
        
        # Smart home actions
        if action == 'turn_on_lights':
            if context['time_of_day'] == 'night':
                return f"[SMART HOME] Turning on dim lights (night mode) - {gesture_name}"
            elif context['time_of_day'] == 'evening':
                return f"[SMART HOME] Turning on warm lights (evening mode) - {gesture_name}"
            else:
                return f"[SMART HOME] Turning on lights - {gesture_name}"
        
        elif action == 'turn_off_lights':
            return f"[SMART HOME] Turning off lights - {gesture_name}"
        
        elif action in ['increase_volume', 'volume_up']:
            return f"[MEDIA] Volume +20% - {gesture_name}"
        
        elif action in ['decrease_volume', 'volume_down']:
            return f"[MEDIA] Volume -20% - {gesture_name}"
        
        elif action == 'next_track':
            return f"[MEDIA] Skipping to next track - {gesture_name}"
        
        elif action == 'previous_track':
            return f"[MEDIA] Going to previous track - {gesture_name}"
        
        # Emergency actions
        elif action == 'check_for_fall':
            if context['elderly_care_mode']:
                return f"[!] ALERT: Monitoring for fall recovery - {gesture_name} (elderly mode)"
            else:
                return f"[INFO] Normal bending motion detected - {gesture_name}"
        
        elif action == 'emergency_check':
            return f"[!] EMERGENCY: Sudden downward motion, checking status - {gesture_name}"
        
        elif action == 'trigger_emergency_services':
            return f"[!!!] CRITICAL: Calling emergency services - {gesture_name}"
        
        # Exercise tracking
        elif action == 'log_exercise':
            return f"[FITNESS] Logged exercise: {gesture_name}"
        
        elif action == 'start_workout_mode':
            return f"[FITNESS] Workout mode activated - {gesture_name}"
        
        # Default
        else:
            return f"[ACTION] {action} - {gesture_name} ({confidence*100:.1f}%)"


def demo_gesture_agent():
    """Demo the gesture-based agentic system."""
    print("="*70)
    print("GESTURE-BASED AGENTIC SYSTEM DEMO")
    print("="*70 + "\n")
    
    # Initialize agent
    agent = GestureAgent(
        config_path='config/gesture_actions.yaml',
        api_key='sk-71f67cd6e695467eb0251aef4f05d734'
    )
    
    print("\n" + "="*70)
    print("TEST SCENARIOS")
    print("="*70 + "\n")
    
    # Scenario 1: Wave to turn on lights
    print("Scenario 1: Smart Home Control")
    print("-"*70)
    result = agent.execute_gesture_action(
        activity_code='A17',  # Wave left
        confidence=0.92,
        additional_context={'room': 'living_room'}
    )
    
    # Scenario 2: Wave to turn off lights
    print("\nScenario 2: Turn Off Lights")
    print("-"*70)
    result = agent.execute_gesture_action(
        activity_code='A18',  # Wave right
        confidence=0.88,
        additional_context={'room': 'living_room', 'user_leaving': True}
    )
    
    # Scenario 3: Raise hand for volume
    print("\nScenario 3: Media Control")
    print("-"*70)
    result = agent.execute_gesture_action(
        activity_code='A13',  # Raise left hand
        confidence=0.85,
        additional_context={'music_playing': True}
    )
    
    # Scenario 4: Emergency - sudden bend
    print("\nScenario 4: Emergency Detection")
    print("-"*70)
    result = agent.execute_gesture_action(
        activity_code='A19',  # Picking up / bending
        confidence=0.78,
        additional_context={
            'user_age': 75,
            'alone_at_home': True,
            'time_since_last_motion': 300  # 5 minutes
        }
    )
    
    # Scenario 5: Exercise tracking
    print("\nScenario 5: Exercise Tracking")
    print("-"*70)
    result = agent.execute_gesture_action(
        activity_code='A26',  # Jumping
        confidence=0.91,
        additional_context={'workout_session_active': True}
    )
    
    print("\n" + "="*70)
    print("DEMO COMPLETE!")
    print("="*70)
    print("\nYour gesture-based agentic system is working!")
    print("\nCustomize actions in: config/gesture_actions.yaml")


if __name__ == '__main__':
    demo_gesture_agent()

