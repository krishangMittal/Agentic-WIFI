"""
Stage 3: The "Agentic Logic" (The Brain) - Simple Version

Direct API calls to DeepSeek without LangChain dependency.
Much simpler, faster, and more transparent.
"""

import os
import json
import requests
from typing import List, Tuple, Dict, Optional, Any


class RFCommandAgent:
    """
    Agentic AI that interprets RF command classifications and executes actions.
    
    Simple version using direct API calls (no LangChain needed).
    """
    
    # Command to action mapping
    ACTION_MAP = {
        "help": "trigger_help_alert",
        "ambulance": "trigger_emergency_services",
        "police": "trigger_police_alert",
        "fire": "trigger_fire_alert",
        "emergency": "trigger_emergency_services",
        "stop": "stop_current_action",
        "yes": "confirm_action",
        "no": "cancel_action",
        "left": "navigate_left",
        "right": "navigate_right",
        "forward": "navigate_forward",
        "backward": "navigate_backward",
        "up": "adjust_up",
        "down": "adjust_down",
        "home": "return_home"
    }
    
    def __init__(
        self,
        use_llm: bool = True,
        api_key: Optional[str] = None,
        confidence_threshold: float = 0.7
    ):
        """
        Initialize the RF command agent.
        
        Args:
            use_llm: Use LLM for fuzzy matching and reasoning
            api_key: DeepSeek API key (or set DEEPSEEK_API_KEY env var)
            confidence_threshold: Minimum confidence to execute without LLM
        """
        self.use_llm = use_llm
        self.confidence_threshold = confidence_threshold
        self.context_history = []
        
        if self.use_llm:
            self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
            if not self.api_key:
                print("Warning: DEEPSEEK_API_KEY not set. LLM features disabled.")
                print("Set with: export DEEPSEEK_API_KEY='your-key'")
                self.use_llm = False
            else:
                print(f"[OK] DeepSeek API key configured (ends with ...{self.api_key[-8:]})")
        
        if not self.use_llm:
            print("Running in rule-based mode (no LLM)")
    
    def _call_deepseek(self, prompt: str) -> str:
        """
        Call DeepSeek API directly (no LangChain needed).
        
        Args:
            prompt: The prompt to send to DeepSeek
            
        Returns:
            Response text from DeepSeek
        """
        url = "https://api.deepseek.com/v1/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 500
        }
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
        
        except requests.exceptions.RequestException as e:
            print(f"Error calling DeepSeek API: {e}")
            return ""
    
    def interpret_command(
        self,
        predictions: List[Tuple[str, float]],
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Interpret classifier predictions and decide on action.
        
        Args:
            predictions: List of (command, confidence) from classifier
            context: Additional context (e.g., "user in medical emergency")
            
        Returns:
            Dict with action, confidence, reasoning
        """
        top_command, top_confidence = predictions[0]
        
        # Check if command is valid (exists in ACTION_MAP)
        is_valid_command = top_command in self.ACTION_MAP
        
        # High confidence AND valid command - execute directly
        if top_confidence >= self.confidence_threshold and is_valid_command:
            return {
                "action": self.ACTION_MAP.get(top_command),
                "command": top_command,
                "confidence": top_confidence,
                "reasoning": "High confidence prediction",
                "requires_confirmation": False
            }
        
        # Low confidence OR invalid command - use LLM for reasoning if available
        if self.use_llm:
            return self._llm_interpret(predictions, context)
        else:
            return self._fuzzy_match(predictions, context)
    
    def _fuzzy_match(
        self,
        predictions: List[Tuple[str, float]],
        context: Optional[str]
    ) -> Dict[str, Any]:
        """
        Simple fuzzy matching without LLM.
        """
        top_command, top_confidence = predictions[0]
        second_command, second_confidence = predictions[1] if len(predictions) > 1 else ("", 0.0)
        
        # If top two are very close, flag for confirmation
        if second_confidence > 0.0 and (top_confidence - second_confidence) < 0.1:
            return {
                "action": "request_clarification",
                "command": f"{top_command} or {second_command}",
                "confidence": top_confidence,
                "reasoning": f"Ambiguous between {top_command} and {second_command}",
                "requires_confirmation": True,
                "alternatives": [top_command, second_command]
            }
        
        # Context-based boosting
        if context and "emergency" in context.lower():
            if "help" in top_command or "ambulance" in top_command:
                return {
                    "action": self.ACTION_MAP.get(top_command, "trigger_emergency_services"),
                    "command": top_command,
                    "confidence": min(top_confidence + 0.2, 1.0),
                    "reasoning": f"Emergency context detected, boosting {top_command}",
                    "requires_confirmation": False
                }
        
        # Default
        return {
            "action": self.ACTION_MAP.get(top_command, "unknown"),
            "command": top_command,
            "confidence": top_confidence,
            "reasoning": "Best guess with moderate confidence",
            "requires_confirmation": True
        }
    
    def _llm_interpret(
        self,
        predictions: List[Tuple[str, float]],
        context: Optional[str]
    ) -> Dict[str, Any]:
        """
        Use DeepSeek LLM to reason about uncertain predictions.
        
        Direct API call - no LangChain needed!
        """
        # Format predictions for LLM
        pred_str = "\n".join([
            f"- {cmd}: {conf*100:.1f}% confidence"
            for cmd, conf in predictions
        ])
        
        context_str = f"\nContext: {context}" if context else ""
        
        prompt = f"""You are an AI assistant for an RF-based voice command system.
The RF signal classifier detected these possible commands:

{pred_str}{context_str}

Your task:
1. Determine the most likely intended command
2. Consider context if provided
3. Handle fuzzy matches (e.g., "amblance" likely means "ambulance")
4. Recommend an action from: {list(self.ACTION_MAP.keys())}

Respond in EXACTLY this format:
Command: <most likely command>
Action: <action to take>
Confidence: <your confidence 0-1>
Reasoning: <brief explanation>
"""
        
        # Call DeepSeek directly
        response = self._call_deepseek(prompt)
        
        if not response:
            # Fallback to rule-based if API fails
            return self._fuzzy_match(predictions, context)
        
        # Parse LLM response
        try:
            lines = response.strip().split('\n')
            result = {}
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    result[key] = value
            
            command = result.get('command', predictions[0][0])
            action = result.get('action', self.ACTION_MAP.get(command, 'unknown'))
            
            # Parse confidence
            conf_str = result.get('confidence', str(predictions[0][1]))
            try:
                confidence = float(conf_str)
            except:
                confidence = predictions[0][1]
            
            reasoning = result.get('reasoning', 'LLM interpretation')
            
            return {
                "action": action,
                "command": command,
                "confidence": confidence,
                "reasoning": reasoning,
                "requires_confirmation": confidence < 0.8,
                "llm_response": response
            }
        
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return self._fuzzy_match(predictions, context)
    
    def execute_action(self, interpretation: Dict[str, Any]) -> str:
        """Execute the interpreted action."""
        action = interpretation["action"]
        command = interpretation["command"]
        confidence = interpretation["confidence"]
        
        # Store in context history
        self.context_history.append({
            "command": command,
            "action": action,
            "confidence": confidence
        })
        
        # Simulate action execution
        if action == "trigger_emergency_services":
            return f"[!] EMERGENCY: Calling ambulance (triggered by '{command}' with {confidence*100:.1f}% confidence)"
        
        elif action == "trigger_help_alert":
            return f"[!] ALERT: Sending help notification ('{command}' - {confidence*100:.1f}%)"
        
        elif action == "trigger_police_alert":
            return f"[!] POLICE: Contacting authorities ('{command}' - {confidence*100:.1f}%)"
        
        elif action.startswith("navigate_"):
            direction = action.replace("navigate_", "")
            return f"[>] NAVIGATE: Moving {direction} ('{command}' - {confidence*100:.1f}%)"
        
        elif action == "request_clarification":
            alternatives = interpretation.get("alternatives", [])
            return f"[?] CLARIFY: Did you say {' or '.join(alternatives)}? (Confidence: {confidence*100:.1f}%)"
        
        elif action == "stop_current_action":
            return f"[X] STOP: Halting current action ('{command}' - {confidence*100:.1f}%)"
        
        else:
            return f"Action: {action} | Command: {command} | Confidence: {confidence*100:.1f}%"


def demo_agent():
    """Demo the agent with simulated classifier outputs."""
    print("="*70)
    print("RF COMMAND AGENT DEMO (Simple Version - No LangChain!)")
    print("="*70 + "\n")
    
    # Check for API key
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("[WARNING] DEEPSEEK_API_KEY not set.")
        print("To enable LLM features:")
        print("  export DEEPSEEK_API_KEY='your-key-here'")
        print("\nRunning in rule-based mode for demo...\n")
    
    agent = RFCommandAgent(use_llm=bool(api_key))
    
    # Test scenarios
    scenarios = [
        {
            "name": "High Confidence - Clear Command",
            "predictions": [("ambulance", 0.95), ("help", 0.03), ("emergency", 0.02)],
            "context": None
        },
        {
            "name": "Low Confidence - Ambiguous",
            "predictions": [("help", 0.45), ("home", 0.42), ("stop", 0.13)],
            "context": None
        },
        {
            "name": "Emergency Context Boost",
            "predictions": [("help", 0.65), ("home", 0.25), ("yes", 0.10)],
            "context": "User has fallen and pressed emergency button"
        },
        {
            "name": "Fuzzy Match Needed (LLM test)",
            "predictions": [("amblance", 0.70), ("help", 0.20), ("emergency", 0.10)],
            "context": "Medical emergency detected"
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\nScenario {i}: {scenario['name']}")
        print("-" * 70)
        print(f"Predictions: {scenario['predictions']}")
        if scenario['context']:
            print(f"Context: {scenario['context']}")
        
        # Interpret
        interpretation = agent.interpret_command(
            scenario['predictions'],
            scenario['context']
        )
        
        print(f"\nInterpretation:")
        print(f"  Command: {interpretation['command']}")
        print(f"  Action: {interpretation['action']}")
        print(f"  Confidence: {interpretation['confidence']*100:.1f}%")
        print(f"  Reasoning: {interpretation['reasoning']}")
        
        if agent.use_llm and 'llm_response' in interpretation:
            print(f"\n  LLM Response:")
            for line in interpretation['llm_response'].split('\n'):
                print(f"    {line}")
        
        # Execute
        result = agent.execute_action(interpretation)
        print(f"\nExecution: {result}")
    
    print("\n" + "="*70)
    print("Demo complete!")
    print("="*70)


if __name__ == "__main__":
    demo_agent()

