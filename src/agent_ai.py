"""
Stage 3: The "Agentic Logic" (The Brain)

This module implements the LLM-based agent that takes classifier outputs
and decides what actions to execute, with fuzzy matching and context awareness.
"""

import os
from typing import List, Tuple, Dict, Optional, Any
from pathlib import Path

# Conditional imports for AI components
try:
    from langchain.agents import Tool, AgentExecutor, create_structured_chat_agent
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain.memory import ConversationBufferMemory
    from langchain_openai import ChatOpenAI  # For DeepSeek (OpenAI-compatible)
    
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("Warning: LangChain not available. Install: pip install langchain langchain-openai")

# Try to import Claude (optional)
try:
    from langchain_anthropic import ChatAnthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False


class RFCommandAgent:
    """
    Agentic AI that interprets RF command classifications and executes actions.
    
    Key features:
    - Fuzzy matching for uncertain predictions
    - Context-aware decision making
    - Tool execution based on commands
    - Handles ambiguity with LLM reasoning
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
        llm_provider: str = "deepseek",  # "deepseek" or "claude"
        model_name: Optional[str] = None,
        confidence_threshold: float = 0.7,
        api_key: Optional[str] = None
    ):
        """
        Initialize the RF command agent.
        
        Args:
            use_llm: Use LLM for fuzzy matching and reasoning
            llm_provider: LLM provider ("deepseek" or "claude")
            model_name: Model name (defaults based on provider)
            confidence_threshold: Minimum confidence to execute without LLM
            api_key: API key (or set via environment variable)
        """
        self.use_llm = use_llm and LANGCHAIN_AVAILABLE
        self.confidence_threshold = confidence_threshold
        self.context_history = []
        self.llm_provider = llm_provider.lower()
        
        if self.use_llm:
            if self.llm_provider == "deepseek":
                # DeepSeek configuration
                deepseek_api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
                if not deepseek_api_key:
                    print("Warning: DEEPSEEK_API_KEY not set. LLM features disabled.")
                    print("Set with: export DEEPSEEK_API_KEY='your-key'")
                    self.use_llm = False
                else:
                    # DeepSeek uses OpenAI-compatible API
                    model = model_name or "deepseek-chat"
                    self.llm = ChatOpenAI(
                        model=model,
                        openai_api_key=deepseek_api_key,
                        openai_api_base="https://api.deepseek.com/v1",
                        temperature=0.3,  # Lower for more deterministic responses
                        max_tokens=500
                    )
                    print(f"✓ Initialized with DeepSeek ({model}) - Very cost effective!")
                    
            elif self.llm_provider == "claude":
                # Claude configuration
                if not CLAUDE_AVAILABLE:
                    print("Warning: langchain-anthropic not installed.")
                    print("Install: pip install langchain-anthropic")
                    self.use_llm = False
                else:
                    anthropic_api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
                    if not anthropic_api_key:
                        print("Warning: ANTHROPIC_API_KEY not set. LLM features disabled.")
                        self.use_llm = False
                    else:
                        model = model_name or "claude-3-5-sonnet-20241022"
                        self.llm = ChatAnthropic(
                            model=model,
                            anthropic_api_key=anthropic_api_key,
                            temperature=0.3
                        )
                        print(f"✓ Initialized with Claude ({model})")
            else:
                print(f"Warning: Unknown LLM provider '{llm_provider}'. Use 'deepseek' or 'claude'.")
                self.use_llm = False
        
        if not self.use_llm:
            print("Running in rule-based mode (no LLM)")
    
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
        
        # High confidence - execute directly
        if top_confidence >= self.confidence_threshold:
            return {
                "action": self.ACTION_MAP.get(top_command, "unknown"),
                "command": top_command,
                "confidence": top_confidence,
                "reasoning": "High confidence prediction",
                "requires_confirmation": False
            }
        
        # Low confidence - use LLM for reasoning if available
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
        
        Uses similarity and context to make best guess.
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
                # Boost emergency-related commands in emergency context
                return {
                    "action": self.ACTION_MAP.get(top_command, "trigger_emergency_services"),
                    "command": top_command,
                    "confidence": min(top_confidence + 0.2, 1.0),  # Boost confidence
                    "reasoning": f"Emergency context detected, boosting {top_command}",
                    "requires_confirmation": False
                }
        
        # Default: execute with caution flag
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
        Use LLM to reason about uncertain predictions.
        
        The LLM can handle:
        - Fuzzy matching (amblance -> ambulance)
        - Context awareness
        - Multi-step reasoning
        """
        # Format predictions for LLM
        pred_str = "\n".join([
            f"- {cmd}: {conf*100:.1f}% confidence"
            for cmd, conf in predictions
        ])
        
        context_str = f"\nContext: {context}" if context else ""
        
        prompt = f"""You are an AI assistant for an RF-based voice command system.
The RF signal classifier has detected these possible commands:

{pred_str}{context_str}

Your task:
1. Determine the most likely intended command
2. Consider context if provided
3. Handle fuzzy matches (e.g., "amblance" likely means "ambulance")
4. Recommend an action from: {list(self.ACTION_MAP.keys())}

Respond in this format:
Command: <most likely command>
Action: <action to take>
Confidence: <your confidence 0-1>
Reasoning: <brief explanation>
"""
        
        try:
            response = self.llm.invoke(prompt)
            content = response.content
            
            # Parse LLM response
            lines = content.strip().split('\n')
            result = {}
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    result[key] = value
            
            command = result.get('command', predictions[0][0])
            action = result.get('action', self.ACTION_MAP.get(command, 'unknown'))
            confidence = float(result.get('confidence', predictions[0][1]))
            reasoning = result.get('reasoning', 'LLM interpretation')
            
            return {
                "action": action,
                "command": command,
                "confidence": confidence,
                "reasoning": reasoning,
                "requires_confirmation": confidence < 0.8,
                "llm_response": content
            }
        
        except Exception as e:
            print(f"LLM interpretation failed: {e}")
            return self._fuzzy_match(predictions, context)
    
    def execute_action(self, interpretation: Dict[str, Any]) -> str:
        """
        Execute the interpreted action.
        
        In a real system, this would trigger actual functions.
        Here we simulate the execution.
        """
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
            return f"🚨 EMERGENCY: Calling ambulance (triggered by '{command}' with {confidence*100:.1f}% confidence)"
        
        elif action == "trigger_help_alert":
            return f"📞 ALERT: Sending help notification ('{command}' - {confidence*100:.1f}%)"
        
        elif action == "trigger_police_alert":
            return f"🚔 POLICE: Contacting authorities ('{command}' - {confidence*100:.1f}%)"
        
        elif action.startswith("navigate_"):
            direction = action.replace("navigate_", "")
            return f"🧭 NAVIGATE: Moving {direction} ('{command}' - {confidence*100:.1f}%)"
        
        elif action == "request_clarification":
            alternatives = interpretation.get("alternatives", [])
            return f"❓ CLARIFY: Did you say {' or '.join(alternatives)}? (Confidence: {confidence*100:.1f}%)"
        
        elif action == "stop_current_action":
            return f"⛔ STOP: Halting current action ('{command}' - {confidence*100:.1f}%)"
        
        else:
            return f"Action: {action} | Command: {command} | Confidence: {confidence*100:.1f}%"


def demo_agent():
    """Demo the agent with simulated classifier outputs."""
    print("="*70)
    print("RF COMMAND AGENT DEMO")
    print("="*70 + "\n")
    
    # Try DeepSeek first (cheaper), fall back to Claude, then rule-based
    agent = RFCommandAgent(
        use_llm=True,
        llm_provider="deepseek",  # Use DeepSeek by default
        confidence_threshold=0.7
    )
    
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
            "name": "Fuzzy Match Needed",
            "predictions": [("amblance", 0.70), ("ambulance", 0.20), ("help", 0.10)],
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
        print(f"  Needs confirmation: {interpretation['requires_confirmation']}")
        
        # Execute
        result = agent.execute_action(interpretation)
        print(f"\nExecution Result:")
        print(f"  {result}")
    
    print("\n" + "="*70)
    print("Agent demo complete!")
    print("="*70)


if __name__ == "__main__":
    demo_agent()

