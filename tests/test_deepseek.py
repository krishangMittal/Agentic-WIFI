"""
Quick test of DeepSeek integration
"""
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set API key for this test
os.environ['DEEPSEEK_API_KEY'] = 'sk-71f67cd6e695467eb0251aef4f05d734'

# Import and test
from agent_simple import RFCommandAgent

print("="*70)
print("TESTING DEEPSEEK FUZZY MATCHING")
print("="*70 + "\n")

agent = RFCommandAgent(use_llm=True)

# Test fuzzy match: "amblance" should correct to "ambulance"
print("Test: Fuzzy match correction")
print("-" * 70)
predictions = [("amblance", 0.70), ("help", 0.20), ("emergency", 0.10)]
context = "Medical emergency detected"

print(f"Input predictions: {predictions}")
print(f"Context: {context}\n")

interpretation = agent.interpret_command(predictions, context)

print("LLM Interpretation:")
print(f"  Command: {interpretation['command']}")
print(f"  Action: {interpretation['action']}")
print(f"  Confidence: {interpretation['confidence']*100:.1f}%")
print(f"  Reasoning: {interpretation['reasoning']}\n")

if 'llm_response' in interpretation:
    print("Full LLM Response:")
    print(interpretation['llm_response'])

result = agent.execute_action(interpretation)
print(f"\nExecution: {result}")

# Check if it worked
if "ambulance" in interpretation['command'].lower():
    print("\n" + "="*70)
    print("[SUCCESS] Fuzzy matching worked!")
    print("'amblance' was correctly interpreted as 'ambulance'")
    print("="*70)
else:
    print("\n" + "="*70)
    print("[FAILED] Fuzzy matching did not work")
    print(f"Expected: ambulance, Got: {interpretation['command']}")
    print("="*70)

