"""
"Siri for WiFi" - Complete Integrated Workflow

This script demonstrates the full three-stage pipeline:
Stage 1: RF "Ear" → Preprocessing
Stage 2: Command Classifier → Inference
Stage 3: Agentic Logic → Action Execution
"""

from pathlib import Path
from typing import Optional
import sys

# Import our components
from classifier import RFCommandClassifier
from agent_ai import RFCommandAgent


class SiriForWiFi:
    """
    Complete RF-based voice assistant system.
    
    RF Signals → Spectrogram → Classifier → LLM Agent → Action
    """
    
    def __init__(
        self,
        classifier_model_path: Optional[str] = None,
        use_llm_agent: bool = True,
        llm_provider: str = "deepseek",
        confidence_threshold: float = 0.7,
        deepseek_api_key: Optional[str] = None
    ):
        """
        Initialize the Siri for WiFi system.
        
        Args:
            classifier_model_path: Path to fine-tuned classifier weights
            use_llm_agent: Use LLM for intelligent action selection
            llm_provider: LLM provider ("deepseek" or "claude")
            confidence_threshold: Confidence threshold for direct execution
            deepseek_api_key: DeepSeek API key (optional)
        """
        print("Initializing Siri for WiFi...")
        
        # Stage 2: Classifier
        print("  [Stage 2] Loading command classifier...")
        self.classifier = RFCommandClassifier(
            model_path=classifier_model_path,
            use_pretrained=True
        )
        
        # Stage 3: Agent
        print("  [Stage 3] Initializing agentic AI...")
        self.agent = RFCommandAgent(
            use_llm=use_llm_agent,
            llm_provider=llm_provider,
            confidence_threshold=confidence_threshold,
            api_key=deepseek_api_key
        )
        
        print("✓ System ready!\n")
    
    def process_rf_signal(
        self,
        spectrogram_path: str,
        context: Optional[str] = None,
        verbose: bool = True
    ) -> dict:
        """
        Complete pipeline: spectrogram → prediction → action.
        
        Args:
            spectrogram_path: Path to RF spectrogram image
            context: Additional context for the agent
            verbose: Print detailed information
            
        Returns:
            Dict with classification, interpretation, and execution result
        """
        if verbose:
            print("="*70)
            print(f"Processing: {Path(spectrogram_path).name}")
            print("="*70)
        
        # Stage 2: Classify the RF signal
        if verbose:
            print("\n[Stage 2: Classification]")
        
        predictions = self.classifier.predict(spectrogram_path, top_k=3)
        
        if verbose:
            print("Top-3 Predictions:")
            for i, (cmd, conf) in enumerate(predictions, 1):
                print(f"  {i}. {cmd:15s} {conf*100:5.2f}%")
        
        # Stage 3: Interpret and decide action
        if verbose:
            print("\n[Stage 3: Agentic Interpretation]")
        
        interpretation = self.agent.interpret_command(predictions, context)
        
        if verbose:
            print(f"Command: {interpretation['command']}")
            print(f"Action: {interpretation['action']}")
            print(f"Confidence: {interpretation['confidence']*100:.1f}%")
            print(f"Reasoning: {interpretation['reasoning']}")
            if interpretation['requires_confirmation']:
                print("⚠ Action requires confirmation")
        
        # Execute
        if verbose:
            print("\n[Execution]")
        
        execution_result = self.agent.execute_action(interpretation)
        
        if verbose:
            print(execution_result)
            print()
        
        return {
            "spectrogram": spectrogram_path,
            "predictions": predictions,
            "interpretation": interpretation,
            "execution": execution_result,
            "success": interpretation['action'] != "unknown"
        }
    
    def process_batch(
        self,
        spectrogram_dir: str,
        context: Optional[str] = None
    ) -> list:
        """
        Process multiple RF spectrograms.
        
        Args:
            spectrogram_dir: Directory containing spectrogram images
            context: Context for the batch
            
        Returns:
            List of processing results
        """
        spectrogram_paths = list(Path(spectrogram_dir).glob("*.png"))
        
        if not spectrogram_paths:
            print(f"No spectrograms found in {spectrogram_dir}")
            return []
        
        print(f"Processing {len(spectrogram_paths)} spectrograms...\n")
        
        results = []
        for path in spectrogram_paths:
            result = self.process_rf_signal(str(path), context, verbose=False)
            results.append(result)
            
            # Print summary
            interp = result['interpretation']
            print(f"✓ {path.name:30s} → {interp['command']:12s} "
                  f"({interp['confidence']*100:5.1f}%) → {interp['action']}")
        
        return results


def demo_workflow():
    """Demo the complete workflow."""
    print("="*70)
    print("SIRI FOR WIFI - COMPLETE WORKFLOW DEMO")
    print("="*70 + "\n")
    
    # Initialize system with DeepSeek (cheaper!)
    system = SiriForWiFi(
        use_llm_agent=True,
        llm_provider="deepseek"
    )
    
    # Check for sample data
    data_dir = Path("data/images")
    
    if data_dir.exists() and list(data_dir.glob("*.png")):
        print("Found sample spectrograms. Processing...\n")
        system.process_batch(str(data_dir))
    else:
        print("No sample data found. Demonstrating with simulated predictions...\n")
        
        # Simulate a high-confidence prediction
        print("Simulation 1: High Confidence Emergency")
        print("-" * 70)
        print("Simulated RF signal → 'ambulance' detected (95% confidence)")
        
        # Manually create interpretation
        predictions = [("ambulance", 0.95), ("help", 0.03), ("emergency", 0.02)]
        interpretation = system.agent.interpret_command(predictions, None)
        result = system.agent.execute_action(interpretation)
        
        print(f"Action: {interpretation['action']}")
        print(f"Result: {result}\n")
        
        # Simulate an ambiguous prediction
        print("Simulation 2: Ambiguous Command")
        print("-" * 70)
        print("Simulated RF signal → uncertain between 'help' and 'home'")
        
        predictions = [("help", 0.52), ("home", 0.48)]
        interpretation = system.agent.interpret_command(predictions, "User in living room")
        result = system.agent.execute_action(interpretation)
        
        print(f"Action: {interpretation['action']}")
        print(f"Result: {result}\n")
    
    print("="*70)
    print("Next Steps:")
    print("1. Generate spectrograms using Stage 1 preprocessing")
    print("2. Fine-tune the classifier on RVTALL data")
    print("3. Deploy in real-world RF sensing scenario")
    print("="*70)


if __name__ == "__main__":
    demo_workflow()

