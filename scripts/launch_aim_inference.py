"""
Launch Multi-Model Inference Extension for SNN.
Supports HuggingFace models and local checkpoints with conversation tracking.
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from aim_multi_model_inference import MultiModelInferenceExtension


def load_model_universal(model_id: str):
    """
    Universal model loader - supports both HuggingFace and local models.
    
    Args:
        model_id: Either HuggingFace model ID or path to local checkpoint
    
    Returns:
        (model, tokenizer) tuple
    """
    import torch
    
    # Check if it's a local path
    if Path(model_id).exists() or model_id in ["nano", "tiny", "default"]:
        print(f"📦 Loading local SNN model: {model_id}")
        from src.Core_Models.builders import build_model_from_config
        from utils import SimpleTokenizer
        
        if model_id == "nano" or model_id == "default":
            config = {
                "model_name": "snn-nano",
                "task": "lm",
                "vocab_size": 4000,
                "embedding_dim": 128,
                "hidden_dim": 128,
                "num_micro_layers": 4,
                "num_heads": 4,
                "max_seq_len": 256,
            }
            model = build_model_from_config(config)
        elif model_id == "tiny":
            config = {
                "model_name": "snn-tiny",
                "task": "lm",
                "vocab_size": 8000,
                "embedding_dim": 256,
                "hidden_dim": 256,
                "num_micro_layers": 8,
                "num_heads": 8,
                "max_seq_len": 512,
            }
            model = build_model_from_config(config)
        else:
            from utils.model_io import load_model
            model, _ = load_model(model_id)
        
        model.eval()
        tokenizer = SimpleTokenizer()
        
        return model.to("cpu"), tokenizer
    
    # HuggingFace model
    print(f"📦 Loading HuggingFace model: {model_id}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        model.eval()
        
        # Set pad token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return model, tokenizer
    
    except Exception as e:
        print(f"❌ Error loading {model_id}: {e}")
        print("💡 Falling back to gpt2...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        model.eval()
        return model, tokenizer


def run_inference_universal(model, tokenizer, input_text: str, conversation_history: list):
    """
    Universal inference function - works with both HF and local models.
    
    Args:
        model: Model object
        tokenizer: Tokenizer object
        input_text: User input
        conversation_history: List of previous turns
    
    Returns:
        Generated text
    """
    import torch
    
    # Check if it's a HuggingFace model
    try:
        from transformers import PreTrainedModel
        is_hf_model = isinstance(model, PreTrainedModel)
    except:
        is_hf_model = False
    
    if is_hf_model:
        # HuggingFace inference with conversation context
        # Build prompt from history
        context = ""
        for turn in conversation_history[-5:]:  # Last 5 turns
            context += f"User: {turn['input']}\nAI: {turn['output']}\n"
        
        prompt = context + f"User: {input_text}\nAI:"
        
        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the new response
        if "AI:" in generated:
            response = generated.split("AI:")[-1].strip()
        else:
            response = generated[len(prompt):].strip()
        
        return response
    
    else:
        # Local SNN model inference
        from utils import SimpleTokenizer
        
        token_ids = tokenizer.encode(input_text, add_special_tokens=True)
        input_tensor = torch.tensor([token_ids], dtype=torch.long)
        
        with torch.no_grad():
            outputs = model(input_tensor)
        
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
        
        predictions = torch.argmax(logits, dim=-1)
        predicted_ids = predictions[0].tolist()
        
        try:
            output_text = tokenizer.decode(predicted_ids)
        except:
            output_text = str(predicted_ids[:20])
        
        return output_text


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Multi-Model Inference with Aim Tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launch_aim_inference.py
  python launch_aim_inference.py --port 8000
  python launch_aim_inference.py --models gpt2 distilgpt2 "microsoft/DialoGPT-small"
  
Available models:
  - HuggingFace: Any model ID from huggingface.co/models
  - Local: nano, tiny, or path to .safetensors checkpoint
  
Once running:
  - Web UI: http://localhost:53801/ui
  - API docs: http://localhost:53801/docs
  - Aim dashboard: http://localhost:53800
        """
    )
    parser.add_argument(
        "--port",
        type=int,
        default=53801,
        help="Port for inference API"
    )
    parser.add_argument(
        "--aim-repo",
        type=str,
        default=".aim_project",
        help="Path to Aim repository"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Default HuggingFace models to make available"
    )
    
    args = parser.parse_args()
    
    # Set environment
    os.environ.setdefault("AIM_TRACKING_URI", "aim://localhost:53800")
    os.environ.setdefault("SNN_TRACKING_BACKEND", "aim")
    os.environ.setdefault("SNN_TRACKING_MODE", "enabled")
    
    print("=" * 70)
    print("Multi-Model Inference Extension for Aim")
    print("=" * 70)
    
    # Create extension
    extension = MultiModelInferenceExtension(
        model_loader_fn=load_model_universal,
        inference_fn=run_inference_universal,
        aim_repo_path=args.aim_repo,
        port=args.port,
        default_hf_models=args.models,
    )
    
    # Run
    extension.run()
