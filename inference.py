#!/usr/bin/env python3
"""
ONN Inference - Test trained models with generation

Usage:
    python inference.py --model models/phi-4 --prompt "What is 2+2?"
    python inference.py --model models/phi-4 --lora output/training_v3/checkpoints/final.safetensors
    python inference.py --interactive --model models/phi-4
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, List

import torch
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.hf_tokenizer import load_tokenizer


class SimpleGenerator:
    """Minimal generator using embeddings and LoRA."""
    
    def __init__(self, model_path: str, lora_path: Optional[str] = None, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_path = Path(model_path)
        
        print(f"🔧 Loading model from {model_path}...")
        self._load_tokenizer()
        self._load_embeddings()
        
        if lora_path:
            self._load_lora(lora_path)
        else:
            self.lora_weights = None
    
    def _load_tokenizer(self):
        """Load tokenizer."""
        self.tokenizer = load_tokenizer(self.model_path)
        self.vocab_size = self.tokenizer.vocab_size
        print(f"   Tokenizer: vocab_size={self.vocab_size}")
    
    def _load_embeddings(self):
        """Load embedding weights."""
        index_path = self.model_path / "model.safetensors.index.json"
        
        if index_path.exists():
            with open(index_path) as f:
                index = json.load(f)
            weight_map = index.get("weight_map", {})
            
            for key, file in weight_map.items():
                if "embed_tokens" in key and "weight" in key:
                    embed_file = self.model_path / file
                    print(f"   Loading embeddings from {embed_file.name}...")
                    weights = load_safetensors(str(embed_file))
                    self.embeddings = weights[key].to(self.device)
                    break
        
        self.vocab_size, self.hidden_size = self.embeddings.shape
        print(f"   Embeddings: vocab={self.vocab_size}, hidden={self.hidden_size}")
    
    def _load_lora(self, lora_path: str):
        """Load LoRA weights."""
        print(f"   Loading LoRA from {lora_path}...")
        self.lora_weights = load_safetensors(lora_path)
        print(f"   LoRA layers: {len([k for k in self.lora_weights.keys() if 'lora_A' in k])}")
    
    def get_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Get embeddings for token IDs."""
        token_ids = token_ids.clamp(0, self.vocab_size - 1)
        return F.embedding(token_ids, self.embeddings)
    
    def apply_lora(self, hidden: torch.Tensor, layer_idx: int = 0) -> torch.Tensor:
        """Apply LoRA transformation."""
        if self.lora_weights is None:
            return hidden
        
        lora_A_key = f"layers.{layer_idx}.lora_A"
        lora_B_key = f"layers.{layer_idx}.lora_B"
        
        if lora_A_key not in self.lora_weights:
            return hidden
        
        lora_A = self.lora_weights[lora_A_key].to(self.device)
        lora_B = self.lora_weights[lora_B_key].to(self.device)
        
        # Assuming alpha=16, rank=8
        scale = 16 / 8
        
        x_float = hidden.float()
        lora_out = F.linear(F.linear(x_float, lora_A), lora_B)
        return hidden + (lora_out * scale).to(hidden.dtype)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass to get logits."""
        hidden = self.get_embeddings(input_ids)
        
        # Apply LoRA layers
        num_lora_layers = len([k for k in (self.lora_weights or {}).keys() if 'lora_A' in k])
        for i in range(num_lora_layers):
            hidden = self.apply_lora(hidden, i)
        
        # Compute logits using tied embeddings
        logits = torch.matmul(hidden.float(), self.embeddings.t().float())
        return logits
    
    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 50, 
                 temperature: float = 0.7, top_p: float = 0.9) -> str:
        """Generate text from prompt."""
        # Tokenize
        input_ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        
        generated = input_ids
        
        for _ in range(max_new_tokens):
            # Forward pass
            logits = self.forward(generated)
            
            # Get last token logits
            next_token_logits = logits[0, -1, :]
            
            # Apply temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            
            # Top-p sampling
            probs = F.softmax(next_token_logits, dim=-1)
            
            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # Remove tokens with cumulative probability above top_p
                sorted_indices_to_remove = cumsum_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                probs[indices_to_remove] = 0.0
                probs = probs / probs.sum()
            
            # Sample
            next_token = torch.multinomial(probs, num_samples=1).unsqueeze(0)
            
            # Append to generated
            generated = torch.cat([generated, next_token], dim=-1)
            
            # Stop on EOS
            if self.tokenizer.eos_id and next_token.item() == self.tokenizer.eos_id:
                break
        
        # Decode
        output_ids = generated[0].tolist()
        output_text = self.tokenizer.decode(output_ids)
        
        return output_text
    
    def interactive(self):
        """Interactive generation mode."""
        print("\n🤖 Interactive Mode (type 'quit' to exit)")
        print("-" * 50)
        
        while True:
            try:
                prompt = input("\n📝 You: ")
                if prompt.lower() in ['quit', 'exit', 'q']:
                    break
                
                print("🤖 Model: ", end="", flush=True)
                output = self.generate(prompt, max_new_tokens=100)
                
                # Remove prompt from output if present
                if output.startswith(prompt):
                    output = output[len(prompt):]
                
                print(output.strip())
                
            except KeyboardInterrupt:
                break
        
        print("\n👋 Goodbye!")


def main():
    parser = argparse.ArgumentParser(description="ONN Inference")
    parser.add_argument("--model", default="models/phi-4", help="Model path")
    parser.add_argument("--lora", help="Path to LoRA checkpoint")
    parser.add_argument("--prompt", help="Input prompt for generation")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--max-tokens", type=int, default=50, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    
    args = parser.parse_args()
    
    print("="*60)
    print("ONN INFERENCE")
    print("="*60)
    
    # Initialize generator
    generator = SimpleGenerator(args.model, args.lora)
    
    if args.interactive:
        generator.interactive()
    elif args.prompt:
        print(f"\n📝 Prompt: {args.prompt}")
        output = generator.generate(
            args.prompt, 
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )
        print(f"\n🤖 Output: {output}")
    else:
        print("\n⚠️ Please provide --prompt or --interactive")


if __name__ == "__main__":
    main()
