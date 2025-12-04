"""
Layer-Parallel Batch Processing - Amortize weight loading across samples.

The Key Insight:
---------------
Instead of processing each sample through all layers (loading weights 40 times per sample),
we process ALL samples through each layer (loading weights 40 times TOTAL).

Traditional: Sample × Layers = N × 40 weight loads
Layer-Parallel: Layers × 1 = 40 weight loads (regardless of N samples!)

This transforms the problem from "slow because of weight loading" to 
"fast because we load once, compute many times."

Memory Model:
- Embeddings: ~1GB (kept on GPU)
- One layer weights: ~400MB (loaded, used, discarded)
- Hidden states for ALL samples: B × S × H × 2 bytes
  - For B=32 samples, S=32 seq, H=5120: 32×32×5120×2 = 10MB
  - This is TINY compared to weights!

So we can process many samples simultaneously through each layer!
"""

import torch
import torch.nn.functional as F
from torch import nn
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass
import logging
import json
import math
import gc
import time

logger = logging.getLogger(__name__)


@dataclass
class LayerParallelConfig:
    """Configuration."""
    batch_size: int = 16  # Process this many samples per layer
    max_seq_length: int = 64
    lora_r: int = 4
    lora_alpha: int = 8
    learning_rate: float = 1e-4
    num_epochs: int = 1


def get_mem():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9
    return 0


def clear_mem():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class CompactLoRA(nn.Module):
    """Memory-efficient LoRA."""
    
    def __init__(self, in_dim: int, out_dim: int, r: int = 4, alpha: int = 8):
        super().__init__()
        self.scale = alpha / r
        # Small matrices, FP32 for gradients
        self.A = nn.Parameter(torch.randn(r, in_dim) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_dim, r))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, in_dim]
        # Use same dtype as input for computation
        A = self.A.to(x.dtype)
        B = self.B.to(x.dtype)
        return (x @ A.T @ B.T) * self.scale


class WeightLoader:
    """Fast weight loading from safetensors."""
    
    def __init__(self, model_path: str, device: torch.device):
        self.path = Path(model_path)
        self.device = device
        
        with open(self.path / "model.safetensors.index.json") as f:
            self._index = json.load(f)
        
        self._handles = {}
    
    def _handle(self, fname: str):
        from safetensors import safe_open
        if fname not in self._handles:
            self._handles[fname] = safe_open(
                str(self.path / fname), framework="pt", device="cpu"
            )
        return self._handles[fname]
    
    def load(self, name: str, dtype=torch.float16) -> Optional[torch.Tensor]:
        """Load weight directly to GPU."""
        fname = self._index["weight_map"].get(name)
        if not fname:
            return None
        
        # Load to GPU directly
        tensor = self._handle(fname).get_tensor(name)
        return tensor.to(device=self.device, dtype=dtype)
    
    def close(self):
        self._handles.clear()


class LayerParallelTrainer:
    """
    Trainer that processes all samples through each layer before moving to next.
    
    This minimizes weight loading overhead by loading each layer's weights only ONCE
    per batch, regardless of how many samples are in the batch.
    """
    
    def __init__(self, model_path: str, config: Optional[LayerParallelConfig] = None):
        self.config = config or LayerParallelConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model config
        from transformers import AutoConfig, AutoTokenizer
        
        self.model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Weight loader
        self.loader = WeightLoader(model_path, self.device)
        
        # Model dimensions
        self.hidden_size = self.model_config.hidden_size
        self.num_layers = self.model_config.num_hidden_layers
        self.num_heads = self.model_config.num_attention_heads
        self.num_kv_heads = getattr(self.model_config, 'num_key_value_heads', self.num_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.vocab_size = self.model_config.vocab_size
        
        # LoRA adapters for all layers (tiny, always in GPU memory)
        q_out = self.num_heads * self.head_dim
        v_out = self.num_kv_heads * self.head_dim
        
        self.lora_q = nn.ModuleList([
            CompactLoRA(self.hidden_size, q_out, self.config.lora_r, self.config.lora_alpha)
            for _ in range(self.num_layers)
        ]).to(self.device)
        
        self.lora_v = nn.ModuleList([
            CompactLoRA(self.hidden_size, v_out, self.config.lora_r, self.config.lora_alpha)
            for _ in range(self.num_layers)
        ]).to(self.device)
        
        # Load embeddings (needed for every forward pass)
        embed_w = self.loader.load("model.embed_tokens.weight")
        self.embeddings = nn.Embedding.from_pretrained(embed_w, freeze=True)
        del embed_w
        
        # Optimizer
        lora_params = list(self.lora_q.parameters()) + list(self.lora_v.parameters())
        self.optimizer = torch.optim.AdamW(lora_params, lr=self.config.learning_rate)
        
        logger.info(f"LayerParallelTrainer initialized")
        logger.info(f"Layers: {self.num_layers}, Hidden: {self.hidden_size}")
        logger.info(f"LoRA params: {sum(p.numel() for p in lora_params):,}")
        logger.info(f"VRAM after init: {get_mem():.2f}GB")
    
    def _forward_layer(
        self,
        hidden: torch.Tensor,  # [B, S, H]
        layer_idx: int,
        qkv_w: torch.Tensor,
        o_w: torch.Tensor,
        ln1_w: torch.Tensor,
        ln2_w: torch.Tensor,
        gate_up_w: torch.Tensor,
        down_w: torch.Tensor,
    ) -> torch.Tensor:
        """Forward through one layer with pre-loaded weights."""
        B, S, H = hidden.shape
        residual = hidden
        
        # LayerNorm 1
        hidden = F.layer_norm(hidden, (H,), weight=ln1_w)
        
        # QKV projection
        qkv = F.linear(hidden, qkv_w)
        
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        
        q = qkv[..., :q_size]
        k = qkv[..., q_size:q_size + kv_size]
        v = qkv[..., q_size + kv_size:]
        del qkv
        
        # Apply LoRA (these are always in memory, tiny!)
        q = q + self.lora_q[layer_idx](residual)
        v = v + self.lora_v[layer_idx](residual)
        
        # Reshape for attention
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # GQA expansion
        if self.num_kv_heads < self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)
        
        # Flash attention
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        del q, k, v
        
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, H)
        
        # Output projection
        attn_out = F.linear(attn_out, o_w)
        hidden = residual + attn_out
        del attn_out
        
        # LayerNorm 2
        residual = hidden
        hidden = F.layer_norm(hidden, (H,), weight=ln2_w)
        
        # MLP
        gate_up = F.linear(hidden, gate_up_w)
        inter_size = gate_up_w.shape[0] // 2
        gate = gate_up[..., :inter_size]
        up = gate_up[..., inter_size:]
        del gate_up
        
        hidden = F.silu(gate) * up
        del gate, up
        
        hidden = F.linear(hidden, down_w)
        hidden = residual + hidden
        
        return hidden
    
    def _batch_forward(
        self,
        input_ids: torch.Tensor,  # [B, S]
    ) -> torch.Tensor:
        """
        Forward pass processing all samples through each layer before moving to next.
        
        Uses gradient checkpointing to avoid storing all activations.
        """
        from torch.utils.checkpoint import checkpoint
        
        B, S = input_ids.shape
        
        # Embed all samples
        hidden = self.embeddings(input_ids)  # [B, S, H]
        
        # Process through all layers with checkpointing
        for layer_idx in range(self.num_layers):
            prefix = f"model.layers.{layer_idx}."
            
            # Load all weights for this layer
            qkv_w = self.loader.load(prefix + "self_attn.qkv_proj.weight")
            o_w = self.loader.load(prefix + "self_attn.o_proj.weight")
            ln1_w = self.loader.load(prefix + "input_layernorm.weight")
            ln2_w = self.loader.load(prefix + "post_attention_layernorm.weight")
            gate_up_w = self.loader.load(prefix + "mlp.gate_up_proj.weight")
            down_w = self.loader.load(prefix + "mlp.down_proj.weight")
            
            # Use gradient checkpointing - recompute forward during backward
            # This trades compute for memory
            hidden = checkpoint(
                self._forward_layer,
                hidden, layer_idx,
                qkv_w, o_w, ln1_w, ln2_w, gate_up_w, down_w,
                use_reentrant=False,
            )
            
            # Discard weights immediately
            del qkv_w, o_w, ln1_w, ln2_w, gate_up_w, down_w
            clear_mem()
        
        # Final norm
        final_ln = self.loader.load("model.norm.weight")
        hidden = F.layer_norm(hidden, (self.hidden_size,), weight=final_ln)
        del final_ln
        
        return hidden
    
    def compute_loss(
        self,
        hidden: torch.Tensor,  # [B, S, H]
        labels: torch.Tensor,  # [B, S]
    ) -> torch.Tensor:
        """Compute causal LM loss."""
        # Use embedding weight for output projection (tied weights)
        logits = F.linear(hidden, self.embeddings.weight)
        
        # Shift for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.tokenizer.pad_token_id,
        )
        
        return loss
    
    def train_batch(self, texts: List[str]) -> float:
        """Train on a batch of texts."""
        self.optimizer.zero_grad()
        
        # Tokenize all texts
        inputs = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.config.max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        input_ids = inputs["input_ids"].to(self.device)
        labels = input_ids.clone()
        
        # Forward pass (layer-parallel)
        hidden = self._batch_forward(input_ids)
        
        # Compute loss
        loss = self.compute_loss(hidden, labels)
        
        # Backward
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            list(self.lora_q.parameters()) + list(self.lora_v.parameters()),
            1.0
        )
        
        # Update
        self.optimizer.step()
        
        clear_mem()
        
        return loss.item()
    
    def train(self, data: List[str]) -> Dict[str, Any]:
        """Train on dataset."""
        batch_size = self.config.batch_size
        num_batches = (len(data) + batch_size - 1) // batch_size
        
        logger.info(f"Training on {len(data)} samples, {num_batches} batches")
        
        total_loss = 0.0
        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            for batch_idx in range(num_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, len(data))
                batch_texts = data[start:end]
                
                t0 = time.time()
                loss = self.train_batch(batch_texts)
                elapsed = time.time() - t0
                
                total_loss += loss
                
                logger.info(f"Batch {batch_idx + 1}/{num_batches}: "
                          f"loss={loss:.4f}, time={elapsed:.1f}s, VRAM={get_mem():.2f}GB")
        
        return {
            "loss": total_loss / num_batches,
            "num_batches": num_batches,
        }
    
    def save_lora(self, path: str):
        """Save LoRA weights."""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        state_dict = {
            "lora_q": self.lora_q.state_dict(),
            "lora_v": self.lora_v.state_dict(),
        }
        torch.save(state_dict, save_path / "lora.pt")
        logger.info(f"Saved LoRA to {save_path}")
    
    def generate(self, prompt: str, max_new_tokens: int = 20) -> str:
        """Generate text."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward
                hidden = self._batch_forward(input_ids)
                
                # Get logits for last position
                logits = F.linear(hidden[:, -1:, :], self.embeddings.weight)
                next_token = logits.argmax(dim=-1)
                
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)


def load_math_data(path: str, max_samples: int = None) -> List[str]:
    """Load math dataset."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            item = json.loads(line)
            if "problem" in item and "answer" in item:
                data.append(f"Problem: {item['problem']}\nAnswer: {item['answer']}")
    return data
