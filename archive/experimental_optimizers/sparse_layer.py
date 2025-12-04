"""
Sparse Layer Training - Train only a subset of layers.

Radical Insight:
---------------
Do we REALLY need to compute through all 40 layers to train LoRA?

In practice, LoRA on just the LAST few layers often works surprisingly well.
What if we:
1. Forward through first few layers to get "features" 
2. Skip middle layers entirely (use identity / random projection)
3. Forward through last few layers with LoRA

This reduces compute from 40 layers to ~10 layers = 4x speedup!

Even more radical: What if middle layers just do identity?
The residual connections mean: output ≈ input + small_delta
So skipping layers is like setting delta=0, which is not crazy.

Memory: Same as before (only load one layer at a time)
Speed: 4x faster (only compute 10 layers instead of 40)
Quality: Slightly lower but still useful for fine-tuning
"""

import torch
import torch.nn.functional as F
from torch import nn
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass
import logging
import json
import math
import gc
import time

logger = logging.getLogger(__name__)


@dataclass
class SparseConfig:
    """Configuration."""
    max_seq_length: int = 64
    lora_r: int = 8
    lora_alpha: int = 16
    learning_rate: float = 2e-4
    
    # Sparse layer selection
    compute_first_n: int = 3   # Compute first N layers
    compute_last_n: int = 5    # Compute last N layers
    # Middle layers: identity (skip)


def get_mem():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9
    return 0


def clear_mem():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class QuickLoRA(nn.Module):
    """Fast LoRA implementation."""
    
    def __init__(self, in_dim: int, out_dim: int, r: int = 8, alpha: int = 16):
        super().__init__()
        self.scale = alpha / r
        self.A = nn.Parameter(torch.randn(r, in_dim) * (1 / math.sqrt(in_dim)))
        self.B = nn.Parameter(torch.zeros(out_dim, r))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x.float() @ self.A.T @ self.B.T * self.scale).to(x.dtype)


class FastWeightLoader:
    """Weight loader optimized for repeated access."""
    
    def __init__(self, model_path: str, device: torch.device):
        self.path = Path(model_path)
        self.device = device
        
        with open(self.path / "model.safetensors.index.json") as f:
            self._index = json.load(f)
        
        # Keep handles open for faster repeated access
        self._handles = {}
        
        # Pre-open all files
        from safetensors import safe_open
        for fname in set(self._index["weight_map"].values()):
            self._handles[fname] = safe_open(
                str(self.path / fname), framework="pt", device="cpu"
            )
    
    def load(self, name: str, dtype=torch.float16) -> Optional[torch.Tensor]:
        fname = self._index["weight_map"].get(name)
        if not fname or fname not in self._handles:
            return None
        return self._handles[fname].get_tensor(name).to(device=self.device, dtype=dtype)
    
    def close(self):
        self._handles.clear()


class SparseLayerTrainer:
    """
    Trainer that only computes through a subset of layers.
    
    Strategy:
    - Compute first N layers (get low-level features)
    - SKIP middle layers (identity transform)
    - Compute last M layers (where fine-tuning matters most)
    
    This gives us ~4x speedup with minimal quality loss!
    """
    
    def __init__(self, model_path: str, config: Optional[SparseConfig] = None):
        self.config = config or SparseConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        from transformers import AutoConfig, AutoTokenizer
        
        self.model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.loader = FastWeightLoader(model_path, self.device)
        
        # Dimensions
        self.hidden_size = self.model_config.hidden_size
        self.num_layers = self.model_config.num_hidden_layers
        self.num_heads = self.model_config.num_attention_heads
        self.num_kv_heads = getattr(self.model_config, 'num_key_value_heads', self.num_heads)
        self.head_dim = self.hidden_size // self.num_heads
        
        # Which layers to actually compute
        self.active_layers = (
            list(range(self.config.compute_first_n)) + 
            list(range(self.num_layers - self.config.compute_last_n, self.num_layers))
        )
        
        q_out = self.num_heads * self.head_dim
        v_out = self.num_kv_heads * self.head_dim
        
        # LoRA only for active layers
        self.lora_q = nn.ModuleDict({
            str(i): QuickLoRA(self.hidden_size, q_out, self.config.lora_r, self.config.lora_alpha)
            for i in self.active_layers
        }).to(self.device)
        
        self.lora_v = nn.ModuleDict({
            str(i): QuickLoRA(self.hidden_size, v_out, self.config.lora_r, self.config.lora_alpha)
            for i in self.active_layers
        }).to(self.device)
        
        # Embeddings
        embed_w = self.loader.load("model.embed_tokens.weight")
        self.embeddings = nn.Embedding.from_pretrained(embed_w, freeze=True)
        del embed_w
        
        # Optimizer
        params = list(self.lora_q.parameters()) + list(self.lora_v.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=self.config.learning_rate)
        
        logger.info(f"SparseLayerTrainer initialized")
        logger.info(f"Total layers: {self.num_layers}")
        logger.info(f"Active layers: {self.active_layers} ({len(self.active_layers)} layers)")
        logger.info(f"Skipping {self.num_layers - len(self.active_layers)} middle layers!")
        logger.info(f"LoRA params: {sum(p.numel() for p in params):,}")
        logger.info(f"VRAM: {get_mem():.2f}GB")
    
    def _forward_layer(
        self,
        hidden: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """Forward through one layer."""
        prefix = f"model.layers.{layer_idx}."
        B, S, H = hidden.shape
        residual = hidden
        
        # Load weights
        qkv_w = self.loader.load(prefix + "self_attn.qkv_proj.weight")
        o_w = self.loader.load(prefix + "self_attn.o_proj.weight")
        ln1_w = self.loader.load(prefix + "input_layernorm.weight")
        ln2_w = self.loader.load(prefix + "post_attention_layernorm.weight")
        gate_up_w = self.loader.load(prefix + "mlp.gate_up_proj.weight")
        down_w = self.loader.load(prefix + "mlp.down_proj.weight")
        
        # LayerNorm 1
        hidden = F.layer_norm(hidden, (H,), weight=ln1_w)
        
        # QKV
        qkv = F.linear(hidden, qkv_w)
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        
        q = qkv[..., :q_size]
        k = qkv[..., q_size:q_size + kv_size]
        v = qkv[..., q_size + kv_size:]
        del qkv
        
        # LoRA (only for active layers)
        if str(layer_idx) in self.lora_q:
            q = q + self.lora_q[str(layer_idx)](residual)
            v = v + self.lora_v[str(layer_idx)](residual)
        
        # Attention
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        if self.num_kv_heads < self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)
        
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        del q, k, v
        
        attn = attn.transpose(1, 2).contiguous().view(B, S, H)
        attn = F.linear(attn, o_w)
        hidden = residual + attn
        del attn
        
        # MLP
        residual = hidden
        hidden = F.layer_norm(hidden, (H,), weight=ln2_w)
        
        gate_up = F.linear(hidden, gate_up_w)
        inter_size = gate_up_w.shape[0] // 2
        gate = gate_up[..., :inter_size]
        up = gate_up[..., inter_size:]
        del gate_up
        
        hidden = F.silu(gate) * up
        del gate, up
        
        hidden = F.linear(hidden, down_w)
        hidden = residual + hidden
        
        # Cleanup
        del qkv_w, o_w, ln1_w, ln2_w, gate_up_w, down_w
        clear_mem()
        
        return hidden
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass with sparse layer computation."""
        from torch.utils.checkpoint import checkpoint
        
        hidden = self.embeddings(input_ids)
        
        for layer_idx in range(self.num_layers):
            if layer_idx in self.active_layers:
                # Actually compute this layer with gradient checkpointing
                hidden = checkpoint(
                    self._forward_layer,
                    hidden, layer_idx,
                    use_reentrant=False,
                )
            # Else: identity (just keep hidden as-is via residual)
            # This is the "skip" - we don't load or compute!
        
        # Final norm
        final_ln = self.loader.load("model.norm.weight")
        hidden = F.layer_norm(hidden, (self.hidden_size,), weight=final_ln)
        del final_ln
        
        return hidden
    
    def train_step(self, input_ids: torch.Tensor) -> float:
        """Training step."""
        self.optimizer.zero_grad()
        
        hidden = self.forward(input_ids)
        
        # Loss
        logits = F.linear(hidden, self.embeddings.weight)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.tokenizer.pad_token_id,
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.lora_q.parameters()) + list(self.lora_v.parameters()),
            1.0
        )
        self.optimizer.step()
        
        clear_mem()
        return loss.item()
    
    def train(self, texts: List[str]) -> Dict[str, float]:
        """Train on texts."""
        logger.info(f"Training on {len(texts)} samples (sparse layers)")
        
        total_loss = 0.0
        for i, text in enumerate(texts):
            inputs = self.tokenizer(
                text, truncation=True, max_length=self.config.max_seq_length,
                padding="max_length", return_tensors="pt"
            )
            input_ids = inputs["input_ids"].to(self.device)
            
            t0 = time.time()
            loss = self.train_step(input_ids)
            elapsed = time.time() - t0
            
            total_loss += loss
            logger.info(f"Sample {i+1}/{len(texts)}: loss={loss:.4f}, "
                       f"time={elapsed:.1f}s, VRAM={get_mem():.2f}GB")
        
        return {"avg_loss": total_loss / len(texts)}
    
    def save_lora(self, path: str):
        """Save LoRA weights."""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            "lora_q": self.lora_q.state_dict(),
            "lora_v": self.lora_v.state_dict(),
            "active_layers": self.active_layers,
        }, save_path / "lora.pt")
        
        logger.info(f"Saved to {save_path}")
