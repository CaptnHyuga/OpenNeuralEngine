"""
Cached Layer Training - Keep frequently-used layers on GPU

The insight: We have 2.95GB free after initialization.
Each layer is ~680MB, so we can cache ~4 layers.

By caching the LAST 4 layers on GPU:
- They're used in both forward AND backward
- We eliminate 8 layer loads (4 forward + 4 backward)
- That's 8 * 380ms = 3+ seconds saved!

Expected: ~3.5s per sample (vs 6s before)
"""

import torch
import torch.nn.functional as F
from torch import nn
from typing import Dict, List, Optional
from pathlib import Path
from dataclasses import dataclass
import logging
import json
import math
import gc
import time

logger = logging.getLogger(__name__)


@dataclass  
class CachedConfig:
    max_seq_length: int = 32
    lora_r: int = 4
    lora_alpha: int = 8
    learning_rate: float = 2e-4
    # Sparse layers
    compute_first_n: int = 2   # Stream these
    compute_last_n: int = 4    # CACHE these on GPU


def get_mem():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9
    return 0


def clear_mem():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class QuickLoRA(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, r: int = 4, alpha: int = 8):
        super().__init__()
        self.scale = alpha / r
        self.A = nn.Parameter(torch.randn(r, in_dim) * (0.01 / math.sqrt(in_dim)))
        self.B = nn.Parameter(torch.zeros(out_dim, r))
    
    def forward(self, x):
        return (x.float() @ self.A.T @ self.B.T * self.scale).to(x.dtype)


class CachedLayerTrainer:
    """
    Training with layer caching.
    
    Strategy:
    - Stream: first N layers (load from disk each time)
    - Cache: last M layers (keep permanently on GPU)
    
    This eliminates loading time for cached layers!
    """
    
    def __init__(self, model_path: str, config: Optional[CachedConfig] = None):
        self.config = config or CachedConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        from transformers import AutoConfig, AutoTokenizer
        
        self.model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Model dimensions
        self.hidden_size = self.model_config.hidden_size
        self.num_layers = self.model_config.num_hidden_layers
        self.num_heads = self.model_config.num_attention_heads
        self.num_kv_heads = getattr(self.model_config, 'num_key_value_heads', self.num_heads)
        self.head_dim = self.hidden_size // self.num_heads
        
        # Layers
        first_n = self.config.compute_first_n
        last_n = self.config.compute_last_n
        self.stream_layers = list(range(first_n))  # Load from disk
        self.cached_layers = list(range(self.num_layers - last_n, self.num_layers))  # Keep on GPU
        self.all_layers = self.stream_layers + self.cached_layers
        
        logger.info(f"Stream layers: {self.stream_layers}")
        logger.info(f"Cached layers: {self.cached_layers}")
        
        # Weight loader for streaming layers
        self.path = Path(model_path)
        with open(self.path / "model.safetensors.index.json") as f:
            self._index = json.load(f)
        
        from safetensors import safe_open
        self._handles = {}
        for fname in set(self._index["weight_map"].values()):
            self._handles[fname] = safe_open(
                str(self.path / fname), framework="pt", device="cpu"
            )
        
        # Pinned buffer cache
        self._pinned: Dict[tuple, torch.Tensor] = {}
        
        # LoRA
        q_out = self.num_heads * self.head_dim
        v_out = self.num_kv_heads * self.head_dim
        
        self.lora_q = nn.ModuleDict({
            str(i): QuickLoRA(self.hidden_size, q_out, self.config.lora_r, self.config.lora_alpha)
            for i in self.all_layers
        }).to(self.device)
        
        self.lora_v = nn.ModuleDict({
            str(i): QuickLoRA(self.hidden_size, v_out, self.config.lora_r, self.config.lora_alpha)
            for i in self.all_layers
        }).to(self.device)
        
        # Embeddings
        embed_w = self._load_weight("model.embed_tokens.weight")
        self.embeddings = nn.Embedding.from_pretrained(embed_w, freeze=True)
        del embed_w
        
        self.final_ln = self._load_weight("model.norm.weight")
        self.lm_head = self.embeddings.weight
        
        logger.info(f"After embeddings: {get_mem():.2f}GB")
        
        # CACHE the last N layers on GPU
        self.cached_weights: Dict[int, Dict[str, torch.Tensor]] = {}
        logger.info("Caching layers on GPU...")
        
        for layer_idx in self.cached_layers:
            self.cached_weights[layer_idx] = self._load_layer(layer_idx)
            logger.info(f"  Cached layer {layer_idx}: {get_mem():.2f}GB")
        
        # Optimizer
        params = list(self.lora_q.parameters()) + list(self.lora_v.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=self.config.learning_rate)
        
        logger.info(f"CachedLayerTrainer initialized")
        logger.info(f"Total layers: {len(self.all_layers)} ({len(self.stream_layers)} streamed, {len(self.cached_layers)} cached)")
        logger.info(f"LoRA params: {sum(p.numel() for p in params):,}")
        logger.info(f"Final VRAM: {get_mem():.2f}GB")
    
    def _get_pinned(self, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
        key = (shape, dtype)
        if key not in self._pinned:
            self._pinned[key] = torch.empty(shape, dtype=dtype, pin_memory=True)
        return self._pinned[key]
    
    def _load_weight(self, name: str) -> Optional[torch.Tensor]:
        fname = self._index["weight_map"].get(name)
        if not fname or fname not in self._handles:
            return None
        cpu = self._handles[fname].get_tensor(name)
        pinned = self._get_pinned(cpu.shape, cpu.dtype)
        pinned.copy_(cpu)
        return pinned.to(device=self.device, dtype=torch.float16)
    
    def _load_layer(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """Load all weights for a layer."""
        prefix = f"model.layers.{layer_idx}."
        weights = {}
        
        suffixes = [
            "input_layernorm.weight",
            "self_attn.qkv_proj.weight",
            "self_attn.o_proj.weight",
            "post_attention_layernorm.weight",
            "mlp.gate_up_proj.weight",
            "mlp.down_proj.weight",
        ]
        
        for suffix in suffixes:
            w = self._load_weight(prefix + suffix)
            if w is not None:
                weights[prefix + suffix] = w
        
        torch.cuda.synchronize()
        return weights
    
    def _get_layer_weights(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """Get weights for a layer - cached or loaded."""
        if layer_idx in self.cached_weights:
            return self.cached_weights[layer_idx]
        return self._load_layer(layer_idx)
    
    def _forward_layer(self, hidden: torch.Tensor, layer_idx: int, weights: Dict[str, torch.Tensor]) -> torch.Tensor:
        prefix = f"model.layers.{layer_idx}."
        B, S, H = hidden.shape
        residual = hidden
        
        ln1_w = weights.get(prefix + "input_layernorm.weight")
        if ln1_w is not None:
            hidden = F.layer_norm(hidden, (H,), weight=ln1_w)
        
        qkv_w = weights.get(prefix + "self_attn.qkv_proj.weight")
        if qkv_w is None:
            return hidden
        
        qkv = F.linear(hidden, qkv_w)
        
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        q = qkv[..., :q_size]
        k = qkv[..., q_size:q_size + kv_size]
        v = qkv[..., q_size + kv_size:q_size + 2*kv_size]
        del qkv
        
        q = q + self.lora_q[str(layer_idx)](residual)
        v = v + self.lora_v[str(layer_idx)](residual)
        
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
        
        o_w = weights.get(prefix + "self_attn.o_proj.weight")
        if o_w is not None:
            attn = F.linear(attn, o_w)
        
        hidden = residual + attn
        del attn
        
        residual = hidden
        ln2_w = weights.get(prefix + "post_attention_layernorm.weight")
        if ln2_w is not None:
            hidden = F.layer_norm(hidden, (H,), weight=ln2_w)
        
        gate_up_w = weights.get(prefix + "mlp.gate_up_proj.weight")
        if gate_up_w is not None:
            gate_up = F.linear(hidden, gate_up_w)
            inter_size = gate_up_w.shape[0] // 2
            gate = gate_up[..., :inter_size]
            up = gate_up[..., inter_size:]
            del gate_up
            hidden = F.silu(gate) * up
            del gate, up
            
            down_w = weights.get(prefix + "mlp.down_proj.weight")
            if down_w is not None:
                hidden = F.linear(hidden, down_w)
        
        return residual + hidden
    
    def train_step(self, input_ids: torch.Tensor) -> float:
        B, S = input_ids.shape
        self.optimizer.zero_grad()
        
        # Forward
        hidden = self.embeddings(input_ids)
        hidden_states = [hidden.detach()]
        
        for layer_idx in self.all_layers:
            weights = self._get_layer_weights(layer_idx)
            
            with torch.no_grad():
                hidden = self._forward_layer(hidden, layer_idx, weights)
            hidden_states.append(hidden.detach())
            
            # Only delete if streamed (not cached)
            if layer_idx not in self.cached_weights:
                del weights
        
        with torch.no_grad():
            hidden = F.layer_norm(hidden, (self.hidden_size,), weight=self.final_ln)
            logits = F.linear(hidden, self.lm_head)
        
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.tokenizer.pad_token_id,
        )
        loss_val = loss.item()
        
        # Backward
        grad = torch.zeros(B, S, self.hidden_size, device=self.device, dtype=hidden.dtype)
        probs = F.softmax(shift_logits.detach(), dim=-1)
        grad_logits = probs.clone()
        grad_logits.scatter_add_(
            2, shift_labels.unsqueeze(-1),
            -torch.ones_like(shift_labels.unsqueeze(-1), dtype=grad_logits.dtype)
        )
        grad[:, :-1, :] = grad_logits @ self.lm_head / (S - 1)
        
        for i, layer_idx in enumerate(reversed(self.all_layers)):
            weights = self._get_layer_weights(layer_idx)
            
            h_in = hidden_states[-(i + 2)].requires_grad_(True)
            
            prefix = f"model.layers.{layer_idx}."
            residual = h_in
            
            ln1_w = weights.get(prefix + "input_layernorm.weight")
            h = F.layer_norm(h_in, (self.hidden_size,), weight=ln1_w) if ln1_w is not None else h_in
            
            qkv_w = weights.get(prefix + "self_attn.qkv_proj.weight")
            if qkv_w is None:
                continue
            
            qkv = F.linear(h, qkv_w)
            q_size = self.num_heads * self.head_dim
            kv_size = self.num_kv_heads * self.head_dim
            q = qkv[..., :q_size]
            k = qkv[..., q_size:q_size + kv_size]
            v = qkv[..., q_size + kv_size:q_size + 2*kv_size]
            
            q = q + self.lora_q[str(layer_idx)](residual)
            v = v + self.lora_v[str(layer_idx)](residual)
            
            q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = v.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
            
            if self.num_kv_heads < self.num_heads:
                n_rep = self.num_heads // self.num_kv_heads
                k = k.repeat_interleave(n_rep, dim=1)
                v = v.repeat_interleave(n_rep, dim=1)
            
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            attn = attn.transpose(1, 2).contiguous().view(B, S, self.hidden_size)
            
            o_w = weights.get(prefix + "self_attn.o_proj.weight")
            if o_w is not None:
                attn = F.linear(attn, o_w)
            
            out = residual + attn
            
            residual_mlp = out
            ln2_w = weights.get(prefix + "post_attention_layernorm.weight")
            out2 = F.layer_norm(out, (self.hidden_size,), weight=ln2_w) if ln2_w is not None else out
            
            gate_up_w = weights.get(prefix + "mlp.gate_up_proj.weight")
            if gate_up_w is not None:
                gate_up = F.linear(out2, gate_up_w)
                inter_size = gate_up_w.shape[0] // 2
                gate, up = gate_up[..., :inter_size], gate_up[..., inter_size:]
                out2 = F.silu(gate) * up
                
                down_w = weights.get(prefix + "mlp.down_proj.weight")
                if down_w is not None:
                    out2 = F.linear(out2, down_w)
            
            output = residual_mlp + out2
            
            loss_contrib = (output * grad).sum()
            loss_contrib.backward()
            
            grad = h_in.grad.detach() if h_in.grad is not None else grad
            
            if layer_idx not in self.cached_weights:
                del weights
        
        torch.nn.utils.clip_grad_norm_(
            list(self.lora_q.parameters()) + list(self.lora_v.parameters()),
            1.0
        )
        self.optimizer.step()
        
        del hidden_states
        clear_mem()
        
        return loss_val
    
    def train(self, texts: List[str], epochs: int = 1) -> Dict:
        logger.info(f"Training on {len(texts)} samples (cached layers)")
        
        losses = []
        times = []
        
        for epoch in range(epochs):
            for i, text in enumerate(texts):
                t0 = time.time()
                
                enc = self.tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=self.config.max_seq_length,
                    truncation=True,
                    padding="max_length",
                )
                input_ids = enc["input_ids"].to(self.device)
                
                loss = self.train_step(input_ids)
                losses.append(loss)
                
                dt = time.time() - t0
                times.append(dt)
                logger.info(f"[{epoch+1}/{epochs}] Sample {i+1}/{len(texts)}: "
                           f"loss={loss:.4f}, time={dt:.1f}s, VRAM={get_mem():.2f}GB")
        
        return {
            "losses": losses,
            "avg_loss": sum(losses) / len(losses),
            "avg_time": sum(times) / len(times),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
    
    logger.info("=" * 60)
    logger.info("CACHED LAYER TRAINING")
    logger.info("Last 4 layers cached on GPU = instant access")
    logger.info("=" * 60)
    
    config = CachedConfig(max_seq_length=32)
    trainer = CachedLayerTrainer("models/phi-4", config)
    
    texts = []
    with open("data/Dataset/math.jsonl") as f:
        for i, line in enumerate(f):
            if i >= 8:
                break
            item = json.loads(line)
            texts.append(f"Problem: {item['problem']}\nAnswer: {item['answer']}")
    
    logger.info(f"\nLoaded {len(texts)} samples")
    
    results = trainer.train(texts)
    
    logger.info("=" * 60)
    logger.info(f"Avg loss: {results['avg_loss']:.4f}")
    logger.info(f"Avg time: {results['avg_time']:.1f}s/sample")
    logger.info("=" * 60)
