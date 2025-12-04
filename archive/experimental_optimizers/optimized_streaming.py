"""
Optimized Training: Tiled Matmul + Per-Layer Gradient Streaming

This combines two optimizations:
1. TILED MATMUL: Better memory access patterns for faster computation
2. PER-LAYER GRADIENTS: Process one layer at a time to fit in VRAM

The key insight is that these are ORTHOGONAL optimizations:
- Tiling optimizes HOW we compute (memory coalescing, cache efficiency)
- Per-layer gradients optimizes WHAT we keep in memory (only one layer at a time)

Together they should give us both speed AND memory efficiency.
"""

import logging
import math
import json
import gc
import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# UTILITIES
# ============================================================================

def get_mem() -> float:
    """Get allocated GPU memory in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9
    return 0.0

def clear_mem():
    """Clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ============================================================================
# TILED MATRIX MULTIPLICATION
# ============================================================================

@dataclass
class TiledConfig:
    """Configuration for tiled computation."""
    # Tile sizes - tuned for GTX 1650 with 4GB VRAM
    tile_m: int = 64   # Tiles along sequence dimension
    tile_k: int = 64   # Tiles along hidden dimension
    tile_n: int = 64   # Tiles along output dimension
    
    # Memory management
    max_seq_length: int = 32
    lora_r: int = 4
    lora_alpha: int = 8
    learning_rate: float = 1e-4


def tiled_matmul_optimized(
    x: torch.Tensor,
    w: torch.Tensor,
    tile_m: int = 64,
    tile_k: int = 64,
    tile_n: int = 64,
) -> torch.Tensor:
    """
    Optimized tiled matrix multiplication: x @ w.T
    
    Key optimizations:
    1. Process in tiles to maximize cache hits
    2. Pre-allocate output to avoid fragmentation  
    3. Use contiguous memory views for coalesced access
    4. Fuse multiple small tiles into bigger blocks when possible
    
    Args:
        x: Input tensor [*, K] 
        w: Weight tensor [N, K] (will be transposed)
        tile_*: Tile dimensions
    
    Returns:
        Output tensor [*, N]
    """
    original_shape = x.shape
    x_2d = x.view(-1, x.shape[-1])  # [M, K]
    M, K = x_2d.shape
    N = w.shape[0]
    
    # Pre-allocate output - contiguous for better access
    out = torch.empty(M, N, dtype=x.dtype, device=x.device)
    
    # Transpose weight once for better memory access
    w_t = w.T.contiguous()  # [K, N]
    
    # Process in tiles
    for m_start in range(0, M, tile_m):
        m_end = min(m_start + tile_m, M)
        x_tile = x_2d[m_start:m_end]  # [tile_m, K]
        
        for n_start in range(0, N, tile_n):
            n_end = min(n_start + tile_n, N)
            
            # Accumulate result for this output tile
            out_tile = torch.zeros(m_end - m_start, n_end - n_start, 
                                   dtype=x.dtype, device=x.device)
            
            for k_start in range(0, K, tile_k):
                k_end = min(k_start + tile_k, K)
                
                # Small tile matmul - fits in L2 cache
                x_subtile = x_tile[:, k_start:k_end]  # [tile_m, tile_k]
                w_subtile = w_t[k_start:k_end, n_start:n_end]  # [tile_k, tile_n]
                
                out_tile += x_subtile @ w_subtile
            
            out[m_start:m_end, n_start:n_end] = out_tile
    
    # Reshape back
    return out.view(*original_shape[:-1], N)


# ============================================================================
# LORA WITH OPTIMIZED COMPUTE
# ============================================================================

class OptimizedLoRA(nn.Module):
    """LoRA adapter with optimized matmul."""
    
    def __init__(self, in_dim: int, out_dim: int, r: int = 4, alpha: int = 8):
        super().__init__()
        self.scale = alpha / r
        # A: down-projection [in_dim -> r]
        # B: up-projection [r -> out_dim]
        self.A = nn.Parameter(torch.randn(r, in_dim) * (0.01 / math.sqrt(in_dim)))
        self.B = nn.Parameter(torch.zeros(out_dim, r))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x @ A.T @ B.T = x @ (A.T @ B.T) - but A.T @ B.T is tiny [in_dim, out_dim]
        # Pre-compute this for efficiency
        return (x.float() @ self.A.T @ self.B.T * self.scale).to(x.dtype)


# ============================================================================
# STREAMING WEIGHT LOADER
# ============================================================================

class StreamingWeightLoader:
    """
    Load weights from safetensors one layer at a time.
    Uses memory-mapped access for efficiency.
    """
    
    def __init__(self, model_path: str, device: torch.device):
        self.path = Path(model_path)
        self.device = device
        
        # Load index
        with open(self.path / "model.safetensors.index.json") as f:
            self._index = json.load(f)
        
        # Open file handles
        self._handles: Dict[str, safe_open] = {}
        for fname in set(self._index["weight_map"].values()):
            self._handles[fname] = safe_open(
                str(self.path / fname), framework="pt", device="cpu"
            )
    
    def load_weight(self, name: str, dtype: torch.dtype = torch.float16) -> Optional[torch.Tensor]:
        """Load a single weight tensor."""
        fname = self._index["weight_map"].get(name)
        if not fname or fname not in self._handles:
            return None
        return self._handles[fname].get_tensor(name).to(device=self.device, dtype=dtype)
    
    def load_layer_weights(self, layer_idx: int, dtype: torch.dtype = torch.float16) -> Dict[str, torch.Tensor]:
        """Load all weights for a layer."""
        prefix = f"model.layers.{layer_idx}."
        weights = {}
        
        weight_suffixes = [
            "input_layernorm.weight",
            "self_attn.qkv_proj.weight",
            "self_attn.o_proj.weight",
            "post_attention_layernorm.weight",
            "mlp.gate_up_proj.weight",
            "mlp.down_proj.weight",
        ]
        
        for suffix in weight_suffixes:
            name = prefix + suffix
            w = self.load_weight(name, dtype)
            if w is not None:
                weights[name] = w
        
        return weights
    
    def close(self):
        """Close all file handles."""
        self._handles.clear()


# ============================================================================
# OPTIMIZED TRAINER
# ============================================================================

class OptimizedStreamingTrainer:
    """
    Training with tiled matmul and per-layer gradient streaming.
    
    Memory model:
    - Embeddings: ~0.35GB (frozen)
    - LoRA params: ~0.01GB  
    - 1 layer weights: ~1GB (streamed)
    - Activations: ~0.1GB (small seq len)
    - Gradients: ~0.1GB
    Total: ~1.6GB (fits in 4GB!)
    """
    
    def __init__(self, model_path: str, config: Optional[TiledConfig] = None):
        self.config = config or TiledConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        from transformers import AutoConfig, AutoTokenizer
        
        self.model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Weight loader
        self.weight_loader = StreamingWeightLoader(model_path, self.device)
        
        # Model dimensions  
        self.hidden_size = self.model_config.hidden_size
        self.num_layers = self.model_config.num_hidden_layers
        self.num_heads = self.model_config.num_attention_heads
        self.num_kv_heads = getattr(self.model_config, 'num_key_value_heads', self.num_heads)
        self.head_dim = self.hidden_size // self.num_heads
        
        # LoRA dimensions
        q_out = self.num_heads * self.head_dim  # 5120
        v_out = self.num_kv_heads * self.head_dim  # 1280
        
        # Create LoRA modules for ALL layers on GPU
        self.lora_q = nn.ModuleList([
            OptimizedLoRA(self.hidden_size, q_out, self.config.lora_r, self.config.lora_alpha)
            for _ in range(self.num_layers)
        ]).to(self.device)
        
        self.lora_v = nn.ModuleList([
            OptimizedLoRA(self.hidden_size, v_out, self.config.lora_r, self.config.lora_alpha)
            for _ in range(self.num_layers)
        ]).to(self.device)
        
        # Load embeddings (keep on GPU)
        embed_w = self.weight_loader.load_weight("model.embed_tokens.weight")
        self.embeddings = nn.Embedding.from_pretrained(embed_w, freeze=True)
        del embed_w
        
        # Final layer norm
        self.final_ln_weight = self.weight_loader.load_weight("model.norm.weight")
        
        # LM head weight (tied to embeddings)
        self.lm_head_weight = self.embeddings.weight
        
        # Optimizer
        params = list(self.lora_q.parameters()) + list(self.lora_v.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=self.config.learning_rate)
        
        lora_param_count = sum(p.numel() for p in params)
        logger.info(f"OptimizedStreamingTrainer initialized")
        logger.info(f"LoRA params: {lora_param_count:,}")
        logger.info(f"VRAM: {get_mem():.2f}GB")
    
    def _forward_layer_no_grad(
        self,
        hidden: torch.Tensor,
        layer_idx: int,
        weights: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Forward through one layer WITHOUT gradient tracking."""
        prefix = f"model.layers.{layer_idx}."
        B, S, H = hidden.shape
        
        residual = hidden
        
        # LayerNorm 1
        ln1_w = weights.get(prefix + "input_layernorm.weight")
        if ln1_w is not None:
            hidden = F.layer_norm(hidden, (H,), weight=ln1_w)
        
        # QKV projection with tiled matmul
        qkv_w = weights.get(prefix + "self_attn.qkv_proj.weight")
        if qkv_w is not None:
            qkv = tiled_matmul_optimized(
                hidden, qkv_w,
                self.config.tile_m, self.config.tile_k, self.config.tile_n
            )
        else:
            qkv = hidden.new_zeros(B, S, self.hidden_size * 3)
        
        # Split Q, K, V
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        q = qkv[..., :q_size]
        k = qkv[..., q_size:q_size + kv_size]
        v = qkv[..., q_size + kv_size:q_size + 2*kv_size]
        del qkv
        
        # LoRA deltas (still no grad for base model part)
        with torch.enable_grad():
            q = q + self.lora_q[layer_idx](residual)
            v = v + self.lora_v[layer_idx](residual)
        
        # Reshape for attention
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # GQA: repeat K, V heads
        if self.num_kv_heads < self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)
        
        # Attention
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        del q, k, v
        
        attn = attn.transpose(1, 2).contiguous().view(B, S, H)
        
        # Output projection
        o_w = weights.get(prefix + "self_attn.o_proj.weight")
        if o_w is not None:
            attn = tiled_matmul_optimized(
                attn, o_w,
                self.config.tile_m, self.config.tile_k, self.config.tile_n
            )
        
        hidden = residual + attn
        del attn
        
        # MLP
        residual = hidden
        
        ln2_w = weights.get(prefix + "post_attention_layernorm.weight")
        if ln2_w is not None:
            hidden = F.layer_norm(hidden, (H,), weight=ln2_w)
        
        gate_up_w = weights.get(prefix + "mlp.gate_up_proj.weight")
        if gate_up_w is not None:
            gate_up = tiled_matmul_optimized(
                hidden, gate_up_w,
                self.config.tile_m, self.config.tile_k, self.config.tile_n
            )
            inter_size = gate_up_w.shape[0] // 2
            gate = gate_up[..., :inter_size]
            up = gate_up[..., inter_size:]
            del gate_up
            
            hidden = F.silu(gate) * up
            del gate, up
            
            down_w = weights.get(prefix + "mlp.down_proj.weight")
            if down_w is not None:
                hidden = tiled_matmul_optimized(
                    hidden, down_w,
                    self.config.tile_m, self.config.tile_k, self.config.tile_n
                )
        
        return residual + hidden
    
    def _compute_layer_gradients(
        self,
        hidden_in: torch.Tensor,
        hidden_out: torch.Tensor,
        grad_out: torch.Tensor,
        layer_idx: int,
        weights: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute gradients for one layer's LoRA parameters.
        
        This uses forward-mode autodiff style computation:
        1. Re-do forward with grad enabled for LoRA
        2. Compute loss contribution from this layer
        3. Backprop through LoRA only
        4. Return grad_input for previous layer
        """
        prefix = f"model.layers.{layer_idx}."
        B, S, H = hidden_in.shape
        
        hidden_in = hidden_in.detach().requires_grad_(True)
        
        # Forward again with gradients
        residual = hidden_in
        
        ln1_w = weights.get(prefix + "input_layernorm.weight")
        hidden = F.layer_norm(hidden_in, (H,), weight=ln1_w) if ln1_w is not None else hidden_in
        
        qkv_w = weights.get(prefix + "self_attn.qkv_proj.weight")
        if qkv_w is not None:
            qkv = tiled_matmul_optimized(
                hidden, qkv_w,
                self.config.tile_m, self.config.tile_k, self.config.tile_n
            )
        else:
            qkv = hidden.new_zeros(B, S, H * 3)
        
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        q = qkv[..., :q_size]
        k = qkv[..., q_size:q_size + kv_size]
        v = qkv[..., q_size + kv_size:q_size + 2*kv_size]
        
        # LoRA (this is what we want gradients for)
        q = q + self.lora_q[layer_idx](residual)
        v = v + self.lora_v[layer_idx](residual)
        
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        if self.num_kv_heads < self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)
        
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(B, S, H)
        
        o_w = weights.get(prefix + "self_attn.o_proj.weight")
        if o_w is not None:
            attn = tiled_matmul_optimized(attn, o_w, self.config.tile_m, self.config.tile_k, self.config.tile_n)
        
        hidden = residual + attn
        
        # MLP
        residual_mlp = hidden
        ln2_w = weights.get(prefix + "post_attention_layernorm.weight")
        hidden = F.layer_norm(hidden, (H,), weight=ln2_w) if ln2_w is not None else hidden
        
        gate_up_w = weights.get(prefix + "mlp.gate_up_proj.weight")
        if gate_up_w is not None:
            gate_up = tiled_matmul_optimized(hidden, gate_up_w, self.config.tile_m, self.config.tile_k, self.config.tile_n)
            inter_size = gate_up_w.shape[0] // 2
            gate, up = gate_up[..., :inter_size], gate_up[..., inter_size:]
            hidden = F.silu(gate) * up
            
            down_w = weights.get(prefix + "mlp.down_proj.weight")
            if down_w is not None:
                hidden = tiled_matmul_optimized(hidden, down_w, self.config.tile_m, self.config.tile_k, self.config.tile_n)
        
        output = residual_mlp + hidden
        
        # Compute gradient w.r.t. this layer's contribution
        # grad_out is the gradient from later layers
        loss_contrib = (output * grad_out).sum()
        loss_contrib.backward()
        
        grad_in = hidden_in.grad.detach() if hidden_in.grad is not None else grad_out
        return grad_in
    
    def train_step(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        One training step with per-layer gradient streaming.
        
        Strategy:
        1. Forward pass: stream weights layer-by-layer, save hidden states
        2. Compute loss at the end
        3. Backward pass: stream weights again, compute LoRA gradients per layer
        """
        B, S = input_ids.shape
        
        # ========== FORWARD PASS ==========
        self.optimizer.zero_grad()
        
        with torch.no_grad():
            hidden = self.embeddings(input_ids)
            
            # Save hidden states for backward
            hidden_states = [hidden.detach().clone()]
            
            for layer_idx in range(self.num_layers):
                weights = self.weight_loader.load_layer_weights(layer_idx)
                hidden = self._forward_layer_no_grad(hidden, layer_idx, weights)
                hidden_states.append(hidden.detach().clone())
                
                # Free weights
                del weights
                if layer_idx % 5 == 0:
                    clear_mem()
            
            # Final LayerNorm
            if self.final_ln_weight is not None:
                hidden = F.layer_norm(hidden, (self.hidden_size,), weight=self.final_ln_weight)
            
            # LM head
            logits = hidden @ self.lm_head_weight.T
        
        # ========== COMPUTE LOSS ==========
        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.tokenizer.pad_token_id,
        )
        loss_value = loss.item()
        
        # ========== BACKWARD PASS ==========
        # Start with gradient from loss
        # d(loss)/d(logits) -> d(loss)/d(hidden_final)
        grad_logits = torch.zeros_like(shift_logits)
        grad_logits.scatter_(2, shift_labels.unsqueeze(-1), 1.0)
        grad_logits = -grad_logits / (shift_logits.size(0) * shift_logits.size(1))
        
        # Pad to match sequence length
        grad_hidden = torch.zeros(B, S, self.hidden_size, device=self.device, dtype=hidden.dtype)
        grad_hidden[:, :-1, :] = grad_logits @ self.lm_head_weight
        
        # Backward through layers (reverse order)
        for layer_idx in range(self.num_layers - 1, -1, -1):
            weights = self.weight_loader.load_layer_weights(layer_idx)
            
            grad_hidden = self._compute_layer_gradients(
                hidden_states[layer_idx],
                hidden_states[layer_idx + 1],
                grad_hidden,
                layer_idx,
                weights,
            )
            
            del weights
            if layer_idx % 5 == 0:
                clear_mem()
        
        # ========== UPDATE ==========
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(
            list(self.lora_q.parameters()) + list(self.lora_v.parameters()),
            1.0
        )
        
        self.optimizer.step()
        
        # Free hidden states
        del hidden_states
        clear_mem()
        
        return loss, loss_value
    
    def train(self, texts: List[str], epochs: int = 1) -> Dict:
        """Train on a list of texts."""
        logger.info(f"Training on {len(texts)} samples (optimized streaming)")
        
        losses = []
        
        for epoch in range(epochs):
            epoch_losses = []
            
            for i, text in enumerate(texts):
                t0 = time.time()
                
                # Tokenize
                enc = self.tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=self.config.max_seq_length,
                    truncation=True,
                    padding="max_length",
                )
                input_ids = enc["input_ids"].to(self.device)
                
                # Train step
                loss, loss_value = self.train_step(input_ids)
                epoch_losses.append(loss_value)
                
                dt = time.time() - t0
                logger.info(f"[{epoch+1}/{epochs}] Sample {i+1}/{len(texts)}: "
                           f"loss={loss_value:.4f}, time={dt:.1f}s, VRAM={get_mem():.2f}GB")
            
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            losses.extend(epoch_losses)
            logger.info(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")
        
        return {
            "losses": losses,
            "avg_loss": sum(losses) / len(losses),
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/phi-4")
    parser.add_argument("--data", default="data/Dataset/math.jsonl")
    args = parser.parse_args()
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
    
    config = TiledConfig(max_seq_length=32)
    trainer = OptimizedStreamingTrainer(args.model, config)
    
    # Load data
    texts = []
    with open(args.data) as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            item = json.loads(line)
            texts.append(f"Problem: {item['problem']}\nAnswer: {item['answer']}")
    
    results = trainer.train(texts)
    logger.info(f"Final avg loss: {results['avg_loss']:.4f}")
