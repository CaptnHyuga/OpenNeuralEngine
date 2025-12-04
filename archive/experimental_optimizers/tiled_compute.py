"""
Tiled Matrix Multiplication with Memory Optimization

The key insight: Standard PyTorch matmul is generic and not optimized for our specific case.
We know:
1. Weight matrices are FIXED (we're doing inference/forward pass)
2. We load weights from disk sequentially
3. We can pre-process weights into optimal layout

Strategy:
---------
1. Tile weights into GPU-cache-friendly blocks
2. Stream tiles with double-buffering (load next while computing current)  
3. Accumulate partial results
4. Use Triton for custom kernels without writing raw CUDA

This should give us 2-4x speedup by maximizing memory bandwidth utilization.
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

# Check if Triton is available for custom kernels
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    logger.warning("Triton not available, falling back to PyTorch")


@dataclass
class TiledConfig:
    """Configuration for tiled computation."""
    max_seq_length: int = 64
    lora_r: int = 8
    lora_alpha: int = 16
    learning_rate: float = 2e-4
    
    # Tiling parameters - tuned for GTX 1650 (Turing, 4GB, 896 CUDA cores)
    # L2 cache: 1MB, Shared memory per block: 48KB
    tile_m: int = 64   # Tile size for batch*seq dimension
    tile_n: int = 64   # Tile size for output dimension  
    tile_k: int = 32   # Tile size for reduction (inner) dimension
    
    # Double buffering
    num_buffers: int = 2
    
    # Prefetch depth
    prefetch_tiles: int = 2


def get_mem():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9
    return 0


def clear_mem():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if HAS_TRITON:
    @triton.jit
    def tiled_matmul_kernel(
        # Pointers
        x_ptr, w_ptr, out_ptr,
        # Dimensions
        M, N, K,
        # Strides
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_outm, stride_outn,
        # Tile sizes (must be constexpr for Triton)
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        """
        Tiled matrix multiplication: out = x @ w.T
        
        x: [M, K]
        w: [N, K]  
        out: [M, N]
        
        Each block computes a BLOCK_M x BLOCK_N tile of the output.
        """
        # Block indices
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        
        # Offsets for this block
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        
        # Pointers to first tile
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
        
        # Accumulator
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        # Loop over K dimension in tiles
        for k in range(0, K, BLOCK_K):
            # Load tiles with masking for edge cases
            mask_k = (k + offs_k) < K
            mask_m = offs_m < M
            mask_n = offs_n < N
            
            x_tile = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            w_tile = tl.load(w_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
            
            # Compute partial dot product: acc += x_tile @ w_tile.T
            acc += tl.dot(x_tile, tl.trans(w_tile))
            
            # Advance pointers
            x_ptrs += BLOCK_K * stride_xk
            w_ptrs += BLOCK_K * stride_wk
        
        # Write output
        out_ptrs = out_ptr + offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn
        mask_out = (offs_m < M)[:, None] & (offs_n < N)[None, :]
        tl.store(out_ptrs, acc.to(tl.float16), mask=mask_out)


def tiled_matmul(x: torch.Tensor, w: torch.Tensor, config: TiledConfig) -> torch.Tensor:
    """
    Optimized matrix multiplication using tiling.
    
    x: [batch, seq, in_features] or [M, K]
    w: [out_features, in_features] or [N, K]
    
    Returns: [batch, seq, out_features] or [M, N]
    """
    # Reshape to 2D for computation
    original_shape = x.shape
    if x.dim() == 3:
        batch, seq, K = x.shape
        M = batch * seq
        x = x.view(M, K)
    else:
        M, K = x.shape
    
    N = w.shape[0]
    
    if HAS_TRITON and x.is_cuda:
        # Use Triton kernel
        out = torch.empty(M, N, dtype=x.dtype, device=x.device)
        
        # Grid of blocks
        grid = (
            triton.cdiv(M, config.tile_m),
            triton.cdiv(N, config.tile_n),
        )
        
        tiled_matmul_kernel[grid](
            x, w, out,
            M, N, K,
            x.stride(0), x.stride(1),
            w.stride(0), w.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=config.tile_m,
            BLOCK_N=config.tile_n,
            BLOCK_K=config.tile_k,
        )
    else:
        # Fallback to PyTorch
        out = F.linear(x, w)
    
    # Reshape back
    if len(original_shape) == 3:
        out = out.view(original_shape[0], original_shape[1], N)
    
    return out


class StreamingWeightBuffer:
    """
    Double-buffered weight streaming.
    
    While GPU computes on buffer A, we load next weights into buffer B.
    This hides memory latency almost completely.
    """
    
    def __init__(self, model_path: str, device: torch.device, num_buffers: int = 2):
        self.path = Path(model_path)
        self.device = device
        self.num_buffers = num_buffers
        
        with open(self.path / "model.safetensors.index.json") as f:
            self._index = json.load(f)
        
        # Pre-open all files
        from safetensors import safe_open
        self._handles = {}
        for fname in set(self._index["weight_map"].values()):
            self._handles[fname] = safe_open(
                str(self.path / fname), framework="pt", device="cpu"
            )
        
        # Double buffers on GPU
        self._buffers: List[Dict[str, torch.Tensor]] = [{} for _ in range(num_buffers)]
        self._current_buffer = 0
        
        # CUDA stream for async operations
        if device.type == 'cuda':
            self._stream = torch.cuda.Stream()
        else:
            self._stream = None
    
    def prefetch(self, weight_names: List[str], buffer_idx: int = None):
        """Prefetch weights into a buffer asynchronously."""
        if buffer_idx is None:
            buffer_idx = (self._current_buffer + 1) % self.num_buffers
        
        buffer = self._buffers[buffer_idx]
        buffer.clear()
        
        stream = self._stream or torch.cuda.current_stream()
        
        with torch.cuda.stream(stream) if self._stream else nullcontext():
            for name in weight_names:
                fname = self._index["weight_map"].get(name)
                if fname and fname in self._handles:
                    # Load to GPU in background stream
                    tensor = self._handles[fname].get_tensor(name)
                    buffer[name] = tensor.to(device=self.device, dtype=torch.float16, non_blocking=True)
    
    def get_buffer(self, buffer_idx: int = None) -> Dict[str, torch.Tensor]:
        """Get a buffer, waiting for any pending transfers."""
        if buffer_idx is None:
            buffer_idx = self._current_buffer
        
        if self._stream:
            self._stream.synchronize()
        
        return self._buffers[buffer_idx]
    
    def swap_buffers(self):
        """Swap to the next buffer."""
        self._current_buffer = (self._current_buffer + 1) % self.num_buffers
    
    def load_sync(self, name: str) -> Optional[torch.Tensor]:
        """Synchronous load for one-off weights."""
        fname = self._index["weight_map"].get(name)
        if not fname or fname not in self._handles:
            return None
        return self._handles[fname].get_tensor(name).to(device=self.device, dtype=torch.float16)
    
    def close(self):
        self._handles.clear()
        self._buffers.clear()


from contextlib import nullcontext


class TiledLoRA(nn.Module):
    """LoRA optimized for tiled computation."""
    
    def __init__(self, in_dim: int, out_dim: int, r: int = 8, alpha: int = 16):
        super().__init__()
        self.scale = alpha / r
        # Initialize with small values for stability
        self.A = nn.Parameter(torch.randn(r, in_dim) * (0.01 / math.sqrt(in_dim)))
        self.B = nn.Parameter(torch.zeros(out_dim, r))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Two small matmuls - these are compute-bound, not memory-bound
        # so tiling doesn't help much here
        return (x.float() @ self.A.T @ self.B.T * self.scale).to(x.dtype)


class TiledTrainer:
    """
    Trainer using tiled matrix multiplication and double-buffered weight streaming.
    
    Key optimizations:
    1. Tiled matmul using Triton kernels
    2. Double-buffered weight loading (hide latency)
    3. Optimal data layout for coalesced access
    """
    
    def __init__(self, model_path: str, config: Optional[TiledConfig] = None):
        self.config = config or TiledConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        from transformers import AutoConfig, AutoTokenizer
        
        self.model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Double-buffered weight loading
        self.weight_buffer = StreamingWeightBuffer(
            model_path, self.device, self.config.num_buffers
        )
        
        # Dimensions
        self.hidden_size = self.model_config.hidden_size
        self.num_layers = self.model_config.num_hidden_layers
        self.num_heads = self.model_config.num_attention_heads
        self.num_kv_heads = getattr(self.model_config, 'num_key_value_heads', self.num_heads)
        self.head_dim = self.hidden_size // self.num_heads
        
        q_out = self.num_heads * self.head_dim
        v_out = self.num_kv_heads * self.head_dim
        
        # LoRA for each layer
        self.lora_q = nn.ModuleList([
            TiledLoRA(self.hidden_size, q_out, self.config.lora_r, self.config.lora_alpha)
            for _ in range(self.num_layers)
        ]).to(self.device)
        
        self.lora_v = nn.ModuleList([
            TiledLoRA(self.hidden_size, v_out, self.config.lora_r, self.config.lora_alpha)
            for _ in range(self.num_layers)
        ]).to(self.device)
        
        # Embeddings
        embed_w = self.weight_buffer.load_sync("model.embed_tokens.weight")
        self.embeddings = nn.Embedding.from_pretrained(embed_w, freeze=True)
        del embed_w
        
        # Optimizer
        params = list(self.lora_q.parameters()) + list(self.lora_v.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=self.config.learning_rate)
        
        logger.info(f"TiledTrainer initialized")
        logger.info(f"Triton available: {HAS_TRITON}")
        logger.info(f"Tile sizes: M={self.config.tile_m}, N={self.config.tile_n}, K={self.config.tile_k}")
        logger.info(f"LoRA params: {sum(p.numel() for p in params):,}")
        logger.info(f"VRAM: {get_mem():.2f}GB")
    
    def _get_layer_weight_names(self, layer_idx: int) -> List[str]:
        """Get all weight names for a layer."""
        prefix = f"model.layers.{layer_idx}."
        return [
            prefix + "self_attn.qkv_proj.weight",
            prefix + "self_attn.o_proj.weight",
            prefix + "input_layernorm.weight",
            prefix + "post_attention_layernorm.weight",
            prefix + "mlp.gate_up_proj.weight",
            prefix + "mlp.down_proj.weight",
        ]
    
    def _forward_layer(
        self,
        hidden: torch.Tensor,
        layer_idx: int,
        weights: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Forward through one layer using tiled matmul."""
        prefix = f"model.layers.{layer_idx}."
        B, S, H = hidden.shape
        residual = hidden
        
        # LayerNorm 1
        ln1_w = weights.get(prefix + "input_layernorm.weight")
        if ln1_w is not None:
            hidden = F.layer_norm(hidden, (H,), weight=ln1_w)
        
        # QKV - use tiled matmul
        qkv_w = weights.get(prefix + "self_attn.qkv_proj.weight")
        if qkv_w is not None:
            qkv = tiled_matmul(hidden, qkv_w, self.config)
        else:
            qkv = hidden.new_zeros(B, S, self.hidden_size * 3)
        
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        
        q = qkv[..., :q_size]
        k = qkv[..., q_size:q_size + kv_size]
        v = qkv[..., q_size + kv_size:]
        del qkv
        
        # LoRA
        q = q + self.lora_q[layer_idx](residual)
        v = v + self.lora_v[layer_idx](residual)
        
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
        
        # Output projection
        o_w = weights.get(prefix + "self_attn.o_proj.weight")
        if o_w is not None:
            attn = tiled_matmul(attn, o_w, self.config)
        
        hidden = residual + attn
        del attn
        
        # MLP
        residual = hidden
        
        ln2_w = weights.get(prefix + "post_attention_layernorm.weight")
        if ln2_w is not None:
            hidden = F.layer_norm(hidden, (H,), weight=ln2_w)
        
        gate_up_w = weights.get(prefix + "mlp.gate_up_proj.weight")
        if gate_up_w is not None:
            gate_up = tiled_matmul(hidden, gate_up_w, self.config)
            inter_size = gate_up_w.shape[0] // 2
            gate = gate_up[..., :inter_size]
            up = gate_up[..., inter_size:]
            del gate_up
            
            hidden = F.silu(gate) * up
            del gate, up
            
            down_w = weights.get(prefix + "mlp.down_proj.weight")
            if down_w is not None:
                hidden = tiled_matmul(hidden, down_w, self.config)
        
        hidden = residual + hidden
        
        return hidden
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward with double-buffered weight streaming.
        
        NOTE: We can't use gradient checkpointing with streamed weights because
        the weights won't be available during recomputation. Instead, we compute
        forward through all layers, keeping only final output for backward.
        """
        hidden = self.embeddings(input_ids)
        
        # Prefetch first layer
        self.weight_buffer.prefetch(self._get_layer_weight_names(0), buffer_idx=0)
        
        for layer_idx in range(self.num_layers):
            # Prefetch next layer while computing current
            if layer_idx + 1 < self.num_layers:
                self.weight_buffer.prefetch(
                    self._get_layer_weight_names(layer_idx + 1),
                    buffer_idx=(layer_idx + 1) % self.config.num_buffers
                )
            
            # Get current layer weights (waits for prefetch if needed)
            weights = self.weight_buffer.get_buffer(layer_idx % self.config.num_buffers)
            
            # Forward without checkpointing - weights aren't saved between passes
            hidden = self._forward_layer(hidden, layer_idx, weights)
            
            # Clear processed weights from buffer
            weights.clear()
            
            if layer_idx % 5 == 0:
                clear_mem()
        
        # Final norm
        final_ln = self.weight_buffer.load_sync("model.norm.weight")
        hidden = F.layer_norm(hidden, (self.hidden_size,), weight=final_ln)
        
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
        
        return loss.item()
    
    def train(self, texts: List[str]) -> Dict[str, float]:
        """Train on texts."""
        logger.info(f"Training on {len(texts)} samples (tiled matmul)")
        
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
            
            clear_mem()
        
        return {"avg_loss": total_loss / len(texts)}
