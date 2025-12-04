"""
Maximum GPU Utilization Training

Goal: Keep GPU busy 80%+ of the time instead of 10%

Key strategies:
1. CUDA Streams: Overlap data transfer with compute
2. Double Buffering: Load next layer while computing current
3. Persistent Caching: Keep frequently-used layers on GPU
4. Fused Operations: Reduce kernel launch overhead
5. Memory Pool: Pre-allocate to avoid allocation stalls

Target: <3s/sample (vs 6s current)
"""

import torch
import torch.nn.functional as F
from torch import nn
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import logging
import json
import math
import gc
import time
import threading
from queue import Queue

logger = logging.getLogger(__name__)


@dataclass
class MaxUtilConfig:
    max_seq_length: int = 32
    lora_r: int = 4
    lora_alpha: int = 8
    learning_rate: float = 2e-4
    # Sparse config
    compute_first_n: int = 3
    compute_last_n: int = 5
    # Cache config - layers to keep permanently on GPU
    cache_layers: List[int] = None  # Will be set to first 2 layers
    # Prefetch config
    prefetch_ahead: int = 2  # How many layers to prefetch


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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x.float() @ self.A.T @ self.B.T * self.scale).to(x.dtype)


class AsyncDoubleBuffer:
    """
    Double-buffered async weight loading with CUDA streams.
    
    While GPU computes on buffer A, we load into buffer B.
    Then swap and repeat.
    """
    
    def __init__(self, model_path: str, device: torch.device):
        self.path = Path(model_path)
        self.device = device
        
        with open(self.path / "model.safetensors.index.json") as f:
            self._index = json.load(f)
        
        from safetensors import safe_open
        self._handles = {}
        for fname in set(self._index["weight_map"].values()):
            self._handles[fname] = safe_open(
                str(self.path / fname), framework="pt", device="cpu"
            )
        
        # Double buffers
        self._buffers = [{}, {}]
        self._current_buffer = 0
        
        # CUDA streams for async transfer
        self._transfer_stream = torch.cuda.Stream()
        self._compute_stream = torch.cuda.default_stream()
        
        # Pre-allocated pinned memory pools (reuse to avoid allocation)
        self._pinned_pools: Dict[Tuple[int, ...], torch.Tensor] = {}
        
        # Background loading thread
        self._load_queue = Queue()
        self._load_thread = None
        self._stop_thread = False
        
        # Cached layers (persistent on GPU)
        self._cached_layers: Dict[int, Dict[str, torch.Tensor]] = {}
    
    def _get_pinned_buffer(self, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
        """Get or create a pinned memory buffer."""
        key = (shape, dtype)
        if key not in self._pinned_pools:
            self._pinned_pools[key] = torch.empty(shape, dtype=dtype, pin_memory=True)
        return self._pinned_pools[key]
    
    def cache_layer(self, layer_idx: int):
        """Permanently cache a layer on GPU."""
        if layer_idx in self._cached_layers:
            return
        
        weights = self._load_layer_sync(layer_idx)
        self._cached_layers[layer_idx] = weights
        logger.info(f"Cached layer {layer_idx} on GPU, VRAM: {get_mem():.2f}GB")
    
    def is_cached(self, layer_idx: int) -> bool:
        return layer_idx in self._cached_layers
    
    def get_cached(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        return self._cached_layers.get(layer_idx, {})
    
    def _load_layer_sync(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """Synchronously load a layer's weights."""
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
            name = prefix + suffix
            fname = self._index["weight_map"].get(name)
            if fname and fname in self._handles:
                cpu_tensor = self._handles[fname].get_tensor(name)
                
                # Use pinned buffer for fast transfer
                pinned = self._get_pinned_buffer(cpu_tensor.shape, cpu_tensor.dtype)
                pinned.copy_(cpu_tensor)
                
                # Transfer with non-blocking
                weights[name] = pinned.to(
                    device=self.device, 
                    dtype=torch.float16,
                    non_blocking=True
                )
        
        return weights
    
    def prefetch_layer(self, layer_idx: int, buffer_idx: int):
        """
        Start loading a layer into specified buffer using transfer stream.
        This runs async - doesn't block compute.
        """
        if self.is_cached(layer_idx):
            return  # Already on GPU
        
        buffer = self._buffers[buffer_idx]
        buffer.clear()
        
        prefix = f"model.layers.{layer_idx}."
        suffixes = [
            "input_layernorm.weight",
            "self_attn.qkv_proj.weight",
            "self_attn.o_proj.weight", 
            "post_attention_layernorm.weight",
            "mlp.gate_up_proj.weight",
            "mlp.down_proj.weight",
        ]
        
        with torch.cuda.stream(self._transfer_stream):
            for suffix in suffixes:
                name = prefix + suffix
                fname = self._index["weight_map"].get(name)
                if fname and fname in self._handles:
                    cpu_tensor = self._handles[fname].get_tensor(name)
                    pinned = self._get_pinned_buffer(cpu_tensor.shape, cpu_tensor.dtype)
                    pinned.copy_(cpu_tensor)
                    buffer[name] = pinned.to(
                        device=self.device,
                        dtype=torch.float16, 
                        non_blocking=True
                    )
    
    def get_layer(self, layer_idx: int, buffer_idx: int) -> Dict[str, torch.Tensor]:
        """Get layer weights, waiting for transfer if needed."""
        if self.is_cached(layer_idx):
            return self.get_cached(layer_idx)
        
        # Wait for transfer to complete
        self._transfer_stream.synchronize()
        return self._buffers[buffer_idx]
    
    def load_single(self, name: str) -> Optional[torch.Tensor]:
        fname = self._index["weight_map"].get(name)
        if not fname:
            return None
        cpu = self._handles[fname].get_tensor(name)
        pinned = self._get_pinned_buffer(cpu.shape, cpu.dtype)
        pinned.copy_(cpu)
        return pinned.to(device=self.device, dtype=torch.float16)


class MaxGPUUtilizationTrainer:
    """
    Trainer optimized for maximum GPU utilization.
    
    Key optimizations:
    1. Cache first 2 layers permanently on GPU (most accessed)
    2. Double-buffer remaining layers with async transfer
    3. Overlap compute and transfer using CUDA streams
    4. Pre-allocated memory pools to avoid allocation stalls
    """
    
    def __init__(self, model_path: str, config: Optional[MaxUtilConfig] = None):
        self.config = config or MaxUtilConfig()
        if self.config.cache_layers is None:
            self.config.cache_layers = [0, 1]  # Cache first 2 layers
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        from transformers import AutoConfig, AutoTokenizer
        
        self.model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.loader = AsyncDoubleBuffer(model_path, self.device)
        
        # Dimensions
        self.hidden_size = self.model_config.hidden_size
        self.num_layers = self.model_config.num_hidden_layers
        self.num_heads = self.model_config.num_attention_heads
        self.num_kv_heads = getattr(self.model_config, 'num_key_value_heads', self.num_heads)
        self.head_dim = self.hidden_size // self.num_heads
        
        # Active layers (sparse)
        first_n = self.config.compute_first_n
        last_n = self.config.compute_last_n
        self.active_layers = (
            list(range(first_n)) + 
            list(range(self.num_layers - last_n, self.num_layers))
        )
        
        # LoRA modules
        q_out = self.num_heads * self.head_dim
        v_out = self.num_kv_heads * self.head_dim
        
        self.lora_q = nn.ModuleDict({
            str(i): QuickLoRA(self.hidden_size, q_out, self.config.lora_r, self.config.lora_alpha)
            for i in self.active_layers
        }).to(self.device)
        
        self.lora_v = nn.ModuleDict({
            str(i): QuickLoRA(self.hidden_size, v_out, self.config.lora_r, self.config.lora_alpha)
            for i in self.active_layers
        }).to(self.device)
        
        # Embeddings
        embed_w = self.loader.load_single("model.embed_tokens.weight")
        torch.cuda.synchronize()
        self.embeddings = nn.Embedding.from_pretrained(embed_w, freeze=True)
        
        self.final_ln = self.loader.load_single("model.norm.weight")
        torch.cuda.synchronize()
        self.lm_head = self.embeddings.weight
        
        # Cache specified layers
        logger.info(f"Caching layers {self.config.cache_layers} on GPU...")
        for layer_idx in self.config.cache_layers:
            if layer_idx in self.active_layers:
                self.loader.cache_layer(layer_idx)
        
        # Optimizer
        params = list(self.lora_q.parameters()) + list(self.lora_v.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=self.config.learning_rate)
        
        logger.info("MaxGPUUtilizationTrainer initialized")
        logger.info(f"Active layers: {self.active_layers}")
        logger.info(f"Cached layers: {self.config.cache_layers}")
        logger.info(f"LoRA params: {sum(p.numel() for p in params):,}")
        logger.info(f"VRAM: {get_mem():.2f}GB")
    
    def _forward_layer(
        self,
        hidden: torch.Tensor,
        layer_idx: int,
        weights: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Forward through one layer - optimized for speed."""
        prefix = f"model.layers.{layer_idx}."
        B, S, H = hidden.shape
        residual = hidden
        
        # LayerNorm + QKV in one block
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
        
        # LoRA
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
        
        o_w = weights.get(prefix + "self_attn.o_proj.weight")
        if o_w is not None:
            attn = F.linear(attn, o_w)
        
        hidden = residual + attn
        del attn
        
        # MLP
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
        """
        Training step with maximum GPU utilization.
        
        Strategy:
        - Start prefetching layer N+1 while computing layer N
        - Use double buffering to avoid blocking
        - Cached layers don't need prefetch
        """
        B, S = input_ids.shape
        self.optimizer.zero_grad()
        
        # Forward pass with overlapped loading
        hidden = self.embeddings(input_ids)
        hidden_states = [hidden.detach()]
        
        # Start prefetch for first non-cached layer
        buffer_idx = 0
        non_cached = [l for l in self.active_layers if not self.loader.is_cached(l)]
        if non_cached:
            self.loader.prefetch_layer(non_cached[0], buffer_idx)
        
        for i, layer_idx in enumerate(self.active_layers):
            # Find next layer to prefetch
            remaining = [l for l in self.active_layers[i+1:] if not self.loader.is_cached(l)]
            if remaining:
                next_buffer = 1 - buffer_idx
                self.loader.prefetch_layer(remaining[0], next_buffer)
            
            # Get current layer weights
            if self.loader.is_cached(layer_idx):
                weights = self.loader.get_cached(layer_idx)
            else:
                weights = self.loader.get_layer(layer_idx, buffer_idx)
                buffer_idx = 1 - buffer_idx  # Swap buffer
            
            # Compute
            with torch.no_grad():
                hidden = self._forward_layer(hidden, layer_idx, weights)
            hidden_states.append(hidden.detach())
        
        # Final layers
        with torch.no_grad():
            hidden = F.layer_norm(hidden, (self.hidden_size,), weight=self.final_ln)
            logits = F.linear(hidden, self.lm_head)
        
        # Loss
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.tokenizer.pad_token_id,
        )
        loss_val = loss.item()
        
        # Backward with overlapped loading
        grad = torch.zeros(B, S, self.hidden_size, device=self.device, dtype=hidden.dtype)
        probs = F.softmax(shift_logits.detach(), dim=-1)
        grad_logits = probs.clone()
        grad_logits.scatter_add_(
            2, shift_labels.unsqueeze(-1),
            -torch.ones_like(shift_labels.unsqueeze(-1), dtype=grad_logits.dtype)
        )
        grad[:, :-1, :] = grad_logits @ self.lm_head / (S - 1)
        
        # Backward through layers (reverse)
        buffer_idx = 0
        reversed_layers = list(reversed(self.active_layers))
        non_cached_rev = [l for l in reversed_layers if not self.loader.is_cached(l)]
        if non_cached_rev:
            self.loader.prefetch_layer(non_cached_rev[0], buffer_idx)
        
        for i, layer_idx in enumerate(reversed_layers):
            # Prefetch next
            remaining = [l for l in reversed_layers[i+1:] if not self.loader.is_cached(l)]
            if remaining:
                next_buffer = 1 - buffer_idx
                self.loader.prefetch_layer(remaining[0], next_buffer)
            
            # Get weights
            if self.loader.is_cached(layer_idx):
                weights = self.loader.get_cached(layer_idx)
            else:
                weights = self.loader.get_layer(layer_idx, buffer_idx)
                buffer_idx = 1 - buffer_idx
            
            h_in = hidden_states[-(i+2)].requires_grad_(True)
            
            # Recompute forward for gradient
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
            
            # MLP
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
        
        # Update
        torch.nn.utils.clip_grad_norm_(
            list(self.lora_q.parameters()) + list(self.lora_v.parameters()),
            1.0
        )
        self.optimizer.step()
        
        del hidden_states
        clear_mem()
        
        return loss_val
    
    def train(self, texts: List[str], epochs: int = 1) -> Dict:
        logger.info(f"Training on {len(texts)} samples (max GPU utilization)")
        
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
    
    # Cache layers 0, 1 (most accessed in forward + backward)
    config = MaxUtilConfig(
        max_seq_length=32,
        cache_layers=[0, 1],  # Cache first 2 layers permanently
    )
    trainer = MaxGPUUtilizationTrainer("models/phi-4", config)
    
    # Load data
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
