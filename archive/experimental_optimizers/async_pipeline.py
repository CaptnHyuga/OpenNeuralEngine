"""
Async Pipeline Training - Overlap IO with Compute

Key optimization: While GPU computes layer N, CPU loads layer N+1 weights.
This hides the disk-to-GPU transfer time.

On GTX 1650 with 4GB VRAM:
- Layer weights: ~600MB-1GB each
- PCIe transfer: ~10GB/s (takes ~60-100ms per layer)
- Layer compute: ~30-50ms per layer

Without pipelining: 90-150ms per layer (transfer + compute sequentially)
With pipelining: ~max(60ms, 50ms) = 60ms per layer (overlapped)

Expected speedup: ~1.5-2x
"""

import logging
import math
import json
import gc
import time
import threading
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
from queue import Queue
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_mem() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9
    return 0.0


def clear_mem():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@dataclass
class PipelineConfig:
    max_seq_length: int = 32
    lora_r: int = 4
    lora_alpha: int = 8
    learning_rate: float = 1e-4


class AsyncWeightLoader:
    """
    Asynchronous weight loading with double buffering.
    
    Uses a background thread to load next layer's weights while
    GPU computes on current layer.
    """
    
    def __init__(self, model_path: str, device: torch.device):
        self.path = Path(model_path)
        self.device = device
        
        with open(self.path / "model.safetensors.index.json") as f:
            self._index = json.load(f)
        
        # Open file handles
        self._handles: Dict[str, Any] = {}
        for fname in set(self._index["weight_map"].values()):
            self._handles[fname] = safe_open(
                str(self.path / fname), framework="pt", device="cpu"
            )
        
        # Double buffer
        self._buffers = [None, None]
        self._current = 0
        
        # Background loading
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._pending_future = None
        
        # CUDA stream for async transfer
        if device.type == 'cuda':
            self._transfer_stream = torch.cuda.Stream()
        else:
            self._transfer_stream = None
    
    def _load_layer_cpu(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """Load layer weights to CPU (runs in background thread)."""
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
                weights[name] = self._handles[fname].get_tensor(name)
        
        return weights
    
    def _transfer_to_gpu(self, cpu_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Transfer weights to GPU using async stream."""
        gpu_weights = {}
        
        stream = self._transfer_stream or torch.cuda.current_stream()
        with torch.cuda.stream(stream):
            for name, tensor in cpu_weights.items():
                gpu_weights[name] = tensor.to(
                    device=self.device, 
                    dtype=torch.float16,
                    non_blocking=True
                )
        
        return gpu_weights
    
    def prefetch(self, layer_idx: int):
        """Start loading next layer in background."""
        if self._pending_future is not None:
            return  # Already loading
        
        next_buffer = (self._current + 1) % 2
        self._pending_future = self._executor.submit(self._load_layer_cpu, layer_idx)
        self._pending_buffer = next_buffer
    
    def get_layer(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """Get layer weights, waiting for prefetch if needed."""
        # If we have a pending load, get it
        if self._pending_future is not None:
            cpu_weights = self._pending_future.result()
            self._pending_future = None
            
            # Transfer to GPU
            gpu_weights = self._transfer_to_gpu(cpu_weights)
            
            # Sync transfer
            if self._transfer_stream:
                self._transfer_stream.synchronize()
            
            return gpu_weights
        
        # Synchronous load (first layer or no prefetch)
        cpu_weights = self._load_layer_cpu(layer_idx)
        gpu_weights = self._transfer_to_gpu(cpu_weights)
        
        if self._transfer_stream:
            self._transfer_stream.synchronize()
        
        return gpu_weights
    
    def load_single(self, name: str) -> Optional[torch.Tensor]:
        """Load a single weight synchronously."""
        fname = self._index["weight_map"].get(name)
        if not fname or fname not in self._handles:
            return None
        return self._handles[fname].get_tensor(name).to(
            device=self.device, dtype=torch.float16
        )
    
    def close(self):
        self._executor.shutdown(wait=False)


class SimpleLoRA(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, r: int = 4, alpha: int = 8):
        super().__init__()
        self.scale = alpha / r
        self.A = nn.Parameter(torch.randn(r, in_dim) * (0.01 / math.sqrt(in_dim)))
        self.B = nn.Parameter(torch.zeros(out_dim, r))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x.float() @ self.A.T @ self.B.T * self.scale).to(x.dtype)


class PipelineTrainer:
    """
    Training with async pipeline:
    1. Load layer N+1 weights while computing layer N
    2. Process one layer at a time for memory efficiency
    3. Per-layer gradient computation
    """
    
    def __init__(self, model_path: str, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        from transformers import AutoConfig, AutoTokenizer
        
        self.model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.loader = AsyncWeightLoader(model_path, self.device)
        
        # Dimensions
        self.hidden_size = self.model_config.hidden_size
        self.num_layers = self.model_config.num_hidden_layers
        self.num_heads = self.model_config.num_attention_heads
        self.num_kv_heads = getattr(self.model_config, 'num_key_value_heads', self.num_heads)
        self.head_dim = self.hidden_size // self.num_heads
        
        q_out = self.num_heads * self.head_dim
        v_out = self.num_kv_heads * self.head_dim
        
        # LoRA modules
        self.lora_q = nn.ModuleList([
            SimpleLoRA(self.hidden_size, q_out, self.config.lora_r, self.config.lora_alpha)
            for _ in range(self.num_layers)
        ]).to(self.device)
        
        self.lora_v = nn.ModuleList([
            SimpleLoRA(self.hidden_size, v_out, self.config.lora_r, self.config.lora_alpha)
            for _ in range(self.num_layers)
        ]).to(self.device)
        
        # Embeddings
        embed_w = self.loader.load_single("model.embed_tokens.weight")
        self.embeddings = nn.Embedding.from_pretrained(embed_w, freeze=True)
        del embed_w
        
        self.final_ln = self.loader.load_single("model.norm.weight")
        self.lm_head = self.embeddings.weight
        
        # Optimizer
        params = list(self.lora_q.parameters()) + list(self.lora_v.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=self.config.learning_rate)
        
        logger.info(f"PipelineTrainer initialized")
        logger.info(f"LoRA params: {sum(p.numel() for p in params):,}")
        logger.info(f"VRAM: {get_mem():.2f}GB")
    
    @torch.no_grad()
    def _forward_layer(
        self,
        hidden: torch.Tensor,
        layer_idx: int,
        weights: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Forward through one layer."""
        prefix = f"model.layers.{layer_idx}."
        B, S, H = hidden.shape
        residual = hidden
        
        # LayerNorm
        ln1_w = weights.get(prefix + "input_layernorm.weight")
        if ln1_w is not None:
            hidden = F.layer_norm(hidden, (H,), weight=ln1_w)
        
        # QKV
        qkv_w = weights.get(prefix + "self_attn.qkv_proj.weight")
        if qkv_w is not None:
            qkv = F.linear(hidden, qkv_w)
        else:
            return hidden
        
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        q = qkv[..., :q_size]
        k = qkv[..., q_size:q_size + kv_size]
        v = qkv[..., q_size + kv_size:q_size + 2*kv_size]
        del qkv
        
        # LoRA (enable grad just for this)
        with torch.enable_grad():
            residual_grad = residual.detach().requires_grad_(False)
            q = q + self.lora_q[layer_idx](residual_grad)
            v = v + self.lora_v[layer_idx](residual_grad)
        
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
    
    def _compute_lora_grads(
        self,
        hidden_in: torch.Tensor,
        layer_idx: int,
        weights: Dict[str, torch.Tensor],
        grad_out: torch.Tensor,
    ) -> torch.Tensor:
        """Compute gradients for LoRA parameters."""
        prefix = f"model.layers.{layer_idx}."
        B, S, H = hidden_in.shape
        
        hidden_in = hidden_in.detach().requires_grad_(True)
        residual = hidden_in
        
        ln1_w = weights.get(prefix + "input_layernorm.weight")
        hidden = F.layer_norm(hidden_in, (H,), weight=ln1_w) if ln1_w is not None else hidden_in
        
        qkv_w = weights.get(prefix + "self_attn.qkv_proj.weight")
        if qkv_w is None:
            return grad_out
        
        qkv = F.linear(hidden, qkv_w)
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        q = qkv[..., :q_size]
        k = qkv[..., q_size:q_size + kv_size]
        v = qkv[..., q_size + kv_size:q_size + 2*kv_size]
        
        # LoRA
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
            attn = F.linear(attn, o_w)
        
        hidden = residual + attn
        
        # MLP (simplified - no LoRA here)
        residual_mlp = hidden
        ln2_w = weights.get(prefix + "post_attention_layernorm.weight")
        hidden = F.layer_norm(hidden, (H,), weight=ln2_w) if ln2_w is not None else hidden
        
        gate_up_w = weights.get(prefix + "mlp.gate_up_proj.weight")
        if gate_up_w is not None:
            gate_up = F.linear(hidden, gate_up_w)
            inter_size = gate_up_w.shape[0] // 2
            gate, up = gate_up[..., :inter_size], gate_up[..., inter_size:]
            hidden = F.silu(gate) * up
            
            down_w = weights.get(prefix + "mlp.down_proj.weight")
            if down_w is not None:
                hidden = F.linear(hidden, down_w)
        
        output = residual_mlp + hidden
        
        # Backward
        loss = (output * grad_out).sum()
        loss.backward()
        
        return hidden_in.grad.detach() if hidden_in.grad is not None else grad_out
    
    def train_step(self, input_ids: torch.Tensor) -> float:
        """One training step with async pipeline."""
        B, S = input_ids.shape
        self.optimizer.zero_grad()
        
        # ========== FORWARD with pipelining ==========
        t_forward = time.time()
        
        with torch.no_grad():
            hidden = self.embeddings(input_ids)
            hidden_states = [hidden.clone()]
            
            # Start prefetching layer 1 while processing layer 0
            self.loader.prefetch(1) if self.num_layers > 1 else None
            
            for layer_idx in range(self.num_layers):
                # Get current layer weights
                weights = self.loader.get_layer(layer_idx)
                
                # Start prefetching next layer
                if layer_idx + 2 < self.num_layers:
                    self.loader.prefetch(layer_idx + 2)
                
                # Compute
                hidden = self._forward_layer(hidden, layer_idx, weights)
                hidden_states.append(hidden.clone())
                
                # Free weights immediately
                del weights
                
                if layer_idx % 10 == 0:
                    clear_mem()
            
            # Final
            hidden = F.layer_norm(hidden, (self.hidden_size,), weight=self.final_ln)
            logits = F.linear(hidden, self.lm_head)
        
        forward_time = time.time() - t_forward
        
        # ========== LOSS ==========
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.tokenizer.pad_token_id,
        )
        loss_val = loss.item()
        
        # ========== BACKWARD with pipelining ==========
        t_backward = time.time()
        
        # Gradient from loss
        grad = torch.zeros(B, S, self.hidden_size, device=self.device, dtype=hidden.dtype)
        
        # Simple gradient approximation through softmax
        probs = F.softmax(shift_logits, dim=-1)
        grad_logits = probs.clone()
        grad_logits.scatter_add_(
            2, shift_labels.unsqueeze(-1),
            -torch.ones_like(shift_labels.unsqueeze(-1), dtype=grad_logits.dtype)
        )
        grad[:, :-1, :] = grad_logits @ self.lm_head / (S - 1)
        
        # Start prefetch for backward
        self.loader.prefetch(self.num_layers - 1)
        
        for layer_idx in range(self.num_layers - 1, -1, -1):
            weights = self.loader.get_layer(layer_idx)
            
            if layer_idx > 0:
                self.loader.prefetch(layer_idx - 1)
            
            grad = self._compute_lora_grads(
                hidden_states[layer_idx],
                layer_idx,
                weights,
                grad,
            )
            
            del weights
            if layer_idx % 10 == 0:
                clear_mem()
        
        backward_time = time.time() - t_backward
        
        # ========== UPDATE ==========
        torch.nn.utils.clip_grad_norm_(
            list(self.lora_q.parameters()) + list(self.lora_v.parameters()),
            1.0
        )
        self.optimizer.step()
        
        del hidden_states
        clear_mem()
        
        logger.debug(f"  Forward: {forward_time:.1f}s, Backward: {backward_time:.1f}s")
        
        return loss_val
    
    def train(self, texts: List[str], epochs: int = 1) -> Dict:
        """Train on texts."""
        logger.info(f"Training on {len(texts)} samples (async pipeline)")
        
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
        
        avg_time = sum(times) / len(times)
        return {
            "losses": losses,
            "avg_loss": sum(losses) / len(losses),
            "avg_time": avg_time,
        }


if __name__ == "__main__":
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
    
    config = PipelineConfig(max_seq_length=32)
    trainer = PipelineTrainer("models/phi-4", config)
    
    # Load data
    texts = []
    with open("data/Dataset/math.jsonl") as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            item = json.loads(line)
            texts.append(f"Problem: {item['problem']}\nAnswer: {item['answer']}")
    
    results = trainer.train(texts)
    
    logger.info("=" * 60)
    logger.info(f"Avg loss: {results['avg_loss']:.4f}")
    logger.info(f"Avg time: {results['avg_time']:.1f}s/sample")
    logger.info("=" * 60)
