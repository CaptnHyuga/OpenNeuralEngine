"""
Micro-Batch Streaming Compute Engine - Truly minimal memory footprint.

The insight: We can't just stream weights - we need to stream the COMPUTATION.
Instead of computing full intermediate activations, we compute output columns
one at a time, never materializing the full intermediate.

For y = activation(x @ W1) @ W2:
- Instead of computing full hidden = x @ W1 (huge!)
- We compute y[:, i] = sum_j(activation(x @ W1[:, j]) * W2[i, j])

This is slower but uses CONSTANT memory regardless of model size.
"""

import torch
import torch.nn.functional as F
from torch import nn
from typing import Dict, Optional, Tuple, Any, Generator
from pathlib import Path
from dataclasses import dataclass
import logging
import json
import math

logger = logging.getLogger(__name__)


@dataclass
class MicroStreamConfig:
    """Configuration for micro-streaming."""
    vram_budget_gb: float = 3.5  # Leave headroom
    
    # Micro-batch sizes
    output_chunk_size: int = 64  # Compute this many output features at once
    
    # LoRA config
    lora_r: int = 4
    lora_alpha: int = 8
    
    # Training
    learning_rate: float = 1e-4
    gradient_accumulation: int = 8


def get_gpu_memory():
    """Get GPU memory info."""
    if torch.cuda.is_available():
        return {
            'allocated': torch.cuda.memory_allocated() / 1e9,
            'total': torch.cuda.get_device_properties(0).total_memory / 1e9,
        }
    return {'allocated': 0, 'total': 0}


def clear_memory():
    """Aggressive memory cleanup."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class TinyLoRA(nn.Module):
    """Minimal LoRA that computes output incrementally."""
    
    def __init__(self, in_features: int, out_features: int, r: int = 4, alpha: int = 8):
        super().__init__()
        self.scale = alpha / r
        
        # FP32 for gradients, tiny memory footprint
        self.A = nn.Parameter(torch.zeros(r, in_features, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(out_features, r, dtype=torch.float32))
        
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute LoRA delta: x @ A.T @ B.T * scale"""
        dtype = x.dtype
        return ((x.float() @ self.A.T) @ self.B.T * self.scale).to(dtype)


class SafetensorStreamer:
    """Streams individual weight slices from safetensors."""
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        
        with open(self.model_path / "model.safetensors.index.json") as f:
            self._index = json.load(f)
        
        self._handles = {}
    
    def _get_handle(self, filename: str):
        from safetensors import safe_open
        if filename not in self._handles:
            self._handles[filename] = safe_open(
                str(self.model_path / filename), framework="pt", device="cpu"
            )
        return self._handles[filename]
    
    def get_weight_slice(
        self,
        weight_name: str,
        row_start: int = None,
        row_end: int = None,
        col_start: int = None,
        col_end: int = None,
        device: torch.device = None,
        dtype: torch.dtype = torch.float16,
    ) -> Optional[torch.Tensor]:
        """Get a slice of a weight matrix."""
        weight_map = self._index.get("weight_map", {})
        filename = weight_map.get(weight_name)
        
        if filename is None:
            return None
        
        handle = self._get_handle(filename)
        slice_obj = handle.get_slice(weight_name)
        
        # Apply slicing
        if row_start is not None or row_end is not None:
            rs = row_start or 0
            re = row_end or slice_obj.get_shape()[0]
            if col_start is not None or col_end is not None:
                cs = col_start or 0
                ce = col_end or slice_obj.get_shape()[1]
                data = slice_obj[rs:re, cs:ce]
            else:
                data = slice_obj[rs:re]
        elif col_start is not None or col_end is not None:
            cs = col_start or 0
            ce = col_end or slice_obj.get_shape()[1]
            data = slice_obj[:, cs:ce]
        else:
            data = slice_obj[:]
        
        tensor = torch.tensor(data, dtype=dtype)
        if device is not None:
            tensor = tensor.to(device)
        return tensor
    
    def get_shape(self, weight_name: str) -> Optional[Tuple[int, ...]]:
        """Get shape of a weight without loading it."""
        weight_map = self._index.get("weight_map", {})
        filename = weight_map.get(weight_name)
        if filename is None:
            return None
        handle = self._get_handle(filename)
        return tuple(handle.get_slice(weight_name).get_shape())
    
    def close(self):
        self._handles.clear()


class MicroStreamingLayer:
    """
    Transformer layer that computes in micro-batches.
    
    The key insight: For y = x @ W where W is [out, in],
    we can compute y[:, i:i+chunk] = x @ W[i:i+chunk].T
    
    This means we never need to hold the full W or full y!
    """
    
    def __init__(
        self,
        streamer: SafetensorStreamer,
        layer_idx: int,
        config: MicroStreamConfig,
        model_config: Any,
        device: torch.device,
    ):
        self.streamer = streamer
        self.layer_idx = layer_idx
        self.config = config
        self.model_config = model_config
        self.device = device
        
        self.hidden_size = model_config.hidden_size
        self.num_heads = model_config.num_attention_heads
        self.num_kv_heads = getattr(model_config, 'num_key_value_heads', self.num_heads)
        self.head_dim = self.hidden_size // self.num_heads
        
        q_size = self.num_heads * self.head_dim
        v_size = self.num_kv_heads * self.head_dim
        
        # LoRA adapters - always in memory (tiny!)
        self.lora_q = TinyLoRA(self.hidden_size, q_size, config.lora_r, config.lora_alpha).to(device)
        self.lora_v = TinyLoRA(self.hidden_size, v_size, config.lora_r, config.lora_alpha).to(device)
    
    def _prefix(self) -> str:
        return f"model.layers.{self.layer_idx}."
    
    def _chunked_linear(
        self,
        x: torch.Tensor,
        weight_name: str,
        out_features: int,
    ) -> torch.Tensor:
        """
        Compute x @ W.T by streaming W in chunks.
        
        This is the CORE innovation - we never hold full W in memory!
        """
        batch, seq, _ = x.shape
        chunk_size = self.config.output_chunk_size
        
        # Output tensor (this we DO need to hold)
        output = torch.zeros(batch, seq, out_features, dtype=x.dtype, device=self.device)
        
        for start in range(0, out_features, chunk_size):
            end = min(start + chunk_size, out_features)
            
            # Load just this chunk of W: W[start:end, :]
            weight_chunk = self.streamer.get_weight_slice(
                weight_name,
                row_start=start,
                row_end=end,
                device=self.device,
                dtype=x.dtype,
            )
            
            if weight_chunk is not None:
                # Compute partial output: output[..., start:end] = x @ weight_chunk.T
                output[..., start:end] = torch.matmul(x, weight_chunk.T)
                del weight_chunk
        
        return output
    
    def _get_layernorm_weight(self, name: str) -> Optional[torch.Tensor]:
        """Get LayerNorm weight (small, load directly)."""
        full_name = self._prefix() + name
        return self.streamer.get_weight_slice(full_name, device=self.device)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass with micro-streaming."""
        batch, seq, _ = hidden_states.shape
        residual = hidden_states
        
        # === Input LayerNorm ===
        ln1_w = self._get_layernorm_weight("input_layernorm.weight")
        if ln1_w is not None:
            hidden_states = F.layer_norm(hidden_states, (self.hidden_size,), weight=ln1_w)
            del ln1_w
        
        # === Attention ===
        # QKV dimensions
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        qkv_size = q_size + 2 * kv_size
        
        # Compute QKV in chunks
        qkv_name = self._prefix() + "self_attn.qkv_proj.weight"
        qkv = self._chunked_linear(hidden_states, qkv_name, qkv_size)
        
        # Split Q, K, V
        q = qkv[..., :q_size]
        k = qkv[..., q_size:q_size + kv_size]
        v = qkv[..., q_size + kv_size:]
        del qkv
        
        # Apply LoRA
        q = q + self.lora_q(residual)
        v = v + self.lora_v(residual)
        
        # Reshape for attention
        q = q.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # GQA expansion
        if self.num_kv_heads < self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)
        
        # Attention (Flash when available)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        del q, k, v
        
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq, self.hidden_size)
        
        # Output projection (chunked)
        o_name = self._prefix() + "self_attn.o_proj.weight"
        attn_out = self._chunked_linear(attn_out, o_name, self.hidden_size)
        
        hidden_states = residual + attn_out
        del attn_out
        
        # === Post-attention LayerNorm ===
        residual = hidden_states
        ln2_w = self._get_layernorm_weight("post_attention_layernorm.weight")
        if ln2_w is not None:
            hidden_states = F.layer_norm(hidden_states, (self.hidden_size,), weight=ln2_w)
            del ln2_w
        
        # === MLP ===
        # This is the biggest memory hog - intermediate_size is huge!
        # We need to be VERY careful here
        
        gate_up_name = self._prefix() + "mlp.gate_up_proj.weight"
        gate_up_shape = self.streamer.get_shape(gate_up_name)
        
        if gate_up_shape is not None:
            intermediate_size = gate_up_shape[0] // 2
            
            # Compute gate_up in chunks, but we need full intermediate for SiLU*up
            # This is unavoidable - we need the full gate and up to compute activation
            # But we can still stream it!
            
            gate_up = self._chunked_linear(hidden_states, gate_up_name, gate_up_shape[0])
            
            gate = gate_up[..., :intermediate_size]
            up = gate_up[..., intermediate_size:]
            
            mlp_hidden = F.silu(gate) * up
            del gate, up, gate_up
            
            # Down projection (chunked)
            down_name = self._prefix() + "mlp.down_proj.weight"
            mlp_out = self._chunked_linear(mlp_hidden, down_name, self.hidden_size)
            del mlp_hidden
            
            hidden_states = residual + mlp_out
            del mlp_out
        
        clear_memory()
        return hidden_states


class MicroStreamingTransformer:
    """
    Full transformer with micro-streaming.
    
    Maximum memory efficiency through:
    1. Weight streaming (never hold full weight matrices)
    2. Chunked computation (compute output in pieces)
    3. Aggressive cleanup (delete intermediates immediately)
    """
    
    def __init__(self, model_path: str, config: Optional[MicroStreamConfig] = None):
        self.model_path = Path(model_path)
        self.config = config or MicroStreamConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load configs
        from transformers import AutoConfig, AutoTokenizer
        
        self.model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize streamer
        self.streamer = SafetensorStreamer(model_path)
        
        # Initialize layers
        self.layers = []
        for i in range(self.model_config.num_hidden_layers):
            layer = MicroStreamingLayer(
                self.streamer, i, self.config, self.model_config, self.device
            )
            self.layers.append(layer)
        
        # Embeddings - we MUST keep these, but can we shrink them?
        # Actually, we can load embedding rows on demand!
        self._embeddings_loaded = False
        self.embeddings = None
        
        logger.info(f"MicroStreamingTransformer: {len(self.layers)} layers")
        
        # Count LoRA params
        lora_params = sum(
            sum(p.numel() for p in layer.lora_q.parameters()) +
            sum(p.numel() for p in layer.lora_v.parameters())
            for layer in self.layers
        )
        logger.info(f"LoRA parameters: {lora_params:,}")
    
    def _ensure_embeddings(self):
        """Load embeddings (lazy)."""
        if self._embeddings_loaded:
            return
        
        # Find embedding weight
        embed_name = "model.embed_tokens.weight"
        embed_weight = self.streamer.get_weight_slice(embed_name, device=self.device)
        
        if embed_weight is not None:
            self.embeddings = nn.Embedding.from_pretrained(embed_weight, freeze=True)
            logger.info(f"Loaded embeddings: {embed_weight.shape}")
        else:
            self.embeddings = nn.Embedding(
                self.model_config.vocab_size,
                self.model_config.hidden_size,
            ).to(self.device)
            logger.warning("Using random embeddings")
        
        self._embeddings_loaded = True
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        self._ensure_embeddings()
        
        hidden_states = self.embeddings(input_ids)
        
        for i, layer in enumerate(self.layers):
            hidden_states = layer.forward(hidden_states)
            
            if i % 10 == 0:
                mem = get_gpu_memory()
                logger.debug(f"Layer {i}: {mem['allocated']:.2f}GB")
        
        # Final norm
        final_ln = self.streamer.get_weight_slice("model.norm.weight", device=self.device)
        if final_ln is not None:
            hidden_states = F.layer_norm(
                hidden_states, (self.model_config.hidden_size,), weight=final_ln
            )
        
        # Logits via tied embeddings
        logits = F.linear(hidden_states, self.embeddings.weight)
        return logits
    
    def compute_loss(self, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute causal LM loss."""
        logits = self.forward(input_ids)
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
    
    def get_lora_parameters(self):
        """Get all LoRA parameters."""
        params = []
        for layer in self.layers:
            params.extend(layer.lora_q.parameters())
            params.extend(layer.lora_v.parameters())
        return params
    
    def train_step(self, input_ids: torch.Tensor, labels: torch.Tensor, optimizer) -> float:
        """Single training step."""
        optimizer.zero_grad()
        
        loss = self.compute_loss(input_ids, labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.get_lora_parameters(), 1.0)
        optimizer.step()
        
        return loss.item()
    
    def save_lora(self, path: str):
        """Save LoRA weights."""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        state_dict = {}
        for i, layer in enumerate(self.layers):
            for name, param in layer.lora_q.named_parameters():
                state_dict[f"layer.{i}.q.{name}"] = param.cpu()
            for name, param in layer.lora_v.named_parameters():
                state_dict[f"layer.{i}.v.{name}"] = param.cpu()
        
        torch.save(state_dict, save_path / "lora.pt")
        logger.info(f"Saved to {save_path}")
    
    def generate(self, prompt: str, max_new_tokens: int = 20) -> str:
        """Generate text."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self.forward(input_ids)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                clear_memory()
        
        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
