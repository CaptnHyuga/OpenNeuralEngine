"""HuggingFace-compatible model wrapper.

This module provides a model architecture that can load weights from
HuggingFace SmolLM/SmolVLM checkpoints while maintaining compatibility
with our custom training pipeline.
"""
from __future__ import annotations

import math
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Check for Flash Attention / SDPA availability
_SDPA_AVAILABLE = hasattr(F, "scaled_dot_product_attention")


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (used by LLaMA-style models)."""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())
    
    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            self.max_seq_len = seq_len
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply rotary position embeddings to query and key tensors."""
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, dim)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class HFAttention(nn.Module):
    """Multi-head attention compatible with HuggingFace SmolLM weights.
    
    Supports Flash Attention via PyTorch SDPA for O(1) memory complexity.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: Optional[int] = None,
        max_seq_len: int = 2048,
        use_flash_attention: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim or hidden_size // num_heads
        self.num_key_value_groups = num_heads // num_kv_heads
        self.use_flash_attention = use_flash_attention and _SDPA_AVAILABLE
        
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)
        
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = hidden_states.shape
        
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Compute position offset for KV-cache
        kv_seq_len = seq_len
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[2]
        
        # Apply rotary embeddings with correct position offset
        cos, sin = self.rotary_emb(kv_seq_len)
        cos, sin = cos.to(q.device), sin.to(q.device)
        
        # For cached generation, only apply RoPE to new positions
        if past_key_value is not None:
            position_offset = past_key_value[0].shape[2]
            cos_new = cos[position_offset:position_offset + seq_len]
            sin_new = sin[position_offset:position_offset + seq_len]
        else:
            cos_new = cos[:seq_len]
            sin_new = sin[:seq_len]
        
        q, k = apply_rotary_pos_emb(q, k, cos_new, sin_new)
        
        # KV-cache: concatenate past keys/values
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        
        # Store new cache if requested
        new_cache = (k, v) if use_cache else None
        
        # Repeat k/v heads if using grouped-query attention
        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)
        
        # Use Flash Attention (SDPA) when available for O(1) memory
        if self.use_flash_attention and seq_len > 1:
            # SDPA handles causal masking internally
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True,
            )
        else:
            # Standard attention for single-token generation or when SDPA unavailable
            scale = 1.0 / math.sqrt(self.head_dim)
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
            
            # Causal mask (for full seq during prefill, or no mask needed for single token)
            full_seq_len = k.shape[2]
            if attention_mask is None and seq_len > 1:
                causal_mask = torch.triu(
                    torch.full((seq_len, full_seq_len), float('-inf'), device=hidden_states.device),
                    diagonal=full_seq_len - seq_len + 1
                )
                attn_weights = attn_weights + causal_mask
            elif attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
            attn_output = torch.matmul(attn_weights, v)
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        
        return self.o_proj(attn_output), new_cache


class HFMLP(nn.Module):
    """MLP layer compatible with HuggingFace SmolLM (SwiGLU activation)."""
    
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class HFDecoderLayer(nn.Module):
    """Transformer decoder layer compatible with HuggingFace SmolLM."""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.self_attn = HFAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            max_seq_len=max_seq_len,
        )
        self.mlp = HFMLP(hidden_size, intermediate_size)
        self.input_layernorm = RMSNorm(hidden_size)
        self.post_attention_layernorm = RMSNorm(hidden_size)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, new_cache = self.self_attn(
            hidden_states, attention_mask, past_key_value, use_cache
        )
        hidden_states = residual + hidden_states
        
        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, new_cache


class HFCompatibleLM(nn.Module):
    """Language model compatible with HuggingFace SmolLM/SmolVLM weights.
    
    This model can:
    1. Load pretrained HuggingFace weights directly
    2. Be used for inference (chat/generation)
    3. Be fine-tuned with our custom training pipeline
    """
    
    def __init__(
        self,
        vocab_size: int = 49280,
        hidden_size: int = 960,
        num_layers: int = 32,
        num_heads: int = 15,
        num_kv_heads: int = 5,
        intermediate_size: int = 2560,
        max_seq_len: int = 2048,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len
        
        # Embedding
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            HFDecoderLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                intermediate_size=intermediate_size,
                max_seq_len=max_seq_len,
            )
            for _ in range(num_layers)
        ])
        
        # Final norm and LM head
        self.norm = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Tie weights (common in LLMs)
        self.lm_head.weight = self.embed_tokens.weight
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        use_cache: bool = False,
    ):
        """Forward pass returning logits (and optionally KV-cache).
        
        Args:
            input_ids: (batch_size, seq_len) token IDs
            attention_mask: Optional attention mask
            past_key_values: Cached key/value states for fast generation
            use_cache: Whether to return updated cache
            
        Returns:
            If use_cache=False: logits (batch_size, seq_len, vocab_size)
            If use_cache=True: (logits, new_cache) tuple
        """
        hidden_states = self.embed_tokens(input_ids)
        
        new_cache = () if use_cache else None
        
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            hidden_states, layer_cache = layer(hidden_states, attention_mask, past_kv, use_cache)
            if use_cache:
                new_cache = new_cache + (layer_cache,)
        
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        # Backward compatible: only return tuple when cache requested
        if use_cache:
            return logits, new_cache
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        eos_token_id: Optional[int] = None,
        min_new_tokens: int = 1,
        repetition_penalty: float = 1.1,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Generate tokens autoregressively with KV-cache.
        
        Uses key-value caching for 10-50x speedup over naive generation.
        
        Args:
            input_ids: (batch_size, seq_len) input token IDs
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            eos_token_id: Stop generation on this token
            min_new_tokens: Minimum tokens to generate before allowing EOS
            repetition_penalty: Penalty for repeating tokens (>1 = less repetition)
            use_cache: Use KV-cache for faster generation (default: True)
            
        Returns:
            Generated token IDs including input
        """
        self.eval()
        generated = input_ids.clone()
        past_key_values = None
        
        # Prefill: process all input tokens at once
        if use_cache:
            logits, past_key_values = self.forward(generated, use_cache=True)
        else:
            logits = self.forward(generated, use_cache=False)
        
        for i in range(max_new_tokens):
            # Get logits for last position only
            next_logits = logits[:, -1, :] / max(temperature, 1e-8)
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for b in range(generated.size(0)):
                    for token_id in set(generated[b].tolist()):
                        if token_id < next_logits.size(-1):
                            if next_logits[b, token_id] > 0:
                                next_logits[b, token_id] /= repetition_penalty
                            else:
                                next_logits[b, token_id] *= repetition_penalty
            
            # Suppress EOS until min_new_tokens reached
            if eos_token_id is not None and i < min_new_tokens:
                next_logits[:, eos_token_id] = float('-inf')
            
            # Top-k filtering
            if top_k is not None and top_k > 0:
                values, indices = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits = torch.full_like(next_logits, float('-inf'))
                next_logits.scatter_(1, indices, values)
            
            # Top-p (nucleus) filtering
            if top_p is not None and 0 < top_p < 1:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                for b in range(next_logits.size(0)):
                    indices_to_remove = sorted_indices[b, sorted_indices_to_remove[b]]
                    next_logits[b, indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop on EOS (after min_new_tokens)
            if eos_token_id is not None and i >= min_new_tokens and (next_token == eos_token_id).all():
                break
            
            # Forward pass for next token only (using cache)
            if use_cache and i < max_new_tokens - 1:
                # Truncate cache if exceeding max length
                if generated.shape[1] >= self.max_seq_len:
                    # Reset cache and recompute from truncated context
                    context = generated[:, -self.max_seq_len + 1:]
                    logits, past_key_values = self.forward(context, use_cache=True)
                else:
                    logits, past_key_values = self.forward(
                        next_token, past_key_values=past_key_values, use_cache=True
                    )
            elif not use_cache and i < max_new_tokens - 1:
                # Fallback: no cache, recompute everything (slow)
                if generated.shape[1] >= self.max_seq_len:
                    context = generated[:, -self.max_seq_len + 1:]
                else:
                    context = generated
                logits = self.forward(context, use_cache=False)
        
        return generated

    def generate_streaming(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        eos_token_id: Optional[int] = None,
        min_new_tokens: int = 1,
        repetition_penalty: float = 1.1,
        use_cache: bool = True,
    ):
        """Generate tokens one at a time, yielding each token ID.
        
        Use this for streaming responses in web interfaces.
        
        Args:
            Same as generate()
            
        Yields:
            int: Each generated token ID
        """
        self.eval()
        generated = input_ids.clone()
        past_key_values = None
        
        # Prefill: process all input tokens at once
        if use_cache:
            logits, past_key_values = self.forward(generated, use_cache=True)
        else:
            logits = self.forward(generated, use_cache=False)
        
        for i in range(max_new_tokens):
            # Get logits for last position only
            next_logits = logits[:, -1, :] / max(temperature, 1e-8)
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for b in range(generated.size(0)):
                    for token_id in set(generated[b].tolist()):
                        if token_id < next_logits.size(-1):
                            if next_logits[b, token_id] > 0:
                                next_logits[b, token_id] /= repetition_penalty
                            else:
                                next_logits[b, token_id] *= repetition_penalty
            
            # Suppress EOS until min_new_tokens reached
            if eos_token_id is not None and i < min_new_tokens:
                next_logits[:, eos_token_id] = float('-inf')
            
            # Top-k filtering
            if top_k is not None and top_k > 0:
                values, indices = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits = torch.full_like(next_logits, float('-inf'))
                next_logits.scatter_(1, indices, values)
            
            # Top-p (nucleus) filtering
            if top_p is not None and 0 < top_p < 1:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                for b in range(next_logits.size(0)):
                    indices_to_remove = sorted_indices[b, sorted_indices_to_remove[b]]
                    next_logits[b, indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            # Yield the new token
            yield next_token.item()
            
            # Stop on EOS (after min_new_tokens)
            if eos_token_id is not None and i >= min_new_tokens and (next_token == eos_token_id).all():
                break
            
            # Forward pass for next token only (using cache)
            if use_cache and i < max_new_tokens - 1:
                if generated.shape[1] >= self.max_seq_len:
                    context = generated[:, -self.max_seq_len + 1:]
                    logits, past_key_values = self.forward(context, use_cache=True)
                else:
                    logits, past_key_values = self.forward(
                        next_token, past_key_values=past_key_values, use_cache=True
                    )
            elif not use_cache and i < max_new_tokens - 1:
                if generated.shape[1] >= self.max_seq_len:
                    context = generated[:, -self.max_seq_len + 1:]
                else:
                    context = generated
                logits = self.forward(context, use_cache=False)

    def generate_multimodal(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        eos_token_id: Optional[int] = None,
        min_new_tokens: int = 1,
        repetition_penalty: float = 1.1,
        use_cache: bool = True,
    ):
        """Generate text conditioned on both text and images (multimodal).
        
        This is a placeholder implementation that processes images as context
        but currently only does text generation. For full multimodal support,
        you need to load a multimodal model architecture.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            pixel_values: Preprocessed image tensor [batch, channels, height, width]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            eos_token_id: End-of-sequence token ID
            min_new_tokens: Minimum tokens before EOS
            repetition_penalty: Penalty for repeating tokens
            use_cache: Enable KV caching
            
        Returns:
            Generated token IDs [batch, seq_len + new_tokens]
        """
        # Note: This is a text-only model, so we ignore pixel_values
        # For true multimodal support, use a multimodal model architecture
        if pixel_values is not None:
            # In a full multimodal model, you would:
            # 1. Encode images with vision encoder
            # 2. Project to text embedding space
            # 3. Prepend to text embeddings
            # For now, we just do text generation
            pass
        
        # Fall back to standard text generation
        return self.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=eos_token_id,
            min_new_tokens=min_new_tokens,
            repetition_penalty=repetition_penalty,
            use_cache=use_cache,
        )


def load_hf_checkpoint(model: HFCompatibleLM, checkpoint_path: str) -> Dict[str, int]:
    """Load HuggingFace SmolLM/SmolVLM weights into our compatible model.
    
    Args:
        model: HFCompatibleLM instance
        checkpoint_path: Path to .safetensors file
        
    Returns:
        Dict with loading statistics
    """
    from safetensors.torch import load_file
    
    hf_weights = load_file(checkpoint_path)
    model_state = model.state_dict()
    
    # Mapping from HF names to our names
    mapped = 0
    skipped = 0
    
    key_mapping = {
        "model.text_model.embed_tokens.weight": "embed_tokens.weight",
        "model.text_model.norm.weight": "norm.weight",
        "lm_head.weight": "lm_head.weight",
    }
    
    # Add layer mappings
    for i in range(model.num_layers):
        hf_prefix = f"model.text_model.layers.{i}"
        our_prefix = f"layers.{i}"
        
        layer_mappings = {
            f"{hf_prefix}.input_layernorm.weight": f"{our_prefix}.input_layernorm.weight",
            f"{hf_prefix}.post_attention_layernorm.weight": f"{our_prefix}.post_attention_layernorm.weight",
            f"{hf_prefix}.self_attn.q_proj.weight": f"{our_prefix}.self_attn.q_proj.weight",
            f"{hf_prefix}.self_attn.k_proj.weight": f"{our_prefix}.self_attn.k_proj.weight",
            f"{hf_prefix}.self_attn.v_proj.weight": f"{our_prefix}.self_attn.v_proj.weight",
            f"{hf_prefix}.self_attn.o_proj.weight": f"{our_prefix}.self_attn.o_proj.weight",
            f"{hf_prefix}.mlp.gate_proj.weight": f"{our_prefix}.mlp.gate_proj.weight",
            f"{hf_prefix}.mlp.up_proj.weight": f"{our_prefix}.mlp.up_proj.weight",
            f"{hf_prefix}.mlp.down_proj.weight": f"{our_prefix}.mlp.down_proj.weight",
        }
        key_mapping.update(layer_mappings)
    
    # Load weights
    new_state = {}
    for hf_key, our_key in key_mapping.items():
        if hf_key in hf_weights and our_key in model_state:
            if hf_weights[hf_key].shape == model_state[our_key].shape:
                new_state[our_key] = hf_weights[hf_key]
                mapped += 1
            else:
                print(f"Shape mismatch: {hf_key} {hf_weights[hf_key].shape} vs {our_key} {model_state[our_key].shape}")
                skipped += 1
        elif hf_key in hf_weights:
            skipped += 1
    
    # Load the mapped weights
    model.load_state_dict(new_state, strict=False)
    
    return {"mapped": mapped, "skipped": skipped, "total_hf": len(hf_weights)}


def create_hf_compatible_model(checkpoint_path: Optional[str] = None) -> HFCompatibleLM:
    """Create HF-compatible model, optionally loading weights.
    
    Auto-detects model configuration from checkpoint if provided.
    Handles both original HuggingFace format and our fine-tuned format.
    """
    # Default SmolLM-360M config
    config = {
        "vocab_size": 49280,
        "hidden_size": 960,
        "num_layers": 32,
        "num_heads": 15,
        "num_kv_heads": 5,
        "intermediate_size": 2560,
        "max_seq_len": 2048,
    }
    
    is_our_format = False
    
    # Try to infer config from checkpoint
    if checkpoint_path:
        from safetensors.torch import load_file
        weights = load_file(checkpoint_path)
        
        # Check if this is our format (keys like "layers.0.self_attn...")
        # or HuggingFace format (keys like "model.text_model.layers.0...")
        if "embed_tokens.weight" in weights:
            # Our format - direct loading
            is_our_format = True
            embed = weights["embed_tokens.weight"]
            config["vocab_size"] = embed.shape[0]
            config["hidden_size"] = embed.shape[1]
            
            # Infer num_layers
            layer_keys = [k for k in weights.keys() if k.startswith("layers.") and ".self_attn." in k]
            if layer_keys:
                max_layer = max(int(k.split("layers.")[1].split(".")[0]) for k in layer_keys)
                config["num_layers"] = max_layer + 1
            
            # Infer intermediate_size
            if "layers.0.mlp.gate_proj.weight" in weights:
                config["intermediate_size"] = weights["layers.0.mlp.gate_proj.weight"].shape[0]
            
            # Infer num_heads
            if "layers.0.self_attn.q_proj.weight" in weights and "layers.0.self_attn.k_proj.weight" in weights:
                q_out = weights["layers.0.self_attn.q_proj.weight"].shape[0]
                k_out = weights["layers.0.self_attn.k_proj.weight"].shape[0]
                head_dim = 64  # Common default
                config["num_heads"] = q_out // head_dim
                config["num_kv_heads"] = k_out // head_dim
                
        elif "model.text_model.embed_tokens.weight" in weights:
            # HuggingFace format
            embed = weights["model.text_model.embed_tokens.weight"]
            config["vocab_size"] = embed.shape[0]
            config["hidden_size"] = embed.shape[1]
            
            # Infer num_layers
            layer_keys = [k for k in weights.keys() if "text_model.layers." in k]
            if layer_keys:
                max_layer = max(int(k.split("layers.")[1].split(".")[0]) for k in layer_keys)
                config["num_layers"] = max_layer + 1
            
            # Infer intermediate_size from MLP
            mlp_key = "model.text_model.layers.0.mlp.gate_proj.weight"
            if mlp_key in weights:
                config["intermediate_size"] = weights[mlp_key].shape[0]
            
            # Infer num_heads from attention
            q_key = "model.text_model.layers.0.self_attn.q_proj.weight"
            k_key = "model.text_model.layers.0.self_attn.k_proj.weight"
            if q_key in weights and k_key in weights:
                q_out = weights[q_key].shape[0]
                k_out = weights[k_key].shape[0]
                head_dim = 64
                config["num_heads"] = q_out // head_dim
                config["num_kv_heads"] = k_out // head_dim
    
    model = HFCompatibleLM(**config)
    
    if checkpoint_path:
        if is_our_format:
            # Direct loading for our format
            from safetensors.torch import load_file
            weights = load_file(checkpoint_path)
            
            # Filter to matching keys
            model_state = model.state_dict()
            filtered = {k: v for k, v in weights.items() if k in model_state and v.shape == model_state[k].shape}
            model.load_state_dict(filtered, strict=False)
            print(f"[OK] Loaded HF checkpoint: {len(filtered)} tensors mapped (our format)")
        else:
            # HuggingFace format - use mapping
            stats = load_hf_checkpoint(model, checkpoint_path)
            print(f"[OK] Loaded HF checkpoint: {stats['mapped']} tensors mapped")
    
    return model
