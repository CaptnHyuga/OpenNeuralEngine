"""ONN Inference Engine - Hardware-optimized text generation.

Provides multiple backends for inference:
1. vLLM (preferred) - Production-grade, highly optimized
2. HuggingFace Transformers - Fallback, broad compatibility
3. ONNX Runtime - CPU inference optimization

Automatically selects the best backend based on:
- Model architecture compatibility
- Hardware capabilities
- Installed dependencies
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

# Check available backends
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        pipeline,
        TextGenerationPipeline,
    )
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# ONN imports
try:
    from ..orchestration import get_hardware_profile
    ONN_ORCHESTRATION = True
except ImportError:
    ONN_ORCHESTRATION = False


@dataclass
class InferenceConfig:
    """Configuration for inference engine."""
    
    # Model settings
    model_path: str = ""
    trust_remote_code: bool = True
    
    # Generation parameters
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    
    # vLLM specific
    tensor_parallel_size: int = 1  # Multi-GPU
    gpu_memory_utilization: float = 0.9
    
    # Performance
    dtype: str = "auto"  # "auto", "float16", "bfloat16", "float32"
    
    # Backend selection
    backend: str = "auto"  # "auto", "vllm", "hf", "onnx"


@dataclass
class GenerationResult:
    """Result from text generation."""
    
    prompt: str
    generated_text: str
    full_text: str  # prompt + generated
    
    # Metadata
    tokens_generated: int = 0
    generation_time_seconds: float = 0.0
    tokens_per_second: float = 0.0
    
    # Sampling info
    finish_reason: str = "length"  # "length", "stop", "error"
    
    def __post_init__(self):
        if self.generation_time_seconds > 0:
            self.tokens_per_second = self.tokens_generated / self.generation_time_seconds


class OnnInferenceEngine:
    """Hardware-optimized inference engine with automatic backend selection.
    
    Automatically chooses the best available backend:
    1. vLLM for maximum throughput (if installed)
    2. HuggingFace as fallback
    
    Configures based on hardware profile (VRAM, precision support, etc.)
    """
    
    def __init__(
        self,
        model: str,
        config: Optional[InferenceConfig] = None,
    ):
        """Initialize inference engine.
        
        Args:
            model: Model path or HuggingFace model ID.
            config: Inference configuration (auto-configured if None).
        """
        self.model_path = model
        self.config = config or InferenceConfig(model_path=model)
        
        # Auto-configure based on hardware
        self._configure()
        
        # Initialize backend
        self._init_backend()
    
    def _configure(self):
        """Configure engine based on hardware capabilities."""
        profile = None
        if ONN_ORCHESTRATION:
            try:
                profile = get_hardware_profile()
            except Exception:
                pass
        
        # Auto-select dtype
        if self.config.dtype == "auto":
            if profile and profile.supports_bf16:
                self.config.dtype = "bfloat16"
            elif torch.cuda.is_available():
                self.config.dtype = "float16"
            else:
                self.config.dtype = "float32"
        
        # Auto-select backend
        if self.config.backend == "auto":
            if VLLM_AVAILABLE and torch.cuda.is_available():
                self.config.backend = "vllm"
            elif HF_AVAILABLE:
                self.config.backend = "hf"
            else:
                raise RuntimeError(
                    "No inference backend available. "
                    "Install transformers or vllm."
                )
        
        # Configure tensor parallelism for multi-GPU
        if profile and len(profile.gpus) > 1:
            self.config.tensor_parallel_size = min(
                len(profile.gpus),
                self.config.tensor_parallel_size or len(profile.gpus)
            )
        
        # Adjust GPU memory utilization based on available VRAM
        if profile and profile.has_gpu:
            vram_gb = profile.available_vram_mb / 1024
            if vram_gb < 8:
                self.config.gpu_memory_utilization = 0.85  # More conservative
    
    def _init_backend(self):
        """Initialize the selected backend."""
        print(f"🚀 Initializing inference engine...")
        print(f"   Model: {self.model_path}")
        print(f"   Backend: {self.config.backend}")
        print(f"   Dtype: {self.config.dtype}")
        
        if self.config.backend == "vllm":
            self._init_vllm()
        elif self.config.backend == "hf":
            self._init_hf()
        else:
            raise ValueError(f"Unknown backend: {self.config.backend}")
    
    def _init_vllm(self):
        """Initialize vLLM backend."""
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM not installed. Install with: pip install vllm")
        
        dtype_map = {
            "float16": "float16",
            "bfloat16": "bfloat16",
            "float32": "float32",
            "auto": "auto",
        }
        
        self.engine = LLM(
            model=self.model_path,
            trust_remote_code=self.config.trust_remote_code,
            tensor_parallel_size=self.config.tensor_parallel_size,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            dtype=dtype_map.get(self.config.dtype, "auto"),
        )
        
        print(f"   ✅ vLLM engine ready")
    
    def _init_hf(self):
        """Initialize HuggingFace backend."""
        if not HF_AVAILABLE:
            raise ImportError("Transformers not installed. Install with: pip install transformers")
        
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = dtype_map.get(self.config.dtype, torch.float16)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.config.trust_remote_code,
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=self.config.trust_remote_code,
            torch_dtype=torch_dtype,
            device_map="auto" if device == "cuda" else None,
        )
        
        if device == "cpu":
            self.model = self.model.to(device)
        
        self.engine = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )
        
        print(f"   ✅ HuggingFace engine ready")
    
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
    ) -> GenerationResult:
        """Generate text from prompt.
        
        Args:
            prompt: Input text prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0 = deterministic).
            top_p: Nucleus sampling threshold.
            stop: Stop sequences.
            
        Returns:
            GenerationResult with generated text and metadata.
        """
        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature if temperature is not None else self.config.temperature
        top_p = top_p if top_p is not None else self.config.top_p
        
        start_time = time.time()
        
        if self.config.backend == "vllm":
            result = self._generate_vllm(prompt, max_tokens, temperature, top_p, stop)
        else:
            result = self._generate_hf(prompt, max_tokens, temperature, top_p, stop)
        
        result.generation_time_seconds = time.time() - start_time
        result.tokens_per_second = (
            result.tokens_generated / result.generation_time_seconds
            if result.generation_time_seconds > 0 else 0
        )
        
        return result
    
    def _generate_vllm(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[List[str]],
    ) -> GenerationResult:
        """Generate using vLLM backend."""
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            repetition_penalty=self.config.repetition_penalty,
        )
        
        outputs = self.engine.generate([prompt], sampling_params)
        output = outputs[0]
        generated = output.outputs[0].text
        
        return GenerationResult(
            prompt=prompt,
            generated_text=generated,
            full_text=prompt + generated,
            tokens_generated=len(output.outputs[0].token_ids),
            finish_reason=output.outputs[0].finish_reason or "length",
        )
    
    def _generate_hf(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[List[str]],
    ) -> GenerationResult:
        """Generate using HuggingFace backend."""
        # HF pipeline
        outputs = self.engine(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else None,
            top_p=top_p,
            do_sample=temperature > 0,
            repetition_penalty=self.config.repetition_penalty,
            return_full_text=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        
        full_text = outputs[0]["generated_text"]
        generated = full_text[len(prompt):]
        
        # Estimate tokens (rough)
        tokens_generated = len(self.tokenizer.encode(generated))
        
        return GenerationResult(
            prompt=prompt,
            generated_text=generated,
            full_text=full_text,
            tokens_generated=tokens_generated,
            finish_reason="length",
        )
    
    def generate_batch(
        self,
        prompts: List[str],
        **kwargs,
    ) -> List[GenerationResult]:
        """Generate text for multiple prompts (batched for efficiency).
        
        Args:
            prompts: List of input prompts.
            **kwargs: Generation parameters.
            
        Returns:
            List of GenerationResult objects.
        """
        if self.config.backend == "vllm":
            # vLLM handles batching efficiently
            max_tokens = kwargs.get("max_tokens", self.config.max_tokens)
            temperature = kwargs.get("temperature", self.config.temperature)
            top_p = kwargs.get("top_p", self.config.top_p)
            
            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=self.config.repetition_penalty,
            )
            
            start_time = time.time()
            outputs = self.engine.generate(prompts, sampling_params)
            total_time = time.time() - start_time
            
            results = []
            for prompt, output in zip(prompts, outputs):
                generated = output.outputs[0].text
                tokens = len(output.outputs[0].token_ids)
                results.append(GenerationResult(
                    prompt=prompt,
                    generated_text=generated,
                    full_text=prompt + generated,
                    tokens_generated=tokens,
                    generation_time_seconds=total_time / len(prompts),
                    finish_reason=output.outputs[0].finish_reason or "length",
                ))
            return results
        else:
            # Fallback to sequential for HF
            return [self.generate(p, **kwargs) for p in prompts]


# ============================================================================
# Convenience Functions
# ============================================================================

# Global engine cache for convenience functions
_engine_cache: Dict[str, OnnInferenceEngine] = {}


def generate(
    model: str,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
    **kwargs,
) -> str:
    """Quick text generation from any model.
    
    One-liner for text generation:
    
        output = generate("gpt2", "Once upon a time")
    
    Args:
        model: Model path or HuggingFace ID.
        prompt: Input text prompt.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        **kwargs: Additional generation parameters.
        
    Returns:
        Generated text (not including prompt).
    """
    global _engine_cache
    
    if model not in _engine_cache:
        _engine_cache[model] = OnnInferenceEngine(model)
    
    engine = _engine_cache[model]
    result = engine.generate(prompt, max_tokens=max_tokens, temperature=temperature, **kwargs)
    return result.generated_text


def generate_batch(
    model: str,
    prompts: List[str],
    **kwargs,
) -> List[str]:
    """Batch text generation for efficiency.
    
    Args:
        model: Model path or HuggingFace ID.
        prompts: List of input prompts.
        **kwargs: Generation parameters.
        
    Returns:
        List of generated texts.
    """
    global _engine_cache
    
    if model not in _engine_cache:
        _engine_cache[model] = OnnInferenceEngine(model)
    
    engine = _engine_cache[model]
    results = engine.generate_batch(prompts, **kwargs)
    return [r.generated_text for r in results]
