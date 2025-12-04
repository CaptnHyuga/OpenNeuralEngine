"""Low VRAM Inference - Memory-Efficient Generation for Large Models.

Enables inference on 15B+ models with 4GB VRAM through:
1. INT4 quantization (4x memory reduction)
2. Layer-wise CPU offloading (sequential computation)
3. KV-cache optimization (reduced cache memory)
4. Dynamic attention (Flash Attention when available)

Memory Comparison (15B model inference):
- FP16: ~30GB VRAM
- INT8: ~15GB VRAM
- INT4: ~7.5GB VRAM
- INT4 + CPU offload: ~3GB VRAM

Usage:
    inferencer = LowVRAMInference(
        model_path="models/phi-4",
        vram_limit_gb=4.0,
    )
    
    response = inferencer.generate("Solve: 2x + 5 = 15")
"""
from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for low-VRAM inference."""
    
    # Quantization
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    
    # Device management
    device_map: str = "auto"
    max_memory: Optional[Dict[int, str]] = None  # e.g., {0: "3.5GB", "cpu": "16GB"}
    offload_folder: str = "./offload_cache"
    
    # Generation defaults
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    repetition_penalty: float = 1.1
    
    # Memory optimization
    use_cache: bool = True  # KV-cache (faster but more memory)
    low_cpu_mem_usage: bool = True
    
    # Streaming
    stream: bool = False


class LowVRAMInference:
    """Memory-efficient inference for large language models."""
    
    def __init__(
        self,
        model_path: str,
        config: Optional[InferenceConfig] = None,
        vram_limit_gb: Optional[float] = None,
        adapter_path: Optional[str] = None,
    ):
        """Initialize low-VRAM inference engine.
        
        Args:
            model_path: Path to model or HuggingFace model ID.
            config: Inference configuration.
            vram_limit_gb: VRAM limit for auto-configuration.
            adapter_path: Optional path to LoRA adapters.
        """
        self.model_path = model_path
        self.adapter_path = adapter_path
        
        # Auto-configure based on VRAM
        if config is None:
            config = self._auto_configure(vram_limit_gb)
        self.config = config
        
        self.model = None
        self.tokenizer = None
        
        logger.info(f"Low-VRAM inference initialized for {model_path}")
    
    def _auto_configure(self, vram_limit_gb: Optional[float] = None) -> InferenceConfig:
        """Auto-configure based on available VRAM."""
        config = InferenceConfig()
        
        if vram_limit_gb is None:
            if torch.cuda.is_available():
                vram_limit_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            else:
                vram_limit_gb = 0
        
        logger.info(f"Auto-configuring inference for {vram_limit_gb:.1f}GB VRAM")
        
        if vram_limit_gb <= 4:
            # Ultra-low VRAM mode
            config.load_in_4bit = True
            config.bnb_4bit_use_double_quant = True
            config.max_memory = {0: f"{vram_limit_gb * 0.85:.1f}GB", "cpu": "32GB"}
            config.max_new_tokens = 128
            config.use_cache = True
            config.low_cpu_mem_usage = True
            logger.info("Ultra-low VRAM mode: INT4 + CPU offload + small generation")
        
        elif vram_limit_gb <= 8:
            config.load_in_4bit = True
            config.bnb_4bit_use_double_quant = True
            config.max_memory = {0: f"{vram_limit_gb * 0.9:.1f}GB", "cpu": "32GB"}
            config.max_new_tokens = 256
            logger.info("Low VRAM mode: INT4 + partial offload")
        
        elif vram_limit_gb <= 16:
            config.load_in_4bit = True
            config.bnb_4bit_use_double_quant = False
            config.max_new_tokens = 512
            logger.info("Medium VRAM mode: INT4")
        
        else:
            config.load_in_4bit = False  # Can use FP16
            config.max_new_tokens = 1024
            logger.info("High VRAM mode: FP16")
        
        return config
    
    def _setup_quantization_config(self):
        """Create bitsandbytes quantization config."""
        from transformers import BitsAndBytesConfig
        
        compute_dtype = (
            torch.float16 if self.config.bnb_4bit_compute_dtype == "float16"
            else torch.bfloat16
        )
        
        return BitsAndBytesConfig(
            load_in_4bit=self.config.load_in_4bit,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
        )
    
    def load_model(self) -> None:
        """Load model with quantization and optional LoRA adapters."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info(f"Loading model from {self.model_path}...")
        
        # Clean up any existing model
        self._cleanup()
        
        # Load tokenizer
        tokenizer_path = self.adapter_path if self.adapter_path else self.model_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            padding_side="left",  # For generation
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Setup model loading kwargs
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16,
            "low_cpu_mem_usage": self.config.low_cpu_mem_usage,
        }
        
        # Quantization
        if self.config.load_in_4bit:
            model_kwargs["quantization_config"] = self._setup_quantization_config()
        
        # Device mapping
        if self.config.max_memory:
            model_kwargs["device_map"] = "auto"
            model_kwargs["max_memory"] = self.config.max_memory
            model_kwargs["offload_folder"] = self.config.offload_folder
            
            # Create offload folder
            Path(self.config.offload_folder).mkdir(parents=True, exist_ok=True)
        else:
            model_kwargs["device_map"] = self.config.device_map
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            **model_kwargs,
        )
        
        # Load LoRA adapters if provided
        if self.adapter_path:
            from peft import PeftModel
            logger.info(f"Loading LoRA adapters from {self.adapter_path}")
            self.model = PeftModel.from_pretrained(
                self.model,
                self.adapter_path,
            )
        
        # Set to eval mode
        self.model.eval()
        
        # Report memory
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            logger.info(f"Model loaded. VRAM: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    def _cleanup(self) -> None:
        """Clean up memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        do_sample: Optional[bool] = None,
        repetition_penalty: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate text from prompt.
        
        Args:
            prompt: Input prompt.
            max_new_tokens: Max tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            top_k: Top-k sampling parameter.
            do_sample: Whether to use sampling.
            repetition_penalty: Repetition penalty.
            stop_sequences: Stop generation on these sequences.
            system_prompt: Optional system prompt to prepend.
        
        Returns:
            Generated text.
        """
        if self.model is None:
            self.load_model()
        
        # Use defaults from config
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature if temperature is not None else self.config.temperature
        top_p = top_p if top_p is not None else self.config.top_p
        top_k = top_k if top_k is not None else self.config.top_k
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        repetition_penalty = repetition_penalty or self.config.repetition_penalty
        
        # Format prompt with system prompt if provided
        if system_prompt:
            full_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{prompt}\n<|assistant|>\n"
        else:
            full_prompt = prompt
        
        # Tokenize
        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        
        # Move to model device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Setup generation kwargs
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "do_sample": do_sample,
            "repetition_penalty": repetition_penalty,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": self.config.use_cache,
        }
        
        # Handle stop sequences
        if stop_sequences:
            from transformers import StoppingCriteria, StoppingCriteriaList
            
            class StopOnSequences(StoppingCriteria):
                def __init__(self, stop_ids: List[List[int]]):
                    self.stop_ids = stop_ids
                
                def __call__(self, input_ids, scores, **kwargs) -> bool:
                    for stop_id in self.stop_ids:
                        if len(stop_id) <= input_ids.shape[1]:
                            if input_ids[0, -len(stop_id):].tolist() == stop_id:
                                return True
                    return False
            
            stop_ids = [
                self.tokenizer.encode(seq, add_special_tokens=False)
                for seq in stop_sequences
            ]
            gen_kwargs["stopping_criteria"] = StoppingCriteriaList([
                StopOnSequences(stop_ids)
            ])
        
        # Generate
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        
        # Decode only new tokens
        new_tokens = outputs[0, inputs["input_ids"].shape[1]:]
        generated = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        gen_time = time.time() - start_time
        tokens_per_sec = len(new_tokens) / gen_time
        
        logger.debug(f"Generated {len(new_tokens)} tokens in {gen_time:.2f}s ({tokens_per_sec:.1f} tok/s)")
        
        return generated
    
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> Generator[str, None, None]:
        """Stream generated tokens.
        
        Args:
            prompt: Input prompt.
            max_new_tokens: Max tokens to generate.
            temperature: Sampling temperature.
            **kwargs: Additional generation arguments.
        
        Yields:
            Generated tokens as they're produced.
        """
        if self.model is None:
            self.load_model()
        
        from transformers import TextIteratorStreamer
        from threading import Thread
        
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature if temperature is not None else self.config.temperature
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Setup streamer
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        
        # Generation kwargs
        gen_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": self.config.do_sample,
            "streamer": streamer,
        }
        
        # Start generation in thread
        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()
        
        # Yield tokens as they're generated
        for token in streamer:
            yield token
        
        thread.join()
    
    def batch_generate(
        self,
        prompts: List[str],
        max_new_tokens: Optional[int] = None,
        **kwargs,
    ) -> List[str]:
        """Generate for multiple prompts (batched for efficiency).
        
        Args:
            prompts: List of input prompts.
            max_new_tokens: Max tokens per generation.
            **kwargs: Additional generation arguments.
        
        Returns:
            List of generated texts.
        """
        if self.model is None:
            self.load_model()
        
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        
        # Tokenize batch
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        # Decode each output
        results = []
        for i, output in enumerate(outputs):
            # Skip prompt tokens
            prompt_len = inputs["input_ids"][i].ne(self.tokenizer.pad_token_id).sum()
            new_tokens = output[prompt_len:]
            results.append(self.tokenizer.decode(new_tokens, skip_special_tokens=True))
        
        return results
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        stats = {}
        
        if torch.cuda.is_available():
            stats["vram_allocated_gb"] = torch.cuda.memory_allocated() / (1024**3)
            stats["vram_reserved_gb"] = torch.cuda.memory_reserved() / (1024**3)
            stats["vram_total_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            stats["vram_free_gb"] = stats["vram_total_gb"] - stats["vram_allocated_gb"]
        
        import psutil
        mem = psutil.virtual_memory()
        stats["ram_used_gb"] = mem.used / (1024**3)
        stats["ram_total_gb"] = mem.total / (1024**3)
        stats["ram_available_gb"] = mem.available / (1024**3)
        
        return stats


class InferenceServer:
    """Simple inference server for serving models via HTTP."""
    
    def __init__(
        self,
        model_path: str,
        vram_limit_gb: Optional[float] = None,
        adapter_path: Optional[str] = None,
    ):
        """Initialize inference server.
        
        Args:
            model_path: Path to model.
            vram_limit_gb: VRAM limit.
            adapter_path: Optional LoRA adapter path.
        """
        self.inferencer = LowVRAMInference(
            model_path=model_path,
            vram_limit_gb=vram_limit_gb,
            adapter_path=adapter_path,
        )
    
    def start(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        """Start the inference server.
        
        Args:
            host: Host to bind to.
            port: Port to bind to.
        """
        from fastapi import FastAPI
        from pydantic import BaseModel
        import uvicorn
        
        app = FastAPI(title="ONN Inference Server")
        
        class GenerateRequest(BaseModel):
            prompt: str
            max_new_tokens: int = 256
            temperature: float = 0.7
            top_p: float = 0.9
            system_prompt: Optional[str] = None
        
        class GenerateResponse(BaseModel):
            generated_text: str
            tokens_generated: int
        
        @app.on_event("startup")
        async def startup():
            self.inferencer.load_model()
        
        @app.post("/generate", response_model=GenerateResponse)
        async def generate(request: GenerateRequest):
            result = self.inferencer.generate(
                prompt=request.prompt,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                system_prompt=request.system_prompt,
            )
            return GenerateResponse(
                generated_text=result,
                tokens_generated=len(self.inferencer.tokenizer.encode(result)),
            )
        
        @app.get("/health")
        async def health():
            return {"status": "healthy", "memory": self.inferencer.get_memory_usage()}
        
        uvicorn.run(app, host=host, port=port)


# Convenience function
def quick_inference(
    model_path: str,
    prompt: str,
    vram_limit_gb: Optional[float] = None,
    adapter_path: Optional[str] = None,
    **kwargs,
) -> str:
    """Quick inference function for one-off generation.
    
    Args:
        model_path: Path to model.
        prompt: Input prompt.
        vram_limit_gb: VRAM limit.
        adapter_path: Optional LoRA adapter path.
        **kwargs: Generation arguments.
    
    Returns:
        Generated text.
    """
    inferencer = LowVRAMInference(
        model_path=model_path,
        vram_limit_gb=vram_limit_gb,
        adapter_path=adapter_path,
    )
    return inferencer.generate(prompt, **kwargs)
