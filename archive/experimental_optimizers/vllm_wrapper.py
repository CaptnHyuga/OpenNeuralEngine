"""vLLM Inference Wrapper - High-performance inference serving.

Wraps vLLM for efficient large-scale inference:
- Continuous batching
- PagedAttention for efficient KV cache
- Tensor parallelism for multi-GPU
- OpenAI-compatible API server

Falls back to HuggingFace pipeline if vLLM not available.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

import torch


class InferenceBackend(Enum):
    """Available inference backends."""
    VLLM = "vllm"
    HF_PIPELINE = "hf_pipeline"
    HF_GENERATE = "hf_generate"
    ONNX = "onnx"


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    do_sample: bool = True
    
    # Stopping criteria
    stop_sequences: List[str] = field(default_factory=list)
    stop_token_ids: List[int] = field(default_factory=list)
    
    # Streaming
    stream: bool = False
    
    # vLLM specific
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    best_of: int = 1
    use_beam_search: bool = False
    
    def to_vllm_params(self) -> Dict[str, Any]:
        """Convert to vLLM SamplingParams kwargs."""
        params = {
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature if self.do_sample else 0.0,
            "top_p": self.top_p,
            "top_k": self.top_k if self.top_k > 0 else -1,
            "repetition_penalty": self.repetition_penalty,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "stop": self.stop_sequences if self.stop_sequences else None,
            "stop_token_ids": self.stop_token_ids if self.stop_token_ids else None,
        }
        
        if self.use_beam_search:
            params["use_beam_search"] = True
            params["best_of"] = self.best_of
        
        return params
    
    def to_hf_params(self) -> Dict[str, Any]:
        """Convert to HuggingFace generate kwargs."""
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature if self.do_sample else 1.0,
            "top_p": self.top_p,
            "top_k": self.top_k if self.top_k > 0 else None,
            "repetition_penalty": self.repetition_penalty,
            "do_sample": self.do_sample,
        }


@dataclass
class InferenceResult:
    """Result from inference."""
    
    text: str
    prompt: str
    tokens_generated: int = 0
    generation_time_seconds: float = 0.0
    tokens_per_second: float = 0.0
    finish_reason: str = "stop"
    
    # Additional info
    prompt_tokens: int = 0
    total_tokens: int = 0


@dataclass 
class BatchInferenceResult:
    """Result from batch inference."""
    
    results: List[InferenceResult]
    total_time_seconds: float = 0.0
    total_tokens: int = 0
    throughput_tokens_per_second: float = 0.0


class VLLMInferenceWrapper:
    """High-performance inference wrapper using vLLM."""
    
    def __init__(
        self,
        model_name_or_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        dtype: str = "auto",
        quantization: Optional[str] = None,  # "awq", "gptq", "squeezellm"
        max_model_len: Optional[int] = None,
        trust_remote_code: bool = True,
    ):
        """Initialize inference wrapper.
        
        Args:
            model_name_or_path: Model identifier or path.
            tensor_parallel_size: Number of GPUs for tensor parallelism.
            gpu_memory_utilization: Fraction of GPU memory to use.
            dtype: Data type ("auto", "float16", "bfloat16").
            quantization: Quantization method if model is quantized.
            max_model_len: Maximum sequence length (None = auto).
            trust_remote_code: Trust remote code in HF models.
        """
        self.model_name = model_name_or_path
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.dtype = dtype
        self.quantization = quantization
        self.max_model_len = max_model_len
        self.trust_remote_code = trust_remote_code
        
        self._backend: InferenceBackend = InferenceBackend.HF_GENERATE
        self._engine = None
        self._tokenizer = None
        self._model = None
        self._initialized = False
    
    def _check_vllm(self) -> bool:
        """Check if vLLM is available."""
        try:
            import vllm
            return True
        except ImportError:
            return False
    
    def initialize(self):
        """Initialize the inference engine."""
        if self._initialized:
            return
        
        if self._check_vllm():
            self._init_vllm()
        else:
            self._init_hf_fallback()
        
        self._initialized = True
    
    def _init_vllm(self):
        """Initialize vLLM engine."""
        from vllm import LLM, SamplingParams
        
        self._backend = InferenceBackend.VLLM
        
        engine_args = {
            "model": self.model_name,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "trust_remote_code": self.trust_remote_code,
        }
        
        if self.dtype != "auto":
            engine_args["dtype"] = self.dtype
        
        if self.quantization:
            engine_args["quantization"] = self.quantization
        
        if self.max_model_len:
            engine_args["max_model_len"] = self.max_model_len
        
        self._engine = LLM(**engine_args)
        print(f"✓ Initialized vLLM engine with {self.tensor_parallel_size} GPU(s)")
    
    def _init_hf_fallback(self):
        """Initialize HuggingFace fallback."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self._backend = InferenceBackend.HF_GENERATE
        
        print("ℹ️  vLLM not available, using HuggingFace backend")
        
        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        )
        
        # Load model
        model_kwargs = {
            "trust_remote_code": self.trust_remote_code,
            "device_map": "auto",
        }
        
        if self.dtype == "float16":
            model_kwargs["torch_dtype"] = torch.float16
        elif self.dtype == "bfloat16":
            model_kwargs["torch_dtype"] = torch.bfloat16
        
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs,
        )
        
        print(f"✓ Initialized HuggingFace model")
    
    def generate(
        self,
        prompts: Union[str, List[str]],
        config: Optional[GenerationConfig] = None,
    ) -> Union[InferenceResult, BatchInferenceResult]:
        """Generate text for one or more prompts.
        
        Args:
            prompts: Single prompt or list of prompts.
            config: Generation configuration.
        
        Returns:
            InferenceResult for single prompt, BatchInferenceResult for multiple.
        """
        if not self._initialized:
            self.initialize()
        
        config = config or GenerationConfig()
        is_single = isinstance(prompts, str)
        prompt_list = [prompts] if is_single else prompts
        
        start_time = time.time()
        
        if self._backend == InferenceBackend.VLLM:
            results = self._generate_vllm(prompt_list, config)
        else:
            results = self._generate_hf(prompt_list, config)
        
        total_time = time.time() - start_time
        
        if is_single:
            return results[0]
        
        total_tokens = sum(r.tokens_generated for r in results)
        return BatchInferenceResult(
            results=results,
            total_time_seconds=total_time,
            total_tokens=total_tokens,
            throughput_tokens_per_second=total_tokens / total_time if total_time > 0 else 0,
        )
    
    def _generate_vllm(
        self,
        prompts: List[str],
        config: GenerationConfig,
    ) -> List[InferenceResult]:
        """Generate using vLLM engine."""
        from vllm import SamplingParams
        
        params = SamplingParams(**config.to_vllm_params())
        outputs = self._engine.generate(prompts, params)
        
        results = []
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            num_tokens = len(output.outputs[0].token_ids)
            
            results.append(InferenceResult(
                text=generated_text,
                prompt=prompt,
                tokens_generated=num_tokens,
                finish_reason=output.outputs[0].finish_reason or "stop",
                prompt_tokens=len(output.prompt_token_ids),
                total_tokens=len(output.prompt_token_ids) + num_tokens,
            ))
        
        return results
    
    def _generate_hf(
        self,
        prompts: List[str],
        config: GenerationConfig,
    ) -> List[InferenceResult]:
        """Generate using HuggingFace model."""
        results = []
        
        for prompt in prompts:
            start_time = time.time()
            
            inputs = self._tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
            
            prompt_length = inputs["input_ids"].shape[1]
            
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    **config.to_hf_params(),
                    pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
                )
            
            generated_ids = outputs[0][prompt_length:]
            generated_text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            gen_time = time.time() - start_time
            num_tokens = len(generated_ids)
            
            results.append(InferenceResult(
                text=generated_text,
                prompt=prompt,
                tokens_generated=num_tokens,
                generation_time_seconds=gen_time,
                tokens_per_second=num_tokens / gen_time if gen_time > 0 else 0,
                prompt_tokens=prompt_length,
                total_tokens=prompt_length + num_tokens,
            ))
        
        return results
    
    async def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> AsyncIterator[str]:
        """Stream generated text token by token.
        
        Args:
            prompt: Input prompt.
            config: Generation configuration.
        
        Yields:
            Generated text chunks.
        """
        if not self._initialized:
            self.initialize()
        
        config = config or GenerationConfig()
        config.stream = True
        
        if self._backend == InferenceBackend.VLLM:
            async for chunk in self._stream_vllm(prompt, config):
                yield chunk
        else:
            async for chunk in self._stream_hf(prompt, config):
                yield chunk
    
    async def _stream_vllm(
        self,
        prompt: str,
        config: GenerationConfig,
    ) -> AsyncIterator[str]:
        """Stream using vLLM."""
        from vllm import SamplingParams
        
        params = SamplingParams(**config.to_vllm_params())
        
        # vLLM streaming requires async engine
        # For simplicity, we'll batch generate and yield
        # Full streaming requires vllm.AsyncLLMEngine
        result = self._generate_vllm([prompt], config)[0]
        
        # Simulate streaming by yielding chunks
        text = result.text
        chunk_size = 4
        for i in range(0, len(text), chunk_size):
            yield text[i:i + chunk_size]
            await asyncio.sleep(0.01)
    
    async def _stream_hf(
        self,
        prompt: str,
        config: GenerationConfig,
    ) -> AsyncIterator[str]:
        """Stream using HuggingFace with TextIteratorStreamer."""
        try:
            from transformers import TextIteratorStreamer
            import threading
            
            inputs = self._tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
            
            streamer = TextIteratorStreamer(
                self._tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )
            
            generation_kwargs = {
                **inputs,
                **config.to_hf_params(),
                "streamer": streamer,
                "pad_token_id": self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
            }
            
            # Run generation in separate thread
            thread = threading.Thread(
                target=self._model.generate,
                kwargs=generation_kwargs,
            )
            thread.start()
            
            for text in streamer:
                yield text
                await asyncio.sleep(0)  # Allow other tasks
            
            thread.join()
            
        except ImportError:
            # Fallback to non-streaming
            result = self._generate_hf([prompt], config)[0]
            yield result.text
    
    @property
    def backend(self) -> InferenceBackend:
        """Get current backend."""
        return self._backend
    
    def __repr__(self) -> str:
        return (
            f"VLLMInferenceWrapper(model={self.model_name}, "
            f"backend={self._backend.value}, "
            f"tensor_parallel={self.tensor_parallel_size})"
        )


class InferenceServer:
    """OpenAI-compatible inference server using vLLM."""
    
    def __init__(
        self,
        model_name_or_path: str,
        host: str = "0.0.0.0",
        port: int = 8000,
        **engine_kwargs,
    ):
        """Initialize inference server.
        
        Args:
            model_name_or_path: Model to serve.
            host: Server host.
            port: Server port.
            **engine_kwargs: Additional engine arguments.
        """
        self.model_name = model_name_or_path
        self.host = host
        self.port = port
        self.engine_kwargs = engine_kwargs
        self._server = None
    
    def start(self):
        """Start the inference server."""
        if self._check_vllm_server():
            self._start_vllm_server()
        else:
            self._start_fastapi_server()
    
    def _check_vllm_server(self) -> bool:
        """Check if vLLM server is available."""
        try:
            from vllm.entrypoints.openai.api_server import run_server
            return True
        except ImportError:
            return False
    
    def _start_vllm_server(self):
        """Start vLLM OpenAI-compatible server."""
        import subprocess
        import sys
        
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_name,
            "--host", self.host,
            "--port", str(self.port),
        ]
        
        if self.engine_kwargs.get("tensor_parallel_size", 1) > 1:
            cmd.extend(["--tensor-parallel-size", str(self.engine_kwargs["tensor_parallel_size"])])
        
        if self.engine_kwargs.get("quantization"):
            cmd.extend(["--quantization", self.engine_kwargs["quantization"]])
        
        print(f"🚀 Starting vLLM server at http://{self.host}:{self.port}")
        self._server = subprocess.Popen(cmd)
    
    def _start_fastapi_server(self):
        """Start FastAPI server with HuggingFace backend."""
        try:
            from fastapi import FastAPI
            from pydantic import BaseModel
            import uvicorn
        except ImportError as e:
            raise ImportError(
                "FastAPI required for server. Run: pip install fastapi uvicorn"
            ) from e
        
        app = FastAPI(title="ONN Inference Server")
        wrapper = VLLMInferenceWrapper(self.model_name, **self.engine_kwargs)
        wrapper.initialize()
        
        class CompletionRequest(BaseModel):
            prompt: str
            max_tokens: int = 256
            temperature: float = 0.7
            top_p: float = 0.9
            stream: bool = False
        
        class CompletionResponse(BaseModel):
            text: str
            usage: dict
        
        @app.post("/v1/completions")
        async def completions(request: CompletionRequest):
            config = GenerationConfig(
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
            )
            
            result = wrapper.generate(request.prompt, config)
            
            return CompletionResponse(
                text=result.text,
                usage={
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.tokens_generated,
                    "total_tokens": result.total_tokens,
                },
            )
        
        @app.get("/health")
        async def health():
            return {"status": "healthy", "model": self.model_name}
        
        print(f"🚀 Starting FastAPI server at http://{self.host}:{self.port}")
        uvicorn.run(app, host=self.host, port=self.port)
    
    def stop(self):
        """Stop the server."""
        if self._server:
            self._server.terminate()
            self._server = None


# Convenience functions
def create_inference_engine(
    model: str,
    **kwargs,
) -> VLLMInferenceWrapper:
    """Create an inference engine for a model.
    
    Args:
        model: Model name or path.
        **kwargs: Engine configuration.
    
    Returns:
        Initialized inference wrapper.
    """
    wrapper = VLLMInferenceWrapper(model, **kwargs)
    wrapper.initialize()
    return wrapper


def serve_model(
    model: str,
    port: int = 8000,
    **kwargs,
):
    """Start an inference server for a model.
    
    Args:
        model: Model to serve.
        port: Server port.
        **kwargs: Engine configuration.
    """
    server = InferenceServer(model, port=port, **kwargs)
    server.start()
