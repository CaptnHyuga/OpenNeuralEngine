"""ONN Inference Server - OpenAI-compatible API server.

Wraps vLLM's OpenAI-compatible server for production deployment.
Provides a simple interface with automatic hardware optimization.

Features:
- OpenAI-compatible API endpoints
- Automatic model quantization for low VRAM
- Built-in metrics and monitoring
- Easy deployment with one command

Usage:
    # From code
    from src.inference import serve
    serve("./my_model", port=8000)
    
    # From CLI
    onn serve --model ./my_model --port 8000
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Import engine
from .engine import (
    OnnInferenceEngine,
    InferenceConfig,
    GenerationResult,
    generate as engine_generate,
    generate_batch as engine_generate_batch,
)


# ============================================================================
# API Models (OpenAI-compatible)
# ============================================================================

class ChatMessage(BaseModel):
    """Chat message format."""
    role: str = Field(..., description="Role: 'system', 'user', or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: str = Field(..., description="Model to use")
    messages: List[ChatMessage] = Field(..., description="Chat messages")
    max_tokens: Optional[int] = Field(256, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(0.7, description="Sampling temperature")
    top_p: Optional[float] = Field(0.9, description="Nucleus sampling threshold")
    stream: Optional[bool] = Field(False, description="Stream response")
    stop: Optional[List[str]] = Field(None, description="Stop sequences")


class CompletionRequest(BaseModel):
    """OpenAI-compatible completion request."""
    model: str = Field(..., description="Model to use")
    prompt: Union[str, List[str]] = Field(..., description="Input prompt(s)")
    max_tokens: Optional[int] = Field(256, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(0.7, description="Sampling temperature")
    top_p: Optional[float] = Field(0.9, description="Nucleus sampling threshold")
    stream: Optional[bool] = Field(False, description="Stream response")
    stop: Optional[List[str]] = Field(None, description="Stop sequences")


class CompletionChoice(BaseModel):
    """Completion choice in response."""
    index: int
    text: str = ""
    message: Optional[ChatMessage] = None
    finish_reason: str = "length"


class Usage(BaseModel):
    """Token usage information."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class CompletionResponse(BaseModel):
    """OpenAI-compatible completion response."""
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Usage


# ============================================================================
# Server Configuration
# ============================================================================

@dataclass
class ServerConfig:
    """Server configuration."""
    
    model: str = ""
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Model settings
    trust_remote_code: bool = True
    dtype: str = "auto"
    
    # vLLM specific
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    
    # Server settings
    max_concurrent_requests: int = 100
    timeout: float = 300.0
    
    # CORS
    cors_origins: List[str] = field(default_factory=lambda: ["*"])


# ============================================================================
# Inference Server
# ============================================================================

class OnnInferenceServer:
    """OpenAI-compatible inference server.
    
    Wraps OnnInferenceEngine with a FastAPI server providing:
    - /v1/completions - Text completion
    - /v1/chat/completions - Chat completion
    - /v1/models - List available models
    - /health - Health check
    """
    
    def __init__(self, config: ServerConfig):
        """Initialize server.
        
        Args:
            config: Server configuration.
        """
        self.config = config
        self.app = self._create_app()
        self.engine: Optional[OnnInferenceEngine] = None
        self._request_count = 0
        self._start_time = time.time()
    
    def _create_app(self) -> FastAPI:
        """Create FastAPI application."""
        app = FastAPI(
            title="ONN Inference Server",
            description="OpenAI-compatible inference API powered by OpenNeuralEngine",
            version="2.0.0",
        )
        
        # CORS
        app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Routes
        app.get("/health")(self.health)
        app.get("/v1/models")(self.list_models)
        app.post("/v1/completions")(self.completions)
        app.post("/v1/chat/completions")(self.chat_completions)
        
        @app.on_event("startup")
        async def startup():
            await self._load_model()
        
        return app
    
    async def _load_model(self):
        """Load the model on startup."""
        print(f"🚀 Loading model: {self.config.model}")
        
        inference_config = InferenceConfig(
            model_path=self.config.model,
            trust_remote_code=self.config.trust_remote_code,
            dtype=self.config.dtype,
            tensor_parallel_size=self.config.tensor_parallel_size,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
        )
        
        self.engine = OnnInferenceEngine(
            model=self.config.model,
            config=inference_config,
        )
        
        print(f"✅ Model loaded and ready!")
    
    async def health(self) -> Dict[str, Any]:
        """Health check endpoint."""
        uptime = time.time() - self._start_time
        return {
            "status": "healthy",
            "model": self.config.model,
            "uptime_seconds": uptime,
            "requests_served": self._request_count,
        }
    
    async def list_models(self) -> Dict[str, Any]:
        """List available models (OpenAI-compatible)."""
        return {
            "object": "list",
            "data": [
                {
                    "id": self.config.model,
                    "object": "model",
                    "created": int(self._start_time),
                    "owned_by": "onn",
                }
            ],
        }
    
    async def completions(self, request: CompletionRequest) -> CompletionResponse:
        """Text completion endpoint (OpenAI-compatible)."""
        if self.engine is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        self._request_count += 1
        
        # Handle single or batch prompts
        prompts = request.prompt if isinstance(request.prompt, list) else [request.prompt]
        
        # Generate
        results = []
        for prompt in prompts:
            result = self.engine.generate(
                prompt=prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop,
            )
            results.append(result)
        
        # Build response
        choices = [
            CompletionChoice(
                index=i,
                text=r.generated_text,
                finish_reason=r.finish_reason,
            )
            for i, r in enumerate(results)
        ]
        
        # Estimate tokens
        total_completion_tokens = sum(r.tokens_generated for r in results)
        
        return CompletionResponse(
            id=f"cmpl-{int(time.time())}",
            created=int(time.time()),
            model=request.model,
            choices=choices,
            usage=Usage(
                completion_tokens=total_completion_tokens,
                total_tokens=total_completion_tokens,
            ),
        )
    
    async def chat_completions(self, request: ChatCompletionRequest) -> CompletionResponse:
        """Chat completion endpoint (OpenAI-compatible)."""
        if self.engine is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        self._request_count += 1
        
        # Convert messages to prompt
        prompt = self._messages_to_prompt(request.messages)
        
        # Generate
        result = self.engine.generate(
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop,
        )
        
        # Build response
        return CompletionResponse(
            id=f"chatcmpl-{int(time.time())}",
            object="chat.completion",
            created=int(time.time()),
            model=request.model,
            choices=[
                CompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=result.generated_text),
                    finish_reason=result.finish_reason,
                )
            ],
            usage=Usage(
                completion_tokens=result.tokens_generated,
                total_tokens=result.tokens_generated,
            ),
        )
    
    def _messages_to_prompt(self, messages: List[ChatMessage]) -> str:
        """Convert chat messages to a text prompt.
        
        Uses a simple format. For model-specific templates,
        override this method or use the model's chat template.
        """
        parts = []
        for msg in messages:
            if msg.role == "system":
                parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                parts.append(f"Assistant: {msg.content}")
        
        # Add prompt for assistant response
        parts.append("Assistant:")
        return "\n".join(parts)
    
    def run(self):
        """Run the server."""
        print(f"\n🌐 Starting ONN Inference Server")
        print(f"   URL: http://{self.config.host}:{self.config.port}")
        print(f"   Model: {self.config.model}")
        print(f"   Docs: http://{self.config.host}:{self.config.port}/docs")
        print()
        
        uvicorn.run(
            self.app,
            host=self.config.host,
            port=self.config.port,
            log_level="info",
        )


# ============================================================================
# Convenience Functions
# ============================================================================

def serve(
    model: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    dtype: str = "auto",
    **kwargs,
) -> None:
    """Launch inference server with one command.
    
    Starts an OpenAI-compatible server for the given model:
    
        serve("./my_model", port=8000)
    
    Then use with any OpenAI client:
    
        import openai
        client = openai.OpenAI(base_url="http://localhost:8000/v1")
        response = client.completions.create(
            model="./my_model",
            prompt="Hello, world!"
        )
    
    Args:
        model: Model path or HuggingFace ID.
        host: Server host.
        port: Server port.
        dtype: Model dtype ("auto", "float16", "bfloat16").
        **kwargs: Additional server configuration.
    """
    config = ServerConfig(
        model=model,
        host=host,
        port=port,
        dtype=dtype,
        **kwargs,
    )
    
    server = OnnInferenceServer(config)
    server.run()


# Re-export convenience functions from engine
generate = engine_generate
generate_batch = engine_generate_batch
