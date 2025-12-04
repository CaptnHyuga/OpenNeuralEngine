"""ONN Inference Module - Production inference serving.

Wraps vLLM for high-performance inference with automatic optimization.
Provides simple API for both local inference and server deployment.

Usage:
    from src.inference import serve, generate
    
    # Quick generation
    output = generate("./my_model", "Hello, world!")
    
    # Launch inference server
    serve("./my_model", port=8000)
"""

from .server import (
    OnnInferenceServer,
    serve,
    generate,
    generate_batch,
    ServerConfig,
)

from .engine import (
    OnnInferenceEngine,
    InferenceConfig,
    GenerationResult,
)

__all__ = [
    # Server
    "OnnInferenceServer",
    "serve",
    "ServerConfig",
    # Engine
    "OnnInferenceEngine",
    "InferenceConfig",
    "GenerationResult",
    # Generation
    "generate",
    "generate_batch",
]
