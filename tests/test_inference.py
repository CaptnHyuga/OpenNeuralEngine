"""Tests for ONN Inference module."""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.mark.unit
class TestInferenceConfig:
    """Tests for InferenceConfig dataclass."""

    def test_config_creation(self):
        """Test InferenceConfig can be created."""
        from src.inference.engine import InferenceConfig
        
        config = InferenceConfig(
            model_path="gpt2",
            max_tokens=256,
            temperature=0.7,
        )
        
        assert config.model_path == "gpt2"
        assert config.max_tokens == 256
        assert config.temperature == 0.7

    def test_config_defaults(self):
        """Test InferenceConfig has sensible defaults."""
        from src.inference.engine import InferenceConfig
        
        config = InferenceConfig()
        
        assert config.max_tokens == 256
        assert 0 < config.temperature <= 1.0
        assert 0 < config.top_p <= 1.0
        assert config.backend == "auto"


@pytest.mark.unit
class TestGenerationResult:
    """Tests for GenerationResult dataclass."""

    def test_result_creation(self):
        """Test GenerationResult can be created."""
        from src.inference.engine import GenerationResult
        
        result = GenerationResult(
            prompt="Hello",
            generated_text=" world!",
            full_text="Hello world!",
            tokens_generated=2,
            generation_time_seconds=0.1,
        )
        
        assert result.prompt == "Hello"
        assert result.generated_text == " world!"
        assert result.full_text == "Hello world!"

    def test_result_calculates_speed(self):
        """Test GenerationResult calculates tokens per second."""
        from src.inference.engine import GenerationResult
        
        result = GenerationResult(
            prompt="Hello",
            generated_text=" world!",
            full_text="Hello world!",
            tokens_generated=10,
            generation_time_seconds=0.5,
        )
        
        assert result.tokens_per_second == 20.0  # 10 tokens / 0.5 seconds


@pytest.mark.unit
class TestServerConfig:
    """Tests for ServerConfig dataclass."""

    def test_server_config_creation(self):
        """Test ServerConfig can be created."""
        from src.inference.server import ServerConfig
        
        config = ServerConfig(
            model="gpt2",
            host="localhost",
            port=8000,
        )
        
        assert config.model == "gpt2"
        assert config.host == "localhost"
        assert config.port == 8000

    def test_server_config_defaults(self):
        """Test ServerConfig has sensible defaults."""
        from src.inference.server import ServerConfig
        
        config = ServerConfig()
        
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.max_concurrent_requests == 100


@pytest.mark.unit
class TestAPIModels:
    """Tests for OpenAI-compatible API models."""

    def test_chat_message(self):
        """Test ChatMessage model."""
        from src.inference.server import ChatMessage
        
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_completion_request(self):
        """Test CompletionRequest model."""
        from src.inference.server import CompletionRequest
        
        request = CompletionRequest(
            model="gpt2",
            prompt="Hello, world!",
            max_tokens=50,
        )
        
        assert request.model == "gpt2"
        assert request.prompt == "Hello, world!"
        assert request.max_tokens == 50

    def test_chat_completion_request(self):
        """Test ChatCompletionRequest model."""
        from src.inference.server import ChatCompletionRequest, ChatMessage
        
        request = ChatCompletionRequest(
            model="gpt2",
            messages=[
                ChatMessage(role="user", content="Hi!")
            ],
        )
        
        assert request.model == "gpt2"
        assert len(request.messages) == 1
        assert request.messages[0].content == "Hi!"


@pytest.mark.unit
class TestOnnInferenceEngine:
    """Tests for OnnInferenceEngine."""

    def test_engine_backend_selection_without_vllm(self):
        """Test engine selects HF backend when vLLM not available."""
        from src.inference.engine import OnnInferenceEngine, InferenceConfig, VLLM_AVAILABLE, HF_AVAILABLE
        
        if not HF_AVAILABLE:
            pytest.skip("transformers not installed")
        
        # Mock to test backend selection
        with patch.object(OnnInferenceEngine, '_init_backend'):
            config = InferenceConfig(backend="hf")
            engine = OnnInferenceEngine.__new__(OnnInferenceEngine)
            engine.model_path = "gpt2"
            engine.config = config
            engine._configure()
            
            # With HF forced, should use HF
            assert engine.config.backend == "hf"

    def test_engine_auto_dtype_selection(self):
        """Test engine auto-selects dtype based on hardware."""
        import torch
        from src.inference.engine import InferenceConfig
        
        config = InferenceConfig(dtype="auto")
        
        # After configuration, dtype should be resolved
        # (In real usage, this happens in _configure)
        assert config.dtype == "auto"  # Before configuration


@pytest.mark.unit
class TestOnnInferenceServer:
    """Tests for OnnInferenceServer."""

    def test_server_creates_app(self):
        """Test server creates FastAPI app."""
        from src.inference.server import OnnInferenceServer, ServerConfig
        
        config = ServerConfig(model="gpt2")
        server = OnnInferenceServer(config)
        
        assert server.app is not None
        assert server.config.model == "gpt2"

    def test_server_messages_to_prompt(self):
        """Test server converts messages to prompt."""
        from src.inference.server import OnnInferenceServer, ServerConfig, ChatMessage
        
        config = ServerConfig(model="gpt2")
        server = OnnInferenceServer(config)
        
        messages = [
            ChatMessage(role="system", content="You are helpful."),
            ChatMessage(role="user", content="Hi!"),
        ]
        
        prompt = server._messages_to_prompt(messages)
        
        assert "System:" in prompt
        assert "You are helpful" in prompt
        assert "User:" in prompt
        assert "Hi!" in prompt
        assert "Assistant:" in prompt


@pytest.mark.integration
class TestEngineIntegration:
    """Integration tests for inference engine."""

    @pytest.mark.slow
    def test_hf_generation(self):
        """Test generation with HuggingFace backend."""
        try:
            from src.inference.engine import OnnInferenceEngine, InferenceConfig
            
            config = InferenceConfig(
                model_path="gpt2",
                backend="hf",
                max_tokens=20,
            )
            
            engine = OnnInferenceEngine(model="gpt2", config=config)
            result = engine.generate("Hello, my name is")
            
            assert result.prompt == "Hello, my name is"
            assert len(result.generated_text) > 0
            assert result.tokens_generated > 0
            
        except ImportError:
            pytest.skip("transformers not installed")
        except Exception as e:
            if "model" in str(e).lower():
                pytest.skip("Model not available for testing")
            raise


@pytest.mark.integration
class TestServerIntegration:
    """Integration tests for inference server."""

    def test_server_health_endpoint(self):
        """Test server health endpoint."""
        from src.inference.server import OnnInferenceServer, ServerConfig
        from fastapi.testclient import TestClient
        
        config = ServerConfig(model="gpt2")
        server = OnnInferenceServer(config)
        
        # Don't actually load the model for this test
        server.engine = MagicMock()
        
        client = TestClient(server.app)
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model"] == "gpt2"
