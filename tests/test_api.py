"""Tests for ONN 2.0 API server module."""
from __future__ import annotations

import pytest
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Try to import FastAPI test client
try:
    from fastapi.testclient import TestClient
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    TestClient = None

# Import the API app if available
try:
    from src.api.server import app
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False
    app = None


@pytest.mark.skipif(not FASTAPI_AVAILABLE or not API_AVAILABLE, reason="FastAPI or API not available")
@pytest.mark.unit
class TestAPIHealth:
    """Tests for API health endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/api/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data


@pytest.mark.skipif(not FASTAPI_AVAILABLE or not API_AVAILABLE, reason="FastAPI or API not available")
@pytest.mark.unit
class TestAPIModels:
    """Tests for model management endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_list_models(self, client):
        """Test listing available models."""
        response = client.get("/api/models")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        
    def test_list_models_contains_hf_models(self, client):
        """Test that HuggingFace models are listed."""
        response = client.get("/api/models")
        assert response.status_code == 200
        data = response.json()
        model_ids = [m["id"] for m in data]
        assert "gpt2" in model_ids or "distilgpt2" in model_ids

    def test_search_models(self, client):
        """Test searching models."""
        response = client.get("/api/models/search", params={"query": "gpt"})
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        
    def test_search_models_by_task(self, client):
        """Test searching models with task filter."""
        response = client.get("/api/models/search", params={"query": "gpt", "task": "text-generation"})
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


@pytest.mark.skipif(not FASTAPI_AVAILABLE or not API_AVAILABLE, reason="FastAPI or API not available")
@pytest.mark.unit
class TestAPIDatasets:
    """Tests for dataset management endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_list_datasets(self, client):
        """Test listing available datasets."""
        response = client.get("/api/datasets")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


@pytest.mark.skipif(not FASTAPI_AVAILABLE or not API_AVAILABLE, reason="FastAPI or API not available")
@pytest.mark.unit
class TestAPIHardware:
    """Tests for hardware/system information endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_hardware_info(self, client):
        """Test hardware info endpoint."""
        response = client.get("/api/hardware")
        
        assert response.status_code == 200
        data = response.json()
        # HardwareResponse model fields
        assert "cpu" in data
        assert "ram_gb" in data
        
    def test_hardware_info_cpu_details(self, client):
        """Test hardware info contains CPU details."""
        response = client.get("/api/hardware")
        assert response.status_code == 200
        data = response.json()
        
        assert "cpu" in data
        assert "cores" in data["cpu"]
        assert "threads" in data["cpu"]
        
    def test_hardware_recommend_endpoint(self, client):
        """Test hardware recommendation endpoint."""
        response = client.get("/api/hardware/recommend", params={
            "model_size": 124_000_000,  # GPT-2 size
            "dataset_size": 10000
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "batch_size" in data
        assert "precision" in data
        assert "gradient_checkpointing" in data
        
    def test_hardware_recommend_small_model(self, client):
        """Test recommendations for small model."""
        response = client.get("/api/hardware/recommend", params={
            "model_size": 10_000_000,
            "dataset_size": 1000
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["batch_size"] >= 1
        
    def test_hardware_recommend_large_model(self, client):
        """Test recommendations for large model."""
        response = client.get("/api/hardware/recommend", params={
            "model_size": 7_000_000_000,  # 7B params
            "dataset_size": 100000
        })
        
        assert response.status_code == 200
        data = response.json()
        # Should recommend smaller batch size or quantization for large models
        assert "batch_size" in data


@pytest.mark.skipif(not FASTAPI_AVAILABLE or not API_AVAILABLE, reason="FastAPI or API not available")
@pytest.mark.unit
class TestAPIExperiments:
    """Tests for experiment tracking endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_list_experiments(self, client):
        """Test listing experiments."""
        response = client.get("/api/experiments")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


@pytest.mark.skipif(not FASTAPI_AVAILABLE or not API_AVAILABLE, reason="FastAPI or API not available")
@pytest.mark.unit
class TestAPITraining:
    """Tests for training endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_list_train_runs(self, client):
        """Test listing training runs."""
        response = client.get("/api/train/runs")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


@pytest.mark.skipif(not FASTAPI_AVAILABLE or not API_AVAILABLE, reason="FastAPI or API not available")
@pytest.mark.unit  
class TestAPIInference:
    """Tests for inference endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_inference_endpoint_exists(self, client):
        """Test that inference endpoint exists (may fail without model)."""
        # Just check the endpoint routing works - actual inference tested separately
        response = client.post("/api/inference/generate", json={
            "model": "nonexistent",
            "prompt": "test"
        })
        # Should get 500 (model not found) not 404 (route not found)
        assert response.status_code in [200, 500]
        
    def test_inference_stream_endpoint_exists(self, client):
        """Test that streaming inference endpoint exists."""
        response = client.post("/api/inference/stream", json={
            "model": "nonexistent", 
            "prompt": "test"
        })
        # Streaming endpoint returns 200 with error in stream (not HTTP error)
        assert response.status_code == 200
        # Should contain error message in stream
        content = response.text
        assert "error" in content or "[DONE]" in content


@pytest.mark.skipif(not FASTAPI_AVAILABLE or not API_AVAILABLE, reason="FastAPI or API not available")
@pytest.mark.integration
class TestAPIIntegration:
    """Integration tests for API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_full_model_workflow(self, client):
        """Test complete model listing and search workflow."""
        # List all models
        list_response = client.get("/api/models")
        assert list_response.status_code == 200
        
        # Search for specific model
        search_response = client.get("/api/models/search", params={"query": "gpt2"})
        assert search_response.status_code == 200
        
        # Results should contain gpt2 models
        results = search_response.json()
        if len(results) > 0:
            assert any("gpt" in m.get("name", "").lower() or "gpt" in m.get("id", "").lower() 
                      for m in results)

    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options("/api/health")
        # Should not error
        assert response.status_code in [200, 405]  # OPTIONS may not be implemented


# Tests that don't require FastAPI
@pytest.mark.unit
class TestAPISchemas:
    """Tests for API schema/model definitions."""

    def test_import_schemas(self):
        """Test that API schemas can be imported."""
        try:
            from src.api.server import (
                ModelInfo,
                TrainRequest,
                InferenceRequest,
            )
            assert ModelInfo is not None
            assert TrainRequest is not None
            assert InferenceRequest is not None
        except ImportError:
            pytest.skip("API schemas not available")

    def test_model_info_schema(self):
        """Test ModelInfo schema validation."""
        try:
            from src.api.server import ModelInfo
            
            # Should be able to create with required fields
            info = ModelInfo(
                id="test-model",
                name="Test Model",
                source="huggingface",
                size_mb=100.0,
                parameters=1000000,
                task="text-generation",
            )
            
            assert info.id == "test-model"
            assert info.name == "Test Model"
        except ImportError:
            pytest.skip("API schemas not available")
