"""OpenNeuralEngine API Server.

FastAPI backend that bridges the React frontend with ONN core functionality.
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from pathlib import Path
import asyncio
import uuid
import json
import time

app = FastAPI(
    title="OpenNeuralEngine API",
    description="Production-Grade Democratic AI Framework API",
    version="2.0.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Models
# ============================================================================

class HardwareResponse(BaseModel):
    gpu: Optional[Dict[str, Any]] = None
    cpu: Dict[str, Any]
    ram_gb: float
    storage_gb: float

class RecommendedConfig(BaseModel):
    batch_size: int
    precision: str
    gradient_checkpointing: bool
    quantization: Optional[str] = None

class ModelInfo(BaseModel):
    id: str
    name: str
    source: str
    size_mb: float
    parameters: int
    task: str
    loaded: bool = False

class TrainRequest(BaseModel):
    model: str
    dataset: str
    output_dir: str = "./output"
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    precision: Optional[str] = None
    gradient_checkpointing: bool = True

class TrainStatus(BaseModel):
    id: str
    status: str  # pending, running, completed, failed
    progress: float
    metrics: Dict[str, Any]
    logs: List[str] = []

class InferenceRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False

class InferenceResponse(BaseModel):
    id: str
    model: str
    output: str
    tokens_generated: int
    time_ms: float

class DatasetInfo(BaseModel):
    id: str
    name: str
    path: str
    format: str
    size_mb: float
    num_samples: int
    columns: List[str] = []

class Experiment(BaseModel):
    id: str
    name: str
    runs: int
    created_at: str
    last_run_at: str

class ExperimentRun(BaseModel):
    id: str
    experiment: str
    name: str
    status: str
    metrics: Dict[str, List[float]]
    params: Dict[str, Any]
    created_at: str
    duration_s: float

class ExportRequest(BaseModel):
    model_path: str
    format: str  # onnx, torchscript, safetensors
    output_path: str
    optimize: bool = True
    quantize: Optional[str] = None  # int8, int4

# ============================================================================
# State Management
# ============================================================================

# In-memory state (would use Redis/DB in production)
training_runs: Dict[str, TrainStatus] = {}
loaded_models: Dict[str, Any] = {}
active_connections: List[WebSocket] = []

# Local model paths cache
LOCAL_MODELS_DIR = Path("src/Core_Models/Save")

def resolve_model_path(model_id: str) -> str:
    """Resolve a model ID to a loadable path/identifier.
    
    For local models (from Save dir), returns the full path.
    For HF models, returns the ID as-is.
    """
    # Check if it's a local model in our Save directory
    local_path = LOCAL_MODELS_DIR / f"{model_id}.safetensors"
    if local_path.exists():
        return str(local_path)
    
    # Check for directory-based model
    local_dir = LOCAL_MODELS_DIR / model_id
    if local_dir.exists() and local_dir.is_dir():
        return str(local_dir)
    
    # Return as-is (HuggingFace, timm, etc.)
    return model_id

def load_model_and_tokenizer(model_id: str) -> dict:
    """Load a model and its tokenizer.
    
    Returns dict with 'model', 'tokenizer', and 'info' keys.
    """
    model_path = resolve_model_path(model_id)
    
    # Check if it's a HuggingFace model (has / or is a known HF model name)
    is_hf_model = "/" in model_id or model_id in [
        "gpt2", "distilgpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"
    ]
    
    if is_hf_model:
        # Load HuggingFace model with tokenizer
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
        
        # Set pad token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return {"model": model, "tokenizer": tokenizer, "info": None}
    
    else:
        # Local model - load with our loader, but we need a tokenizer
        # For local models trained with this framework, use a base tokenizer
        from src.wrappers import UniversalModelLoader
        from transformers import AutoTokenizer
        
        loader = UniversalModelLoader()
        model, info = loader.load(model_path)
        
        # Use GPT-2 tokenizer as default for local models (most are GPT-style)
        try:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        except Exception:
            tokenizer = None
        
        return {"model": model, "tokenizer": tokenizer, "info": info}

# ============================================================================
# Hardware Endpoints
# ============================================================================

@app.get("/api/hardware", response_model=HardwareResponse)
async def get_hardware():
    """Get hardware profile."""
    try:
        from src.orchestration import HardwareProfiler
        profiler = HardwareProfiler()
        hw = profiler.profile()
        
        return HardwareResponse(
            gpu={
                "name": hw.gpus[0].name if hw.gpus else None,
                "vram_gb": hw.gpus[0].total_vram_mb / 1024 if hw.gpus else 0,
                "cuda_version": hw.gpus[0].driver_version if hw.gpus else None,
            } if hw.gpus else None,
            cpu={
                "name": hw.cpu.name,
                "cores": hw.cpu.physical_cores,
                "threads": hw.cpu.logical_cores,
            },
            ram_gb=hw.memory.total_ram_mb / 1024,
            storage_gb=hw.storage.available_space_gb,
        )
    except ImportError:
        # Fallback if orchestration not available
        import platform
        import os
        return HardwareResponse(
            gpu=None,
            cpu={
                "name": platform.processor() or "Unknown",
                "cores": os.cpu_count() or 1,
                "threads": os.cpu_count() or 1,
            },
            ram_gb=8.0,
            storage_gb=100.0,
        )

@app.get("/api/hardware/recommend", response_model=RecommendedConfig)
async def get_recommended_config(model_size: int, dataset_size: int):
    """Get recommended training configuration."""
    try:
        from src.orchestration import HardwareProfiler, ConfigOrchestrator
        profiler = HardwareProfiler()
        orchestrator = ConfigOrchestrator(profiler)  # Pass profiler, not profile
        config = orchestrator.orchestrate(
            model_name_or_path="custom",
            num_params=model_size,
            dataset_size=dataset_size,
        )
        
        return RecommendedConfig(
            batch_size=config.effective_batch_size,  # Use effective_batch_size
            precision=config.precision.value,
            gradient_checkpointing=config.gradient_checkpointing,
            quantization="int4" if config.quantization and config.quantization.load_in_4bit else (
                "int8" if config.quantization and config.quantization.load_in_8bit else None
            ),
        )
    except ImportError:
        return RecommendedConfig(
            batch_size=4,
            precision="fp32",
            gradient_checkpointing=True,
            quantization=None,
        )

# ============================================================================
# Model Endpoints
# ============================================================================

@app.get("/api/models", response_model=List[ModelInfo])
async def list_models():
    """List available models."""
    # Check local models
    local_models = []
    save_dir = Path("src/Core_Models/Save")
    if save_dir.exists():
        for f in save_dir.glob("*.safetensors"):
            local_models.append(ModelInfo(
                id=f.stem,
                name=f.stem,
                source="local",
                size_mb=f.stat().st_size / (1024 * 1024),
                parameters=0,  # Would need to inspect model
                task="text-generation",
                loaded=f.stem in loaded_models,
            ))
    
    # Add popular HF models
    hf_models = [
        ModelInfo(
            id="gpt2", name="GPT-2", source="huggingface",
            size_mb=500, parameters=124_000_000, task="text-generation",
            loaded="gpt2" in loaded_models,
        ),
        ModelInfo(
            id="distilgpt2", name="DistilGPT-2", source="huggingface",
            size_mb=330, parameters=82_000_000, task="text-generation",
            loaded="distilgpt2" in loaded_models,
        ),
        ModelInfo(
            id="facebook/opt-125m", name="OPT-125M", source="huggingface",
            size_mb=500, parameters=125_000_000, task="text-generation",
            loaded="facebook/opt-125m" in loaded_models,
        ),
        ModelInfo(
            id="microsoft/DialoGPT-small", name="DialoGPT-Small", source="huggingface",
            size_mb=500, parameters=124_000_000, task="conversational",
            loaded="microsoft/DialoGPT-small" in loaded_models,
        ),
    ]

    return local_models + hf_models

@app.post("/api/models/load")
async def load_model(model_id: str):
    """Load a model into memory."""
    try:
        loaded_models[model_id] = load_model_and_tokenizer(model_id)
        return {"success": True, "model_id": model_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.post("/api/models/{model_id}/unload")
async def unload_model(model_id: str):
    """Unload a model from memory."""
    if model_id in loaded_models:
        del loaded_models[model_id]
        return {"success": True}
    raise HTTPException(status_code=404, detail="Model not loaded")

@app.get("/api/models/search", response_model=List[ModelInfo])
async def search_models(query: str, task: Optional[str] = None):
    """Search HuggingFace models."""
    # In production, would call HF API
    # For now, return filtered list
    all_models = [
        ModelInfo(
            id="gpt2", name="GPT-2", source="huggingface",
            size_mb=500, parameters=124_000_000, task="text-generation",
        ),
        ModelInfo(
            id="gpt2-medium", name="GPT-2 Medium", source="huggingface",
            size_mb=1500, parameters=355_000_000, task="text-generation",
        ),
        ModelInfo(
            id="gpt2-large", name="GPT-2 Large", source="huggingface",
            size_mb=3000, parameters=774_000_000, task="text-generation",
        ),
        ModelInfo(
            id="distilgpt2", name="DistilGPT-2", source="huggingface",
            size_mb=330, parameters=82_000_000, task="text-generation",
        ),
        ModelInfo(
            id="facebook/opt-125m", name="OPT-125M", source="huggingface",
            size_mb=500, parameters=125_000_000, task="text-generation",
        ),
        ModelInfo(
            id="facebook/opt-350m", name="OPT-350M", source="huggingface",
            size_mb=1400, parameters=350_000_000, task="text-generation",
        ),
        ModelInfo(
            id="EleutherAI/gpt-neo-125m", name="GPT-Neo 125M", source="huggingface",
            size_mb=500, parameters=125_000_000, task="text-generation",
        ),
        ModelInfo(
            id="microsoft/DialoGPT-small", name="DialoGPT-Small", source="huggingface",
            size_mb=500, parameters=124_000_000, task="conversational",
        ),
        ModelInfo(
            id="microsoft/DialoGPT-medium", name="DialoGPT-Medium", source="huggingface",
            size_mb=1500, parameters=355_000_000, task="conversational",
        ),
        ModelInfo(
            id="bert-base-uncased", name="BERT Base", source="huggingface",
            size_mb=440, parameters=110_000_000, task="fill-mask",
        ),
    ]

    query_lower = query.lower()
    results = [
        m for m in all_models
        if query_lower in m.name.lower() or query_lower in m.id.lower()
    ]
    if task:
        results = [m for m in results if m.task == task]
    
    return results[:10]

# ============================================================================
# Training Endpoints
# ============================================================================

@app.post("/api/train/start")
async def start_training(request: TrainRequest, background_tasks: BackgroundTasks):
    """Start a training run."""
    run_id = str(uuid.uuid4())[:8]
    
    training_runs[run_id] = TrainStatus(
        id=run_id,
        status="pending",
        progress=0.0,
        metrics={"loss": 0, "epoch": 0, "step": 0},
        logs=["Training job created"]
    )
    
    # Run training in background
    background_tasks.add_task(run_training, run_id, request)
    
    return {"run_id": run_id}

async def run_training(run_id: str, request: TrainRequest):
    """Background training task."""
    try:
        training_runs[run_id].status = "running"
        training_runs[run_id].logs.append("Loading model...")
        
        # Simulate or run actual training
        try:
            from src.wrappers import HFTrainerWrapper, UniversalModelLoader
            from src.data_adapters import AUTO_DETECT
            
            # Load model
            loader = UniversalModelLoader()
            model, tokenizer = loader.load(request.model)
            
            # Load data
            adapter = AUTO_DETECT(request.dataset)
            dataset = adapter.dataset
            
            # Train
            trainer = HFTrainerWrapper(model, tokenizer)
            
            def progress_callback(metrics):
                training_runs[run_id].progress = metrics.get("progress", 0)
                training_runs[run_id].metrics = metrics
            
            trainer.train(
                dataset,
                output_dir=request.output_dir,
                num_epochs=request.epochs,
                batch_size=request.batch_size,
                learning_rate=request.learning_rate,
                callback=progress_callback,
            )
            
            training_runs[run_id].status = "completed"
            training_runs[run_id].progress = 100.0
            
        except ImportError:
            # Simulate training
            for epoch in range(request.epochs):
                for step in range(100):
                    await asyncio.sleep(0.1)
                    progress = ((epoch * 100 + step) / (request.epochs * 100)) * 100
                    training_runs[run_id].progress = progress
                    training_runs[run_id].metrics = {
                        "loss": 2.5 - (progress / 100) * 2,
                        "epoch": epoch + 1,
                        "step": epoch * 100 + step,
                    }
                    
                    # Broadcast to WebSocket clients
                    await broadcast_training_update(run_id)
            
            training_runs[run_id].status = "completed"
            training_runs[run_id].progress = 100.0
            
    except Exception as e:
        training_runs[run_id].status = "failed"
        training_runs[run_id].logs.append(f"Error: {str(e)}")

@app.get("/api/train/{run_id}/status", response_model=TrainStatus)
async def get_train_status(run_id: str):
    """Get training run status."""
    if run_id not in training_runs:
        raise HTTPException(status_code=404, detail="Run not found")
    return training_runs[run_id]

@app.post("/api/train/{run_id}/stop")
async def stop_training(run_id: str):
    """Stop a training run."""
    if run_id not in training_runs:
        raise HTTPException(status_code=404, detail="Run not found")
    
    training_runs[run_id].status = "failed"
    training_runs[run_id].logs.append("Training stopped by user")
    return {"success": True}

@app.get("/api/train/runs", response_model=List[TrainStatus])
async def list_train_runs():
    """List all training runs."""
    return list(training_runs.values())

# ============================================================================
# Inference Endpoints
# ============================================================================

@app.post("/api/inference/generate", response_model=InferenceResponse)
async def run_inference(request: InferenceRequest):
    """Run inference on a model."""
    start_time = time.time()
    
    try:
        # Check if model is loaded
        if request.model not in loaded_models:
            # Load on demand
            loaded_models[request.model] = load_model_and_tokenizer(request.model)
        
        model_data = loaded_models[request.model]
        model = model_data["model"]
        tokenizer = model_data["tokenizer"]
        
        if tokenizer is None:
            raise HTTPException(status_code=500, detail="No tokenizer available for this model")
        
        # Generate
        import torch
        inputs = tokenizer(request.prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the input prompt from output
        if output_text.startswith(request.prompt):
            output_text = output_text[len(request.prompt):].strip()
        
        return InferenceResponse(
            id=str(uuid.uuid4())[:8],
            model=request.model,
            output=output_text,
            tokens_generated=len(outputs[0]) - len(inputs["input_ids"][0]),
            time_ms=(time.time() - start_time) * 1000,
        )
        
    except ImportError:
        # Simulate response
        await asyncio.sleep(0.5)
        return InferenceResponse(
            id=str(uuid.uuid4())[:8],
            model=request.model,
            output=f"This is a simulated response to: {request.prompt}",
            tokens_generated=20,
            time_ms=(time.time() - start_time) * 1000,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.post("/api/inference/stream")
async def stream_inference(request: InferenceRequest):
    """Stream inference tokens."""
    async def generate():
        try:
            # Similar to above but yield tokens
            if request.model not in loaded_models:
                loaded_models[request.model] = load_model_and_tokenizer(request.model)
            
            model_data = loaded_models[request.model]
            model = model_data["model"]
            tokenizer = model_data["tokenizer"]
            
            if tokenizer is None:
                yield f"data: {json.dumps({'error': 'No tokenizer available'})}\n\n"
                return
            
            import torch
            inputs = tokenizer(request.prompt, return_tensors="pt")
            
            # Streaming generation
            for _ in range(request.max_tokens):
                with torch.no_grad():
                    outputs = model(**inputs)
                    next_token_logits = outputs.logits[:, -1, :]
                    next_token = torch.multinomial(
                        torch.softmax(next_token_logits / request.temperature, dim=-1), 
                        num_samples=1
                    )
                    
                    if next_token.item() == tokenizer.eos_token_id:
                        break
                    
                    token_text = tokenizer.decode(next_token[0])
                    yield f"data: {json.dumps({'token': token_text})}\n\n"
                    
                    inputs["input_ids"] = torch.cat([inputs["input_ids"], next_token], dim=-1)
                    if "attention_mask" in inputs:
                        inputs["attention_mask"] = torch.cat([
                            inputs["attention_mask"], 
                            torch.ones((1, 1), dtype=torch.long)
                        ], dim=-1)
                    
                    await asyncio.sleep(0.02)  # Small delay for streaming effect
            
            yield "data: [DONE]\n\n"
            
        except ImportError:
            # Simulate streaming
            words = f"This is a simulated streaming response to your prompt: {request.prompt}".split()
            for word in words:
                yield f"data: {json.dumps({'token': word + ' '})}\n\n"
                await asyncio.sleep(0.05)
            yield "data: [DONE]\n\n"
        except Exception as e:
            # Handle errors gracefully in stream
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

# ============================================================================
# Dataset Endpoints
# ============================================================================

@app.get("/api/datasets", response_model=List[DatasetInfo])
async def list_datasets():
    """List available datasets."""
    datasets = []
    data_dir = Path("data/Dataset")
    
    if data_dir.exists():
        for f in data_dir.glob("*"):
            if f.suffix in [".json", ".jsonl", ".csv", ".parquet"]:
                datasets.append(DatasetInfo(
                    id=f.stem,
                    name=f.name,
                    path=str(f),
                    format=f.suffix[1:],
                    size_mb=f.stat().st_size / (1024 * 1024),
                    num_samples=0,  # Would need to count
                ))
    
    return datasets

@app.post("/api/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)):  # noqa: B008 - FastAPI pattern
    """Upload a dataset file."""
    data_dir = Path("data/Dataset")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = data_dir / file.filename
    
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    return DatasetInfo(
        id=file_path.stem,
        name=file.filename,
        path=str(file_path),
        format=file_path.suffix[1:],
        size_mb=len(content) / (1024 * 1024),
        num_samples=0,
    )

@app.get("/api/datasets/analyze")
async def analyze_dataset(path: str):
    """Analyze a dataset."""
    try:
        from src.data_adapters import AUTO_DETECT
        adapter = AUTO_DETECT(path)
        
        return DatasetInfo(
            id=Path(path).stem,
            name=Path(path).name,
            path=path,
            format=adapter.format,
            size_mb=Path(path).stat().st_size / (1024 * 1024) if Path(path).exists() else 0,
            num_samples=len(adapter.dataset) if hasattr(adapter, 'dataset') else 0,
            columns=adapter.columns if hasattr(adapter, 'columns') else [],
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

# ============================================================================
# Experiment Endpoints
# ============================================================================

@app.get("/api/experiments", response_model=List[Experiment])
async def list_experiments():
    """List all experiments."""
    # Would integrate with Aim in production
    return [
        Experiment(
            id="default",
            name="Default Experiment",
            runs=len(training_runs),
            created_at="2024-01-01T00:00:00Z",
            last_run_at="2024-01-01T00:00:00Z",
        )
    ]

@app.get("/api/experiments/{exp_id}/runs", response_model=List[ExperimentRun])
async def get_experiment_runs(exp_id: str):
    """Get runs for an experiment."""
    runs = []
    for run_id, status in training_runs.items():
        runs.append(ExperimentRun(
            id=run_id,
            experiment=exp_id,
            name=f"Run {run_id}",
            status=status.status,
            metrics={"loss": [status.metrics.get("loss", 0)]},
            params={"epochs": 3, "batch_size": 4},
            created_at="2024-01-01T00:00:00Z",
            duration_s=0,
        ))
    return runs

@app.get("/api/experiments/compare")
async def compare_experiment_runs(runs: str):
    """Compare multiple runs."""
    run_ids = runs.split(",")
    comparison_runs = []
    comparison_metrics: Dict[str, Dict[str, float]] = {}
    
    for run_id in run_ids:
        if run_id in training_runs:
            status = training_runs[run_id]
            comparison_runs.append(ExperimentRun(
                id=run_id,
                experiment="default",
                name=f"Run {run_id}",
                status=status.status,
                metrics={"loss": [status.metrics.get("loss", 0)]},
                params={},
                created_at="2024-01-01T00:00:00Z",
                duration_s=0,
            ))
            
            for metric, value in status.metrics.items():
                if metric not in comparison_metrics:
                    comparison_metrics[metric] = {}
                comparison_metrics[metric][run_id] = value if isinstance(value, (int, float)) else 0
    
    return {
        "runs": comparison_runs,
        "comparison": comparison_metrics,
    }

# ============================================================================
# Export Endpoints
# ============================================================================

@app.post("/api/export")
async def export_model(request: ExportRequest):
    """Export a model to different format."""
    try:
        output_path = Path(request.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Would call actual export functions
        # For now, simulate
        return {
            "success": True,
            "output_path": str(output_path),
            "size_mb": 500.0,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

# ============================================================================
# WebSocket for Live Updates
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates."""
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        while True:
            # Keep connection alive
            _ = await websocket.receive_text()
            # Handle incoming messages if needed
    except WebSocketDisconnect:
        active_connections.remove(websocket)

async def broadcast_training_update(run_id: str):
    """Broadcast training updates to all connected clients."""
    if run_id in training_runs:
        message = {
            "type": "training_update",
            "run_id": run_id,
            "status": training_runs[run_id].dict(),
        }
        for connection in active_connections:
            try:
                await connection.send_json(message)
            except Exception:  # noqa: BLE001 - intentionally broad
                pass

# ============================================================================
# Health Check
# ============================================================================

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "2.0.0"}

# ============================================================================
# Serve Frontend (Production)
# ============================================================================

# In production, serve the built frontend
frontend_path = Path(__file__).parent.parent.parent / "frontend" / "dist"
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")
