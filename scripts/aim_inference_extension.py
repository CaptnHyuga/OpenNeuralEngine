"""
Aim Inference Extension - Multi-model inference with conversation tracking.
Run this alongside Aim server to enable inference in the UI.

Features:
- Multiple HuggingFace model support
- Conversation history tracking per model
- Easy model switching
- All conversations logged to Aim for comparison
"""
try:
    from aim import Run
    AIM_AVAILABLE = True
except ImportError:
    AIM_AVAILABLE = False
    Run = None
    print("Warning: Aim package not available. Tracking disabled. Use Docker for Aim on Windows.")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, Any, Dict, List
import uvicorn
from datetime import datetime
import json
import uuid


class InferenceRequest(BaseModel):
    """Inference request schema."""
    input: str
    model_id: str = "default"
    conversation_id: Optional[str] = None
    log_to_aim: bool = True
    notes: str = ""
    experiment: str = "inference"


class InferenceResponse(BaseModel):
    """Inference response schema."""
    success: bool
    predictions: Any
    model_id: str
    conversation_id: str
    turn_number: int
    run_hash: Optional[str] = None
    aim_url: Optional[str] = None
    timestamp: str
    error: Optional[str] = None


class ModelInfo(BaseModel):
    """Model information schema."""
    id: str
    name: str
    source: str  # "huggingface" or "local"
    loaded: bool
    conversation_count: int


class ConversationTurn(BaseModel):
    """Single conversation turn."""
    turn_number: int
    input: str
    output: Any
    timestamp: str
    model_id: str


class AimInferenceExtension:
    """
    Multi-model inference extension with conversation tracking.
    Supports HuggingFace models and local models seamlessly.
    """
    
    def __init__(
        self,
        model_loader_fn,
        inference_fn,
        aim_repo_path: str = ".aim_project",
        port: int = 53801,
        default_hf_models: Optional[List[str]] = None,
    ):
        """
        Initialize the extension.
        
        Args:
            model_loader_fn: Function(model_id) -> model
            inference_fn: Function(model, input, conversation_history) -> predictions
            aim_repo_path: Path to Aim repository
            port: Port for the inference API
            default_hf_models: List of HuggingFace model IDs to preload
        """
        self.model_loader_fn = model_loader_fn
        self.inference_fn = inference_fn
        self.aim_repo_path = aim_repo_path
        self.port = port
        
        # Model registry: {model_id: {"model": model_obj, "info": ModelInfo}}
        self.models: Dict[str, Dict[str, Any]] = {}
        
        # Conversation store: {conversation_id: [ConversationTurn]}
        self.conversations: Dict[str, List[Dict[str, Any]]] = {}
        
        # Default HuggingFace models
        self.default_hf_models = default_hf_models or [
            "gpt2",
            "distilgpt2",
            "microsoft/DialoGPT-small",
        ]
        
        # Create FastAPI app
        self.app = FastAPI(
            title="Aim Multi-Model Inference Extension",
            description="Inference API with multi-model support and conversation tracking",
            version="2.0.0",
        )
        
        # Add CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:53800", "http://localhost:53801"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Register routes
        self._setup_routes()
    
    def _load_model(self, model_id: str):
        """Load a model by ID (lazy loading)."""
        if model_id in self.models and self.models[model_id]["model"] is not None:
            return self.models[model_id]["model"]
        
        print(f"Loading model: {model_id}...")
        model = self.model_loader_fn(model_id)
        
        # Determine source
        source = "huggingface" if model_id in self.default_hf_models or "/" in model_id else "local"
        
        # Register model
        self.models[model_id] = {
            "model": model,
            "info": {
                "id": model_id,
                "name": model_id.split("/")[-1] if "/" in model_id else model_id,
                "source": source,
                "loaded": True,
                "conversation_count": 0,
            }
        }
        
        print(f"Model {model_id} loaded successfully!")
        return model
    
    def _get_conversation(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get or create a conversation."""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        return self.conversations[conversation_id]
    
    def _add_to_conversation(
        self,
        conversation_id: str,
        model_id: str,
        input_text: str,
        output: Any
    ):
        """Add a turn to the conversation."""
        conversation = self._get_conversation(conversation_id)
        turn = {
            "turn_number": len(conversation) + 1,
            "input": input_text,
            "output": output,
            "timestamp": datetime.now().isoformat(),
            "model_id": model_id,
        }
        conversation.append(turn)
        
        # Update model conversation count
        if model_id in self.models:
            self.models[model_id]["info"]["conversation_count"] += 1
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/health")
        async def health():
            """Health check endpoint."""
            loaded_models = sum(1 for m in self.models.values() if m["model"] is not None)
            return {
                "status": "healthy",
                "aim_repo": self.aim_repo_path,
                "models_loaded": loaded_models,
                "total_conversations": len(self.conversations),
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.get("/ready")
        async def ready():
            """Readiness check for deployment orchestration."""
            try:
                loaded_models = sum(1 for m in self.models.values() if m["model"] is not None)
                return {
                    "status": "ready",
                    "models_loaded": loaded_models,
                    "models_available": len(self.default_hf_models),
                    "conversations_active": len(self.conversations),
                    "aim_tracking": "enabled" if self.aim_repo_path else "disabled",
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                raise HTTPException(status_code=503, detail=f"Not ready: {str(e)}")
        
        @self.app.get("/models", response_model=List[ModelInfo])
        async def list_models():
            """List all available models."""
            models_list = []
            
            # Add registered models
            for model_id, data in self.models.items():
                models_list.append(ModelInfo(**data["info"]))
            
            # Add available HF models not yet loaded
            for hf_model in self.default_hf_models:
                if hf_model not in self.models:
                    models_list.append(ModelInfo(
                        id=hf_model,
                        name=hf_model.split("/")[-1],
                        source="huggingface",
                        loaded=False,
                        conversation_count=0,
                    ))
            
            return models_list
        
        @self.app.post("/models/add")
        async def add_model(model_id: str):
            """Add a new HuggingFace model."""
            try:
                if model_id not in self.models:
                    if model_id not in self.default_hf_models:
                        self.default_hf_models.append(model_id)
                return {"success": True, "model_id": model_id}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.get("/conversations")
        async def list_conversations():
            """List all conversations."""
            return [
                {
                    "conversation_id": conv_id,
                    "turns": len(turns),
                    "models_used": list(set(t["model_id"] for t in turns)),
                    "last_updated": turns[-1]["timestamp"] if turns else None,
                }
                for conv_id, turns in self.conversations.items()
            ]
        
        @self.app.get("/conversations/{conversation_id}")
        async def get_conversation(conversation_id: str):
            """Get a specific conversation history."""
            if conversation_id not in self.conversations:
                raise HTTPException(status_code=404, detail="Conversation not found")
            return {
                "conversation_id": conversation_id,
                "turns": self.conversations[conversation_id],
            }
        
        @self.app.delete("/conversations/{conversation_id}")
        async def delete_conversation(conversation_id: str):
            """Delete a conversation."""
            if conversation_id in self.conversations:
                del self.conversations[conversation_id]
            return {"success": True}
        
        @self.app.post("/inference", response_model=InferenceResponse)
        async def run_inference(request: InferenceRequest):
            """Run inference with conversation tracking."""
            try:
                # Generate conversation ID if not provided
                conversation_id = request.conversation_id or str(uuid.uuid4())
                
                # Load model
                model = self._load_model(request.model_id)
                
                # Get conversation history
                conversation_history = self._get_conversation(conversation_id)
                
                # Run inference with history
                predictions = self.inference_fn(
                    model,
                    request.input,
                    conversation_history
                )
                
                # Add to conversation
                self._add_to_conversation(
                    conversation_id,
                    request.model_id,
                    request.input,
                    predictions
                )
                
                turn_number = len(conversation_history)
                
                # Create Aim run if logging enabled
                run_hash = None
                aim_url = None
                
                if request.log_to_aim and AIM_AVAILABLE:
                    session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    
                    run = Run(
                        repo=self.aim_repo_path,
                        experiment=request.experiment,
                    )
                    
                    # Log parameters
                    run["model_id"] = request.model_id
                    run["conversation_id"] = conversation_id
                    run["turn_number"] = turn_number
                    run["timestamp"] = session_id
                    run["input_length"] = len(request.input)
                    run["interface"] = "aim_extension_multi_model"
                    if request.notes:
                        run["notes"] = request.notes
                    
                    # Track conversation context
                    run.track(
                        len(conversation_history),
                        name="conversation_length",
                        context={"conversation_id": conversation_id}
                    )
                    
                    # Track predictions
                    run.track(
                        predictions,
                        name="prediction",
                        context={"model": request.model_id, "turn": turn_number}
                    )
                    
                    # Store input
                    run.track(
                        request.input,
                        name="input_text",
                        context={"model": request.model_id, "turn": turn_number}
                    )
                    
                    # Log full conversation history
                    run.track(
                        json.dumps(conversation_history[-5:]),  # Last 5 turns
                        name="recent_history",
                        context={"conversation_id": conversation_id}
                    )
                    
                    run_hash = run.hash
                    aim_url = f"http://localhost:53800/runs/{run_hash}"
                    
                    run.close()
                
                return InferenceResponse(
                    success=True,
                    predictions=predictions,
                    model_id=request.model_id,
                    conversation_id=conversation_id,
                    turn_number=turn_number,
                    run_hash=run_hash,
                    aim_url=aim_url,
                    timestamp=datetime.now().isoformat(),
                )
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise HTTPException(
                    status_code=500,
                    detail=str(e)
                )
                
                # Create Aim run if logging enabled
                run_hash = None
                aim_url = None
                
                if request.log_to_aim:
                    session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    
                    run = Run(
                        repo=self.aim_repo_path,
                        experiment=request.experiment,
                    )
                    
                    # Log parameters
                    run["inference_type"] = "api"
                    run["timestamp"] = session_id
                    run["input_length"] = len(request.input)
                    run["interface"] = "aim_extension"
                    if request.notes:
                        run["notes"] = request.notes
                    
                    # Track predictions
                    run.track(
                        predictions,
                        name="prediction",
                        context={"session": session_id}
                    )
                    
                    # Store input as text
                    run.track(
                        request.input,
                        name="input_text",
                        context={"session": session_id}
                    )
                    
                    run_hash = run.hash
                    aim_url = f"http://localhost:53800/runs/{run_hash}"
                    
                    run.close()
                
                return InferenceResponse(
                    success=True,
                    predictions=predictions,
                    run_hash=run_hash,
                    aim_url=aim_url,
                    timestamp=datetime.now().isoformat(),
                )
                
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=str(e)
                )
        
        @self.app.get("/")
        async def root():
            """Root endpoint with instructions."""
            return {
                "message": "Aim Inference Extension API",
                "endpoints": {
                    "/health": "Health check",
                    "/inference": "Run inference (POST)",
                    "/ui": "Inference UI",
                },
                "aim_dashboard": "http://localhost:53800",
                "docs": f"http://localhost:{self.port}/docs",
            }
        
        @self.app.get("/ui")
        async def ui_page():
            """Return HTML UI for inference."""
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Aim Inference Panel</title>
                <style>
                    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                    body {{
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        min-height: 100vh;
                        padding: 20px;
                    }}
                    .container {{
                        max-width: 900px;
                        margin: 0 auto;
                        background: white;
                        border-radius: 12px;
                        box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                        overflow: hidden;
                    }}
                    .header {{
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        padding: 30px;
                        text-align: center;
                    }}
                    .header h1 {{
                        font-size: 32px;
                        margin-bottom: 10px;
                    }}
                    .header p {{
                        opacity: 0.9;
                        font-size: 16px;
                    }}
                    .content {{
                        padding: 40px;
                    }}
                    label {{
                        display: block;
                        margin-bottom: 8px;
                        font-weight: 600;
                        color: #333;
                    }}
                    textarea {{
                        width: 100%;
                        min-height: 180px;
                        padding: 15px;
                        border: 2px solid #e0e0e0;
                        border-radius: 8px;
                        font-family: 'Monaco', 'Courier New', monospace;
                        font-size: 14px;
                        margin-bottom: 20px;
                        transition: border-color 0.3s;
                        resize: vertical;
                    }}
                    textarea:focus {{
                        outline: none;
                        border-color: #667eea;
                    }}
                    input[type="text"] {{
                        width: 100%;
                        padding: 12px 15px;
                        border: 2px solid #e0e0e0;
                        border-radius: 8px;
                        margin-bottom: 20px;
                        font-size: 14px;
                        transition: border-color 0.3s;
                    }}
                    input[type="text"]:focus {{
                        outline: none;
                        border-color: #667eea;
                    }}
                    .checkbox-container {{
                        margin-bottom: 25px;
                        display: flex;
                        align-items: center;
                    }}
                    input[type="checkbox"] {{
                        width: 20px;
                        height: 20px;
                        margin-right: 10px;
                        cursor: pointer;
                    }}
                    .checkbox-label {{
                        font-weight: normal;
                        cursor: pointer;
                        user-select: none;
                    }}
                    button {{
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        padding: 15px 40px;
                        border: none;
                        border-radius: 8px;
                        cursor: pointer;
                        font-size: 16px;
                        font-weight: 600;
                        width: 100%;
                        transition: transform 0.2s, box-shadow 0.2s;
                    }}
                    button:hover {{
                        transform: translateY(-2px);
                        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
                    }}
                    button:active {{
                        transform: translateY(0);
                    }}
                    button:disabled {{
                        opacity: 0.6;
                        cursor: not-allowed;
                        transform: none;
                    }}
                    .result {{
                        background: #f5f5f5;
                        padding: 20px;
                        border-radius: 8px;
                        margin-top: 25px;
                        display: none;
                        border-left: 4px solid #667eea;
                    }}
                    .result.error {{
                        background: #ffebee;
                        border-left-color: #f44336;
                    }}
                    .result.success {{
                        background: #e8f5e9;
                        border-left-color: #4caf50;
                    }}
                    .result-content {{
                        white-space: pre-wrap;
                        font-family: 'Monaco', 'Courier New', monospace;
                        font-size: 13px;
                        line-height: 1.6;
                    }}
                    .aim-link {{
                        margin-top: 15px;
                        padding: 12px;
                        background: white;
                        border-radius: 6px;
                        display: inline-block;
                    }}
                    .aim-link a {{
                        color: #667eea;
                        text-decoration: none;
                        font-weight: 600;
                    }}
                    .aim-link a:hover {{
                        text-decoration: underline;
                    }}
                    .spinner {{
                        border: 3px solid rgba(255,255,255,0.3);
                        border-top: 3px solid white;
                        border-radius: 50%;
                        width: 20px;
                        height: 20px;
                        animation: spin 0.8s linear infinite;
                        display: inline-block;
                        margin-right: 10px;
                        vertical-align: middle;
                    }}
                    @keyframes spin {{
                        0% {{ transform: rotate(0deg); }}
                        100% {{ transform: rotate(360deg); }}
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>🎯 SNN Inference Panel</h1>
                        <p>Integrated with Aim Experiment Tracking</p>
                    </div>
                    
                    <div class="content">
                        <label for="input">Input Text:</label>
                        <textarea id="input" placeholder="Enter your input text here..."></textarea>
                        
                        <label for="notes">Session Notes (optional):</label>
                        <input type="text" id="notes" placeholder="Add notes about this inference session...">
                        
                        <label for="experiment">Experiment Name:</label>
                        <input type="text" id="experiment" value="inference" placeholder="inference">
                        
                        <div class="checkbox-container">
                            <input type="checkbox" id="logToAim" checked>
                            <label for="logToAim" class="checkbox-label">Log this inference to Aim</label>
                        </div>
                        
                        <button id="runBtn" onclick="runInference()">
                            🔮 Run Inference
                        </button>
                        
                        <div id="result" class="result">
                            <div id="resultContent" class="result-content"></div>
                        </div>
                    </div>
                </div>
                
                <script>
                    async function runInference() {{
                        const runBtn = document.getElementById('runBtn');
                        const resultDiv = document.getElementById('result');
                        const resultContent = document.getElementById('resultContent');
                        
                        resultDiv.style.display = 'none';
                        
                        const input = document.getElementById('input').value;
                        if (!input.trim()) {{
                            alert('Please enter some input text');
                            return;
                        }}
                        
                        // Disable button and show loading
                        runBtn.disabled = true;
                        runBtn.innerHTML = '<span class="spinner"></span>Running inference...';
                        
                        const data = {{
                            input: input,
                            log_to_aim: document.getElementById('logToAim').checked,
                            notes: document.getElementById('notes').value,
                            experiment: document.getElementById('experiment').value || 'inference',
                        }};
                        
                        try {{
                            const response = await fetch('http://localhost:{self.port}/inference', {{
                                method: 'POST',
                                headers: {{ 'Content-Type': 'application/json' }},
                                body: JSON.stringify(data)
                            }});
                            
                            const result = await response.json();
                            
                            if (result.success) {{
                                let message = '✅ Inference Complete\\n\\n';
                                message += 'Predictions:\\n';
                                message += JSON.stringify(result.predictions, null, 2);
                                message += '\\n\\nTimestamp: ' + result.timestamp;
                                
                                resultDiv.className = 'result success';
                                resultContent.innerHTML = message;
                                
                                if (result.run_hash) {{
                                    const linkDiv = document.createElement('div');
                                    linkDiv.className = 'aim-link';
                                    linkDiv.innerHTML = `
                                        🔗 <a href="${{result.aim_url}}" target="_blank">
                                            View in Aim Dashboard (Run: ${{result.run_hash.substring(0, 8)}})
                                        </a>
                                    `;
                                    resultContent.appendChild(linkDiv);
                                }}
                            }} else {{
                                resultDiv.className = 'result error';
                                resultContent.textContent = '❌ Error: ' + (result.error || 'Unknown error');
                            }}
                            
                            resultDiv.style.display = 'block';
                            
                        }} catch (error) {{
                            resultDiv.className = 'result error';
                            resultContent.textContent = '❌ Request failed: ' + error.message + '\\n\\nMake sure the inference server is running on port {self.port}';
                            resultDiv.style.display = 'block';
                        }} finally {{
                            // Re-enable button
                            runBtn.disabled = false;
                            runBtn.innerHTML = '🔮 Run Inference';
                        }}
                    }}
                    
                    // Allow Enter key in textarea to submit (Ctrl+Enter)
                    document.getElementById('input').addEventListener('keydown', function(e) {{
                        if (e.ctrlKey && e.key === 'Enter') {{
                            runInference();
                        }}
                    }});
                </script>
            </body>
            </html>
            """
            return HTMLResponse(content=html)
    
    def run(self):
        """Start the inference extension server."""
        print(f"\n🚀 Starting Aim Inference Extension on port {self.port}")
        print(f"📊 Aim repository: {self.aim_repo_path}")
        print(f"🎯 Inference UI: http://localhost:{self.port}/ui")
        print(f"📖 API docs: http://localhost:{self.port}/docs")
        print(f"🔗 Aim dashboard: http://localhost:53800\n")
        
        uvicorn.run(
            self.app,
            host="0.0.0.0",
            port=self.port,
            log_level="info"
        )
