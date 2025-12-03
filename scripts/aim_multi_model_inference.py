"""
Aim Multi-Model Inference Extension - HuggingFace integration with conversation tracking.

Features:
- Multiple HuggingFace model support
- Conversation history tracking per session
- Easy model switching in the UI
- All conversations logged to Aim for comparison
- Seamless integration with existing pipeline
"""
try:
    from aim import Run
    AIM_AVAILABLE = True
except Exception:
    AIM_AVAILABLE = False
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
    model_id: str = "gpt2"
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
    source: str
    loaded: bool
    conversation_count: int


class MultiModelInferenceExtension:
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
            model_loader_fn: Function(model_id) -> (model, tokenizer)
            inference_fn: Function(model, tokenizer, input, history) -> output
            aim_repo_path: Path to Aim repository
            port: Port for the inference API
            default_hf_models: List of HuggingFace model IDs available
        """
        self.model_loader_fn = model_loader_fn
        self.inference_fn = inference_fn
        self.aim_repo_path = aim_repo_path
        self.port = port
        
        # Model registry: {model_id: {"model": model, "tokenizer": tok, "info": {...}}}
        self.models: Dict[str, Dict[str, Any]] = {}
        
        # Conversation store: {conversation_id: [turns]}
        self.conversations: Dict[str, List[Dict[str, Any]]] = {}
        
        # Default HuggingFace models
        self.default_hf_models = default_hf_models or [
            "gpt2",
            "distilgpt2",
            "microsoft/DialoGPT-small",
            "microsoft/DialoGPT-medium",
            "facebook/opt-125m",
            "EleutherAI/gpt-neo-125m",
        ]
        
        # Create FastAPI app
        self.app = FastAPI(
            title="Aim Multi-Model Inference",
            description="Multi-model inference with conversation tracking",
            version="2.0.0",
        )
        
        # CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self._setup_routes()
    
    def _load_model(self, model_id: str):
        """Load model and tokenizer (lazy loading)."""
        if model_id in self.models and self.models[model_id]["model"] is not None:
            return self.models[model_id]["model"], self.models[model_id]["tokenizer"]
        
        print(f"📦 Loading model: {model_id}...")
        model, tokenizer = self.model_loader_fn(model_id)
        
        source = "huggingface" if "/" in model_id or model_id in self.default_hf_models else "local"
        
        self.models[model_id] = {
            "model": model,
            "tokenizer": tokenizer,
            "info": {
                "id": model_id,
                "name": model_id.split("/")[-1] if "/" in model_id else model_id,
                "source": source,
                "loaded": True,
                "conversation_count": 0,
            }
        }
        
        print(f"✅ Model {model_id} loaded!")
        return model, tokenizer
    
    def _get_conversation(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get or create conversation."""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        return self.conversations[conversation_id]
    
    def _add_turn(self, conversation_id: str, model_id: str, input_text: str, output: str):
        """Add turn to conversation."""
        conversation = self._get_conversation(conversation_id)
        conversation.append({
            "turn": len(conversation) + 1,
            "input": input_text,
            "output": output,
            "model": model_id,
            "timestamp": datetime.now().isoformat(),
        })
        
        if model_id in self.models:
            self.models[model_id]["info"]["conversation_count"] += 1
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/health")
        async def health():
            """Health check."""
            return {
                "status": "healthy",
                "models_loaded": sum(1 for m in self.models.values() if m["model"]),
                "conversations": len(self.conversations),
            }
        
        @self.app.get("/models", response_model=List[ModelInfo])
        async def list_models():
            """List available models."""
            result = []
            
            # Loaded models
            for mid, data in self.models.items():
                result.append(ModelInfo(**data["info"]))
            
            # Available but not loaded
            for hf_model in self.default_hf_models:
                if hf_model not in self.models:
                    result.append(ModelInfo(
                        id=hf_model,
                        name=hf_model.split("/")[-1],
                        source="huggingface",
                        loaded=False,
                        conversation_count=0,
                    ))
            
            return result
        
        @self.app.post("/models/add")
        async def add_model(model_id: str):
            """Add new HuggingFace model."""
            if model_id not in self.default_hf_models:
                self.default_hf_models.append(model_id)
            return {"success": True, "model_id": model_id}
        
        @self.app.get("/conversations")
        async def list_conversations():
            """List all conversations."""
            return [
                {
                    "id": cid,
                    "turns": len(turns),
                    "models": list(set(t["model"] for t in turns)),
                    "last_updated": turns[-1]["timestamp"] if turns else None,
                }
                for cid, turns in self.conversations.items()
            ]
        
        @self.app.get("/conversations/{conversation_id}")
        async def get_conversation(conversation_id: str):
            """Get conversation history."""
            if conversation_id not in self.conversations:
                raise HTTPException(404, "Conversation not found")
            return {
                "conversation_id": conversation_id,
                "turns": self.conversations[conversation_id],
            }
        
        @self.app.delete("/conversations/{conversation_id}")
        async def delete_conversation(conversation_id: str):
            """Delete conversation."""
            if conversation_id in self.conversations:
                del self.conversations[conversation_id]
            return {"success": True}
        
        @self.app.post("/inference", response_model=InferenceResponse)
        async def run_inference(request: InferenceRequest):
            """Run inference with conversation tracking."""
            try:
                conv_id = request.conversation_id or str(uuid.uuid4())
                
                # Load model
                model, tokenizer = self._load_model(request.model_id)
                
                # Get history
                history = self._get_conversation(conv_id)
                
                # Run inference
                output = self.inference_fn(model, tokenizer, request.input, history)
                
                # Add to conversation
                self._add_turn(conv_id, request.model_id, request.input, output)
                
                turn_num = len(history)
                
                # Log to Aim (only if available)
                run_hash = None
                aim_url = None
                
                if request.log_to_aim and AIM_AVAILABLE:
                    run = Run(repo=self.aim_repo_path, experiment=request.experiment)
                    
                    run["model_id"] = request.model_id
                    run["conversation_id"] = conv_id
                    run["turn"] = turn_num
                    run["notes"] = request.notes
                    run["interface"] = "multi_model"
                    
                    run.track(turn_num, name="turn_number", context={"conv": conv_id})
                    run.track(len(history), name="conv_length", context={"conv": conv_id})
                    run.track(request.input, name="input", context={"model": request.model_id})
                    run.track(output, name="output", context={"model": request.model_id})
                    
                    run_hash = run.hash
                    aim_url = f"http://localhost:53800/runs/{run_hash}"
                    run.close()
                
                return InferenceResponse(
                    success=True,
                    predictions=output,
                    model_id=request.model_id,
                    conversation_id=conv_id,
                    turn_number=turn_num,
                    run_hash=run_hash,
                    aim_url=aim_url,
                    timestamp=datetime.now().isoformat(),
                )
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise HTTPException(500, str(e))
        
        @self.app.get("/")
        async def root():
            """API info."""
            return {
                "title": "Multi-Model Inference API",
                "endpoints": {
                    "/models": "List models",
                    "/conversations": "List conversations",
                    "/inference": "Run inference",
                    "/ui": "Web interface",
                },
                "aim_dashboard": "http://localhost:53800",
            }
        
        @self.app.get("/ui")
        async def ui():
            """Web interface."""
            return HTMLResponse(self._generate_ui())
    
    def _generate_ui(self) -> str:
        """Generate modern UI HTML."""
        return f'''<!DOCTYPE html>
<html><head>
<title>Multi-Model Inference</title>
<style>
* {{margin:0;padding:0;box-sizing:border-box}}
body {{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);min-height:100vh;padding:20px}}
.container {{max-width:1200px;margin:0 auto;background:#fff;border-radius:12px;box-shadow:0 10px 40px rgba(0,0,0,.2);overflow:hidden}}
.header {{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:#fff;padding:30px;text-align:center}}
.header h1 {{font-size:32px;margin-bottom:10px}}
.content {{padding:40px;display:grid;grid-template-columns:300px 1fr;gap:30px}}
.sidebar {{border-right:2px solid #e0e0e0;padding-right:30px}}
.sidebar h3 {{margin-bottom:15px;color:#667eea}}
.model-list {{list-style:none}}
.model-item {{padding:12px;margin:5px 0;border-radius:6px;cursor:pointer;border:2px solid #e0e0e0;transition:all .3s}}
.model-item:hover {{border-color:#667eea;background:#f5f7ff}}
.model-item.selected {{background:#667eea;color:#fff;border-color:#667eea}}
.model-item .badge {{font-size:10px;padding:2px 6px;border-radius:3px;margin-left:5px}}
.loaded {{background:#4caf50;color:#fff}}
.hf {{background:#ff9800;color:#fff}}
.main {{}}
label {{display:block;margin-bottom:8px;font-weight:600;color:#333}}
select, textarea, input {{width:100%;padding:12px;border:2px solid #e0e0e0;border-radius:8px;margin-bottom:15px;font-size:14px}}
textarea {{min-height:150px;font-family:monospace;resize:vertical}}
button {{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:#fff;padding:15px 40px;border:none;border-radius:8px;cursor:pointer;font-size:16px;font-weight:600;width:100%;transition:transform .2s}}
button:hover {{transform:translateY(-2px);box-shadow:0 5px 20px rgba(102,126,234,.4)}}
.history {{background:#f5f5f5;padding:20px;border-radius:8px;margin-top:20px;max-height:400px;overflow-y:auto}}
.turn {{background:#fff;padding:15px;margin:10px 0;border-radius:6px;border-left:4px solid #667eea}}
.turn .meta {{font-size:12px;color:#666;margin-bottom:8px}}
.turn .input {{font-weight:600;margin-bottom:5px}}
.turn .output {{color:#333;font-family:monospace;font-size:13px}}
.spinner {{border:3px solid rgba(255,255,255,.3);border-top:3px solid #fff;border-radius:50%;width:20px;height:20px;animation:spin .8s linear infinite;display:inline-block;margin-right:10px}}
@keyframes spin {{0%{{transform:rotate(0deg)}}100%{{transform:rotate(360deg)}}}}
.new-conv {{background:#4caf50;margin-bottom:15px}}
</style>
</head><body>
<div class="container">
<div class="header">
<h1>🤖 Multi-Model Inference</h1>
<p>HuggingFace models with conversation tracking</p>
</div>
<div class="content">
<div class="sidebar">
<h3>📦 Models</h3>
<button class="new-conv" onclick="newConversation()">➕ New Conversation</button>
<div id="modelList"></div>
</div>
<div class="main">
<label>Conversation ID:</label>
<input type="text" id="convId" readonly>
<label>Input:</label>
<textarea id="input" placeholder="Enter your message..."></textarea>
<label>Notes (optional):</label>
<input type="text" id="notes" placeholder="Add notes...">
<button id="sendBtn" onclick="sendMessage()">🚀 Send</button>
<div class="history" id="history"></div>
</div>
</div>
</div>
<script>
let selectedModel = 'gpt2';
let conversationId = generateUUID();
let history = [];

function generateUUID() {{
return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {{
const r = Math.random() * 16 | 0;
return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
}});
}}

function newConversation() {{
conversationId = generateUUID();
history = [];
document.getElementById('convId').value = conversationId;
document.getElementById('history').innerHTML = '<p style="text-align:center;color:#999">New conversation started</p>';
}}

async function loadModels() {{
try {{
const resp = await fetch('http://localhost:{self.port}/models');
const models = await resp.json();
const list = document.getElementById('modelList');
list.innerHTML = '<ul class="model-list">' + models.map(m => 
`<li class="model-item ${{m.id === selectedModel ? 'selected' : ''}}" onclick="selectModel('${{m.id}}')">
${{m.name}}
<span class="badge ${{m.loaded ? 'loaded' : 'hf'}}">${{m.loaded ? '✓' : 'HF'}}</span>
<br><small style="opacity:0.7">${{m.conversation_count}} convs</small>
</li>`
).join('') + '</ul>';
}} catch(e) {{
console.error('Failed to load models:', e);
}}
}}

function selectModel(modelId) {{
selectedModel = modelId;
loadModels();
}}

async function sendMessage() {{
const input = document.getElementById('input').value.trim();
if (!input) return;

const btn = document.getElementById('sendBtn');
btn.disabled = true;
btn.innerHTML = '<span class="spinner"></span>Generating...';

try {{
const resp = await fetch('http://localhost:{self.port}/inference', {{
method: 'POST',
headers: {{'Content-Type': 'application/json'}},
body: JSON.stringify({{
input: input,
model_id: selectedModel,
conversation_id: conversationId,
notes: document.getElementById('notes').value,
log_to_aim: true,
experiment: 'multi_model_chat'
}})
}});

const result = await resp.json();

if (result.success) {{
history.push({{
input: input,
output: result.predictions,
model: result.model_id,
turn: result.turn_number,
timestamp: new Date().toLocaleTimeString()
}});

document.getElementById('input').value = '';
renderHistory();
loadModels();

if (result.aim_url) {{
console.log('Aim URL:', result.aim_url);
}}
}}
}} catch(e) {{
alert('Error: ' + e.message);
}} finally {{
btn.disabled = false;
btn.innerHTML = '🚀 Send';
}}
}}

function renderHistory() {{
const histDiv = document.getElementById('history');
histDiv.innerHTML = history.map(t => 
`<div class="turn">
<div class="meta">Turn ${{t.turn}} • ${{t.model}} • ${{t.timestamp}}</div>
<div class="input">You: ${{t.input}}</div>
<div class="output">AI: ${{t.output}}</div>
</div>`
).reverse().join('');
}}

document.getElementById('convId').value = conversationId;
loadModels();
setInterval(loadModels, 5000);

document.getElementById('input').addEventListener('keydown', e => {{
if (e.ctrlKey && e.key === 'Enter') sendMessage();
}});
</script>
</body></html>'''
    
    def run(self):
        """Start server."""
        print(f"\n🚀 Multi-Model Inference Extension")
        print(f"📊 Aim: {self.aim_repo_path}")
        print(f"🎯 UI: http://localhost:{self.port}/ui")
        print(f"📖 Docs: http://localhost:{self.port}/docs")
        print(f"🔗 Aim Dashboard: http://localhost:53800\n")
        
        uvicorn.run(self.app, host="0.0.0.0", port=self.port, log_level="info")
