<div align="center">

# 🎯 Quick Start Guide

### Get OpenNeuralEngine running in 60 seconds

[Windows Setup](#-windows-one-click-setup) •
[Manual Setup](#-manual-setup) •
[Train Model](#-train-your-first-model) •
[Troubleshooting](#-troubleshooting)

</div>

---

## 🪟 Windows One-Click Setup

### Step 1: Install Prerequisites

| Prerequisite | Required | Link |
|:-------------|:--------:|:-----|
| **Python 3.11+** | ✅ Required | [Download](https://www.python.org/downloads/) |
| **Docker Desktop** | ⭕ Optional | [Download](https://www.docker.com/products/docker-desktop) |

> **Important**: Check "Add Python to PATH" during Python installation

### Step 2: Launch

**Simply double-click `Start-SNN.bat`** in the project folder! 🚀

<div align="center">

```
Starting ONN 2.0...
✅ Python detected
✅ Virtual environment created
✅ Dependencies installed
✅ Docker services started
✅ API server running
✅ Frontend launched
```

</div>

**Browser opens automatically to:**
- 🌐 **Frontend UI**: http://localhost:53801
- 📊 **Aim Dashboard**: http://localhost:53800
- ⚡ **API Docs**: http://localhost:8000/docs

**First run**: ~30-60 seconds | **Subsequent runs**: ~5-10 seconds

---

## 💻 Manual Setup

### Installation

```bash
# Clone repository
git clone https://github.com/CaptnHyuga/OpenNeuralEngine.git
cd OpenNeuralEngine

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/macOS)
source .venv/bin/activate

# Install dependencies
pip install -e ".[full]"
```

### Start Services

```bash
# Start Aim tracking (Docker required)
cd .aim_project
docker compose up -d
cd ..

# Start API server
python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload

# Start frontend (in new terminal)
cd frontend
npm install
npm run dev
```

---

## 🧠 Train Your First Model

### Using CLI

```bash
# Simple text generation
python onn.py train --model gpt2 --data ./my_data.jsonl

# With all options
python onn.py train \
  --model gpt2 \
  --data ./conversations.jsonl \
  --epochs 3 \
  --output ./my-model \
  --device auto
```

### Using Python Script

```python
from src.orchestration import HardwareProfiler, ConfigOrchestrator
from src.wrappers import UniversalModelLoader, HFTrainerWrapper
from src.data_adapters import AUTO_DETECT

# Auto-configure based on hardware
profiler = HardwareProfiler()
orchestrator = ConfigOrchestrator(profiler)

config = orchestrator.orchestrate(
    model_size_params=125_000_000,
    dataset_size=10000,
    task="text-generation"
)

# Load data and model
data = AUTO_DETECT("./my_data/")
loader = UniversalModelLoader()
model, tokenizer = loader.load("gpt2")

# Train
trainer = HFTrainerWrapper(model, tokenizer)
trainer.train(data.dataset, config=config, output_dir="./output")
```

---

## 🌐 Using the Web Interface

### Frontend UI (Port 53801)

<table>
<tr>
<td width="30%"><b>1️⃣ Select Model</b></td>
<td>Choose from pre-loaded models or add any HuggingFace model</td>
</tr>
<tr>
<td><b>2️⃣ Enter Prompt</b></td>
<td>Type your question or instruction</td>
</tr>
<tr>
<td><b>3️⃣ Run Inference</b></td>
<td>Click "Run" or press <code>Ctrl+Enter</code></td>
</tr>
<tr>
<td><b>4️⃣ Compare Models</b></td>
<td>Switch models mid-conversation to compare responses</td>
</tr>
<tr>
<td><b>5️⃣ View Metrics</b></td>
<td>Click Aim link for detailed session analytics</td>
</tr>
</table>

**Pre-loaded Models:**
- GPT-2 (Small, fast)
- DialoGPT (Conversational)
- DistilGPT-2 (Lightweight)

### Aim Dashboard (Port 53800)

- 📈 Track training runs
- ⚖️ Compare model performance
- 💬 View inference sessions
- 📊 Analyze metrics & logs

---

## 🎓 Next Steps

### Explore Examples

```bash
# Fine-tune on custom data
python onn.py train --model gpt2 --data ./my_conversations.jsonl --epochs 5

# Train vision model
python onn.py train --model microsoft/resnet-50 --data ./images/ --task classification

# Evaluate models
python scripts/eval_model.py --suite all --output results.json
```

### Read Documentation

| Resource | Description |
|:---------|:------------|
| 📘 [README](README.md) | Complete project overview |
| 📗 [Capabilities](docs/CAPABILITIES.md) | Full feature list |
| 📕 [API Reference](docs/INDEX.md) | Documentation index |
| 📙 [FAQ](docs/FAQ.md) | Common questions |
| 📓 [Deployment](docs/DEPLOYMENT.md) | Production guide |

---

## 🔧 Troubleshooting

### Common Issues

<details>
<summary><b>"Python not found"</b></summary>

**Solution**: Install Python 3.11+ from [python.org](https://www.python.org/downloads/)
- ✅ Check "Add Python to PATH" during installation
- Restart terminal after installation
</details>

<details>
<summary><b>"Docker not found"</b></summary>

**Solution**: App works without Docker! Aim tracking is optional.
- Install [Docker Desktop](https://www.docker.com/products/docker-desktop) to enable experiment tracking
- Or continue without it - all core features still work
</details>

<details>
<summary><b>"Port already in use"</b></summary>

**Solution**:
```powershell
# Check what's using the port
netstat -ano | findstr :8000

# Kill the process (replace PID)
taskkill /PID <PID> /F

# Or change the port in Start-SNN.bat
```
</details>

<details>
<summary><b>"Module not found"</b></summary>

**Solution**: Reinstall dependencies
```bash
pip install -e ".[full]"
```
</details>

### Still Having Issues?

- 📖 Check [FAQ](docs/FAQ.md)
- 🐛 [Report Bug](https://github.com/CaptnHyuga/OpenNeuralEngine/issues)
- 💬 [Ask Question](https://github.com/CaptnHyuga/OpenNeuralEngine/discussions)

---

<div align="center">

## 🎉 You're All Set!

**Everything can be managed through the web interface**

No terminal needed • Professional UI • One-click setup

**Enjoy OpenNeuralEngine!** 🚀

[⭐ Star on GitHub](https://github.com/CaptnHyuga/OpenNeuralEngine) •
[📖 Read Docs](docs/) •
[💡 Get Help](https://github.com/CaptnHyuga/OpenNeuralEngine/issues)

</div>

## Windows Users (Easiest!)

### Step 1: Install Prerequisites

1. **Python 3.11+** → [Download Here](https://www.python.org/downloads/)
   - ✅ Check "Add Python to PATH" during installation
   
2. **Docker Desktop** (Optional but recommended) → [Download Here](https://www.docker.com/products/docker-desktop)
   - Enables Aim experiment tracking
   - Not required - app works without it!

### Step 2: Run SNN

**Just double-click `Start-SNN.bat`** in the project folder!

That's it! The browser will open automatically to:
- 🌐 **Inference UI:** http://localhost:53801/ui
- 📊 **Aim Dashboard:** http://localhost:53800 (if Docker is installed)

---

## What Just Happened?

The `Start-SNN.bat` launcher automatically:
- ✓ Checked your Python version
- ✓ Created a virtual environment
- ✓ Installed all dependencies
- ✓ Started Aim tracking (if Docker available)
- ✓ Launched the inference server
- ✓ Opened your browser

**First time:** Takes ~30-60 seconds
**Next time:** Takes ~5-10 seconds!

---

## Using the Web Interface

### Inference UI (Port 53801)

1. **Select a model** from the sidebar
   - Pre-loaded: GPT-2, DialoGPT, DistilGPT-2
   - Add any HuggingFace model
   - Use your trained SNN models

2. **Type your prompt** in the text area

3. **Click "Run Inference"** or press `Ctrl+Enter`

4. **Switch models** mid-conversation to compare responses!

5. **View in Aim** - click the Aim link to see detailed metrics

### Aim Dashboard (Port 53800)

- Track all training runs
- Compare model performance
- View inference sessions
- Analyze conversation patterns

---

## What's Next?

### Train Your Own Model

```powershell
# In PowerShell (or just use the Aim UI later)
python train.py --model-preset nano --device cpu --epochs 5
```

Your model will appear automatically in the inference UI!

### Evaluate Models

```powershell
python scripts/eval_model.py --suite all
```

### Learn More

- 📖 **Full README:** [README.md](README.md)
- 🚀 **Deployment Guide:** [DEPLOYMENT.md](DEPLOYMENT.md)
- 📚 **Documentation:** [docs/INDEX.md](docs/INDEX.md)
- ❓ **FAQ:** [docs/FAQ.md](docs/FAQ.md)

---

## Troubleshooting

### "Python not found"
Install Python 3.11+ and restart your terminal

### "Docker not found"
App still works! Aim tracking is optional. Install Docker Desktop to enable it.

### "Port already in use"
Another SNN instance or app is using the port. Stop it and try again.

### Need Help?
Open an issue on GitHub or check the [FAQ](docs/FAQ.md)

---

## One More Thing...

**You never need to use the terminal again!**

Everything can be managed through the web interface:
- Model selection
- Inference
- Conversation history
- Experiment tracking

Just like Spotify, Unity Hub, or any professional desktop app.

**Enjoy!** 🎉
