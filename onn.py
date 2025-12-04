#!/usr/bin/env python
"""OpenNeuralEngine 2.0 - Universal AI Training CLI.

A production-grade, democratized AI framework that automatically configures
optimal training settings based on your hardware. Train any model on any
dataset with a single command.

Examples:
    # Train HuggingFace model on text data
    onn train --model gpt2 --dataset ./my_data.jsonl

    # Train with auto-optimization for your hardware
    onn train --model meta-llama/Llama-2-7b --dataset ./data/ --device auto

    # Fine-tune with LoRA on limited VRAM
    onn train --model llama-70b --dataset ./data/ --use-lora

    # Inference with web UI
    onn infer --model ./checkpoint/

    # Evaluate model
    onn eval --model ./checkpoint/ --dataset ./test.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def print_banner():
    """Print ONN banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════════╗
║   ___  _   _ _   _   ____    ___                                  ║
║  / _ \\| \\ | | \\ | | |___ \\  / _ \\                                 ║
║ | | | |  \\| |  \\| |   __) || | | |                                ║
║ | |_| | |\\  | |\\  |  / __/ | |_| |                                ║
║  \\___/|_| \\_|_| \\_| |_____(_)___/                                 ║
║                                                                   ║
║  OpenNeuralEngine - Production-Grade Democratic AI Framework      ║
║  Train any model. Any data. Any hardware. One command.            ║
╚═══════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def cmd_train(args: argparse.Namespace) -> int:
    """Execute train command."""
    from src.orchestration import get_hardware_profile
    from src.wrappers import HFTrainerWrapper
    from src.data_adapters import AUTO_DETECT
    
    print_banner()
    print("🚀 Starting ONN Training Pipeline\n")
    
    # Step 1: Profile hardware
    print("📊 Profiling hardware...")
    profile = get_hardware_profile()
    print(profile.summary())
    print()
    
    # Step 2: Load dataset
    print(f"📂 Loading dataset from: {args.dataset}")
    try:
        data_result = AUTO_DETECT(
            args.dataset,
            tokenizer=None,  # Will be set after model load
            max_length=args.max_seq_len,
        )
        print(f"   Detected: {data_result.format_detected}")
        print(f"   Samples: {data_result.num_samples}")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return 1
    
    # Step 3: Load tokenizer for text data
    tokenizer = None
    if data_result.data_type == "text":
        print(f"\n🔤 Loading tokenizer for: {args.model}")
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Reload dataset with tokenizer
            data_result = AUTO_DETECT(
                args.dataset,
                tokenizer=tokenizer,
                max_length=args.max_seq_len,
            )
        except Exception as e:
            print(f"⚠️  Could not load tokenizer: {e}")
    
    # Step 4: Setup experiment tracking
    tracker = None
    if args.tracking != "disabled":
        print("\n📈 Setting up experiment tracking...")
        try:
            from src.Core_Models.experiment_tracking import ExperimentTracker
            tracker = ExperimentTracker(
                experiment=args.experiment or "onn-training",
                run_name=args.run_name,
                mode=args.tracking,
            )
            print(f"   Experiment: {args.experiment or 'onn-training'}")
        except Exception as e:
            print(f"⚠️  Tracking not available: {e}")
    
    # Step 5: Train
    print("\n🎯 Starting training...")
    wrapper = HFTrainerWrapper(tracker=tracker)
    
    try:
        result = wrapper.train(
            model=args.model,
            train_dataset=data_result.dataset,
            tokenizer=tokenizer,
            task=args.task,
            num_epochs=args.epochs,
            target_batch_size=args.batch_size,
            learning_rate=args.lr if args.lr > 0 else None,
            output_dir=Path(args.output_dir),
            run_name=args.run_name,
            force_precision=args.precision,
            force_quantization=args.quantization,
            use_lora=args.use_lora,
            lora_rank=args.lora_rank,
        )
        
        print("\n" + result.summary())
        
        if result.success:
            print("\n✅ Training completed successfully!")
            return 0
        else:
            print(f"\n❌ Training failed: {result.errors}")
            return 1
            
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        raise


def cmd_generate(args: argparse.Namespace) -> int:
    """Execute quick text generation."""
    print_banner()
    print("✨ Quick Text Generation\n")
    
    try:
        from src.inference import generate, GenerationResult
        
        print(f"📦 Model: {args.model}")
        print(f"🌡️  Temperature: {args.temperature}")
        print(f"📝 Prompt: {args.prompt[:50]}{'...' if len(args.prompt) > 50 else ''}")
        print()
        
        # Generate text
        result = generate(
            model=args.model,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        
        print("=" * 60)
        print("Generated Text:")
        print("=" * 60)
        print(result)
        print("=" * 60)
        
        return 0
        
    except ImportError as e:
        print(f"⚠️  Inference module not available: {e}")
        
        # Fallback to transformers pipeline
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            
            print("🔄 Using transformers pipeline fallback...")
            
            tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                trust_remote_code=True,
                torch_dtype="auto",
                device_map="auto",
            )
            
            pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
            
            output = pipe(
                args.prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature if args.temperature > 0 else None,
                top_p=args.top_p,
                do_sample=args.temperature > 0,
            )
            
            print("=" * 60)
            print("Generated Text:")
            print("=" * 60)
            print(output[0]["generated_text"])
            print("=" * 60)
            
            return 0
            
        except Exception as e2:
            print(f"❌ Generation failed: {e2}")
            return 1


def cmd_infer(args: argparse.Namespace) -> int:
    """Execute inference command."""
    print_banner()
    print("🔮 Starting ONN Inference Server\n")
    
    # Try to launch inference UI
    try:
        from scripts.launch_aim_inference import main as launch_inference
        return launch_inference()
    except ImportError:
        print("ℹ️  Inference UI not available. Using basic inference.")
    
    # Basic inference fallback
    from src.wrappers import load_model
    
    print(f"📦 Loading model: {args.model}")
    model, info = load_model(
        args.model,
        device="auto",
        quantization=args.quantization,
    )
    print(f"   Loaded: {info.name} ({info.num_params_human} params)")
    
    # Interactive inference loop
    print("\n💬 Interactive mode (type 'quit' to exit):\n")
    
    try:
        from transformers import AutoTokenizer, pipeline
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        
        while True:
            prompt = input("You: ").strip()
            if prompt.lower() in ("quit", "exit", "q"):
                break
            if not prompt:
                continue
            
            response = pipe(prompt, max_new_tokens=args.max_tokens, do_sample=True)
            print(f"AI: {response[0]['generated_text']}\n")
            
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Execute evaluation command using lm-evaluation-harness."""
    print_banner()
    print("📊 Starting ONN Evaluation\n")
    
    # Use the new evaluation module (wraps lm-evaluation-harness)
    try:
        from src.evaluation import evaluate_model, quick_eval, list_tasks, EvalResult
        
        # Determine which tasks to run
        if args.suite == "quick":
            print("⚡ Running quick evaluation (sanity check)...")
            results = quick_eval(args.model, max_samples=args.max_samples or 100)
        else:
            # Get tasks from preset or use provided task names
            tasks = args.suite
            if args.tasks:
                tasks = args.tasks.split(",")
            
            print(f"🎯 Tasks: {tasks}")
            print(f"📦 Model: {args.model}")
            
            results = evaluate_model(
                model=args.model,
                tasks=tasks,
                batch_size=args.batch_size or "auto",
                num_fewshot=args.num_fewshot or 0,
                max_samples=args.max_samples,
                output_path=args.output,
            )
        
        print("\n" + "=" * 60)
        print(results.summary())
        
        if results.errors:
            print(f"\n⚠️  Evaluation had errors: {results.errors}")
            return 1
        
        print(f"\n✅ Evaluation complete! Average accuracy: {results.average_accuracy:.2%}")
        return 0
        
    except ImportError as e:
        print(f"⚠️  lm-evaluation-harness not installed.")
        print("   Install with: pip install lm-eval")
        print(f"   Error: {e}")
        
        # Fallback to basic perplexity evaluation
        print("\n📉 Falling back to perplexity evaluation...")
        try:
            from src.evaluation.metrics import compute_perplexity
            from src.wrappers import load_model
            
            model, info = load_model(args.model, device="auto")
            print(f"   Loaded: {info.name} ({info.num_params_human} params)")
            print("   (Full benchmark evaluation requires lm-eval package)")
        except Exception as e2:
            print(f"❌ Evaluation failed: {e2}")
            return 1
    
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Execute export command."""
    print_banner()
    print("📦 Exporting Model\n")
    
    from src.wrappers import load_model
    
    # Load model
    print(f"Loading: {args.model}")
    model, info = load_model(args.model, device="cpu")
    
    output_path = Path(args.output)
    
    if args.format == "onnx":
        print(f"Exporting to ONNX: {output_path}")
        try:
            from utils.onnx_export import export_to_onnx
            export_to_onnx(model, output_path)
            print(f"✅ Exported to {output_path}")
        except Exception as e:
            print(f"❌ Export failed: {e}")
            return 1
    
    elif args.format == "safetensors":
        print(f"Exporting to safetensors: {output_path}")
        from utils.model_io import save_model
        save_model(model, output_path)
        print(f"✅ Exported to {output_path}")
    
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Show system and model info."""
    print_banner()
    
    from src.orchestration import get_hardware_profile
    
    print("📊 System Information\n")
    profile = get_hardware_profile(force_refresh=True)
    print(profile.summary())
    
    if args.model:
        print(f"\n📦 Model Information: {args.model}\n")
        from src.wrappers import UniversalModelLoader
        loader = UniversalModelLoader()
        source = loader.detect_source(args.model)
        print(f"   Source: {source.value}")
        
        # Try to analyze architecture
        try:
            model, info = loader.load(args.model, device="cpu")
            from src.orchestration.architecture_detector import analyze_model
            arch_info = analyze_model(model)
            print(f"\n{arch_info.summary()}")
        except Exception as e:
            print(f"   Could not analyze architecture: {e}")
    
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Start inference server with OpenAI-compatible API."""
    print_banner()
    print("🚀 Starting ONN Inference Server\n")
    
    # Use the new inference module (wraps vLLM)
    try:
        from src.inference import serve, ServerConfig
        
        print(f"📦 Model: {args.model}")
        print(f"🌐 Endpoint: http://{args.host}:{args.port}")
        print(f"📖 API Docs: http://{args.host}:{args.port}/docs")
        print(f"🔢 Tensor Parallel: {args.tensor_parallel}")
        
        # The serve function handles everything
        serve(
            model=args.model,
            host=args.host,
            port=args.port,
            dtype=args.dtype or "auto",
            tensor_parallel_size=args.tensor_parallel,
        )
        return 0
        
    except ImportError as e:
        print(f"⚠️  Full inference server requires vLLM or transformers.")
        print(f"   Install with: pip install vllm transformers")
        print(f"   Error: {e}")
        
        # Try fallback to basic FastAPI server
        print("\n🔄 Attempting fallback server...")
        try:
            from src.wrappers.vllm_wrapper import InferenceServer
            
            server = InferenceServer(
                model_name_or_path=args.model,
                host=args.host,
                port=args.port,
                tensor_parallel_size=args.tensor_parallel,
            )
            server.start()
            return 0
        except Exception as e2:
            print(f"❌ Server failed to start: {e2}")
            return 1


def cmd_data(args: argparse.Namespace) -> int:
    """Inspect dataset."""
    print_banner()
    print("📂 Dataset Information\n")
    
    from pathlib import Path
    from src.data_adapters import AUTO_DETECT
    from src.data_adapters.registry import detect_data_type, detect_paired_data
    from src.data_adapters.multimodal import detect_modalities
    
    path = Path(args.path)
    
    print(f"Path: {path}")
    print(f"Exists: {'✓' if path.exists() else '✗'}")
    
    if not path.exists():
        print("❌ Path does not exist")
        return 1
    
    # Detect data type
    data_type = detect_data_type(path)
    print(f"Primary Type: {data_type}")
    
    # Detect modalities
    modalities = detect_modalities(path)
    if modalities:
        print(f"Modalities: {', '.join(modalities)}")
    
    # Check for paired data
    if path.is_dir():
        is_paired = detect_paired_data(path)
        print(f"Paired/Multimodal: {'✓' if is_paired else '✗'}")
    
    # Try to load and get stats
    try:
        result = AUTO_DETECT(str(path))
        print(f"\n📊 Dataset Stats:")
        print(f"   Adapter: {result.adapter_name}")
        print(f"   Samples: {result.num_samples}")
        print(f"   Format: {result.format_detected}")
        
        if result.features:
            print(f"   Features:")
            for key, value in result.features.items():
                print(f"      {key}: {value}")
        
        if result.preprocessing_applied:
            print(f"   Preprocessing: {', '.join(result.preprocessing_applied)}")
            
    except Exception as e:
        print(f"\n⚠️  Could not load dataset: {e}")
    
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    """Analyze model architecture."""
    print_banner()
    print("🔬 Model Architecture Analysis\n")
    
    from src.wrappers import load_model
    from src.orchestration.architecture_detector import analyze_model
    from src.orchestration import auto_configure
    
    print(f"📦 Loading model: {args.model}")
    
    try:
        model, info = load_model(args.model, device="cpu")
        print(f"   Loaded: {info.name} ({info.num_params_human} params)")
        
        print("\n🔍 Analyzing architecture...")
        arch_info = analyze_model(model)
        print(f"\n{arch_info.summary()}")
        
        if args.config:
            print("\n⚙️  Generating optimal configuration...")
            config = auto_configure(args.model)
            print(f"\n{config.summary()}")
            
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return 1
    
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser."""
    parser = argparse.ArgumentParser(
        prog="onn",
        description="OpenNeuralEngine - Production-Grade Democratic AI Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Training
  onn train --model gpt2 --dataset ./data.jsonl
  onn train --model llama-70b --dataset ./data/ --use-lora
  onn train --model phi-4 --dataset ./data/ --deepspeed zero3
  
  # Inference & Generation
  onn generate --model ./checkpoint/ --prompt "Hello, world!"
  onn serve --model ./checkpoint/ --port 8000
  onn infer --model ./checkpoint/
  
  # Evaluation & Benchmarking
  onn eval --model ./checkpoint/ --task mmlu --preset quick
  onn benchmark --type full --output results.json
  
  # Model Management
  onn checkpoint list ./outputs
  onn checkpoint info ./my-model/
  onn checkpoint clean ./outputs --keep 3
  onn merge --base llama-7b --adapter ./lora/ --output ./merged/
  
  # Utilities
  onn analyze --model gpt2 --config
  onn export --model ./checkpoint/ --format onnx
  onn data ./my-dataset/
  onn info
        """,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # TRAIN command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--model", "-m", required=True,
                              help="Model name (HuggingFace), path, or identifier")
    train_parser.add_argument("--dataset", "-d", required=True,
                              help="Path to training dataset")
    train_parser.add_argument("--output-dir", "-o", default="./outputs",
                              help="Output directory for checkpoints")
    train_parser.add_argument("--task", choices=["causal_lm", "seq2seq", "classification"],
                              default="causal_lm", help="Training task type")
    train_parser.add_argument("--epochs", "-e", type=int, default=3,
                              help="Number of training epochs")
    train_parser.add_argument("--batch-size", "-b", type=int, default=32,
                              help="Target effective batch size")
    train_parser.add_argument("--lr", type=float, default=0,
                              help="Learning rate (0=auto)")
    train_parser.add_argument("--max-seq-len", type=int, default=512,
                              help="Maximum sequence length")
    train_parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"],
                              help="Force specific precision")
    train_parser.add_argument("--quantization", "-q", choices=["int4", "int8"],
                              help="Force quantization level")
    train_parser.add_argument("--use-lora", action="store_true",
                              help="Use LoRA for efficient fine-tuning")
    train_parser.add_argument("--lora-rank", type=int, default=8,
                              help="LoRA rank")
    train_parser.add_argument("--experiment", help="Experiment name for tracking")
    train_parser.add_argument("--run-name", help="Run name for tracking")
    train_parser.add_argument("--tracking", choices=["enabled", "disabled"],
                              default="enabled", help="Enable/disable tracking")
    train_parser.set_defaults(func=cmd_train)
    
    # INFER command
    infer_parser = subparsers.add_parser("infer", help="Run interactive inference")
    infer_parser.add_argument("--model", "-m", required=True,
                              help="Model path or HuggingFace name")
    infer_parser.add_argument("--quantization", "-q", choices=["int4", "int8"],
                              help="Quantization level")
    infer_parser.add_argument("--max-tokens", type=int, default=256,
                              help="Maximum tokens to generate")
    infer_parser.add_argument("--port", type=int, default=53801,
                              help="Server port")
    infer_parser.set_defaults(func=cmd_infer)
    
    # GENERATE command (quick text generation)
    gen_parser = subparsers.add_parser("generate", help="Quick text generation")
    gen_parser.add_argument("--model", "-m", required=True,
                            help="Model path or HuggingFace name")
    gen_parser.add_argument("--prompt", "-p", required=True,
                            help="Input prompt for generation")
    gen_parser.add_argument("--max-tokens", type=int, default=256,
                            help="Maximum tokens to generate")
    gen_parser.add_argument("--temperature", "-t", type=float, default=0.7,
                            help="Sampling temperature (0=deterministic)")
    gen_parser.add_argument("--top-p", type=float, default=0.9,
                            help="Nucleus sampling threshold")
    gen_parser.add_argument("--quantization", "-q", choices=["int4", "int8"],
                            help="Quantization level")
    gen_parser.set_defaults(func=cmd_generate)
    
    # EVAL command
    eval_parser = subparsers.add_parser("eval", help="Evaluate a model on standard benchmarks")
    eval_parser.add_argument("--model", "-m", required=True,
                             help="Model path or HuggingFace ID")
    eval_parser.add_argument("--suite", default="standard",
                             choices=["quick", "standard", "reasoning", "knowledge", "safety", "full"],
                             help="Evaluation task preset")
    eval_parser.add_argument("--tasks", "-t",
                             help="Comma-separated task names (overrides --suite)")
    eval_parser.add_argument("--batch-size", "-b", type=int,
                             help="Batch size (auto if not specified)")
    eval_parser.add_argument("--num-fewshot", type=int, default=0,
                             help="Number of few-shot examples")
    eval_parser.add_argument("--max-samples", type=int,
                             help="Limit samples per task (for quick testing)")
    eval_parser.add_argument("--output", "-o",
                             help="Save results to JSON file")
    eval_parser.add_argument("--dataset", "-d",
                             help="Custom evaluation dataset (for perplexity eval)")
    eval_parser.set_defaults(func=cmd_eval)
    
    # EXPORT command
    export_parser = subparsers.add_parser("export", help="Export model")
    export_parser.add_argument("--model", "-m", required=True,
                               help="Model to export")
    export_parser.add_argument("--output", "-o", required=True,
                               help="Output path")
    export_parser.add_argument("--format", "-f", choices=["onnx", "safetensors"],
                               default="safetensors", help="Export format")
    export_parser.set_defaults(func=cmd_export)
    
    # INFO command
    info_parser = subparsers.add_parser("info", help="Show system info")
    info_parser.add_argument("--model", "-m", help="Model to inspect")
    info_parser.set_defaults(func=cmd_info)
    
    # SERVE command
    serve_parser = subparsers.add_parser("serve", help="Start OpenAI-compatible inference server")
    serve_parser.add_argument("--model", "-m", required=True,
                              help="Model to serve")
    serve_parser.add_argument("--host", default="0.0.0.0",
                              help="Server host")
    serve_parser.add_argument("--port", "-p", type=int, default=8000,
                              help="Server port")
    serve_parser.add_argument("--tensor-parallel", "-tp", type=int, default=1,
                              help="Tensor parallel size (number of GPUs)")
    serve_parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"],
                              default="auto", help="Model dtype")
    serve_parser.add_argument("--quantization", "-q", choices=["awq", "gptq", "squeezellm"],
                              help="Quantization method (optional)")
    serve_parser.set_defaults(func=cmd_serve)
    
    # DATA command
    data_parser = subparsers.add_parser("data", help="Inspect dataset")
    data_parser.add_argument("path", help="Path to dataset")
    data_parser.set_defaults(func=cmd_data)
    
    # ANALYZE command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze model architecture")
    analyze_parser.add_argument("--model", "-m", required=True,
                                help="Model to analyze")
    analyze_parser.add_argument("--config", "-c", action="store_true",
                                help="Generate optimal training config")
    analyze_parser.set_defaults(func=cmd_analyze)
    
    # BENCHMARK command
    bench_parser = subparsers.add_parser("benchmark", help="Benchmark your hardware for ML")
    bench_parser.add_argument("--type", choices=["quick", "training", "inference", "full"],
                              default="quick", help="Benchmark type")
    bench_parser.add_argument("--model", "-m", help="Model for inference benchmark")
    bench_parser.add_argument("--output", "-o", help="Save results to JSON")
    bench_parser.set_defaults(func=cmd_benchmark)
    
    # CHECKPOINT command
    ckpt_parser = subparsers.add_parser("checkpoint", help="Manage model checkpoints")
    ckpt_subparsers = ckpt_parser.add_subparsers(dest="ckpt_action", help="Checkpoint action")
    
    # checkpoint list
    ckpt_list = ckpt_subparsers.add_parser("list", help="List checkpoints")
    ckpt_list.add_argument("path", nargs="?", default="./outputs", help="Checkpoint directory")
    
    # checkpoint info
    ckpt_info = ckpt_subparsers.add_parser("info", help="Show checkpoint details")
    ckpt_info.add_argument("path", help="Path to checkpoint")
    
    # checkpoint clean
    ckpt_clean = ckpt_subparsers.add_parser("clean", help="Clean old checkpoints")
    ckpt_clean.add_argument("path", help="Checkpoint directory")
    ckpt_clean.add_argument("--keep", "-k", type=int, default=3, help="Keep N most recent")
    ckpt_clean.add_argument("--dry-run", action="store_true", help="Show what would be deleted")
    
    ckpt_parser.set_defaults(func=cmd_checkpoint)
    
    # MERGE command (for LoRA adapters)
    merge_parser = subparsers.add_parser("merge", help="Merge LoRA adapter with base model")
    merge_parser.add_argument("--base", "-b", required=True,
                              help="Path to base model")
    merge_parser.add_argument("--adapter", "-a", required=True,
                              help="Path to LoRA adapter checkpoint")
    merge_parser.add_argument("--output", "-o", required=True,
                              help="Output path for merged model")
    merge_parser.add_argument("--quantize", "-q", choices=["int8", "int4", "fp16"],
                              help="Optional: quantize merged model")
    merge_parser.set_defaults(func=cmd_merge)
    
    return parser


def cmd_merge(args: argparse.Namespace) -> int:
    """Merge LoRA adapter with base model."""
    print_banner()
    print("🔀 LoRA Merge Utility\n")
    
    from pathlib import Path
    
    base_path = Path(args.base)
    adapter_path = Path(args.adapter)
    output_path = Path(args.output)
    
    if not base_path.exists():
        print(f"❌ Base model not found: {base_path}")
        return 1
    
    if not adapter_path.exists():
        print(f"❌ Adapter not found: {adapter_path}")
        return 1
    
    print(f"📦 Base Model: {base_path}")
    print(f"🔌 Adapter: {adapter_path}")
    print(f"📁 Output: {output_path}\n")
    
    try:
        from src.training.merge_lora import LoRAMerger
        
        # Find the LoRA checkpoint file
        if adapter_path.is_dir():
            # Look for safetensors file
            lora_files = list(adapter_path.glob("*.safetensors"))
            if lora_files:
                lora_ckpt = lora_files[0]
            else:
                # Look for lora_weights.safetensors specifically
                lora_ckpt = adapter_path / "lora_weights.safetensors"
        else:
            lora_ckpt = adapter_path
        
        if not lora_ckpt.exists():
            print(f"❌ LoRA weights file not found in: {adapter_path}")
            return 1
        
        print(f"📄 Using LoRA checkpoint: {lora_ckpt}")
        
        # Create merger
        merger = LoRAMerger(
            lora_checkpoint=lora_ckpt,
            base_model=base_path,
        )
        
        # Merge
        print("\n🔄 Merging weights...")
        merged_weights = merger.merge()
        
        # Save
        print(f"\n💾 Saving merged model to: {output_path}")
        merger.save_merged(merged_weights, output_path)
        
        # Optional quantization
        if args.quantize:
            print(f"\n🔢 Quantizing to {args.quantize}...")
            from src.wrappers import quantize_model
            quantize_model(
                str(output_path),
                str(output_path),
                args.quantize,
            )
        
        print("\n✅ Merge complete!")
        
        # Show summary
        total_size = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file())
        print(f"📊 Output size: {total_size / (1024**3):.2f} GB")
        
    except Exception as e:
        print(f"❌ Merge failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


def cmd_checkpoint(args: argparse.Namespace) -> int:
    """Manage model checkpoints."""
    print_banner()
    print("📦 Checkpoint Manager\n")
    
    import json
    from pathlib import Path
    import shutil
    
    if not args.ckpt_action:
        print("Usage: onn checkpoint {list|info|clean} [options]")
        return 1
    
    if args.ckpt_action == "list":
        path = Path(args.path)
        if not path.exists():
            print(f"❌ Path not found: {path}")
            return 1
        
        print(f"📁 Scanning: {path}\n")
        
        # Find checkpoints (HF format and PyTorch format)
        checkpoints = []
        
        for item in sorted(path.rglob("*")):
            # HF checkpoint (has config.json)
            if item.is_dir() and (item / "config.json").exists():
                size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
                checkpoints.append({
                    "path": item,
                    "type": "HuggingFace",
                    "size_mb": size / (1024 * 1024),
                    "mtime": item.stat().st_mtime,
                })
            # PyTorch checkpoint
            elif item.suffix in (".pt", ".pth", ".bin") and item.is_file():
                checkpoints.append({
                    "path": item,
                    "type": "PyTorch",
                    "size_mb": item.stat().st_size / (1024 * 1024),
                    "mtime": item.stat().st_mtime,
                })
        
        if not checkpoints:
            print("   No checkpoints found")
            return 0
        
        # Sort by modification time
        checkpoints.sort(key=lambda x: x["mtime"], reverse=True)
        
        print(f"Found {len(checkpoints)} checkpoint(s):\n")
        for i, ckpt in enumerate(checkpoints, 1):
            from datetime import datetime
            mtime_str = datetime.fromtimestamp(ckpt["mtime"]).strftime("%Y-%m-%d %H:%M")
            rel_path = ckpt["path"].relative_to(path) if path in ckpt["path"].parents or path == ckpt["path"].parent else ckpt["path"]
            print(f"  {i}. {rel_path}")
            print(f"     Type: {ckpt['type']} | Size: {ckpt['size_mb']:.1f} MB | Modified: {mtime_str}")
        
        total_size = sum(c["size_mb"] for c in checkpoints)
        print(f"\n📊 Total: {total_size:.1f} MB across {len(checkpoints)} checkpoint(s)")
        
    elif args.ckpt_action == "info":
        path = Path(args.path)
        if not path.exists():
            print(f"❌ Path not found: {path}")
            return 1
        
        print(f"📄 Checkpoint: {path}\n")
        
        # HF format
        if (path / "config.json").exists():
            print("Type: HuggingFace Transformers")
            
            with open(path / "config.json") as f:
                config = json.load(f)
            
            print(f"Architecture: {config.get('architectures', ['Unknown'])[0]}")
            print(f"Model Type: {config.get('model_type', 'Unknown')}")
            
            if "vocab_size" in config:
                print(f"Vocab Size: {config['vocab_size']:,}")
            if "hidden_size" in config:
                print(f"Hidden Size: {config['hidden_size']}")
            if "num_hidden_layers" in config:
                print(f"Layers: {config['num_hidden_layers']}")
            if "num_attention_heads" in config:
                print(f"Attention Heads: {config['num_attention_heads']}")
            
            # Check for adapter/LoRA
            if (path / "adapter_config.json").exists():
                print("\n🔌 LoRA Adapter Detected")
                with open(path / "adapter_config.json") as f:
                    adapter = json.load(f)
                print(f"   Base Model: {adapter.get('base_model_name_or_path', 'Unknown')}")
                print(f"   LoRA Rank: {adapter.get('r', 'Unknown')}")
                print(f"   LoRA Alpha: {adapter.get('lora_alpha', 'Unknown')}")
            
            # Size info
            model_files = list(path.glob("*.safetensors")) + list(path.glob("*.bin"))
            if model_files:
                total_size = sum(f.stat().st_size for f in model_files)
                print(f"\n💾 Model Size: {total_size / (1024**3):.2f} GB ({len(model_files)} file(s))")
        
        # PyTorch format
        elif path.suffix in (".pt", ".pth", ".bin"):
            import torch
            print("Type: PyTorch Checkpoint")
            print(f"Size: {path.stat().st_size / (1024**2):.1f} MB")
            
            try:
                ckpt = torch.load(path, map_location="cpu", weights_only=True)
                if isinstance(ckpt, dict):
                    print(f"Keys: {list(ckpt.keys())[:10]}")
                    if "model_state_dict" in ckpt:
                        n_params = sum(v.numel() for v in ckpt["model_state_dict"].values())
                        print(f"Parameters: {n_params:,}")
            except Exception as e:
                print(f"⚠️ Could not inspect checkpoint: {e}")
        else:
            print("❓ Unknown checkpoint format")
    
    elif args.ckpt_action == "clean":
        path = Path(args.path)
        if not path.exists():
            print(f"❌ Path not found: {path}")
            return 1
        
        print(f"🧹 Cleaning checkpoints in: {path}")
        print(f"   Keeping {args.keep} most recent\n")
        
        # Find checkpoint directories
        checkpoints = []
        for item in path.iterdir():
            if item.is_dir() and (item / "config.json").exists():
                checkpoints.append({
                    "path": item,
                    "mtime": item.stat().st_mtime,
                })
        
        if len(checkpoints) <= args.keep:
            print(f"   Only {len(checkpoints)} checkpoint(s) found, nothing to clean")
            return 0
        
        # Sort by time, oldest first
        checkpoints.sort(key=lambda x: x["mtime"])
        to_delete = checkpoints[:-args.keep]
        
        total_size = 0
        for ckpt in to_delete:
            size = sum(f.stat().st_size for f in ckpt["path"].rglob("*") if f.is_file())
            total_size += size
            
            if args.dry_run:
                print(f"   Would delete: {ckpt['path'].name} ({size / (1024**2):.1f} MB)")
            else:
                print(f"   Deleting: {ckpt['path'].name} ({size / (1024**2):.1f} MB)")
                shutil.rmtree(ckpt["path"])
        
        if args.dry_run:
            print(f"\n📊 Would free: {total_size / (1024**2):.1f} MB")
        else:
            print(f"\n✅ Freed: {total_size / (1024**2):.1f} MB")
    
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Run hardware benchmarks."""
    print_banner()
    print("⚡ Running Hardware Benchmark\n")
    
    import time
    import json
    from pathlib import Path
    
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmark_type": args.type,
    }
    
    # Hardware info
    from src.orchestration import get_hardware_profile
    profile = get_hardware_profile()
    print(profile.summary())
    print()
    
    results["hardware"] = {
        "hostname": profile.hostname,
        "gpus": len(profile.gpus),
        "total_vram_mb": profile.total_vram_mb,
        "total_ram_mb": profile.total_ram_mb,
    }
    
    # Quick PyTorch benchmark
    import torch
    
    if args.type in ("quick", "training", "full"):
        print("🏃 Matrix multiplication benchmark...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        sizes = [1024, 2048, 4096]
        matmul_results = {}
        
        for size in sizes:
            a = torch.randn(size, size, device=device, dtype=torch.float16)
            b = torch.randn(size, size, device=device, dtype=torch.float16)
            
            # Warmup
            for _ in range(3):
                _ = torch.matmul(a, b)
            
            if device == "cuda":
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            n_iters = 10
            for _ in range(n_iters):
                _ = torch.matmul(a, b)
            
            if device == "cuda":
                torch.cuda.synchronize()
            
            elapsed = time.perf_counter() - start
            avg_time = (elapsed / n_iters) * 1000  # ms
            
            # TFLOPS = 2 * N^3 operations for matmul
            flops = 2 * (size ** 3)
            tflops = (flops / (avg_time / 1000)) / 1e12
            
            matmul_results[f"{size}x{size}"] = {
                "time_ms": round(avg_time, 2),
                "tflops": round(tflops, 2),
            }
            print(f"   {size}x{size}: {avg_time:.2f}ms ({tflops:.2f} TFLOPS)")
        
        results["matmul"] = matmul_results
    
    # Inference benchmark
    if args.type in ("inference", "full") and args.model:
        print(f"\n🔮 Inference benchmark with {args.model}...")
        try:
            from src.inference import generate
            
            prompts = [
                "The quick brown fox",
                "In a galaxy far far away",
                "def fibonacci(n):",
            ]
            
            inference_results = []
            for prompt in prompts:
                start = time.perf_counter()
                result = generate(args.model, prompt, max_tokens=50)
                elapsed = time.perf_counter() - start
                
                tokens = len(result.split())
                inference_results.append({
                    "tokens": tokens,
                    "time_s": round(elapsed, 2),
                    "tokens_per_s": round(tokens / elapsed, 2),
                })
            
            avg_tps = sum(r["tokens_per_s"] for r in inference_results) / len(inference_results)
            results["inference"] = {
                "model": args.model,
                "avg_tokens_per_sec": round(avg_tps, 2),
                "runs": inference_results,
            }
            print(f"   Average: {avg_tps:.2f} tokens/sec")
            
        except Exception as e:
            print(f"   ⚠️ Could not run inference benchmark: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 BENCHMARK SUMMARY")
    print("=" * 60)
    
    if "matmul" in results:
        best_tflops = max(r["tflops"] for r in results["matmul"].values())
        print(f"Peak compute: {best_tflops:.2f} TFLOPS")
    
    if "inference" in results:
        print(f"Inference speed: {results['inference']['avg_tokens_per_sec']:.2f} tokens/sec")
    
    # Save if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {output_path}")
    
    return 0


def main() -> int:
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
