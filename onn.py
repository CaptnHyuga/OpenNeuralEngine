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
    """Execute evaluation command."""
    print_banner()
    print("📊 Starting ONN Evaluation\n")
    
    from src.wrappers import load_model
    from src.data_adapters import AUTO_DETECT
    
    # Load model
    print(f"📦 Loading model: {args.model}")
    model, info = load_model(args.model, device="auto")
    print(f"   Loaded: {info.name} ({info.num_params_human} params)")
    
    # Load dataset
    print(f"\n📂 Loading evaluation dataset: {args.dataset}")
    data_result = AUTO_DETECT(args.dataset)
    print(f"   Samples: {data_result.num_samples}")
    
    # Run evaluation
    print("\n🔬 Running evaluation...")

    try:
        from src.Core_Models.evaluator import Evaluator
        eval_runner = Evaluator(model=model, device="auto")
        results = eval_runner.evaluate()  # Use the evaluator
        print(f"✅ Evaluation complete: {results}")
    except Exception as e:
        print(f"⚠️  Evaluator not available: {e}")

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
    
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser."""
    parser = argparse.ArgumentParser(
        prog="onn",
        description="OpenNeuralEngine - Production-Grade Democratic AI Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  onn train --model gpt2 --dataset ./data.jsonl
  onn train --model llama-70b --dataset ./data/ --use-lora
  onn infer --model ./checkpoint/
  onn eval --model ./checkpoint/ --dataset ./test.jsonl
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
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument("--model", "-m", required=True,
                              help="Model path or HuggingFace name")
    infer_parser.add_argument("--quantization", "-q", choices=["int4", "int8"],
                              help="Quantization level")
    infer_parser.add_argument("--max-tokens", type=int, default=256,
                              help="Maximum tokens to generate")
    infer_parser.add_argument("--port", type=int, default=53801,
                              help="Server port")
    infer_parser.set_defaults(func=cmd_infer)
    
    # EVAL command
    eval_parser = subparsers.add_parser("eval", help="Evaluate a model")
    eval_parser.add_argument("--model", "-m", required=True,
                             help="Model path")
    eval_parser.add_argument("--dataset", "-d", required=True,
                             help="Evaluation dataset")
    eval_parser.add_argument("--suite", default="all",
                             help="Evaluation suite to run")
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
    
    return parser


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
