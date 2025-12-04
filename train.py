#!/usr/bin/env python3
"""
ONN Training CLI
===============

Unified command-line interface for training with different pipeline versions.
Supports automatic configuration based on hardware profile.

Usage:
    python train.py --dataset data/Dataset/math.jsonl --epochs 3
    python train.py --dataset data/Dataset/math.jsonl --pipeline v2 --epochs 5
    python train.py --dataset data/3d_vision/0000.parquet --pipeline v3 --amp
    python train.py --compare --dataset data/Dataset/math.jsonl --max-samples 50
    python train.py --auto --dataset data/Dataset/math.jsonl  # Auto-configure!
"""

import argparse
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def run_pipeline_v1(args) -> Dict[str, Any]:
    """Run original pipeline (random hidden states - for baseline comparison)."""
    from src.training.pipeline import main as v1_main, TrainingConfig, DatasetLoader, UnifiedTrainer
    
    config = TrainingConfig(
        model_path=args.model,
        output_dir=f"{args.output}/v1",
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        learning_rate=args.lr,
    )
    
    dataset = DatasetLoader.load(args.dataset, config.max_samples)
    trainer = UnifiedTrainer(config)
    summary = trainer.train(dataset)
    summary["pipeline"] = "v1"
    return summary


def run_pipeline_v2(args) -> Dict[str, Any]:
    """Run v2 pipeline (real tokenization + LM loss)."""
    from src.training.pipeline_v2 import TrainingConfig, DatasetLoader, RealLanguageModelTrainer
    
    config = TrainingConfig(
        model_path=args.model,
        output_dir=f"{args.output}/v2",
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        max_seq_len=args.max_seq_len,
        learning_rate=args.lr,
        val_split=args.val_split,
    )
    
    all_data = DatasetLoader.load(args.dataset, config.max_samples)
    
    if config.val_split > 0 and len(all_data) > 10:
        split_idx = int(len(all_data) * (1 - config.val_split))
        train_data = all_data[:split_idx]
        val_data = all_data[split_idx:]
    else:
        train_data, val_data = all_data, None
    
    trainer = RealLanguageModelTrainer(config)
    summary = trainer.train(train_data, val_data)
    summary["pipeline"] = "v2"
    return summary


def run_pipeline_v3(args) -> Dict[str, Any]:
    """Run v3 pipeline (optimized with AMP + gradient accumulation)."""
    from src.training.pipeline_v3 import TrainingConfigV3, DatasetLoader, OptimizedTrainer
    
    # Get resume value if set
    resume_from = getattr(args, 'resume', None)
    
    config = TrainingConfigV3(
        model_path=args.model,
        output_dir=f"{args.output}/v3",
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_samples=args.max_samples,
        max_seq_len=args.max_seq_len,
        learning_rate=args.lr,
        use_amp=args.amp,
        val_split=args.val_split,
        resume_from=resume_from,
    )
    
    all_data = DatasetLoader.load(args.dataset, config.max_samples)
    
    if config.val_split > 0 and len(all_data) > 10:
        split_idx = int(len(all_data) * (1 - config.val_split))
        train_data = all_data[:split_idx]
        val_data = all_data[split_idx:]
    else:
        train_data, val_data = all_data, None
    
    trainer = OptimizedTrainer(config)
    summary = trainer.train(train_data, val_data)
    summary["pipeline"] = "v3"
    return summary


def compare_pipelines(args):
    """Compare all pipeline versions on the same dataset."""
    print("="*70)
    print("PIPELINE COMPARISON MODE")
    print("="*70)
    
    results = {}
    
    # Run v1 (baseline)
    print("\n" + "="*70)
    print("Running Pipeline v1 (baseline - random hidden states)")
    print("="*70)
    try:
        results["v1"] = run_pipeline_v1(args)
    except Exception as e:
        print(f"v1 failed: {e}")
        results["v1"] = {"error": str(e)}
    
    # Run v2 (real tokenization)
    print("\n" + "="*70)
    print("Running Pipeline v2 (real tokenization + LM loss)")
    print("="*70)
    try:
        results["v2"] = run_pipeline_v2(args)
    except Exception as e:
        print(f"v2 failed: {e}")
        results["v2"] = {"error": str(e)}
    
    # Run v3 (optimized)
    print("\n" + "="*70)
    print("Running Pipeline v3 (optimized - AMP + gradient accumulation)")
    print("="*70)
    try:
        results["v3"] = run_pipeline_v3(args)
    except Exception as e:
        print(f"v3 failed: {e}")
        results["v3"] = {"error": str(e)}
    
    # Summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    for name, summary in results.items():
        print(f"\n{name}:")
        if "error" in summary:
            print(f"  ❌ Error: {summary['error']}")
        else:
            print(f"  ⏱️  Time: {summary.get('total_time_seconds', 0):.1f}s")
            print(f"  📉 Final Loss: {summary.get('final_train_loss', summary.get('final_loss', 0)):.4f}")
            if "final_train_ppl" in summary:
                print(f"  📊 Final PPL: {summary['final_train_ppl']:.0f}")
    
    # Save comparison
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "comparison_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📁 Results saved to: {output_dir / 'comparison_results.json'}")


def auto_configure(args) -> Dict[str, Any]:
    """Auto-configure training based on hardware profile."""
    print("="*70)
    print("AUTO-CONFIGURATION MODE")
    print("="*70)
    
    from src.orchestration.config_orchestrator import ConfigOrchestrator
    from src.orchestration.hardware_profiler import get_profiler
    
    # Get hardware profile
    profiler = get_profiler()
    profile = profiler.profile()
    
    print("\n🔧 Hardware Profile:")
    print(f"   GPU: {profile.gpus[0].name if profile.gpus else 'None'}")
    print(f"   VRAM: {profile.available_vram_mb:.0f} MB" if profile.gpus else "   VRAM: N/A")
    if profile.memory:
        print(f"   RAM: {profile.memory.total_ram_mb:.0f} MB")
    if profile.cpu:
        print(f"   CPU: {profile.cpu.physical_cores} cores")
    
    # Run orchestrator
    orchestrator = ConfigOrchestrator(profiler)
    
    # Count dataset samples
    from src.training.pipeline_v2 import DatasetLoader
    data = DatasetLoader.load(args.dataset, 1000)
    dataset_size = len(data)
    
    # Get optimal config
    opt_config = orchestrator.orchestrate(
        model_name_or_path=args.model,
        dataset_size=dataset_size,
        max_seq_len=args.max_seq_len,
        num_epochs=args.epochs,
        target_batch_size=args.batch_size * args.grad_accum if hasattr(args, 'grad_accum') else 16,
    )
    
    print("\n📋 Recommended Configuration:")
    print(f"   Precision: {opt_config.precision.value}")
    print(f"   Batch size: {opt_config.per_device_batch_size}")
    print(f"   Gradient accumulation: {opt_config.gradient_accumulation_steps}")
    print(f"   Effective batch: {opt_config.effective_batch_size}")
    print(f"   Learning rate: {opt_config.learning_rate:.2e}")
    print(f"   Gradient checkpointing: {opt_config.gradient_checkpointing}")
    
    print("\n💡 Reasoning:")
    for reason in opt_config.config_reasoning[:5]:
        print(f"   • {reason}")
    
    # Apply config to args
    args.batch_size = opt_config.per_device_batch_size
    args.grad_accum = opt_config.gradient_accumulation_steps
    args.lr = opt_config.learning_rate
    args.amp = opt_config.precision.value in ["fp16", "bf16"]
    
    # Use v3 for optimized pipeline
    print(f"\n🚀 Running with v3 pipeline (optimized)...")
    return run_pipeline_v3(args)


def main():
    parser = argparse.ArgumentParser(
        description="ONN Unified Training CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default pipeline (v2 - recommended)
  python train.py --dataset data/Dataset/math.jsonl --epochs 3
  
  # Train with optimized pipeline (v3)
  python train.py --dataset data/Dataset/math.jsonl --pipeline v3 --amp
  
  # Auto-configure based on hardware
  python train.py --auto --dataset data/Dataset/math.jsonl --epochs 3
  
  # Compare all pipelines
  python train.py --compare --dataset data/Dataset/math.jsonl --max-samples 50 --epochs 2
  
  # Train on multimodal data
  python train.py --dataset data/3d_vision/0000.parquet --pipeline v2
"""
    )
    
    # Core arguments
    parser.add_argument("--dataset", required=True, help="Path to dataset file (.jsonl, .parquet, .json)")
    parser.add_argument("--model", default="models/phi-4", help="Path to model directory")
    parser.add_argument("--output", default="output/training", help="Output directory")
    
    # Pipeline selection
    parser.add_argument("--pipeline", choices=["v1", "v2", "v3"], default="v2",
                        help="Pipeline version: v1=baseline, v2=real tokenization, v3=optimized")
    parser.add_argument("--compare", action="store_true", help="Compare all pipelines")
    parser.add_argument("--auto", action="store_true", help="Auto-configure based on hardware")
    
    # Training params
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--max-samples", type=int, default=100, help="Max samples to use")
    parser.add_argument("--max-seq-len", type=int, default=128, help="Max sequence length")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split")
    
    # V3-specific
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps (v3)")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision (v3)")
    parser.add_argument("--resume", type=str, default=None, 
                        help="Resume from checkpoint: 'latest', checkpoint name, or path (v3 only)")
    parser.add_argument("--find-lr", action="store_true", 
                        help="Run learning rate finder before training")
    
    args = parser.parse_args()
    
    print("="*70)
    print("ONN UNIFIED TRAINING CLI")
    print("="*70)
    print(f"  Dataset: {args.dataset}")
    print(f"  Model: {args.model}")
    mode = "AUTO" if args.auto else ("COMPARE ALL" if args.compare else args.pipeline)
    print(f"  Mode: {mode}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    
    # Run LR finder if requested
    if hasattr(args, 'find_lr') and args.find_lr:
        print("\n" + "="*70)
        print("LEARNING RATE FINDER")
        print("="*70)
        
        from src.training.lr_finder import find_lr
        result = find_lr(
            dataset_path=args.dataset,
            model_path=args.model,
            output_dir=f"{args.output}/lr_finder",
            max_samples=args.max_samples,
        )
        
        suggested_lr = result["optimal_lr"]
        print(f"\n💡 Using suggested learning rate: {suggested_lr:.2e}")
        args.lr = suggested_lr
    
    if args.auto:
        summary = auto_configure(args)
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print(f"  Time: {summary.get('total_time_seconds', 0):.1f}s")
        print(f"  Final Loss: {summary.get('final_train_loss', summary.get('final_loss', 0)):.4f}")
    elif args.compare:
        compare_pipelines(args)
        return
    else:
        # Run selected pipeline
        if args.pipeline == "v1":
            summary = run_pipeline_v1(args)
        elif args.pipeline == "v2":
            summary = run_pipeline_v2(args)
        elif args.pipeline == "v3":
            summary = run_pipeline_v3(args)
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print(f"  Time: {summary.get('total_time_seconds', 0):.1f}s")
        print(f"  Final Loss: {summary.get('final_train_loss', summary.get('final_loss', 0)):.4f}")


if __name__ == "__main__":
    main()
