"""HuggingFace Trainer Wrapper - Intelligent training orchestration.

Wraps HuggingFace Trainer with automatic configuration from the
orchestration layer. Users get one-command training that works
optimally on any hardware.

Features:
- Auto-configured from hardware profiler
- Seamless DeepSpeed integration
- Automatic quantization
- Integrated Aim tracking
- OOM recovery
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import Dataset

# HuggingFace imports
try:
    from transformers import (
        Trainer,
        TrainingArguments,
        TrainerCallback,
        TrainerState,
        TrainerControl,
        PreTrainedModel,
        PreTrainedTokenizer,
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        DataCollatorWithPadding,
        BitsAndBytesConfig,
    )
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    Trainer = TrainingArguments = None

try:
    from peft import (
        prepare_model_for_kbit_training,
        LoraConfig,
        get_peft_model,
        TaskType,
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

from ..orchestration import (
    ConfigOrchestrator,
    TrainingConfig,
    HardwareProfiler,
    ResourceMonitor,
    get_profiler,
)


@dataclass
class TrainingResult:
    """Result of a training run."""
    
    success: bool = True
    epochs_completed: int = 0
    final_loss: float = float("inf")
    best_loss: float = float("inf")
    total_steps: int = 0
    training_time_seconds: float = 0.0
    checkpoint_path: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    config_used: Optional[TrainingConfig] = None
    
    def summary(self) -> str:
        status = "✅ Success" if self.success else "❌ Failed"
        return (
            f"Training {status}\n"
            f"  Epochs: {self.epochs_completed}\n"
            f"  Final Loss: {self.final_loss:.4f}\n"
            f"  Best Loss: {self.best_loss:.4f}\n"
            f"  Steps: {self.total_steps}\n"
            f"  Time: {self.training_time_seconds:.1f}s\n"
            f"  Checkpoint: {self.checkpoint_path or 'None'}"
        )


class AimCallback(TrainerCallback):
    """Callback to log to Aim tracking."""
    
    def __init__(self, tracker):
        self.tracker = tracker
    
    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs and self.tracker:
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    self.tracker.log_metric(key, value, step=state.global_step)
    
    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if self.tracker:
            self.tracker.log_metric("final_step", state.global_step)


class HFTrainerWrapper:
    """Intelligent HuggingFace Trainer wrapper with auto-configuration."""
    
    def __init__(
        self,
        profiler: Optional[HardwareProfiler] = None,
        orchestrator: Optional[ConfigOrchestrator] = None,
        tracker=None,  # ExperimentTracker instance
        output_dir: Optional[Path] = None,
    ):
        """Initialize wrapper.
        
        Args:
            profiler: Hardware profiler (uses global if None).
            orchestrator: Config orchestrator (creates one if None).
            tracker: Aim experiment tracker for logging.
            output_dir: Default output directory for checkpoints.
        """
        if not HF_AVAILABLE:
            raise ImportError(
                "HuggingFace Transformers not installed. "
                "Run: pip install transformers accelerate"
            )
        
        self.profiler = profiler or get_profiler()
        self.orchestrator = orchestrator or ConfigOrchestrator(profiler=self.profiler)
        self.tracker = tracker
        self.output_dir = output_dir or Path("./outputs")
        self.monitor = ResourceMonitor()
    
    def train(
        self,
        model: Union[str, nn.Module, PreTrainedModel],
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        task: str = "causal_lm",
        num_epochs: int = 3,
        target_batch_size: int = 32,
        learning_rate: Optional[float] = None,
        output_dir: Optional[Path] = None,
        run_name: Optional[str] = None,
        # Override options
        force_precision: Optional[str] = None,
        force_quantization: Optional[str] = None,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        # Extra HF args
        extra_training_args: Optional[Dict[str, Any]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
    ) -> TrainingResult:
        """Train a model with automatic configuration.
        
        Args:
            model: Model name (HuggingFace), path, or nn.Module instance.
            train_dataset: Training dataset.
            eval_dataset: Optional evaluation dataset.
            tokenizer: Tokenizer (auto-loaded if model is string).
            task: Task type ("causal_lm", "seq2seq", "classification").
            num_epochs: Number of training epochs.
            target_batch_size: Desired effective batch size.
            learning_rate: Learning rate (None = auto).
            output_dir: Output directory for this run.
            run_name: Name for this training run.
            force_precision: Force specific precision ("fp32", "fp16", "bf16").
            force_quantization: Force quantization ("int4", "int8").
            use_lora: Whether to use LoRA for efficient fine-tuning.
            lora_rank: LoRA rank (if use_lora=True).
            lora_alpha: LoRA alpha (if use_lora=True).
            extra_training_args: Extra TrainingArguments kwargs.
            callbacks: Additional trainer callbacks.
        
        Returns:
            TrainingResult with training outcomes.
        """
        result = TrainingResult()
        start_time = time.time()
        
        output_dir = output_dir or self.output_dir / (run_name or "run")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Step 1: Get model name for orchestration
            model_name = model if isinstance(model, str) else "custom_model"
            
            # Step 2: Orchestrate configuration
            print("🔧 Auto-configuring training...")
            config = self.orchestrator.orchestrate(
                model_name_or_path=model_name,
                task=task,
                max_seq_len=getattr(train_dataset, "max_length", 512),
                target_batch_size=target_batch_size,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                force_precision=force_precision,
                force_quantization=force_quantization,
            )
            print(config.summary())
            result.config_used = config
            
            # Step 3: Load model and tokenizer
            print("📦 Loading model...")
            model_obj, tokenizer = self._load_model_and_tokenizer(
                model=model,
                tokenizer=tokenizer,
                task=task,
                config=config,
            )
            
            # Step 4: Apply LoRA if requested
            if use_lora and PEFT_AVAILABLE:
                print("🔗 Applying LoRA...")
                model_obj = self._apply_lora(model_obj, task, lora_rank, lora_alpha)
            
            # Step 5: Build training arguments
            training_args = self._build_training_args(
                config=config,
                output_dir=output_dir,
                run_name=run_name,
                has_eval=eval_dataset is not None,
                extra_args=extra_training_args,
            )
            
            # Step 6: Get data collator
            data_collator = self._get_data_collator(task, tokenizer)
            
            # Step 7: Build callbacks
            all_callbacks = callbacks or []
            if self.tracker:
                all_callbacks.append(AimCallback(self.tracker))
            
            # Step 8: Create trainer
            trainer = Trainer(
                model=model_obj,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
                callbacks=all_callbacks,
            )
            
            # Step 9: Train with monitoring
            print("🚀 Starting training...")
            self.monitor.start()
            
            try:
                train_result = trainer.train()
                
                result.success = True
                result.epochs_completed = num_epochs
                result.total_steps = train_result.global_step
                result.final_loss = train_result.training_loss
                result.metrics = train_result.metrics
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self.monitor.handle_oom()
                    result.errors.append(f"OOM Error: {e}")
                    result.success = False
                else:
                    raise
            finally:
                self.monitor.stop()
            
            # Step 10: Save final checkpoint
            if result.success:
                checkpoint_path = output_dir / "final_checkpoint"
                trainer.save_model(str(checkpoint_path))
                result.checkpoint_path = str(checkpoint_path)
                print(f"💾 Saved checkpoint to {checkpoint_path}")
            
        except Exception as e:
            result.success = False
            result.errors.append(str(e))
            raise
        
        finally:
            result.training_time_seconds = time.time() - start_time
        
        return result
    
    def _load_model_and_tokenizer(
        self,
        model: Union[str, nn.Module, PreTrainedModel],
        tokenizer: Optional[PreTrainedTokenizer],
        task: str,
        config: TrainingConfig,
    ) -> tuple:
        """Load model and tokenizer with appropriate configuration."""
        
        # Handle model loading
        if isinstance(model, str):
            # Load from HuggingFace or local path
            model_kwargs = {}
            
            # Apply quantization config
            if config.quantization and config.quantization.enabled:
                quant_config = config.to_quantization_config()
                if quant_config:
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        **quant_config
                    )
            
            # Set device map
            if config.device != "cpu":
                model_kwargs["device_map"] = "auto"
            
            # Set dtype
            if config.precision.value in ("fp16", "bf16"):
                model_kwargs["torch_dtype"] = (
                    torch.float16 if config.precision.value == "fp16" else torch.bfloat16
                )
            
            # Load appropriate model class
            if task == "causal_lm":
                model_obj = AutoModelForCausalLM.from_pretrained(model, **model_kwargs)
            elif task == "seq2seq":
                model_obj = AutoModelForSeq2SeqLM.from_pretrained(model, **model_kwargs)
            elif task == "classification":
                model_obj = AutoModelForSequenceClassification.from_pretrained(model, **model_kwargs)
            else:
                model_obj = AutoModelForCausalLM.from_pretrained(model, **model_kwargs)
            
            # Load tokenizer
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(model)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
        
        else:
            model_obj = model
        
        # Prepare for quantized training if needed
        if config.quantization and config.quantization.enabled and PEFT_AVAILABLE:
            model_obj = prepare_model_for_kbit_training(model_obj)
        
        return model_obj, tokenizer
    
    def _apply_lora(
        self,
        model: PreTrainedModel,
        task: str,
        rank: int,
        alpha: int,
    ) -> PreTrainedModel:
        """Apply LoRA to model for efficient fine-tuning."""
        if not PEFT_AVAILABLE:
            raise ImportError("PEFT not installed. Run: pip install peft")
        
        task_type_map = {
            "causal_lm": TaskType.CAUSAL_LM,
            "seq2seq": TaskType.SEQ_2_SEQ_LM,
            "classification": TaskType.SEQ_CLS,
        }
        
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            task_type=task_type_map.get(task, TaskType.CAUSAL_LM),
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        return model
    
    def _build_training_args(
        self,
        config: TrainingConfig,
        output_dir: Path,
        run_name: Optional[str],
        has_eval: bool,
        extra_args: Optional[Dict[str, Any]],
    ) -> TrainingArguments:
        """Build HuggingFace TrainingArguments from config."""
        
        args_dict = config.to_hf_training_args()
        args_dict["output_dir"] = str(output_dir)
        
        if run_name:
            args_dict["run_name"] = run_name
        
        if has_eval:
            args_dict["evaluation_strategy"] = "steps"
            args_dict["eval_steps"] = args_dict.get("eval_steps", 500)
            args_dict["load_best_model_at_end"] = True
        else:
            args_dict["evaluation_strategy"] = "no"
        
        if extra_args:
            args_dict.update(extra_args)
        
        return TrainingArguments(**args_dict)
    
    def _get_data_collator(self, task: str, tokenizer):
        """Get appropriate data collator for task."""
        if task in ("causal_lm", "seq2seq"):
            return DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
            )
        else:
            return DataCollatorWithPadding(tokenizer=tokenizer)


def train(
    model: Union[str, nn.Module],
    dataset,
    **kwargs
) -> TrainingResult:
    """Convenience function for training.
    
    Usage:
        result = train("gpt2", my_dataset, num_epochs=3)
    """
    wrapper = HFTrainerWrapper()
    return wrapper.train(model=model, train_dataset=dataset, **kwargs)
