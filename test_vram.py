"""Quick test of VRAM estimation."""
from src.wrappers.qlora_trainer import QLoRAConfig, estimate_vram_for_model

# phi-4 has ~14B parameters
num_params = 14_000_000_000

print("=" * 60)
print("VRAM Estimation for phi-4 (14B params) with QLoRA")
print("=" * 60)

# Standard estimation
print("\n--- Without CPU Model Offload ---")
for vram in [4, 6, 8, 12]:
    result = estimate_vram_for_model(num_params, vram, seq_length=256)
    fits = "✓ YES" if result["fits_in_vram"] else "✗ NO"
    print(f"{vram}GB: {result['estimated_vram_gb']:.2f}GB needed - Fits: {fits}")

# With aggressive 4GB settings
print("\n--- Ultra-Low VRAM Mode (4GB target) ---")
config = QLoRAConfig(
    lora_r=4,  # Minimal rank
    lora_alpha=8,
    max_seq_length=128,  # Short sequences
    gradient_checkpointing=True,
    cpu_offload_optimizer=True,
    cpu_offload_params=True,  # Key: offload model layers
    bnb_4bit_use_double_quant=True,
)
memory = config.estimate_memory_usage(num_params)
print(f"\nWith CPU offload + r=4 + seq=128:")
for key, value in memory.items():
    print(f"  {key}: {value:.2f} GB")

# Check if it fits
vram_available = 4.0
safety_margin = 0.9  # Use 90% of VRAM
fits = memory["total_gb"] < (vram_available * safety_margin)
print(f"\nFits in 4GB VRAM (with 10% headroom): {'✓ YES' if fits else '✗ NO'}")

if fits:
    print("\n" + "=" * 60)
    print("🎉 14B MODEL CAN RUN ON 4GB VRAM WITH QLORA!")
    print("=" * 60)
else:
    print(f"\nNeeds ~{memory['total_gb'] - vram_available:.1f}GB more")
    print("Consider using a slightly smaller model (7B-10B)")

