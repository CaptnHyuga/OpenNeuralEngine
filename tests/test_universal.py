"""Test Universal Optimizer integration."""
import torch
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_universal_config():
    """Test that universal config works with batched trainer."""
    # Load universal config
    config_path = Path("models/phi-4/universal_config.json")
    with open(config_path) as f:
        config = json.load(f)

    print("Universal Config:")
    print(f"  Chunk size: {config['config']['chunk_size']}")
    print(f"  Batch size: {config['config']['batch_size']}")
    print(f"  Estimated: {config['config']['estimated_time_per_sample_ms']:.0f}ms/sample")
    print(f"  Active layers: {config['config']['active_layers']}")
    print()

    # Test with batched sparse trainer
    from src.wrappers.batched_sparse import BatchedSparseTrainer

    trainer = BatchedSparseTrainer(
        model_path="models/phi-4",
        chunk_size=config["config"]["chunk_size"],
        sparse_layers=config["config"]["active_layers"],
    )
    
    # Get batch size from config
    batch_size = config["config"]["batch_size"]

    # Test batch
    data = [{"input": f"test {i}", "output": f"result {i}"} for i in range(8)]
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    print("Testing real training...")
    start.record()
    loss = trainer.train_step(data)
    end.record()
    torch.cuda.synchronize()

    elapsed = start.elapsed_time(end)
    print(f"  Loss: {loss:.4f}")
    print(f"  Total time: {elapsed:.0f}ms")
    print(f"  Per sample: {elapsed/8:.0f}ms")


if __name__ == "__main__":
    test_universal_config()
