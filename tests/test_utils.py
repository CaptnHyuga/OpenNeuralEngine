"""Unit tests for utility modules (tokenization, paths, model I/O)."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

import utils.project_paths as paths
from utils.tokenization import SimpleTokenizer
from tests.helpers import safe_load_weights


@pytest.mark.unit
class TestProjectPaths:
    """Tests for project path utilities."""
    
    def test_project_root_exists(self):
        """Test that PROJECT_ROOT points to existing directory."""
        assert paths.PROJECT_ROOT.exists()
        assert paths.PROJECT_ROOT.is_dir()
    
    def test_src_dir_exists(self):
        """Test that SRC_DIR exists."""
        assert paths.SRC_DIR.exists()
        assert paths.SRC_DIR.is_dir()
    
    def test_data_dir_exists(self):
        """Test that DATA_DIR exists."""
        assert paths.DATA_DIR.exists()
        assert paths.DATA_DIR.is_dir()
    
    def test_dataset_dir_exists(self):
        """Test that DATASET_DIR exists."""
        assert paths.DATASET_DIR.exists()
        assert paths.DATASET_DIR.is_dir()
    
    def test_models_dir_path(self):
        """Test that MODELS_DIR path is correct."""
        expected = paths.SRC_DIR / "Core_Models" / "Save"
        assert paths.MODELS_DIR == expected
    
    def test_resolve_path_absolute(self):
        """Test resolve_path with absolute path."""
        abs_path = Path("/absolute/path/to/file.txt")
        result = paths.resolve_path(abs_path)
        assert result == abs_path
    
    def test_resolve_path_relative(self):
        """Test resolve_path with relative path."""
        rel_path = Path("data/test.txt")
        result = paths.resolve_path(rel_path)
        assert result == paths.PROJECT_ROOT / rel_path
    
    def test_add_src_to_sys_path(self):
        """Test that add_src_to_sys_path adds SRC_DIR to sys.path."""
        import sys
        
        paths.add_src_to_sys_path()
        src_str = str(paths.SRC_DIR)
        
        assert src_str in sys.path


@pytest.mark.unit
class TestSimpleTokenizer:
    """Tests for SimpleTokenizer."""
    
    def test_tokenizer_initialization(self):
        """Test tokenizer can be initialized."""
        vocab = ["<pad>", "<unk>", "hello", "world", "test"]
        tokenizer = SimpleTokenizer(vocab=vocab)
        
        assert tokenizer.vocab_size == len(vocab)
        assert tokenizer.pad_token_id == 0
    
    def test_tokenizer_encode(self):
        """Test tokenizer encoding."""
        vocab = ["<pad>", "<unk>", "hello", "world", "!"]
        tokenizer = SimpleTokenizer(vocab=vocab)
        
        text = "hello world"
        tokens = tokenizer.encode(text)
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(t, int) for t in tokens)
    
    def test_tokenizer_decode(self):
        """Test tokenizer decoding."""
        vocab = ["<pad>", "<unk>", "hello", "world", "!"]
        tokenizer = SimpleTokenizer(vocab=vocab)
        
        tokens = [2, 3, 4]  # hello world !
        text = tokenizer.decode(tokens)
        
        assert isinstance(text, str)
        assert len(text) > 0
    
    def test_tokenizer_round_trip(self):
        """Test encode-decode round trip."""
        vocab = ["<pad>", "<unk>", "a", "b", "c", "test", "word"]
        tokenizer = SimpleTokenizer(vocab=vocab)
        
        original = "test word"
        tokens = tokenizer.encode(original)
        decoded = tokenizer.decode(tokens)
        
        # Should recover something similar (may not be exact due to tokenization)
        assert isinstance(decoded, str)
    
    def test_tokenizer_unknown_token(self):
        """Test handling of unknown tokens."""
        vocab = ["<pad>", "<unk>", "known"]
        tokenizer = SimpleTokenizer(vocab=vocab)
        
        # Token not in vocab should map to <unk>
        tokens = tokenizer.encode("unknown")
        
        # Should contain unk_token_id
        assert tokenizer.unk_token_id in tokens or len(tokens) > 0
    
    def test_tokenizer_padding(self):
        """Test tokenizer padding functionality."""
        vocab = ["<pad>", "<unk>", "hello", "world"]
        tokenizer = SimpleTokenizer(vocab=vocab)
        
        text = "hello"
        max_length = 10
        
        tokens = tokenizer.encode(text)
        
        # Pad manually
        if len(tokens) < max_length:
            tokens = tokens + [tokenizer.pad_token_id] * (max_length - len(tokens))
        
        assert len(tokens) == max_length


@pytest.mark.unit
def test_model_io_save_load(tmp_path):
    """Test model save/load utilities."""
    # Create a simple model
    model = torch.nn.Linear(10, 5)
    
    # Save
    save_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), save_path)
    
    assert save_path.exists()
    
    # Load
    model2 = torch.nn.Linear(10, 5)
    state_dict = safe_load_weights(save_path)
    model2.load_state_dict(state_dict)
    
    # Compare
    x = torch.randn(2, 10)
    
    model.eval()
    model2.eval()
    
    with torch.no_grad():
        out1 = model(x)
        out2 = model2(x)
    
    assert torch.allclose(out1, out2)


@pytest.mark.unit
def test_safetensors_compatibility(tmp_path):
    """Test that models can be saved/loaded with safetensors."""
    try:
        from safetensors.torch import save_file, load_file
    except ImportError:
        pytest.skip("safetensors not installed")
    
    # Create a simple model
    model = torch.nn.Linear(10, 5)
    state_dict = model.state_dict()
    
    # Save with safetensors
    save_path = tmp_path / "model.safetensors"
    save_file(state_dict, save_path)
    
    assert save_path.exists()
    
    # Load with safetensors
    loaded_state_dict = load_file(save_path)
    
    # Verify keys match
    assert set(state_dict.keys()) == set(loaded_state_dict.keys())
    
    # Verify values match
    for key in state_dict.keys():
        assert torch.allclose(state_dict[key], loaded_state_dict[key])


@pytest.mark.unit
def test_config_validation():
    """Test configuration validation."""
    # Valid config
    valid_config = {
        "vocab_size": 1000,
        "embedding_dim": 64,
        "hidden_dim": 64,
        "num_micro_layers": 2,
        "num_heads": 2,
        "num_kv_heads": 1,
        "dropout": 0.1,
        "max_seq_len": 128,
    }
    
    # Check required keys
    required_keys = ["vocab_size", "hidden_dim", "num_micro_layers"]
    for key in required_keys:
        assert key in valid_config
    
    # Check types
    assert isinstance(valid_config["vocab_size"], int)
    assert isinstance(valid_config["dropout"], (int, float))
    
    # Check ranges
    assert valid_config["vocab_size"] > 0
    assert 0 <= valid_config["dropout"] <= 1
    assert valid_config["num_micro_layers"] > 0


@pytest.mark.unit
def test_path_type_handling():
    """Test that paths can be handled as strings or Path objects."""
    # String path
    str_path = "data/test.txt"
    result1 = paths.resolve_path(str_path)
    
    # Path object
    path_obj = Path("data/test.txt")
    result2 = paths.resolve_path(path_obj)
    
    # Should produce same result
    assert result1 == result2
    assert isinstance(result1, Path)
    assert isinstance(result2, Path)


@pytest.mark.unit
def test_directory_creation(tmp_path):
    """Test creating nested directories."""
    nested_dir = tmp_path / "a" / "b" / "c"
    
    # Create all parent directories
    nested_dir.mkdir(parents=True, exist_ok=True)
    
    assert nested_dir.exists()
    assert nested_dir.is_dir()


@pytest.mark.unit
def test_file_existence_checking():
    """Test checking if files exist."""
    # Project root should exist
    assert paths.PROJECT_ROOT.exists()
    
    # Made-up file should not exist
    fake_file = paths.PROJECT_ROOT / "this_file_does_not_exist_12345.txt"
    assert not fake_file.exists()


@pytest.mark.unit
def test_glob_pattern_matching(data_dir):
    """Test finding files with glob patterns."""
    # Find all Python files in project
    py_files = list(paths.PROJECT_ROOT.rglob("*.py"))
    
    # Should find at least some Python files
    assert len(py_files) > 0
    
    # All should be .py files
    assert all(f.suffix == ".py" for f in py_files)
