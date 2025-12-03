# OpenNeuralEngine Test Suite

Comprehensive pytest-based test suite for the OpenNeuralEngine project.

## Quick Start

```bash
# Install testing dependencies
pip install pytest pytest-cov pytest-xdist

# Run all tests
pytest

# Or use the test runner
python run_tests.py --all
```

## Test Organization

### Test Files

- **`test_layers.py`** - Unit tests for custom neural network layers (SwiGLU, RMSNorm, RoPE, GQA)
- **`test_models.py`** - Tests for model architectures (PuzzleModel, Multimodal models)
- **`test_training.py`** - Integration tests for training functionality
- **`test_inference.py`** - Tests for inference and generation
- **`test_utils.py`** - Tests for utility modules (tokenization, paths, I/O)
- **`test_api.py`** - API endpoint tests (legacy)
- **`test_comprehensive.py`** - Comprehensive integration tests (legacy)

### Test Markers

Tests are organized with pytest markers for selective execution:

- `@pytest.mark.unit` - Fast unit tests for individual components
- `@pytest.mark.integration` - Integration tests across multiple modules
- `@pytest.mark.slow` - Tests that take longer to run (>5s)
- `@pytest.mark.gpu` - Tests that require GPU/CUDA
- `@pytest.mark.model` - Model architecture tests
- `@pytest.mark.training` - Training functionality tests
- `@pytest.mark.inference` - Inference functionality tests
- `@pytest.mark.multimodal` - Multimodal feature tests

## Running Tests

### Basic Usage

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_layers.py

# Run specific test class
pytest tests/test_layers.py::TestSwiGLU

# Run specific test function
pytest tests/test_layers.py::TestSwiGLU::test_swiglu_forward
```

### Using Test Runner Script

```bash
# Run all tests
python run_tests.py --all

# Run only unit tests (fast)
python run_tests.py --unit

# Run integration tests
python run_tests.py --integration

# Skip slow tests
python run_tests.py --fast

# Run with coverage
python run_tests.py --coverage

# Run specific file
python run_tests.py --file tests/test_models.py

# Verbose output
python run_tests.py --all -v

# Stop on first failure
python run_tests.py --all -x

# Run in parallel (4 workers)
python run_tests.py --all --parallel 4
```

### Selective Test Execution

```bash
# Run only unit tests
pytest -m unit

# Run only GPU tests
pytest -m gpu

# Run all except slow tests
pytest -m "not slow"

# Run unit tests but skip slow ones
pytest -m "unit and not slow"

# Run training or inference tests
pytest -m "training or inference"
```

## Coverage Reports

```bash
# Generate coverage report
pytest --cov=src --cov-report=html --cov-report=term-missing

# View HTML report
# Open htmlcov/index.html in browser

# Or use test runner
python run_tests.py --coverage
```

## Test Configuration

### pytest.ini

Configuration file with:
- Test discovery patterns
- Default command-line options
- Marker definitions
- Coverage settings
- Warning filters

### conftest.py

Shared fixtures and configuration:
- `device` - Auto-detect best device (cuda/mps/cpu)
- `model_dir` - Path to model checkpoints
- `data_dir` - Path to data directory
- `small_batch` - Synthetic batch for testing
- `tiny_model_config` - Minimal config for fast tests
- Helper fixtures for shape/dtype/device assertions
- Skip conditions for GPU/model requirements

## Writing New Tests

### Test File Template

```python
"""Description of test module."""
from __future__ import annotations

import pytest
import torch

from Core_Models.your_module import YourClass


@pytest.mark.unit  # or integration, slow, etc.
class TestYourClass:
    """Tests for YourClass."""
    
    def test_initialization(self):
        """Test class can be initialized."""
        obj = YourClass(param=value)
        assert obj is not None
    
    def test_forward_pass(self, random_seed):
        """Test forward pass with fixtures."""
        obj = YourClass(hidden_dim=64)
        x = torch.randn(2, 16, 64)
        output = obj(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
```

### Using Fixtures

```python
def test_with_fixtures(device, small_batch, assert_shape):
    """Example using common fixtures."""
    model = MyModel().to(device)
    
    output = model(small_batch["input_ids"])
    
    # Use assert_shape helper
    assert_shape(output, (2, 16, 1000), "model output")
```

### Parametrized Tests

```python
@pytest.mark.parametrize("hidden_dim", [64, 128, 256])
def test_different_sizes(hidden_dim):
    """Test with multiple parameter values."""
    layer = MyLayer(hidden_dim=hidden_dim)
    x = torch.randn(1, 10, hidden_dim)
    output = layer(x)
    assert output.shape == x.shape
```

## Continuous Integration

Tests can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    pip install -r requirements.txt
    pytest -m "not slow and not gpu" --cov=src
```

## Test Best Practices

1. **Keep tests fast** - Use `tiny_model_config` for unit tests
2. **Mark slow tests** - Use `@pytest.mark.slow` for long-running tests
3. **Use fixtures** - Leverage conftest.py fixtures for common setup
4. **Test edge cases** - Include tests for boundary conditions
5. **Check for NaN/Inf** - Always verify numerical stability
6. **Use meaningful names** - Test names should describe what they test
7. **One assertion concept per test** - Keep tests focused
8. **Clean up resources** - Use `tmp_path` fixture for temporary files

## Troubleshooting

### Tests fail with CUDA errors
- Check GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`
- Run CPU-only tests: `pytest -m "not gpu"`

### Tests are slow
- Run only unit tests: `pytest -m unit`
- Skip slow tests: `pytest -m "not slow"`
- Use parallel execution: `pytest -n auto`

### Import errors
- Ensure virtual environment is activated
- Install test dependencies: `pip install -r requirements.txt`
- Check Python path includes project root

### Coverage reports incomplete
- Make sure all source files are in `src/` directory
- Check `.coveragerc` configuration
- Run with `--cov-report=term-missing` to see missing lines

## Contributing

When adding new features:
1. Write tests first (TDD approach)
2. Ensure tests pass: `pytest`
3. Check coverage: `pytest --cov=src`
4. Add appropriate markers
5. Update this README if adding new test categories

## Resources

- [Pytest documentation](https://docs.pytest.org/)
- [Pytest fixtures](https://docs.pytest.org/en/stable/fixture.html)
- [Pytest markers](https://docs.pytest.org/en/stable/mark.html)
- [Coverage.py](https://coverage.readthedocs.io/)
